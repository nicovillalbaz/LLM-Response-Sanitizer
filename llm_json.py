"""Conservative sanitizer for JSON produced by LLMs.

The parser intentionally supports only a small set of safe repairs:

- stripping a surrounding markdown fence when the entire input is one
  complete fenced block
- removing trailing commas that appear outside strings immediately before
  ``}`` or ``]`` (ignoring intervening whitespace)
- appending missing closing braces/brackets when the existing prefix is
  otherwise structurally complete

It does not attempt speculative recovery for incomplete strings, numbers,
literals, keys, or values.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Final

__all__ = ["parse_llm_json"]

_NUMBER_RE: Final[re.Pattern[str]] = re.compile(
    r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
)
_FENCED_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    r"\A```[^\r\n]*\r?\n(?P<body>.*)\r?\n```\Z",
    re.DOTALL,
)
_JSON_LITERALS: Final[tuple[str, ...]] = ("true", "false", "null")


class _RootState:
    EXPECT_VALUE = "expect_value"
    EXPECT_END = "expect_end"


class _ContainerKind:
    OBJECT = "object"
    ARRAY = "array"


class _ObjectState:
    EXPECT_KEY_OR_END = "expect_key_or_end"
    EXPECT_KEY = "expect_key"
    EXPECT_COLON = "expect_colon"
    EXPECT_VALUE = "expect_value"
    EXPECT_COMMA_OR_END = "expect_comma_or_end"


class _ArrayState:
    EXPECT_VALUE_OR_END = "expect_value_or_end"
    EXPECT_VALUE = "expect_value"
    EXPECT_COMMA_OR_END = "expect_comma_or_end"


_RECOVERABLE_OBJECT_STATES: Final[frozenset[str]] = frozenset(
    {
        _ObjectState.EXPECT_KEY_OR_END,
        _ObjectState.EXPECT_COMMA_OR_END,
    }
)
_RECOVERABLE_ARRAY_STATES: Final[frozenset[str]] = frozenset(
    {
        _ArrayState.EXPECT_VALUE_OR_END,
        _ArrayState.EXPECT_COMMA_OR_END,
    }
)


class _RepairError(ValueError):
    """Raised when the input cannot be repaired conservatively."""


@dataclass(slots=True)
class _Frame:
    kind: str
    state: str


def parse_llm_json(text: str) -> dict:
    """
    Parse LLM-produced JSON using only conservative, structural repairs.

    Allowed repairs:
    - strip a surrounding markdown code fence when the entire input is one
      complete fenced block
    - remove a trailing comma only when its next non-whitespace character is
      ``}`` or ``]``
    - append missing closing braces/brackets when the existing prefix is
      already structurally complete
    """

    prepared = _prepare_input(text)

    try:
        candidate = _StructuralRepairer(prepared).repair()
    except _RepairError as exc:
        raise ValueError(str(exc)) from None

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Input is not valid JSON after conservative repair: {exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object at the root")

    return data


def _prepare_input(text: str) -> str:
    prepared = _strip_complete_fenced_block(text)
    prepared = _remove_trailing_commas(prepared).strip()

    if not prepared:
        raise ValueError("Expected a JSON object")

    return prepared


def _strip_complete_fenced_block(text: str) -> str:
    stripped = text.strip()
    match = _FENCED_BLOCK_RE.fullmatch(stripped)
    if match is None:
        return stripped
    return match.group("body").strip()


def _remove_trailing_commas(text: str) -> str:
    pieces: list[str] = []
    index = 0

    while index < len(text):
        char = text[index]

        if char == '"':
            # Once a string starts, commas inside it are always literal content.
            # If the string never closes, the structural pass rejects it later.
            string_end = _scan_string_token(text, index, allow_unterminated=True)
            pieces.append(text[index:string_end])
            index = string_end
            continue

        next_char = _next_non_whitespace_char(text, index + 1)
        if char == "," and next_char and next_char in "}]":
            index += 1
            continue

        pieces.append(char)
        index += 1

    return "".join(pieces)


def _next_non_whitespace_char(text: str, index: int) -> str:
    while index < len(text) and text[index].isspace():
        index += 1
    return text[index] if index < len(text) else ""


def _is_value_terminator(char: str) -> bool:
    return not char or char.isspace() or char in ",]}"


def _scan_string_token(text: str, start: int, *, allow_unterminated: bool) -> int:
    index = start + 1

    while index < len(text):
        char = text[index]

        if char == "\\":
            index += 1
            if index >= len(text):
                if allow_unterminated:
                    return len(text)
                raise _RepairError("Unterminated string literal")
            index += 1
            continue

        if char == '"':
            return index + 1

        index += 1

    if allow_unterminated:
        return len(text)
    raise _RepairError("Unterminated string literal")


class _StructuralRepairer:
    """Validate structure and append only unambiguous closing delimiters."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0
        self.root_state = _RootState.EXPECT_VALUE
        self.stack: list[_Frame] = []

    def repair(self) -> str:
        while True:
            self._skip_whitespace()

            if self._at_end():
                break

            if not self.stack:
                self._step_root()
                continue

            frame = self.stack[-1]
            if frame.kind == _ContainerKind.OBJECT:
                self._step_object(frame)
            else:
                self._step_array(frame)

        return self._finish()

    def _at_end(self) -> bool:
        return self.index >= len(self.text)

    def _current_char(self) -> str:
        return self.text[self.index]

    def _skip_whitespace(self) -> None:
        while not self._at_end() and self._current_char().isspace():
            self.index += 1

    def _step_root(self) -> None:
        if self.root_state != _RootState.EXPECT_VALUE:
            raise _RepairError("Extra data after the top-level JSON value")
        self._consume_value()

    def _step_object(self, frame: _Frame) -> None:
        char = self._current_char()

        if frame.state == _ObjectState.EXPECT_KEY_OR_END:
            if char == "}":
                self._close_container()
                return
            if char != '"':
                raise _RepairError("Expected a string key or '}' in object")
            self.index = _scan_string_token(
                self.text,
                self.index,
                allow_unterminated=False,
            )
            frame.state = _ObjectState.EXPECT_COLON
            return

        if frame.state == _ObjectState.EXPECT_KEY:
            if char != '"':
                raise _RepairError("Expected a string key after ',' in object")
            self.index = _scan_string_token(
                self.text,
                self.index,
                allow_unterminated=False,
            )
            frame.state = _ObjectState.EXPECT_COLON
            return

        if frame.state == _ObjectState.EXPECT_COLON:
            if char != ":":
                raise _RepairError("Expected ':' after object key")
            frame.state = _ObjectState.EXPECT_VALUE
            self.index += 1
            return

        if frame.state == _ObjectState.EXPECT_VALUE:
            self._consume_value()
            return

        if char == ",":
            frame.state = _ObjectState.EXPECT_KEY
            self.index += 1
            return

        if char == "}":
            self._close_container()
            return

        raise _RepairError("Expected ',' or '}' after object value")

    def _step_array(self, frame: _Frame) -> None:
        char = self._current_char()

        if frame.state == _ArrayState.EXPECT_VALUE_OR_END:
            if char == "]":
                self._close_container()
                return
            self._consume_value()
            return

        if frame.state == _ArrayState.EXPECT_VALUE:
            self._consume_value()
            return

        if char == ",":
            frame.state = _ArrayState.EXPECT_VALUE
            self.index += 1
            return

        if char == "]":
            self._close_container()
            return

        raise _RepairError("Expected ',' or ']' after array value")

    def _close_container(self) -> None:
        self.stack.pop()
        self.index += 1

    def _consume_value(self) -> None:
        if self._at_end():
            raise _RepairError("Input ended while a JSON value was still required")

        self._begin_value()
        char = self._current_char()

        if char == "{":
            self.stack.append(
                _Frame(_ContainerKind.OBJECT, _ObjectState.EXPECT_KEY_OR_END)
            )
            self.index += 1
            return

        if char == "[":
            self.stack.append(
                _Frame(_ContainerKind.ARRAY, _ArrayState.EXPECT_VALUE_OR_END)
            )
            self.index += 1
            return

        if char == '"':
            self.index = _scan_string_token(
                self.text,
                self.index,
                allow_unterminated=False,
            )
            return

        if char in "-0123456789":
            self.index = self._scan_number()
            return

        if char in "tfn":
            self.index = self._scan_literal()
            return

        raise _RepairError("Unexpected token where a JSON value was expected")

    def _begin_value(self) -> None:
        if not self.stack:
            if self.root_state != _RootState.EXPECT_VALUE:
                raise _RepairError("Extra data after the top-level JSON value")
            self.root_state = _RootState.EXPECT_END
            return

        frame = self.stack[-1]

        if frame.kind == _ContainerKind.OBJECT:
            if frame.state != _ObjectState.EXPECT_VALUE:
                raise _RepairError("Object is not in a position where a value is allowed")
            frame.state = _ObjectState.EXPECT_COMMA_OR_END
            return

        if frame.state not in {
            _ArrayState.EXPECT_VALUE_OR_END,
            _ArrayState.EXPECT_VALUE,
        }:
            raise _RepairError("Array is not in a position where a value is allowed")
        frame.state = _ArrayState.EXPECT_COMMA_OR_END

    def _scan_number(self) -> int:
        match = _NUMBER_RE.match(self.text, self.index)
        if match is None:
            raise _RepairError("Invalid or incomplete number")

        end = match.end()
        if end < len(self.text) and not _is_value_terminator(self.text[end]):
            raise _RepairError("Invalid or incomplete number")

        return end

    def _scan_literal(self) -> int:
        for literal in _JSON_LITERALS:
            if self.text.startswith(literal, self.index):
                end = self.index + len(literal)
                if end < len(self.text) and not _is_value_terminator(self.text[end]):
                    raise _RepairError("Invalid literal")
                return end

        remainder = self.text[self.index :]
        if any(literal.startswith(remainder) for literal in _JSON_LITERALS):
            raise _RepairError("Incomplete literal")

        raise _RepairError("Invalid literal")

    def _finish(self) -> str:
        if self.root_state != _RootState.EXPECT_END:
            raise _RepairError("Input ended before a top-level JSON value was complete")

        closers: list[str] = []
        for frame in reversed(self.stack):
            # Only append delimiters when every still-open container is already
            # in a state where closing it is the only missing structural step.
            if frame.kind == _ContainerKind.OBJECT:
                if frame.state not in _RECOVERABLE_OBJECT_STATES:
                    raise _RepairError(
                        "Object ended before the next key, colon, or value was complete"
                    )
                closers.append("}")
                continue

            if frame.state not in _RECOVERABLE_ARRAY_STATES:
                raise _RepairError("Array ended before the next value was complete")
            closers.append("]")

        return self.text + "".join(closers)
