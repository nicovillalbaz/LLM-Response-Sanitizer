import pytest

from llm_json import parse_llm_json


def assert_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        parse_llm_json(text)


def test_parses_clean_valid_object_json() -> None:
    assert parse_llm_json('{"name": "Ada", "active": true, "score": 3}') == {
        "name": "Ada",
        "active": True,
        "score": 3,
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('```json\n{"answer": 42}\n```', {"answer": 42}),
        ('```\n{"answer": 42}\n```', {"answer": 42}),
        ('\n  ```json\n{"answer": 42}\n```\t\n', {"answer": 42}),
    ],
    ids=[
        "json_fence",
        "plain_fence",
        "fence_with_surrounding_whitespace",
    ],
)
def test_strips_complete_fenced_block(text: str, expected: dict) -> None:
    assert parse_llm_json(text) == expected


def test_repairs_fenced_json_with_trailing_comma() -> None:
    text = '```json\n{"answer": 42,}\n```'

    assert parse_llm_json(text) == {"answer": 42}


def test_repairs_fenced_json_with_trailing_comma_and_truncation() -> None:
    text = '```json\n{"items": [1, 2,], "meta": {"count": 2}\n```'

    assert parse_llm_json(text) == {
        "items": [1, 2],
        "meta": {"count": 2},
    }


def test_removes_trailing_comma_in_object() -> None:
    assert parse_llm_json('{"a": 1,}') == {"a": 1}


def test_removes_trailing_commas_in_nested_arrays_and_objects() -> None:
    text = '{"items": [{"id": 1,}, {"id": 2,},], "meta": {"count": 2,},}'

    assert parse_llm_json(text) == {
        "items": [{"id": 1}, {"id": 2}],
        "meta": {"count": 2},
    }


def test_removes_trailing_commas_with_newlines_and_indentation() -> None:
    text = """
    {
      "items": [
        {
          "id": 1,
        },
        {
          "id": 2,
        },
      ],
      "count": 2,
    }
    """

    assert parse_llm_json(text) == {
        "items": [{"id": 1}, {"id": 2}],
        "count": 2,
    }


def test_preserves_commas_inside_strings() -> None:
    text = '{"text": "keep , } and , ] inside strings", "ok": true,}'

    assert parse_llm_json(text) == {
        "text": "keep , } and , ] inside strings",
        "ok": True,
    }


def test_repairs_truncation_by_appending_missing_closing_braces() -> None:
    text = '{"outer": {"inner": {"value": 1}'

    assert parse_llm_json(text) == {"outer": {"inner": {"value": 1}}}


def test_repairs_truncation_by_appending_missing_closing_brackets() -> None:
    text = '{"items": [[1, 2], [3, 4]'

    assert parse_llm_json(text) == {"items": [[1, 2], [3, 4]]}


@pytest.mark.parametrize(
    "text",
    [
        '{"message": "hello}',
        '{"message": "hello\\',
    ],
    ids=["unterminated_quote", "dangling_escape"],
)
def test_incomplete_string_fails(text: str) -> None:
    assert_rejected(text)


@pytest.mark.parametrize(
    "text",
    ['{"ok": tru}', '{"ok": fal}', '{"ok": nul}'],
    ids=["tru", "fal", "nul"],
)
def test_incomplete_literal_fails(text: str) -> None:
    assert_rejected(text)


@pytest.mark.parametrize(
    "text",
    [
        '{"n": 1e}',
        '{"n": 1e-}',
        '{"n": 1.}',
        '{"n": -}',
    ],
    ids=["missing_exponent_digits", "dangling_exponent_sign", "dangling_decimal", "bare_minus"],
)
def test_incomplete_number_fails(text: str) -> None:
    assert_rejected(text)


def test_invalid_number_with_leading_zero_fails() -> None:
    assert_rejected('{"n": 01}')


@pytest.mark.parametrize(
    "text",
    [
        '{"answer": }',
        '{"answer": ',
        '{"items": [1, , 2]}',
        '{"items": [1,',
        '{"a": 1,',
    ],
    ids=[
        "object_value_missing_before_closer",
        "object_value_missing_at_eof",
        "array_value_missing_between_commas",
        "array_value_missing_at_eof",
        "object_key_missing_after_comma_at_eof",
    ],
)
def test_missing_value_or_ambiguous_truncation_fails(text: str) -> None:
    assert_rejected(text)


@pytest.mark.parametrize(
    "text",
    [
        '{"items": [1 2]}',
        '{"a" "b"}',
        '{"a": 1 "b": 2}',
    ],
    ids=[
        "missing_comma_in_array",
        "missing_colon_in_object",
        "missing_comma_between_object_members",
    ],
)
def test_malformed_internal_structure_fails(text: str) -> None:
    assert_rejected(text)


def test_extra_trailing_non_whitespace_data_fails() -> None:
    assert_rejected('{"a": 1} trailing')


@pytest.mark.parametrize(
    "text",
    ['["a", "b"]', '"value"', "123", "true", "null"],
    ids=["array", "string", "number", "true", "null"],
)
def test_non_dict_json_root_fails(text: str) -> None:
    assert_rejected(text)


def test_unmatched_extra_closing_brace_fails() -> None:
    assert_rejected('{"a": 1}}')


def test_unmatched_extra_closing_bracket_fails() -> None:
    assert_rejected('{"items": [1, 2]]}')


def test_mixed_prose_plus_fenced_block_fails() -> None:
    text = 'Here is the JSON:\n```json\n{"a": 1}\n```'

    assert_rejected(text)


def test_incomplete_fenced_block_fails() -> None:
    text = '```json\n{"a": 1,}\n'

    assert_rejected(text)


@pytest.mark.parametrize("text", ["", "   \n\t  "], ids=["empty", "whitespace_only"])
def test_empty_or_whitespace_only_input_fails(text: str) -> None:
    assert_rejected(text)
