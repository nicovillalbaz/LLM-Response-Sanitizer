# Option C - LLM Response Sanitizer

This exercise implements `parse_llm_json(text: str) -> dict` for JSON returned
by an LLM when the response is close to valid JSON but may contain a small set
of predictable formatting issues.

## What `parse_llm_json` does

`parse_llm_json` accepts a string, applies a conservative preprocessing and
repair pass, then parses the result with the Python standard library JSON
decoder.

The function accepts only JSON objects at the root and raises `ValueError` for
everything else.

## What it repairs

The parser allows only safe structural repairs:

- strip a surrounding markdown fenced block when the entire input is exactly one
  complete fenced block
- remove trailing commas only when they appear outside strings and the next
  non-whitespace character is `}` or `]`
- repair truncation only by appending missing `}` and `]` when the structure
  before EOF is already valid and complete apart from those closing delimiters

## What it intentionally does not repair

The parser rejects inputs that would require guessing the author's intent. It
does not repair:

- incomplete strings
- incomplete literals such as `tru`, `fal`, or `nul`
- incomplete or invalid numbers such as `1e`, `1e-`, `1.`, or `-`
- missing values, missing keys, or missing punctuation
- malformed internal structure
- extra trailing data after a valid root object
- unmatched extra closing delimiters
- mixed prose plus a fenced JSON block
- valid JSON whose root is not an object

## Conservative repair philosophy

The goal is not to "fix anything that looks close enough." The goal is to make
only repairs that are unambiguous and structural. If recovery would require
inventing tokens, values, strings, keys, or literals, the function raises
`ValueError` instead.

## Running the tests

Install the development dependency:

```bash
python -m pip install -r requirements-dev.txt
```

Run the test suite:

```bash
pytest
```
