# Validation

Convert an expected parsing exception into a typed error, then apply domain
validation.

## Run it

From the repository root:

```bash
uv run --package example-validation python examples/validation/main.py
```

Expected output:

```text
'8080' -> Ok(value=8080)
'not-a-port' -> Err(value="invalid literal for int() with base 10: 'not-a-port'")
'70000' -> Err(value='port must be between 1 and 65535')
```

## How it works

`try_result` is the exception boundary around `int(raw)`. Its `catch=str`
mapper turns a `ValueError` into a string error. The returned `Result` then
uses `and_then(validate_port)` to keep range validation in a separate function.
This means parsing failures and domain failures have the same visible result
shape, while unexpected exceptions would still propagate.

The complete flow is in [`main.py`](main.py).
