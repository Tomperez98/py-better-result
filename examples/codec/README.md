# Codec boundaries

Encode and decode `Result` values at a JSON-like application boundary while
keeping schema failures typed.

## Run it

From the repository root:

```bash
uv run --package example-codec python examples/codec/main.py
```

Expected output:

```text
Ok(value={'status': 'ok', 'value': {'id': 42, 'name': 'Ada'}})
Ok(value=User(user_id=42, name='Ada'))
Err(value='not_found')
Err(value=ResultDeserializationError(value={'status': 'ok', 'value': {'id': 'wrong'}}, issues=[...]))
Ok(value={'status': 'error', 'error': {'code': 'not_found'}})
```

The exact representation of `ResultDeserializationError` can include its
structured issue details.

## How it works

The codec has one serializer and deserializer for the success branch and one
for the error branch. A valid `Ok(User)` becomes an envelope with `status` and
`value`; a wire-level error decodes back to `Err`. A malformed user envelope
returns a `ResultDeserializationError` instead of raising, and serialization
uses the same `Result` shape for schema failures.

The complete flow is in [`main.py`](main.py).
