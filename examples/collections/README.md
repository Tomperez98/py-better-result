# Result collections

Collect multiple `Result` values, retain both branches, or remove one nested
result layer.

## Run it

From the repository root:

```bash
uv run --package example-collections python examples/collections/main.py
```

Expected output:

```text
Err(value='bad input')
values=[1, 3], errors=['bad input']
Ok(value=42)
```

## How it works

- `all_results` returns one tuple when every item is `Ok`; otherwise it
  short-circuits to the first error.
- `partition_results` returns both success and error lists, preserving the
  relative order within each list.
- `flatten_result` turns `Ok(Ok(42))` into `Ok(42)` and joins nested error
  types when either layer fails.

The complete flow is in [`main.py`](main.py).
