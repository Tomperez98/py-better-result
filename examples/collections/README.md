# Result collections

Collect multiple `Result` values, retain both branches, accumulate every error,
or remove one nested result layer.

## Run it

From the repository root:

```bash
uv run better-result-example collections
```

Expected output:

```text
all_results: Err(value='bad input')
collect_results: Err(value=('bad input',))
partition_results: values=[1, 3], errors=['bad input']
flatten_result: Ok(value=42)
```

## How it works

- `all_results` returns one tuple when every item is `Ok`; otherwise it
  short-circuits to the first error.
- `collect_results` evaluates every item and returns all errors as a tuple,
  preserving input order. Use it for validation when every independent check
  should be reported.
- `partition_results` returns both success and error lists, preserving the
  relative order within each list.
- `flatten_result` turns `Ok(Ok(42))` into `Ok(42)` and joins nested error
  types when either layer fails.

The complete flow is in [`main.py`](main.py).