# Async boundaries

Capture async failures, collect async results, map errors asynchronously, and
use an async codec at a wire boundary.

## Run it

From the repository root:

```bash
uv run better-result-example async-boundaries
```

Expected output:

```text
collected: Err(value=("score error: invalid literal for int() with base 10: 'bad'",))
partitioned: values=[10, 20], errors=["score error: invalid literal for int() with base 10: 'bad'"]
encoded: Ok(value={'status': 'ok', 'value': {'score': 42}})
decoded: Ok(value=42)
decoded error: Err(value='offline')
```

## How it works

`capture_async` turns the expected parsing exception into a `Result`.
`map_err_async` asynchronously enriches only the active error branch.
`collect_results_async` accumulates every error, while
`partition_results_async` returns successful values and errors separately.
Finally, `async_codec` keeps asynchronous serialization and deserialization
schema failures in the `Result` channel.

The complete flow is in [`main.py`](main.py).
