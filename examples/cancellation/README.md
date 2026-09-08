# Cancellation

Stop an in-flight asynchronous operation with `CancellationToken`.

## Run it

From the repository root:

```bash
uv run --package example-cancellation python examples/cancellation/main.py
```

Expected output:

```text
operation cancelled
```

## How it works

The operation waits for 60 seconds, but `try_async` monitors the token while
the operation is running. After the operation starts, `main` calls `token.cancel()`
and awaits the task. The token cancels the in-flight operation and the example
handles the resulting `asyncio.CancelledError`.

Cancellation is separate from a domain `Err`: the operation did not finish
with an application-level failure; it was interrupted. The complete flow is in
[`main.py`](main.py).
