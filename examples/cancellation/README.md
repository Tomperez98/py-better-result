# Cancellation

Stop in-flight asynchronous work and retry waits with `CancellationToken`.

## Run it

From the repository root:

```bash
uv run better-result-example cancellation
```

Expected output:

```text
operation cancelled
retry wait cancelled
```

## How it works

The first operation waits for 60 seconds, but `try_async` monitors the token
while the operation is running. After the operation starts, `main` calls
`token.cancel()` and awaits the task. The token cancels the in-flight operation
and the example handles the resulting `asyncio.CancelledError`.

The second operation fails with a transient exception and enters a 60-second
retry wait. A separate task cancels its token after 10 milliseconds, so the
retry wait is interrupted instead of sleeping for the full delay.

Cancellation is separate from a domain `Err`: the operation did not finish
with an application-level failure; it was interrupted. The complete flow is in
[`main.py`](main.py).
