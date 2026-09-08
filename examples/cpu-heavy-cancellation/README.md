# CPU-heavy cancellation

Cancel CPU-heavy asynchronous work at explicit chunk boundaries.

## Run it

From the repository root:

```bash
uv run better-result-example cpu-heavy-cancellation
```

Expected output:

```text
CPU work cancelled at a chunk boundary
```

## How it works

`compute_checksum` processes the input in bounded chunks. After each chunk it
checks `context.cancel_token` and yields with `asyncio.sleep(0)`. Those two
steps give cancellation a predictable place to be observed even though the
actual checksum calculation is synchronous CPU work.

For ordinary awaitable operations, `try_async` can cancel the in-flight task
itself. CPU-bound code still needs yield points (and, when useful, explicit
token checks) so it does not monopolize the event loop. The complete flow is in
[`main.py`](main.py).
