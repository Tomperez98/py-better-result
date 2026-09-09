# Thread offload

Use `asyncio.to_thread` with an async `Result` combinator when synchronous work
would otherwise run on and block the event-loop thread.

## Run it

From the repository root:

```bash
uv run better-result-example thread-offload
```

Expected output:

```text
Ok(value=4800196)
```

## How it works

`map_async` accepts a callback returning an awaitable. The example adapts the
synchronous `checksum` function with `asyncio.to_thread`, then `map_async`
awaits it and wraps the value back in `Ok`.

The `threading.Event` gate makes the responsiveness check deterministic: the
worker signals that it started and waits for the event-loop coroutine to
release it. The assertion confirms that the `Result` task is still pending
while the worker is waiting. Calling `checksum` directly on the event-loop
thread would block before reaching that assertion.

`to_thread` is useful for keeping the event loop responsive, but it does not
usually provide true parallelism for CPU-heavy pure-Python code because of the
GIL. For CPU work that must run in parallel, use a process pool instead. The
complete flow is in [`main.py`](main.py).
