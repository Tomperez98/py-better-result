# Async traversal

Apply an asynchronous `Result`-returning operation to a collection while
preserving input order and limiting concurrent work.

## Run it

From the repository root:

```bash
uv run better-result-example async-traversal
```

Expected output:

```text
Err(value='user 404 not found')
```

## How it works

`traverse_async` invokes `fetch_user` for every ID and runs at most two calls
at once. Results are collected in input order, so the returned domain error is
based on the input sequence rather than completion timing.

An `Err` is a domain result, not an exception: sibling operations are not
silently cancelled when one operation returns an `Err`. Use the cancellation
facilities in `try_async` when cancellation is part of the operation contract.

The complete flow is in [`main.py`](main.py).
