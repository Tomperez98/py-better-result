# Async workflows

Compose asynchronous operations and collect several results concurrently while
preserving input order.

## Run it

From the repository root:

```bash
uv run better-result-example async-workflows
```

Expected output:

```text
Err(value='user 404 not found')
```

## How it works

- `and_then_async` awaits `fetch_user` only for an `Ok(user_id)`.
- `map_async` awaits the title-formatting callback only when the user exists.
- `all_results_async` accepts awaitables, runs them concurrently, and returns
  the first error in input order. The request for user `404` therefore makes
  the whole collection an `Err`, even though the other requests succeed.

Change the `404` entry in `main.py` to a valid ID to see the successful tuple
of names. The complete flow is in [`main.py`](main.py).
