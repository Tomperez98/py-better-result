# Recovery and error translation

Translate a driver error into a domain error, then recover from a primary-store failure with a cache.

## Run it

From the repository root:

```bash
uv run better-result-example recovery
```

Expected output:

```text
loaded Ada
1: Ok(value=User(user_id=1, name='Ada'))
loaded Ada (cached)
503: Ok(value=User(user_id=503, name='Ada (cached)'))
load failed: user 404 was not found
404: Err(value=UserNotFound(user_id=404))
```

## How it works

`map_err` keeps the driver vocabulary inside the repository seam. `or_else`
runs only after the primary result is an `Err`, allowing the cache to recover
the temporary outage without hiding a not-found error. `inspect` and
`inspect_err` observe the active branch without changing the result.

The complete flow is in [`main.py`](main.py).
