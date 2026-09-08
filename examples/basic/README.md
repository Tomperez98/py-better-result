# Basic workflow

Parse a user ID, load the user, and format a greeting without raising for
expected failures.

## Run it

From the repository root:

```bash
uv run --package example-basic python examples/basic/main.py
```

Expected output:

```text
Hello, Ada!
Could not load user: user id must be a number
Could not load user: user not found
using cached data (service unavailable)
```

## How it works

1. `parse_user_id` returns `Ok(int)` for decimal input and `Err(str)` for
   invalid input.
2. `and_then(load_user)` calls the loader only when parsing succeeds.
3. `map` formats the successful `User`; an existing `Err` passes through
   unchanged.
4. Pattern matching handles the final `Ok` or `Err` explicitly, while
   `unwrap_or_else` demonstrates choosing a fallback at a boundary.

The complete flow is in [`main.py`](main.py).
