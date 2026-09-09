# Examples workspace

These examples mirror the Rust workspace examples and exercise the public
`py-better-result` API.

Run one example from the repository root:

```bash
uv run better-result-example basic
uv run better-result-example sync-retry
uv run better-result-example async-retry
```

List examples:

```bash
uv run better-result-example --list
```

Run every example:

```bash
uv run better-result-example --all
```

## Examples

- `basic`: the smallest synchronous retry workflow.
- `sync-retry`: immediate, fixed, linear, exponential, predicate, and custom
  schedule retries.
- `async-retry`: asynchronous retries and normal asyncio task cancellation.
