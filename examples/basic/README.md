# Basic retry

Run the smallest synchronous `Result` retry workflow.

```bash
uv run better-result-example basic
```

The operation returns a temporary `Err` once, then succeeds under a fixed
zero-second retry policy.
