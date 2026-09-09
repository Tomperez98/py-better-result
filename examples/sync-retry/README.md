# Synchronous retry

Exercise immediate, fixed, linear, exponential, predicate, and custom-schedule
retry policies.

```bash
uv run better-result-example sync-retry
```

The operation returns typed `Err` values until the policy permits a successful
attempt. Raised exceptions are not part of this API; expected failures are
returned as `Result` values.
