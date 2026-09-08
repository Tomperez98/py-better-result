# Retry

Retry an expected exception with a bounded, typed exponential-backoff policy.

## Run it

From the repository root:

```bash
uv run --package example-retry python examples/retry/main.py
```

Expected output:

```text
Ok(value='response body')
attempts=3
RetryPolicy(times=2, schedule=DynamicDelay(function=<function main.<locals>.delay_for at 0x...>), should_retry=None)
```

The function address in the final line varies between runs.

## How it works

`request` fails twice, then succeeds. `try_result` catches the exceptions as
strings, and the policy permits up to three retries while the mapped error
contains `temporary`. With `initial_delay=0`, the example runs immediately but
still demonstrates the attempt count and policy boundary.

The final policy uses a dynamic schedule to show how a delay can depend on the
retry context. The complete flow is in [`main.py`](main.py).
