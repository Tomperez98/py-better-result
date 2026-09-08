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
Ok(value='dynamic response')
dynamic_attempts=3
dynamic_delays=[0.1, 0.2]
```

## How it works

`request` fails twice, then succeeds. `try_result` catches the exceptions as
strings, and the policy permits up to three retries while the mapped error
contains `temporary`. With a short `initial_delay=0.01`, the example also
executes the exponential schedule without making the example slow.

`dynamic_request` exercises a dynamic schedule for real. Its delay callback
receives the mapped error and attempt number, records delays of `0.1` and
`0.2` seconds, and the operation succeeds on the third attempt. The complete
flow is in [`main.py`](main.py).
