# Application workflow

Compose a registration workflow with a domain error vocabulary, then map the
final result to an HTTP response at the application edge.

## Run it

From the repository root:

```bash
uv run better-result-example application-workflow
```

Expected output:

```text
success: status=201
invalid input: status=400
email taken: status=409
store down: status=503
publish fails: status=503
```

## How it works

- `parse_create_user` converts untrusted input into a domain command once at
  the edge.
- Each workflow step returns its own named error type.
- `register_user` composes those steps and exposes the complete
  `RegisterUserError` union in its return type.
- `to_http_response` translates that domain vocabulary into transport details
  exactly once. The `assert_never` branch makes a newly added error impossible
  to forget during type checking.

The complete flow is in [`main.py`](main.py).
