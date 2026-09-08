# Accumulating validation

Validate independent fields and return every issue instead of stopping at the
first invalid field.

## Run it

From the repository root:

```bash
uv run better-result-example validation-collection
```

Expected output:

```text
Err(value=('name is required', 'email must contain @', 'age must be at least 18'))
Ok(value=CreateUser(name='Ada', email='ada@example.com', age=36))
```

## How it works

Each field validator returns a `Result`. `collect_results` evaluates every
validator and returns all errors as an ordered tuple. Only after the collection
succeeds does `parse_create_user` construct the domain command.

Use `all_results` instead when the operation is dependent or should stop at the
first failure. The complete flow is in [`main.py`](main.py).
