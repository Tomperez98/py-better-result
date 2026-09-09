# Capture exceptions

Convert ordinary synchronous and asynchronous exception-based functions into
`Result` values at an explicit boundary.

```bash
uv run better-result-example capture
```

Use the `catch` mapper to translate an exception into the error vocabulary of
the current layer. Without a mapper, `capture` stores the original exception
in `Err`.
