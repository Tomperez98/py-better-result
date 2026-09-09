# Examples workspace

Run a focused `py-better-result` example without installing anything globally.
Each example is a small uv workspace member with its own dependency metadata,
script, and README.

## Set up the workspace

From the repository root:

```bash
uv sync --all-packages
```

The root project remains the library package. Every example declares
`py-better-result` as a workspace dependency, so `uv` runs the examples against
the checked-out source rather than a separately downloaded release.

Run one example by name:

```bash
uv run better-result-example basic
```

List all runnable examples with:

```bash
uv run better-result-example --list
```

Run every example, as CI does, with:

```bash
uv run better-result-example --all
```
