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

Run one example with its workspace package name:

```bash
uv run --package example-basic python examples/basic/main.py
```

## Choose an example

| Example | Task | Run |
| --- | --- | --- |
| [`basic/`](basic/README.md) | Compose `Ok` and `Err` values | `uv run --package example-basic python examples/basic/main.py` |
| [`validation/`](validation/README.md) | Capture and validate parsing errors | `uv run --package example-validation python examples/validation/main.py` |
| [`async-workflows/`](async-workflows/README.md) | Compose and collect async operations | `uv run --package example-async-workflows python examples/async-workflows/main.py` |
| [`cancellation/`](cancellation/README.md) | Cancel an in-flight operation | `uv run --package example-cancellation python examples/cancellation/main.py` |
| [`cpu-heavy-cancellation/`](cpu-heavy-cancellation/README.md) | Add cancellation points to CPU work | `uv run --package example-cpu-heavy-cancellation python examples/cpu-heavy-cancellation/main.py` |
| [`collections/`](collections/README.md) | Collect, partition, and flatten results | `uv run --package example-collections python examples/collections/main.py` |
| [`retry/`](retry/README.md) | Retry expected exceptions | `uv run --package example-retry python examples/retry/main.py` |
| [`codec/`](codec/README.md) | Encode and decode result envelopes | `uv run --package example-codec python examples/codec/main.py` |

Each directory README explains the problem, expected output, and control flow.
Start with [`basic/`](basic/README.md) for the smallest complete workflow.
