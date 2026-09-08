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

## Choose an example

| Example | Task | Run |
| --- | --- | --- |
| [`basic/`](basic/README.md) | Compose `Ok` and `Err` values | `uv run better-result-example basic` |
| [`validation/`](validation/README.md) | Capture and validate parsing errors | `uv run better-result-example validation` |
| [`async-workflows/`](async-workflows/README.md) | Compose and collect async operations | `uv run better-result-example async-workflows` |
| [`async-boundaries/`](async-boundaries/README.md) | Capture async failures, collect async results, and use async codecs | `uv run better-result-example async-boundaries` |
| [`async-traversal/`](async-traversal/README.md) | Traverse async operations with bounded concurrency | `uv run better-result-example async-traversal` |
| [`cancellation/`](cancellation/README.md) | Cancel an in-flight operation | `uv run better-result-example cancellation` |
| [`cpu-heavy-cancellation/`](cpu-heavy-cancellation/README.md) | Add cancellation points to CPU work | `uv run better-result-example cpu-heavy-cancellation` |
| [`collections/`](collections/README.md) | Collect, partition, and flatten results | `uv run better-result-example collections` |
| [`validation-collection/`](validation-collection/README.md) | Accumulate independent validation errors | `uv run better-result-example validation-collection` |
| [`application-workflow/`](application-workflow/README.md) | Compose domain workflows and transport mapping | `uv run better-result-example application-workflow` |
| [`recovery/`](recovery/README.md) | Translate errors and recover from a primary-source failure | `uv run better-result-example recovery` |
| [`retry/`](retry/README.md) | Retry expected exceptions | `uv run better-result-example retry` |
| [`codec/`](codec/README.md) | Encode and decode result envelopes | `uv run better-result-example codec` |

Each directory README explains the problem, expected output, and control flow.
Start with [`basic/`](basic/README.md) for the smallest complete workflow.
