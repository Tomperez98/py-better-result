# Rust Parity Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `py-better-result` to a Python port of the Rust crate’s public retry API, plus a Python `Result` implementation whose synchronous operations match `std::result::Result` rather than exposing Python-only async, codec, collection, capture, or cancellation features.

**Architecture:** Keep `Ok`, `Err`, and `Result` in `src/better_result/_core.py`, using frozen dataclasses and Python equivalents of the applicable `std::result::Result` methods. Replace `_operations.py` with only Rust-parity retry types and `retry`/`retry_async`; use zero-argument callbacks that already return `Result`, so Python does not catch exceptions or add `TryContext`/`CancellationToken`. Delete codec and collection subsystems, then reduce the examples and documentation to the three Rust workspace examples: `basic`, `sync-retry`, and `async-retry`.

**Tech Stack:** Python 3.13+, `asyncio`, `pytest`, Hypothesis, `ty`, Ruff, existing Click example runner. Retry delays use non-negative `float` seconds as the Python representation of Rust `Duration`.

---

## Target public surface

Before implementation, treat this as the compatibility contract.

### Result core

Export:

```python
from better_result import Err, Ok, Result, UnwrapError
```

`Ok` and `Err` remain immutable single-value variants. `Result` exposes only synchronous operations corresponding to Rust `std::result::Result`:

- `is_ok()` / `is_err()`
- `is_ok_and(predicate)` / `is_err_and(predicate)`
- `ok()` / `err()`
- `map()` / `map_or()` / `map_or_else()` / `map_err()`
- `and_()` / `and_then()` / `or_()` / `or_else()` (`and` and `or` are reserved Python keywords)
- `inspect()` / `inspect_err()`
- `unwrap()` / `expect()` / `unwrap_err()` / `expect_err()`
- `unwrap_or()` / `unwrap_or_else()` / `unwrap_or_default()`
- `flatten()` and `transpose()` where their `Result`/`Optional` shapes are representable in Python

Do not expose Python-only `map_async`, `map_err_async`, `and_then_async`, `or_else_async`, `inspect_async`, `inspect_err_async`, `inspect_both`, `inspect_both_async`, `match`, or `unwrap_or_raise`. `UnwrapError` is the Python exception needed to implement Rust’s panic-like unwrap behavior; it is not an additional Result operation.

Remove module-level `is_ok` and `is_err`; callers use `result.is_ok()`, `result.is_err()`, or `isinstance(result, Ok/Err)`.

### Retry API

Export the Rust names and behavior:

```python
from better_result import (
    AlwaysRetry,
    ExponentialBackoff,
    FixedBackoff,
    InvalidBackoffMultiplier,
    InvalidJitterFactor,
    Jittered,
    LinearBackoff,
    RetryContext,
    RetryDecision,
    RetryPolicy,
    RetryPredicate,
    RetrySchedule,
    retry,
    retry_async,
)
```

Python-specific representation decisions:

- Rust `Duration` is represented as a non-negative finite `float` number of seconds.
- `RetryAfter` and `StopRetry` are concrete `RetryDecision` variants.
- `ExponentialBackoff`, `RetryPolicy.exponential`, `Jittered`, and `RetryPolicy.with_jitter` raise dedicated `ValueError` subclasses for invalid configuration, which is the idiomatic Python adaptation of Rust constructor errors.
- Retry operations accept `Callable[[], Result[T, E]]` and async zero-argument callbacks returning `Result[T, E]`; they do not catch raised exceptions.
- `RetryContext` contains only `error` and one-based `attempt`.
- A custom schedule implements `decide(context) -> RetryDecision`, matching the Rust trait. A custom predicate implements `should_retry(context) -> bool`.
- Keep `RetryPolicy.new`, `immediate`, `fixed`, `linear`, `exponential`, `with_predicate`, and `with_jitter`; remove Python-only `constant`, `dynamic`, and `from_schedule` constructors.

---

### Task 1: Lock the reduced public API with failing tests

**Files:**
- Create: `tests/test_public_api.py`
- Modify: `tests/test_core_fresh.py`
- Modify: `tests/test_operations.py`
- Modify: `tests/test_operations_types.py`
- Delete: `tests/test_codec.py`
- Delete: `tests/test_codec_coverage.py`

- [ ] **Step 1: Add public export assertions**

Add a test that imports `better_result.__all__` and asserts it contains only the Result core names and Rust retry names. Assert removed names are absent:

```python
def test_public_api_contains_only_result_and_rust_retry_features() -> None:
    import better_result

    assert set(better_result.__all__) == {
        "AlwaysRetry",
        "Err",
        "ExponentialBackoff",
        "FixedBackoff",
        "InvalidBackoffMultiplier",
        "InvalidJitterFactor",
        "Jittered",
        "LinearBackoff",
        "Ok",
        "Result",
        "RetryContext",
        "RetryDecision",
        "RetryPolicy",
        "RetryPredicate",
        "RetrySchedule",
        "StopRetry",
        "RetryAfter",
        "UnwrapError",
        "retry",
        "retry_async",
    }
    for removed in (
        "AsyncResultCodec",
        "CancellationToken",
        "capture",
        "capture_async",
        "collect_results",
        "collect_results_async",
        "flatten_result",
        "partition_results",
        "partition_results_async",
        "traverse",
        "traverse_async",
        "try_result",
        "try_async",
        "TryContext",
    ):
        assert not hasattr(better_result, removed)
```

`flatten_result` is intentionally not exported as a free function; the supported operation is `Result.flatten()`.

- [ ] **Step 2: Add Result-method parity tests**

Cover both variants for every method in the target surface, including inactive callback non-execution, `and_`/`or_`, predicates, `flatten`, and `transpose`. Assert callbacks that raise are propagated rather than converted to `Err`.

- [ ] **Step 3: Rewrite retry tests around Result-returning callbacks**

Replace `try_result` tests with cases such as:

```python
def test_retry_does_not_catch_callback_exceptions() -> None:
    policy = RetryPolicy.immediate(0)

    with pytest.raises(RuntimeError, match="bug"):
        retry(lambda: _raise_runtime_error(), policy)
```

Add tests for one-shot execution, retry bounds, predicates, custom schedules, invalid constructor results, jitter bounds, and async retry cancellation by cancelling the task.

- [ ] **Step 4: Remove tests for deleted subsystems**

Delete codec tests and remove all test cases covering capture, collection helpers, async Result combinators, traversal, cancellation tokens, and exception-catching retry.

- [ ] **Step 5: Run the focused tests and verify they fail for missing API**

Run:

```bash
cd ../py-better-result
uv run pytest tests/test_public_api.py tests/test_core_fresh.py tests/test_operations.py -q
```

Expected: failures for missing/extra exports and the not-yet-implemented Result/retry parity behavior.

---

### Task 2: Implement the std::result::Result-shaped Python core

**Files:**
- Modify: `src/better_result/_core.py`
- Modify: `src/better_result/__init__.py`
- Test: `tests/test_core_fresh.py`
- Test: `tests/test_public_api.py`

- [ ] **Step 1: Remove Python-only abstract methods and helpers**

Delete async abstract methods, `match`, `inspect_both`, and `unwrap_or_raise`. Delete module-level `is_ok`/`is_err` and the callback return validator only needed for the old async/core-specific behavior.

Keep `Result`, `Ok`, `Err`, and `UnwrapError`; retain frozen dataclasses, equality, positional pattern matching, and unhashability.

- [ ] **Step 2: Add the missing Result methods**

Implement the standard operations with these semantics:

```python
class Result[T, E](ABC):
    @abstractmethod
    def is_ok(self) -> bool: ...

    @abstractmethod
    def is_err(self) -> bool: ...

    @abstractmethod
    def is_ok_and(self, predicate: Callable[[T], bool]) -> bool: ...

    @abstractmethod
    def is_err_and(self, predicate: Callable[[E], bool]) -> bool: ...

    @abstractmethod
    def and_(self, result: Result[U, F]) -> Result[U, F]: ...

    @abstractmethod
    def or_(self, result: Result[U, F]) -> Result[T, F]: ...

    @abstractmethod
    def unwrap_or_default(self) -> T | None: ...

    @abstractmethod
    def flatten(self) -> Result[object, object]: ...

    @abstractmethod
    def transpose(self) -> object: ...
```

Use precise generic annotations in the implementation rather than the abbreviated signatures above. `and_` and `or_` are the Python spellings for Rust’s reserved-keyword methods. `unwrap_or_default()` returns `None` for `Err`, which is the Python adaptation of a defaultable value; document this adaptation explicitly.

`flatten()` must turn `Ok(Ok(value))` into `Ok(value)`, `Ok(Err(error))` into `Err(error)`, and `Err(error)` into `Err(error)`. `transpose()` must implement `Result[Optional[T], E] -> Optional[Result[T, E]]`.

- [ ] **Step 3: Preserve panic-like unwrap behavior without catching defects**

`unwrap`, `expect`, `unwrap_err`, and `expect_err` must raise `UnwrapError` on the wrong variant. Do not catch callback exceptions in `map`, `and_then`, `or_else`, or inspection methods.

- [ ] **Step 4: Update exports**

Export only the core names from `_core.py`; do not export `is_ok` or `is_err`.

- [ ] **Step 5: Run core tests**

Run:

```bash
uv run pytest tests/test_core_fresh.py tests/test_public_api.py -q
```

Expected: all core parity tests pass.

---

### Task 3: Replace Python operations with a direct Rust retry port

**Files:**
- Rewrite: `src/better_result/_operations.py`
- Modify: `src/better_result/__init__.py`
- Test: `tests/test_operations.py`
- Test: `tests/test_properties.py`
- Test: `tests/test_operations_types.py`

- [ ] **Step 1: Define retry decision and schedule types**

Implement the following Python equivalents:

```python
@dataclass(frozen=True, slots=True)
class StopRetry(RetryDecision):
    pass


@dataclass(frozen=True, slots=True)
class RetryAfter(RetryDecision):
    delay: float


@dataclass(frozen=True, slots=True)
class RetryContext[E]:
    error: E
    attempt: int


class RetrySchedule[E](Protocol):
    def decide(self, context: RetryContext[E]) -> RetryDecision: ...


class RetryPredicate[E](Protocol):
    def should_retry(self, context: RetryContext[E]) -> bool: ...
```

Validate retry attempts and delays as finite, non-negative values. `AlwaysRetry.should_retry()` always returns `True`.

- [ ] **Step 2: Implement Rust-named backoffs**

Implement `FixedBackoff.new(delay)`, `LinearBackoff.new(base_delay)`, and `ExponentialBackoff.new(base_delay, multiplier)` with `decide(context)` methods. Use one-based attempts exactly as `src/retry.rs` does:

```python
FixedBackoff.new(delay).decide(context)  # RetryAfter(delay)
LinearBackoff.new(base).decide(context)  # RetryAfter(base * attempt)
ExponentialBackoff.new(base, multiplier)  # base * multiplier ** (attempt - 1)
```

Raise `InvalidBackoffMultiplier` when the multiplier is zero or otherwise invalid. Python constructors use exceptions for invalid configuration; operation failures remain `Result` values.

- [ ] **Step 3: Implement jitter parity**

Implement `Jittered(schedule, factor)` and `RetryPolicy.with_jitter(factor)` as normal Python constructors/methods that raise `InvalidJitterFactor` for invalid input. Preserve the Rust factor range `0.0..=1.0` and multiplicative behavior:

```python
jittered = base_delay * ((1 - factor) + random_value * factor)
```

A wrapped `StopRetry` remains `StopRetry`.

- [ ] **Step 4: Implement RetryPolicy**

Implement:

```python
RetryPolicy.new(max_retries, schedule)
RetryPolicy.immediate(max_retries)
RetryPolicy.fixed(max_retries, delay)
RetryPolicy.linear(max_retries, base_delay)
RetryPolicy.exponential(max_retries, base_delay, multiplier)
policy.with_predicate(predicate)
policy.with_jitter(factor)
```

`max_retries` counts retries after the initial attempt. `policy._decide(error, attempt)` must stop when `attempt > max_retries`, then evaluate the predicate, then call the schedule. This ordering must match `src/retry.rs`.

- [ ] **Step 5: Implement sync and async retry operations**

Implement the direct equivalents of the Rust signatures:

```python
def retry(
    operation: Callable[[], Result[T, E]],
    policy: RetryPolicy[...],
) -> Result[T, E]: ...


async def retry_async(
    operation: Callable[[], Awaitable[Result[T, E]]],
    policy: RetryPolicy[...],
) -> Result[T, E]: ...
```

Call the operation once initially, retry only returned `Err` values, sleep for the scheduled delay, and return the final `Err`. Raised Python exceptions must propagate. `asyncio` task cancellation must propagate through the async retry delay and operation.

- [ ] **Step 6: Rewrite retry property tests**

Replace exception-based properties with generated `Result` operations. Assert no more than `max_retries + 1` calls, attempt numbers observed by schedules are `[1, 2, ...]`, overflow/invalid values produce the same typed constructor failures, and sync/async retries have matching returned Results.

- [ ] **Step 7: Run retry tests and type checks**

Run:

```bash
uv run pytest tests/test_operations.py tests/test_properties.py -q
uv run ty check
uv run ruff check src tests
```

Expected: retry behavior passes with no references to `TryContext`, `CancellationToken`, capture helpers, collection helpers, or codec types.

---

### Task 4: Prune modules, dependencies, and stale tests

**Files:**
- Delete: `src/better_result/_codec.py`
- Delete: `src/better_result/testing/codec_helpers.py`
- Modify: `src/better_result/testing/__init__.py`
- Modify: `src/better_result/__init__.py`
- Modify: `pyproject.toml`
- Delete or rewrite: `tests/test_codec.py`
- Delete or rewrite: `tests/test_codec_coverage.py`
- Rewrite: `tests/test_operations_types.py`
- Rewrite: `tests/test_properties.py`

- [ ] **Step 1: Remove codec implementation and testing helpers**

Delete codec source and helper files. Make `src/better_result/testing/__init__.py` empty or remove the package if it has no remaining purpose.

- [ ] **Step 2: Reduce package exports**

Make `src/better_result/__init__.py` export only the target core and retry names. Ensure importing the package does not import `asyncio`, codec types, Hypothesis helpers, or collection code.

- [ ] **Step 3: Remove stale dependencies and configuration**

The current `pyproject.toml` has no runtime codec dependency, so retain only the existing build metadata and development tools needed by the reduced tests. Remove any coverage source references or package paths that no longer exist.

- [ ] **Step 4: Update typing tests**

Replace async/capture/collection/codec type contracts with contracts for:

- `Result[int, str]` core methods
- `RetryContext[str]`
- `RetryPolicy` constructors and `Result`-returning invalid constructors
- sync `retry`
- async `retry_async`
- custom `RetrySchedule` and `RetryPredicate`

- [ ] **Step 5: Run a stale-symbol search**

Run:

```bash
rg "AsyncResultCodec|CancellationToken|capture_async?|collect_results|partition_results|traverse|try_result|TryContext|SchemaFailure|codec|map_async|and_then_async|inspect_both|match\(" src tests
```

Expected: no matches except intentional historical documentation, which must also be removed in Task 6.

---

### Task 5: Reduce Python examples to the Rust workspace examples

**Files:**
- Delete example directories: `examples/application-workflow/`, `examples/async-boundaries/`, `examples/async-traversal/`, `examples/async-workflows/`, `examples/cancellation/`, `examples/codec/`, `examples/collections/`, `examples/cpu-heavy-cancellation/`, `examples/recovery/`, `examples/retry/`, `examples/thread-offload/`, `examples/validation/`, `examples/validation-collection/`
- Rewrite: `examples/basic/main.py`
- Rewrite: `examples/basic/README.md`
- Create: `examples/sync-retry/main.py`
- Create: `examples/sync-retry/README.md`
- Create: `examples/sync-retry/pyproject.toml`
- Create: `examples/async-retry/main.py`
- Create: `examples/async-retry/README.md`
- Create: `examples/async-retry/pyproject.toml`
- Modify: `examples/README.md`
- Modify: `tools/example-runner/tests/test_cli.py`

- [ ] **Step 1: Make `basic` match the Rust basic example**

Use a zero- or fixed-delay `RetryPolicy` and a zero-argument operation returning `Ok`/`Err`. Do not use application workflow combinators that are not part of the Rust example set.

- [ ] **Step 2: Port the Rust sync retry example**

Use an explicit `FetchError` enum-like Python hierarchy, `RetryPolicy.immediate/fixed/linear/exponential`, a retry predicate, and a custom schedule implementing `decide`. The operation itself must return `Err` rather than raise.

- [ ] **Step 3: Port the Rust async retry example**

Use `retry_async`, a zero-retry operation, an error-aware policy, and task cancellation for both an in-flight pending operation and a long retry delay. Do not expose or use `CancellationToken`.

- [ ] **Step 4: Update example runner tests**

Change expected discovery and CLI invocations from the old full example set to exactly:

```python
assert _discover_examples(examples_dir) == ("async-retry", "basic", "sync-retry")
```

Run all three examples through `--all` and assert their retry output.

- [ ] **Step 5: Run all reduced examples**

Run:

```bash
uv run better-result-example --list
uv run better-result-example --all
```

Expected: only `async-retry`, `basic`, and `sync-retry` are listed and all complete successfully.

---

### Task 6: Rewrite documentation and complete verification

**Files:**
- Rewrite: `README.md`
- Rewrite: `examples/README.md`
- Modify: `examples/*/README.md`
- Modify: `pyproject.toml` version to `2.0.0`
- Modify: `uv.lock` through `uv lock`

- [ ] **Step 1: Document the reduced contract**

Remove sections describing codecs, collection helpers, async Result combinators, capture, `try_result`, cancellation tokens, traversal, and thread offload. Document:

- `Ok`/`Err` and the supported std-like Result methods
- callback exceptions propagate
- retry operates on returned `Result` values
- Rust retry names and constructor behavior
- float-seconds as the Python representation of `Duration`
- `retry_async` cancellation through normal asyncio task cancellation

- [ ] **Step 2: Document the three examples**

Make the example README list and commands match the Rust repository README:

```text
basic
sync-retry
async-retry
```

- [ ] **Step 3: Bump the breaking-release version and refresh the lockfile**

Set `project.version` to `2.0.0` and run:

```bash
uv lock
```

- [ ] **Step 4: Run the complete validation suite**

Run:

```bash
uv run pytest -q
uv run pytest --cov
uv run ruff check .
uv run ty check
uv run better-result-example --all
```

Expected: all tests pass, coverage meets the configured threshold, Ruff and ty are clean, and all three examples succeed.

- [ ] **Step 5: Verify the final API and repository diff**

Run:

```bash
python - <<'PY'
import better_result
print(better_result.__all__)
PY

git status --short
git diff --stat
```

Confirm there are no deleted-feature exports, no stale example directories, and no codec/capture/collection/cancellation implementation files.

---

## Self-review

- The Rust retry surface is covered by Tasks 3 and 5.
- The requested Python Result implementation is covered by Tasks 1 and 2.
- Python-only async Result methods, codecs, collection helpers, exception capture, cancellation tokens, and traversal are explicitly removed in Tasks 1, 4, and 6.
- Existing Python examples are pruned to the three Rust examples in Task 5.
- Invalid Rust constructor paths are represented as Python `Result` values rather than Python-only `ValueError` exceptions.
- Retry callbacks return `Result` directly, so Python exceptions retain the Rust behavior of uncaught programmer defects.
