# Narrow Public API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Turn `better_result` into a small, curated Result library with one supported import surface, one canonical composition style, concrete tagged errors, and hardened codec/retry internals.

**Architecture:** The package root becomes the only supported public entrypoint. Existing implementation modules move behind underscore-prefixed paths; root exports only `Ok`, `Err`, `Result`, `PanicError`, and `TaggedError`. The canonical Result operations are `map`, `map_error`, `and_then`, `and_then_async`, `match`, and `unwrap_or`; convenience/duplicate operations are removed. Codec, retry, and collection functionality remain as internal implementation only for compatibility tests while no longer being public package entrypoints.

**Tech Stack:** Python 3.12+, `pytest`, `ty`, `ruff`, `pytest-cov`.

---

### Task 1: Establish the public API contract

**Files:**
- Modify: `src/better_result/__init__.py`
- Create: `src/better_result/_core.py`
- Create: `src/better_result/_error.py`
- Modify: `src/better_result/core.py`
- Modify: `src/better_result/error.py`
- Test: `tests/test_public_api.py`

- [x] **Step 1: Write tests for the root-only API.**

Assert that `better_result` exports exactly the supported names and that the old public submodules are no longer part of the source package. Assert that `Ok`, `Err`, `Result`, `PanicError`, and `TaggedError` are importable from the root.

- [x] **Step 2: Move implementation imports behind underscore modules.**

Move the implementation currently in `core.py` and `error.py` to `_core.py` and `_error.py`, update all internal imports to use underscore modules, and make the root explicitly define:

```python
from better_result._core import Err, Ok, PanicError, Result
from better_result._error import TaggedError

__all__ = ["Err", "Ok", "PanicError", "Result", "TaggedError"]
```

- [x] **Step 3: Remove the old public module entrypoints.**

Delete `src/better_result/core.py` and `src/better_result/error.py` after all imports and tests use the root or underscore modules. Keep implementation symbols available only through underscore-prefixed modules.

- [x] **Step 4: Run the public API tests.**

Run: `uv run pytest tests/test_public_api.py -q`
Expected: PASS.

---

### Task 2: Reduce Result to one canonical composition API

**Files:**
- Modify: `src/better_result/_core.py`
- Modify: all tests importing core Result operations
- Modify: `README.md`

- [x] **Step 1: Add tests for the retained API.**

Cover `map`, `map_error`, `and_then`, `and_then_async`, `match`, and `unwrap_or`, including callback panics, cancellation, and Result-return validation.

- [x] **Step 2: Remove duplicate Result operations.**

Delete `try_recover`, `try_recover_async`, `flatten`, `tap`, `tap_async`, `tap_error`, `tap_error_async`, `unwrap`, and the custom iterator behavior from `Ok`/`Err`. Remove `is_ok`, `is_err`, and the public `panic` helper from the supported API. Internal panic construction remains available to implementation modules.

- [x] **Step 3: Keep explicit branch matching as the canonical consumer operation.**

Retain `match(on_ok, on_err)` and document that callers should use it rather than helper predicates or direct payload extraction.

- [x] **Step 4: Update documentation examples and tests.**

Rewrite examples using only the retained operations. Replace recovery/flatten/tap tests with tests for the canonical operations and remove tests for unsupported methods.

- [x] **Step 5: Run core tests and type checking.**

Run: `uv run pytest tests/test_core.py tests/test_async_core.py -q`
Expected: PASS.

---

### Task 3: Make TaggedError concrete and closed

**Files:**
- Modify: `src/better_result/_error.py`
- Modify: `src/better_result/__init__.py`
- Modify: tagged-error tests and documentation

- [x] **Step 1: Replace dynamic error construction with subclass-only errors.**

Remove `tagged_error`, `is_tagged_error`, `TaggedError.match`, and `TaggedError.match_partial` from the supported API. A tagged error is declared only by subclassing `TaggedError` with a string class-header tag.

- [x] **Step 2: Validate tags and messages at runtime.**

Require `tag` to be a string and reject empty/whitespace-only tags. Require `message` to be `None` or `str`. Reject malformed values rather than silently producing inconsistent `Exception.args` and `.message` values.

- [x] **Step 3: Prevent property/API collisions.**

Replace unconstrained attribute assignment with a reserved-name check covering all error API names, including `to_dict`, `to_json`, `to_safe_dict`, `to_safe_json`, `match`, `match_partial`, `is_`, `message`, `cause`, `name`, `stack`, `_tag`, `_properties`, `args`, and `__cause__`. Raise `TypeError` for collisions.

- [x] **Step 4: Keep one serialization boundary.**

Retain `to_safe_json()` as the transport serializer and keep diagnostic serialization internal. Ensure nested diagnostic stacks are removed from safe payloads.

- [x] **Step 5: Update error tests.**

Add tests for invalid tag types, whitespace tags, non-string messages, reserved properties, and subclass-only construction. Remove tests for dynamic factory and string-handler dispatch.

- [x] **Step 6: Run error tests.**

Run: `uv run pytest tests/test_error.py tests/test_debug_traces.py -q`
Expected: PASS.

---

### Task 4: Harden the codec implementation without exposing it as core API

**Files:**
- Create: `src/better_result/_codec.py`
- Delete: `src/better_result/codec.py`
- Modify: codec tests and internal imports

- [x] **Step 1: Fix `ResultCodecIssue` typing.**

Use optional TypedDict fields because existing protocol payloads allow either a message-only issue or a value-only issue:

```python
class ResultCodecIssue(TypedDict, total=False):
    message: str
    value: object
```

- [x] **Step 2: Make synchronous async-schema rejection generic.**

When `inspect.isawaitable(validation)` is true in a synchronous codec, close only actual coroutine objects with `inspect.iscoroutine()`. Never assume arbitrary awaitables implement `.close()`. Return a `PanicError` with the documented message for Futures and custom awaitables.

- [x] **Step 3: Make schema typing consistent.**

Define separate synchronous and async schema protocols so callable and object schemas have the same type contract. The async codec may accept synchronous or asynchronous schemas; the sync codec accepts synchronous schemas only.

- [x] **Step 4: Keep codec construction internal.**

Retain codec tests against `better_result._codec` only. Do not export codec constructors or codec classes from the package root.

- [x] **Step 5: Add adversarial codec tests.**

Cover Future/custom-awaitable rejection, cancellation preservation, schema-object typing, malformed envelopes, and both valid issue shapes.

- [x] **Step 6: Run codec tests and type checking.**

Run: `uv run pytest tests/test_codec.py tests/test_codec_goldens.py -q`
Expected: PASS.

---

### Task 5: Move retry and collection helpers behind internal modules

**Files:**
- Create: `src/better_result/_retry.py`
- Create: `src/better_result/_collections.py`
- Delete: `src/better_result/retry.py`
- Delete: `src/better_result/collections.py`
- Modify: retry, collection, and documentation tests

- [x] **Step 1: Move implementation and update imports.**

Move retry and collection implementations behind underscore-prefixed modules. Update all internal and test imports accordingly.

- [x] **Step 2: Preserve one behavior per internal helper.**

Keep retry and collection behavior working for existing internal consumers, but do not re-export `try_result`, `try_async`, `all_results`, `partition`, or their configuration classes from `better_result`.

- [x] **Step 3: Keep retry policy explicit.**

Preserve cancellation and panic semantics, but document these helpers as non-core implementation extensions rather than part of the narrow Result API.

- [x] **Step 4: Run extension tests.**

Run: `uv run pytest tests/test_application_patterns.py tests/test_workflow_composition.py tests/test_async_cancellation.py -q`
Expected: PASS.

---

### Task 6: Rewrite documentation around the canonical API

**Files:**
- Modify: `README.md`
- Modify: documentation example tests

- [x] **Step 1: Remove unsupported examples.**

Delete examples using `try_recover`, `flatten`, taps, dynamic tagged errors, string-handler error matching, public codec imports, and public retry imports.

- [x] **Step 2: Publish one import style.**

All documented examples must use:

```python
from better_result import Err, Ok, Result, TaggedError
```

- [x] **Step 3: Document the canonical workflow.**

Show one flow: construct `Ok`/`Err`, transform with `map`/`and_then`, translate with `map_error`, consume with `match`, and use `unwrap_or` only for a fallback.

- [x] **Step 4: Add an explicit API policy section.**

State that underscore modules are implementation details and only root exports are supported.

- [x] **Step 5: Run documentation tests.**

Run: `uv run pytest tests/test_documentation_examples.py tests/test_documentation_async_examples.py -q`
Expected: PASS.

---

### Task 7: Final validation and packaging checks

**Files:**
- Modify: `pyproject.toml` only if package discovery or version metadata requires it
- Modify: all affected tests

- [x] **Step 1: Run the complete test suite.**

Run: `uv run pytest -q`
Expected: all tests pass.

- [x] **Step 2: Run coverage.**

Run: `uv run pytest --cov`
Expected: 100% configured coverage remains satisfied.

- [x] **Step 3: Run linting and typing.**

Run: `uv run ruff check` and `uv run ty check`
Expected: both pass with zero diagnostics.

- [x] **Step 4: Build the package.**

Run: `uv build --clear --no-create-gitignore`
Expected: wheel and sdist build successfully and contain only the root public API plus underscore implementation modules.

- [x] **Step 5: Verify the final public surface.**

Run a Python inspection script that imports `better_result`, checks `better_result.__all__`, and confirms only the five supported names are exported.

- [x] **Step 6: Review the final diff.**

Run: `git diff --check` and inspect `git status --short`. Confirm no generated coverage/cache files or unrelated CI changes are included.
