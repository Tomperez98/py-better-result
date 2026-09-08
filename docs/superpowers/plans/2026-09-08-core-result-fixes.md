# Core Result Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the fresh `Ok`/`Err` core safe, symmetric, fail-fast on broken callback contracts, and covered by focused tests.

**Architecture:** Keep the existing two concrete variants and the `Result[T, E]` union. Results will be immutable and unhashable because their payloads are arbitrary Python values. Active callbacks execute normally; callbacks required to return a `Result` are validated at the call boundary with `TypeError`. Remove the undocumented exception-based iteration protocol.

**Tech Stack:** Python 3.12+, pytest, ty, ruff, `typing_extensions.TypeIs`.

---

### Task 1: Add failing tests for core invariants

**Files:**
- Create: `tests/test_core_fresh.py`

- [ ] **Step 1: Test construction, equality, pattern matching, and variant predicates.**
- [ ] **Step 2: Test that both variants reject payload reassignment and are unhashable.**
- [ ] **Step 3: Test active/inactive map, map-error, sequencing, fallback, inspection, and async operations.**
- [ ] **Step 4: Test invalid synchronous and asynchronous `and_then` callbacks fail immediately with `TypeError`.**
- [ ] **Step 5: Test callback exceptions propagate instead of being silently converted into an expected `Err`.**
- [ ] **Step 6: Test neither variant exposes the undocumented iterator control-flow behavior.**
- [ ] **Step 7: Run `uv run pytest tests/test_core_fresh.py -q` and confirm failures against the current implementation.**

### Task 2: Repair the core implementation

**Files:**
- Modify: `src/better_result/_core.py`

- [ ] **Step 1: Remove unused type variables and the `DoExceptionError` iterator protocol.**
- [ ] **Step 2: Make `_value` write-once through `object.__setattr__` during construction and reject later assignment.**
- [ ] **Step 3: Set `__hash__ = None` on both variants.**
- [ ] **Step 4: Use exact variant types for equality.**
- [ ] **Step 5: Validate `and_then` and `and_then_async` callback results as `Ok` or `Err`, raising `TypeError` otherwise.**
- [ ] **Step 6: Tighten inactive callback annotations using `Never` and preserve the concrete variant for inspection methods.**
- [ ] **Step 7: Run the focused tests and make them pass.**

### Task 3: Make the typing dependency explicit

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Add `typing-extensions` as a runtime dependency for Python 3.12 support.**
- [ ] **Step 2: Regenerate the lockfile with `uv lock`.**
- [ ] **Step 3: Verify `uv run ty check` and `uv run ruff check`.**

### Task 4: Full validation

**Files:**
- Test: `tests/test_core_fresh.py`

- [ ] **Step 1: Run `uv run pytest -q`.**
- [ ] **Step 2: Run `uv run ty check`.**
- [ ] **Step 3: Run `uv run ruff check`.**
- [ ] **Step 4: Review the diff and confirm no README or legacy panic files were changed.**
