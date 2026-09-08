# Retry and Cancellation Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make retry/cancellation behavior fail-fast and contract-consistent, then align tests, examples, and documentation with the hardened API.

**Architecture:** Keep `try_async` as the public cancellation boundary and make it check the token after retry-policy decisions, including the terminal `StopRetry` path. Keep retry schedules as the typed closed set already exposed by `RetryPolicy`, but reject malformed manually constructed policies at construction time. Exercise dynamic retry and cancellation-during-backoff through public examples rather than private helpers.

**Tech Stack:** Python 3.12, asyncio, pytest/pytest-asyncio, uv, Ruff, ty.

---

### Task 1: Lock down cancellation and policy-construction contracts

**Files:**
- Modify: `tests/test_operations.py` near the existing cancellation and retry validation tests

- [x] **Step 1: Write the failing cancellation regression test**

Add a public-API test where `should_retry` cancels the token and returns `False`; assert that `try_async` raises `asyncio.CancelledError` rather than returning `Err`.

- [x] **Step 2: Write failing runtime-validation tests**

Add tests that construct `RetryPolicy` with a non-`RetrySchedule` object and a non-callable `should_retry`, and assert `ValueError` with messages mentioning `schedule` and `predicate` respectively.

- [x] **Step 3: Run the focused tests**

Run:

```bash
uv run pytest tests/test_operations.py -k 'cancellation or retry_policy' -q
```

Expected: the cancellation test currently fails because the function returns `Err`; the new validation tests fail because construction currently accepts malformed values.

---

### Task 2: Enforce the cancellation and policy invariants

**Files:**
- Modify: `src/better_result/_operations.py:215-227` for `RetryPolicy.__post_init__`
- Modify: `src/better_result/_operations.py:472-482` for the post-decision cancellation check

- [x] **Step 1: Validate policy schedule and predicate at construction**

Extend `RetryPolicy.__post_init__` after the `times` checks:

```python
if not isinstance(
    self.schedule,
    (ConstantDelay, LinearBackoff, ExponentialBackoff, DynamicDelay, Jittered),
):
    _raise_retry_policy_error("retry schedule must be a supported schedule")
if self.should_retry is not None and not callable(self.should_retry):
    _raise_retry_policy_error("retry predicate must be callable")
```

- [x] **Step 2: Check cancellation after policy decisions**

Immediately after `retry_policy.decide(...)` in `try_async`, call `cancel_token.raise_if_cancelled()` when a token exists, before handling `StopRetry`. This preserves cancellation even when the policy declines another retry.

- [x] **Step 3: Run the focused tests**

Run:

```bash
uv run pytest tests/test_operations.py -k 'cancellation or retry_policy' -q
```

Expected: all focused tests pass.

---

### Task 3: Make the retry example exercise dynamic scheduling

**Files:**
- Modify: `examples/retry/main.py`
- Modify: `examples/retry/README.md`

- [x] **Step 1: Replace the passive dynamic-policy printout**

Keep the initial exponential retry demonstration, then run a second operation with a dynamic policy. Record the delays inside `delay_for` and print the resulting `Ok` value, attempt count, and observed delays. This makes the dynamic callback execute and gives deterministic expected output without printing a function memory address.

- [x] **Step 2: Update the retry README output and explanation**

Document the second operation and its observed dynamic delays. Explain that the first example uses zero exponential delay for speed, while the second actually executes a nonzero dynamic schedule.

- [x] **Step 3: Run the retry example**

Run:

```bash
uv run --package example-retry python examples/retry/main.py
```

Expected: deterministic result/attempt/delay lines and no function-address-dependent output.

---

### Task 4: Add a combined async retry/cancellation use case

**Files:**
- Modify: `examples/cancellation/main.py`
- Modify: `examples/cancellation/README.md`

- [x] **Step 1: Add an async retry-wait cancellation flow**

After the in-flight cancellation example, run an operation that fails once, schedule token cancellation, and call `try_async` with a long retry delay. Catch `asyncio.CancelledError` and print a second deterministic message. Ensure the cancellation helper task is awaited/cleaned up.

- [x] **Step 2: Update expected output and explanation**

Explain that cancellation interrupts both active operations and retry backoff waits, and distinguish this from domain `Err` results.

- [x] **Step 3: Run the cancellation example**

Run:

```bash
uv run --package example-cancellation python examples/cancellation/main.py
```

Expected: both cancellation messages, with no 60-second wait.

---

### Task 5: Clarify retry safety, sync blocking, and cancellation guarantees

**Files:**
- Modify: `README.md` in the retry/cancellation section
- Modify: `examples/README.md` only if command/output wording needs synchronization

- [x] **Step 1: Add retry safety guidance**

State that retry counts retry all caught exceptions unless `should_retry` narrows the policy, and recommend transient-error predicates plus idempotent operations.

- [x] **Step 2: Document synchronous retry blocking**

State that delayed `try_result` retries use blocking `time.sleep()` and do not support cancellation tokens; use `try_async` for interruptible async waits.

- [x] **Step 3: Align cancellation wording**

Retain the `CancelledError` contract, describe cancellation checks around policy decisions, and keep the CPU-bound/blocking caveat.

---

### Task 6: Full verification and cleanup

**Files:**
- No additional source files expected

- [x] **Step 1: Run the full test suite and coverage**

```bash
uv run pytest --cov
```

Expected: all tests pass and coverage remains at least the configured 100% threshold.

- [x] **Step 2: Run static checks**

```bash
uv run ruff check .
uv run ty check
```

Expected: both pass.

- [x] **Step 3: Run every example**

Run the seven examples that terminate without external input, plus both cancellation examples, using their documented `uv run --package ...` commands. Confirm expected output is accurate.

- [x] **Step 4: Review the diff**

```bash
git diff --check
git diff --stat
git status --short
```

Confirm only the plan, implementation, tests, examples, and documentation changed; do not alter unrelated user worktree changes.
