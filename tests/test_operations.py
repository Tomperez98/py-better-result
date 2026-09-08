"""Behavioral tests for Result operations outside the core variants."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Never, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

import pytest

from better_result import (
    CancellationToken,
    ConstantDelay,
    DynamicDelay,
    Err,
    ExponentialBackoff,
    Jittered,
    LinearBackoff,
    Ok,
    Result,
    RetryAfter,
    RetryContext,
    RetryPolicy,
    StopRetry,
    TryContext,
    all_results,
    all_results_async,
    capture,
    capture_async,
    collect_results,
    collect_results_async,
    flatten_result,
    partition_results,
    partition_results_async,
    traverse,
    traverse_async,
    try_async,
    try_result,
)


def test_all_results_collects_successes_and_short_circuits_first_error() -> None:
    assert all_results([Ok(1), Ok(2)]) == Ok((1, 2))
    assert all_results([Ok(1), Err("first"), Err("second")]) == Err("first")
    assert all_results([]) == Ok(())


def test_partition_results_preserves_relative_order() -> None:
    assert partition_results([Ok(1), Err("a"), Ok(2), Err("b")]) == (
        [1, 2],
        ["a", "b"],
    )
    assert partition_results([]) == ([], [])


def test_flatten_result_propagates_inner_or_outer_error() -> None:
    assert flatten_result(Ok(Ok(42))) == Ok(42)
    assert flatten_result(Ok(Err("inner"))) == Err("inner")
    assert flatten_result(Err("outer")) == Err("outer")


def test_flatten_result_rejects_unknown_result_variants() -> None:
    invalid = cast("Result[Result[int, str], str]", object())
    with pytest.raises(TypeError, match="expected a concrete Result variant"):
        flatten_result(invalid)


def test_try_result_captures_expected_exceptions_without_forcing_error_types() -> None:
    assert try_result(lambda _: 42) == Ok(42)

    captured = try_result(_raise_zero)
    assert isinstance(captured, Err)
    assert isinstance(captured.err_value, ZeroDivisionError)
    assert try_result(_raise_zero, catch=str) == Err("division by zero")


def test_try_result_exposes_attempt_context_and_retries_immediately() -> None:
    attempts: list[int] = []

    def operation(context: TryContext) -> str:
        attempts.append(context.attempt)
        if len(attempts) < 2:
            message = "temporary"
            raise RuntimeError(message)
        return "ok"

    assert try_result(operation, retry=1) == Ok("ok")
    assert attempts == [1, 2]


def test_retry_policy_decisions_use_typed_schedules() -> None:
    context = RetryContext(error="temporary", attempt=2)

    assert ConstantDelay(2).delay_for(context) == 2
    assert LinearBackoff(2).delay_for(context) == 4
    assert ExponentialBackoff(2).delay_for(context) == 4

    def dynamic_delay(retry_context: RetryContext[str]) -> float:
        return retry_context.attempt / 10

    assert DynamicDelay(dynamic_delay).delay_for(context) == 0.2

    policy = RetryPolicy.from_schedule(
        times=2,
        schedule=ConstantDelay(1),
    )
    assert policy.decide(context) == RetryAfter(1)
    assert policy.decide(RetryContext(error="temporary", attempt=3)) == StopRetry()


def test_jittered_schedule_preserves_its_delay_bounds() -> None:
    context = RetryContext(error="temporary", attempt=2)
    delay = Jittered(ExponentialBackoff(2), factor=0.5).delay_for(context)
    assert 2 <= delay <= 4
    full_jitter_delay = Jittered(ConstantDelay(1), factor=1.0).delay_for(context)
    assert 0 <= full_jitter_delay <= 1


def test_retry_policy_validates_schedule_invariants() -> None:
    invalid_delay: object = object()
    with pytest.raises(ValueError, match="delay"):
        ConstantDelay(cast("float", invalid_delay))

    with pytest.raises(ValueError, match="attempt"):
        RetryContext(error="temporary", attempt=0)
    with pytest.raises(ValueError, match="attempt"):
        RetryContext(error="temporary", attempt=True)
    invalid_attempt: object = object()
    with pytest.raises(ValueError, match="attempt"):
        RetryContext(error="temporary", attempt=cast("int", invalid_attempt))

    invalid_delay_bool: object = True
    with pytest.raises(ValueError, match="delay"):
        ConstantDelay(seconds=cast("float", invalid_delay_bool))

    with pytest.raises(ValueError, match="factor"):
        ExponentialBackoff(1, factor=0)
    with pytest.raises(ValueError, match="factor"):
        ExponentialBackoff(1, factor=True)
    invalid_factor: object = object()
    with pytest.raises(ValueError, match="factor"):
        ExponentialBackoff(1, factor=cast("float", invalid_factor))

    with pytest.raises(ValueError, match="overflow"):
        ExponentialBackoff(1).delay_for(
            RetryContext(error="temporary", attempt=2049),
        )

    with pytest.raises(ValueError, match="jitter"):
        Jittered(ConstantDelay(1), factor=1.1)
    with pytest.raises(ValueError, match="jitter"):
        Jittered(ConstantDelay(1), factor=True)
    invalid_jitter: object = object()
    with pytest.raises(ValueError, match="jitter"):
        Jittered(ConstantDelay(1), factor=cast("float", invalid_jitter))

    invalid_schedule: object = object()
    with pytest.raises(ValueError, match="schedule"):
        RetryPolicy(times=1, schedule=cast("ConstantDelay", invalid_schedule))

    invalid_predicate: object = object()
    with pytest.raises(ValueError, match="predicate"):
        RetryPolicy.constant(
            times=1,
            should_retry=cast(
                "Callable[[RetryContext[str]], bool]",
                invalid_predicate,
            ),
        )

    assert ConstantDelay(2).seconds == 2.0
    assert ExponentialBackoff(1, factor=3).factor == 3.0
    assert RetryAfter(4).delay == 4.0

    assert isinstance(RetryPolicy.constant(times=1).schedule, ConstantDelay)
    assert RetryPolicy(times=1, schedule=ConstantDelay(1)).times == 1
    assert isinstance(
        RetryPolicy.exponential(times=1, initial_delay=1).schedule,
        ExponentialBackoff,
    )
    assert isinstance(
        RetryPolicy.exponential(times=1, initial_delay=1, jitter=0.5).schedule,
        Jittered,
    )
    assert isinstance(
        RetryPolicy[str].dynamic(times=1, delay=lambda _: 1).schedule,
        DynamicDelay,
    )


def test_try_result_without_a_retry_policy_returns_the_mapped_error() -> None:
    assert try_result(_raise_zero, catch=str, retry=None) == Err("division by zero")


def test_try_result_supports_the_same_policy_as_try_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []
    monkeypatch.setattr(time, "sleep", delays.append)
    attempts: list[int] = []

    def operation(context: TryContext) -> str:
        attempts.append(context.attempt)
        if len(attempts) < 3:
            message = "temporary"
            raise RuntimeError(message)
        return "ok"

    result = try_result(
        operation,
        catch=str,
        retry=RetryPolicy.linear(times=2, initial_delay=2),
    )

    assert result == Ok("ok")
    assert attempts == [1, 2, 3]
    assert delays == [2, 4]


def test_try_result_integer_retry_is_immediate_policy_shorthand() -> None:
    attempts: list[int] = []

    def operation(context: TryContext) -> str:
        attempts.append(context.attempt)
        if len(attempts) < 2:
            message = "temporary"
            raise RuntimeError(message)
        return "ok"

    assert try_result(operation, retry=1) == Ok("ok")
    assert attempts == [1, 2]


def test_try_result_propagates_mapper_bugs() -> None:
    def broken_mapper(_: Exception) -> str:
        message = "mapper broken"
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="mapper broken"):
        try_result(_raise_zero, catch=broken_mapper)


# --- capture / capture_async ---


def test_capture_returns_success_without_context() -> None:
    assert capture(lambda: 42) == Ok(42)


def test_capture_maps_expected_exceptions() -> None:
    assert capture(lambda: int("bad"), catch=str) == Err(
        "invalid literal for int() with base 10: 'bad'"
    )


def test_capture_propagates_mapper_defects() -> None:
    def broken_mapper(_: Exception) -> str:
        message = "mapper broken"
        raise RuntimeError(message)

    def divide_one_by_zero() -> float:
        msg = "division by zero"
        raise ZeroDivisionError(msg)

    with pytest.raises(RuntimeError, match="mapper broken"):
        capture(divide_one_by_zero, catch=broken_mapper)


@pytest.mark.asyncio
async def test_capture_async_returns_success() -> None:
    async def operation() -> int:
        return 42

    assert await capture_async(operation) == Ok(42)


@pytest.mark.asyncio
async def test_capture_async_maps_expected_exception() -> None:
    async def operation() -> int:
        message = "expected"
        raise ValueError(message)

    assert await capture_async(operation, catch=str) == Err("expected")


@pytest.mark.asyncio
async def test_capture_async_with_async_catch() -> None:
    async def operation() -> int:
        msg = "async catch"
        raise ValueError(msg)

    async def async_catch(exc: Exception) -> str:
        return str(exc)

    assert await capture_async(operation, catch=async_catch) == Err("async catch")


@pytest.mark.asyncio
async def test_capture_async_awaits_async_catch() -> None:
    """Exercise the `await mapped` branch in capture_async."""

    async def operation() -> int:
        msg = "needs await"
        raise ValueError(msg)

    async def async_catch(exc: Exception) -> str:
        await asyncio.sleep(0)
        return str(exc)

    result = await capture_async(operation, catch=async_catch)
    assert result == Err("needs await")


@pytest.mark.asyncio
async def test_capture_async_propagates_mapper_defects() -> None:
    async def operation() -> int:
        msg = "expected"
        raise ValueError(msg)

    def broken_mapper(_: Exception) -> str:
        msg = "broken"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="broken"):
        await capture_async(operation, catch=broken_mapper)


@pytest.mark.asyncio
async def test_capture_async_returns_exception_without_catch() -> None:
    """Cover the ``catch is None`` branch when the operation raises."""

    async def operation() -> int:
        msg = "no catch"
        raise ValueError(msg)

    result = await capture_async(operation)
    assert isinstance(result, Err)
    assert isinstance(result.err_value, ValueError)


@pytest.mark.asyncio
async def test_capture_async_propagates_cancelled_error() -> None:
    async def operation() -> int:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await capture_async(operation)


# --- collect_results / collect_results_async ---


def test_collect_results_returns_all_successes() -> None:
    assert collect_results([Ok(1), Ok(2)]) == Ok((1, 2))


def test_collect_results_returns_all_errors_in_input_order() -> None:
    assert collect_results([Err("first"), Ok(2), Err("third")]) == Err(
        ("first", "third")
    )


def test_collect_results_empty_input_is_success() -> None:
    assert collect_results([]) == Ok(())


def test_collect_results_preserves_all_errors_when_all_fail() -> None:
    assert collect_results([Err("a"), Err("b"), Err("c")]) == Err(("a", "b", "c"))


@pytest.mark.asyncio
async def test_collect_results_async_with_pending_awaitables() -> None:
    """Exercise the await branch for non-Result items."""

    async def ok_op(value: int) -> Ok[int]:
        await asyncio.sleep(0)
        return Ok(value)

    # Pass tasks (not coroutines) to ensure the awaitable path is hit
    task = asyncio.ensure_future(ok_op(42))
    result = await collect_results_async([task])
    assert result == Ok((42,))


@pytest.mark.asyncio
async def test_collect_results_async_runs_awaitables_concurrently() -> None:
    second_started = asyncio.Event()
    release_first = asyncio.Event()

    async def first() -> Ok[int]:
        await release_first.wait()
        return Ok(1)

    async def second() -> Ok[int]:
        second_started.set()
        return Ok(2)

    task = asyncio.create_task(collect_results_async([first(), second()]))
    try:
        await asyncio.wait_for(second_started.wait(), timeout=1)
    finally:
        release_first.set()
    assert await task == Ok((1, 2))


@pytest.mark.asyncio
async def test_collect_results_async_with_immediate_results() -> None:
    """Cover the ``isinstance(result, Result)`` branch."""
    result = await collect_results_async([Ok(1), Ok(2), Ok(3)])
    assert result == Ok((1, 2, 3))


@pytest.mark.asyncio
async def test_async_collection_operations_preserve_mixed_input_order() -> None:
    async def delayed(value: int) -> Result[int, str]:
        await asyncio.sleep(0)
        return Ok(value)

    def results() -> list[Result[int, str] | Awaitable[Result[int, str]]]:
        return [Ok(1), delayed(2), Err("bad"), delayed(4)]

    assert await all_results_async(results()) == Err("bad")
    assert await collect_results_async(results()) == Err(("bad",))
    assert await partition_results_async(results()) == ([1, 2, 4], ["bad"])


@pytest.mark.asyncio
async def test_async_collection_operations_support_all_awaitable_inputs() -> None:
    async def result(value: int) -> Result[int, str]:
        await asyncio.sleep(0)
        return Ok(value)

    def inputs() -> list[Awaitable[Result[int, str]]]:
        return [result(1), result(2), result(3)]

    assert await all_results_async(inputs()) == Ok((1, 2, 3))
    assert await collect_results_async(inputs()) == Ok((1, 2, 3))
    assert await partition_results_async(inputs()) == ([1, 2, 3], [])


@pytest.mark.asyncio
async def test_collect_results_async_accumulates_errors() -> None:
    async def slow_ok(value: int) -> Ok[int]:
        await asyncio.sleep(0)
        return Ok(value)

    async def slow_err(message: str) -> Err[str]:
        await asyncio.sleep(0)
        return Err(message)

    result = await collect_results_async(
        [slow_ok(1), slow_err("failure"), slow_ok(3), slow_err("also failed")]
    )
    assert result == Err(("failure", "also failed"))


@pytest.mark.asyncio
async def test_collect_results_async_all_succeed() -> None:
    async def slow_ok(value: int) -> Ok[int]:
        await asyncio.sleep(0)
        return Ok(value)

    result = await collect_results_async([slow_ok(1), slow_ok(2)])
    assert result == Ok((1, 2))


@pytest.mark.asyncio
async def test_collect_results_async_empty() -> None:
    assert await collect_results_async([]) == Ok(())


# --- traverse / traverse_async ---


def test_traverse_maps_values_into_one_result() -> None:
    assert traverse((1, 2, 3), lambda value: Ok(value * 2)) == Ok((2, 4, 6))


def test_traverse_shorts_circuits_on_first_error() -> None:
    assert traverse((1, -1, 3), _maybe_fail) == Err("negative")


def test_traverse_empty_input() -> None:
    assert traverse([], Ok) == Ok(())


def _maybe_fail(value: int) -> Result[int, str]:
    if value < 0:
        return Err("negative")
    return Ok(value)


@pytest.mark.asyncio
async def test_traverse_async_concurrent_by_default() -> None:
    steps: list[float] = []
    started_second = asyncio.Event()

    async def slow_op(value: int) -> Ok[int]:
        steps.append(value)
        if value == 1:
            await asyncio.sleep(0.1)
        else:
            started_second.set()
        return Ok(value)

    task = asyncio.create_task(traverse_async([1, 2], slow_op))
    await started_second.wait()
    result = await task
    assert result == Ok((1, 2))


@pytest.mark.asyncio
async def test_traverse_async_returns_first_error() -> None:
    async def fail_on_two(value: int) -> Result[int, str]:
        if value == 2:
            return Err("error at 2")
        return Ok(value)

    result = await traverse_async([1, 2, 3], fail_on_two)
    assert result == Err("error at 2")


@pytest.mark.asyncio
async def test_traverse_async_max_concurrency_limits_active_tasks() -> None:
    active = 0
    max_active = 0
    lock = asyncio.Lock()

    async def track(value: int) -> Ok[int]:
        nonlocal active, max_active
        async with lock:
            active += 1
            max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        async with lock:
            active -= 1
        return Ok(value)

    result = await traverse_async([1, 2, 3, 4], track, max_concurrency=2)
    assert result == Ok((1, 2, 3, 4))
    assert max_active <= 2


@pytest.mark.asyncio
async def test_traverse_async_bounded_consumes_values_lazily() -> None:
    consumed = 0
    release = asyncio.Event()
    started = asyncio.Event()

    def values() -> Iterator[int]:
        nonlocal consumed
        for value in range(10):
            consumed += 1
            yield value

    async def operation(value: int) -> Ok[int]:
        started.set()
        await release.wait()
        return Ok(value)

    task = asyncio.create_task(traverse_async(values(), operation, max_concurrency=2))
    await started.wait()
    await asyncio.sleep(0)
    assert consumed == 2
    release.set()
    assert await task == Ok(tuple(range(10)))


@pytest.mark.asyncio
async def test_traverse_async_bounded_evaluates_all_values_after_err() -> None:
    evaluated: list[int] = []

    async def operation(value: int) -> Result[int, str]:
        evaluated.append(value)
        if value == 1:
            return Err("bad")
        return Ok(value)

    result = await traverse_async(range(5), operation, max_concurrency=2)

    assert result == Err("bad")
    assert evaluated == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_traverse_async_rejects_invalid_concurrency() -> None:
    async def ok_op(value: int) -> Ok[int]:
        return Ok(value)

    with pytest.raises(ValueError, match="concurrency"):
        await traverse_async([1], ok_op, max_concurrency=0)
    with pytest.raises(ValueError, match="concurrency"):
        await traverse_async([1], ok_op, max_concurrency=-1)
    with pytest.raises(ValueError, match="concurrency"):
        await traverse_async([1], ok_op, max_concurrency=True)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_traverse_async_propagates_unexpected_exceptions() -> None:
    async def broken(_value: int) -> Ok[int]:
        msg = "unexpected"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="unexpected"):
        await traverse_async([1], broken)


@pytest.mark.asyncio
async def test_traverse_async_empty() -> None:
    async def never_called(_value: int) -> Ok[int]:
        pytest.fail("should not be called")

    assert await traverse_async([], never_called) == Ok(())


@pytest.mark.asyncio
async def test_traverse_async_max_concurrency_none_is_unrestricted() -> None:
    async def ok_op(value: int) -> Ok[int]:
        return Ok(value)

    result = await traverse_async([1, 2, 3], ok_op, max_concurrency=None)
    assert result == Ok((1, 2, 3))


# --- End of new task tests ---


def _raise_zero(_: TryContext) -> Never:
    message = "division by zero"
    raise ZeroDivisionError(message)


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_try_async_forwards_cancellation_token_and_stops_retry_delay() -> None:
    token = CancellationToken()
    assert not token.is_cancelled
    token.raise_if_cancelled()
    contexts: list[CancellationToken | None] = []
    cancel_task: asyncio.Task[None] | None = None

    async def cancel_later() -> None:
        await asyncio.sleep(0)
        token.cancel()

    async def operation(context: TryContext) -> str:
        nonlocal cancel_task
        contexts.append(context.cancel_token)
        cancel_task = asyncio.create_task(cancel_later())
        message = "temporary"
        raise RuntimeError(message)

    with pytest.raises(asyncio.CancelledError):
        await try_async(
            operation,
            catch=str,
            cancel_token=token,
            retry=RetryPolicy.constant(times=3, delay=10),
        )

    assert contexts == [token]
    assert cancel_task is not None
    await cancel_task
    with pytest.raises(asyncio.CancelledError):
        token.raise_if_cancelled()
    assert token.is_cancelled


@pytest.mark.asyncio
async def test_try_async_cancels_during_a_retry_wait() -> None:
    token = CancellationToken()
    cancel_task: asyncio.Task[None] | None = None

    async def cancel_later() -> None:
        await asyncio.sleep(0.01)
        token.cancel()

    async def operation(_: TryContext) -> str:
        nonlocal cancel_task
        cancel_task = asyncio.create_task(cancel_later())
        message = "temporary"
        raise RuntimeError(message)

    with pytest.raises(asyncio.CancelledError):
        await try_async(
            operation,
            catch=str,
            cancel_token=token,
            retry=RetryPolicy.constant(times=1, delay=10),
        )

    assert cancel_task is not None
    await cancel_task


@pytest.mark.asyncio
async def test_try_async_cancels_an_in_flight_operation_with_the_token() -> None:
    token = CancellationToken()
    operation_started = asyncio.Event()
    operation_cleaned_up = asyncio.Event()

    async def operation(_: TryContext) -> str:
        operation_started.set()
        try:
            await asyncio.sleep(60)
        finally:
            operation_cleaned_up.set()
        return "unreachable"

    task = asyncio.create_task(try_async(operation, cancel_token=token))
    await operation_started.wait()
    token.cancel()

    await asyncio.wait_for(operation_cleaned_up.wait(), timeout=1.0)
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_try_async_propagates_native_task_cancellation() -> None:
    operation_started = asyncio.Event()
    operation_cleaned_up = asyncio.Event()

    async def operation(_: TryContext) -> str:
        operation_started.set()
        try:
            await asyncio.sleep(60)
        finally:
            operation_cleaned_up.set()
        return "unreachable"

    task = asyncio.create_task(try_async(operation, cancel_token=CancellationToken()))
    await operation_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert operation_cleaned_up.is_set()


@pytest.mark.asyncio
async def test_try_async_captures_errors_and_retries_successful_attempts() -> None:
    attempts: list[int] = []

    async def operation(context: TryContext) -> str:
        attempts.append(context.attempt)
        if len(attempts) < 3:
            message = "temporary"
            raise RuntimeError(message)
        return "ok"

    success_token = CancellationToken()
    result = await try_async(
        operation,
        catch=str,
        retry=RetryPolicy.constant(times=3),
        cancel_token=success_token,
    )
    assert result == Ok("ok")
    assert attempts == [1, 2, 3]
    assert await try_async(operation) == Ok("ok")


@pytest.mark.asyncio
async def test_try_async_can_stop_retries_with_a_predicate_and_support_async_catch() -> (
    None
):
    attempts: list[int] = []
    retry_errors: list[str] = []

    async def operation(context: TryContext) -> str:
        attempts.append(context.attempt)
        message = "permanent"
        raise RuntimeError(message)

    async def catch(exc: Exception) -> str:
        return str(exc)

    result = await try_async(
        operation,
        catch=catch,
        retry=RetryPolicy.constant(
            times=3,
            should_retry=lambda context: retry_errors.append(context.error) or False,
        ),
    )
    assert result == Err("permanent")
    assert attempts == [1]
    assert retry_errors == ["permanent"]


@pytest.mark.asyncio
async def test_try_async_applies_backoff_and_dynamic_delays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def always_fails(_: TryContext) -> str:
        message = "failed"
        raise RuntimeError(message)

    result = await try_async(
        always_fails,
        catch=str,
        retry=RetryPolicy.from_schedule(
            times=2,
            schedule=LinearBackoff(2),
        ),
    )
    assert result == Err("failed")
    assert delays == [2, 4]

    delays.clear()
    result = await try_async(
        always_fails,
        catch=str,
        retry=RetryPolicy[str].dynamic(
            times=2,
            delay=lambda context: context.attempt / 10,
        ),
    )
    assert result == Err("failed")
    assert delays == [0.1, 0.2]

    delays.clear()
    result = await try_async(
        always_fails,
        catch=str,
        retry=RetryPolicy.exponential(
            times=2,
            initial_delay=2,
            jitter=0.5,
        ),
    )
    assert result == Err("failed")
    assert len(delays) == 2
    assert 1 <= delays[0] <= 2
    assert 2 <= delays[1] <= 4


@pytest.mark.asyncio
async def test_try_async_validates_retry_policy_and_preserves_cancellation() -> None:
    async def unreachable(_: TryContext) -> str:
        return "unreachable"

    with pytest.raises(ValueError, match="jitter"):
        await try_async(
            unreachable,
            retry=RetryPolicy.exponential(
                times=1,
                initial_delay=0,
                jitter=1.1,
            ),
        )

    async def cancelled(_: TryContext) -> str:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await try_async(cancelled)

    async def fails(_: TryContext) -> str:
        message = "failed"
        raise RuntimeError(message)

    no_mapper = await try_async(fails)
    assert isinstance(no_mapper, Err)
    assert isinstance(no_mapper.err_value, RuntimeError)

    timeout_token = CancellationToken()
    timeout_result = await try_async(
        fails,
        retry=RetryPolicy.constant(times=1),
        cancel_token=timeout_token,
    )
    assert isinstance(timeout_result, Err)

    pre_cancelled = CancellationToken()
    pre_cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await try_async(
            fails,
            retry=RetryPolicy.constant(times=1, delay=10),
            cancel_token=pre_cancelled,
        )

    cancelled_by_policy = CancellationToken()

    def cancel_before_wait(_: RetryContext[str]) -> bool:
        cancelled_by_policy.cancel()
        return True

    with pytest.raises(asyncio.CancelledError):
        await try_async(
            fails,
            retry=RetryPolicy.constant(
                times=1,
                should_retry=cancel_before_wait,
            ),
            cancel_token=cancelled_by_policy,
        )

    cancelled_by_terminal_policy = CancellationToken()

    def cancel_before_stop(_: RetryContext[str]) -> bool:
        cancelled_by_terminal_policy.cancel()
        return False

    with pytest.raises(asyncio.CancelledError):
        await try_async(
            fails,
            retry=RetryPolicy.constant(
                times=1,
                should_retry=cancel_before_stop,
            ),
            cancel_token=cancelled_by_terminal_policy,
        )

    with pytest.raises(ValueError, match="retry"):
        try_result(lambda _: "unused", retry=-1)
    with pytest.raises(ValueError, match="retry"):
        try_result(lambda _: "unused", retry=True)
    with pytest.raises(ValueError, match="retry"):
        await try_async(
            unreachable,
            retry=RetryPolicy(times=-1, schedule=ConstantDelay(0)),
        )
    with pytest.raises(ValueError, match="retry"):
        RetryPolicy(times=True, schedule=ConstantDelay(0))
    with pytest.raises(ValueError, match="delay"):
        await try_async(
            unreachable,
            retry=RetryPolicy.constant(times=1, delay=-1),
        )
    with pytest.raises(ValueError, match="delay"):
        await try_async(
            fails,
            retry=RetryPolicy[Exception].dynamic(times=1, delay=lambda _: -1),
        )


@pytest.mark.asyncio
async def test_async_collection_operations_await_in_input_order() -> None:
    async def result(value: int) -> Ok[int]:
        await asyncio.sleep(0)
        return Ok(value)

    assert await all_results_async([result(1), Ok(2)]) == Ok((1, 2))
    assert await partition_results_async([result(1), Err("bad"), result(2)]) == (
        [1, 2],
        ["bad"],
    )
    assert await all_results_async([]) == Ok(())
    assert await partition_results_async([]) == ([], [])


@pytest.mark.asyncio
async def test_async_collection_operations_propagate_unexpected_rejections() -> None:
    async def broken() -> Ok[int]:
        message = "broken"
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="broken"):
        await all_results_async([broken()])

    with pytest.raises(RuntimeError, match="broken"):
        await partition_results_async([broken()])
