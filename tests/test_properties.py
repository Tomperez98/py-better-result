"""Property-based tests for public Result, retry, and cancellation contracts."""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings, strategies as st

if TYPE_CHECKING:
    from collections.abc import Iterator

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
    ResultDeserializationError,
    RetryAfter,
    RetryContext,
    RetryPolicy,
    StopRetry,
    TryContext,
    all_results,
    async_codec,
    codec,
    flatten_result,
    partition_results,
    try_async,
    try_result,
)


@given(
    is_ok=st.booleans(),
    value=st.integers(),
    error=st.text(),
    left=st.integers(),
    right=st.integers(),
)
def test_result_map_identity_and_composition_laws(
    *,
    is_ok: bool,
    value: int,
    error: str,
    left: int,
    right: int,
) -> None:
    result: Result[int, str] = Ok(value) if is_ok else Err(error)

    assert result.map(lambda item: item) == result
    assert result.map(lambda item: item + left).map(
        lambda item: item * right,
    ) == result.map(lambda item: (item + left) * right)


@settings(max_examples=150)
@given(
    items=st.lists(
        st.tuples(st.booleans(), st.integers(), st.text()),
        max_size=30,
    ),
)
def test_result_collections_match_their_reference_models(
    items: list[tuple[bool, int, str]],
) -> None:
    results: list[Result[int, str]] = [
        Ok(value) if is_ok else Err(error) for is_ok, value, error in items
    ]
    expected_values = [value for is_ok, value, _ in items if is_ok]
    expected_errors = [error for is_ok, _, error in items if not is_ok]

    collected = all_results(results)
    if expected_errors:
        assert collected == Err(expected_errors[0])
    else:
        assert collected == Ok(tuple(expected_values))

    assert partition_results(results) == (expected_values, expected_errors)


@given(
    outer_is_ok=st.booleans(),
    inner_is_ok=st.booleans(),
    value=st.integers(),
    inner_error=st.text(),
    outer_error=st.text(),
)
def test_flatten_result_selects_the_active_error_layer(
    *,
    outer_is_ok: bool,
    inner_is_ok: bool,
    value: int,
    inner_error: str,
    outer_error: str,
) -> None:
    inner: Result[int, str] = Ok(value) if inner_is_ok else Err(inner_error)
    nested: Result[Result[int, str], str] = (
        Ok(inner) if outer_is_ok else Err(outer_error)
    )

    flattened = flatten_result(nested)
    expected: Result[int, str] = inner if outer_is_ok else Err(outer_error)
    assert flattened == expected


@settings(max_examples=100)
@given(failures=st.integers(min_value=0, max_value=12), retries=st.integers(0, 12))
def test_try_result_never_exceeds_its_retry_bound(
    failures: int,
    retries: int,
) -> None:
    attempts: list[int] = []

    def operation(context: TryContext) -> str:
        attempts.append(context.attempt)
        if len(attempts) <= failures:
            message = "temporary"
            raise RuntimeError(message)
        return "ok"

    result = try_result(operation, catch=str, retry=retries)
    expected_attempt_count = min(failures, retries) + 1

    assert attempts == list(range(1, expected_attempt_count + 1))
    if failures <= retries:
        assert result == Ok("ok")
    else:
        assert result == Err("temporary")


@settings(max_examples=100)
@given(
    initial=st.integers(min_value=0, max_value=100),
    factor=st.integers(min_value=1, max_value=4),
    attempt=st.integers(min_value=1, max_value=8),
    jitter=st.floats(min_value=0, max_value=1, allow_nan=False),
)
def test_backoff_schedules_preserve_finite_nonnegative_bounds(
    initial: int,
    factor: int,
    attempt: int,
    jitter: float,
) -> None:
    context = RetryContext(error="temporary", attempt=attempt)
    linear_delay = LinearBackoff(initial).delay_for(context)
    exponential_delay = ExponentialBackoff(initial, factor).delay_for(context)
    jittered_delay = Jittered(
        ExponentialBackoff(initial, factor),
        jitter,
    ).delay_for(context)

    assert linear_delay == initial * attempt
    assert exponential_delay == initial * factor ** (attempt - 1)
    assert math.isfinite(jittered_delay)
    assert 0 <= jittered_delay <= exponential_delay


@settings(max_examples=20)
@given(should_retry=st.booleans())
def test_token_cancellation_never_becomes_a_domain_error(
    *,
    should_retry: bool,
) -> None:
    async def run() -> None:
        token = CancellationToken()

        async def operation(_: TryContext) -> str:
            message = "temporary"
            raise RuntimeError(message)

        def predicate(_: RetryContext[str]) -> bool:
            token.cancel()
            return should_retry

        with pytest.raises(asyncio.CancelledError):
            await try_async(
                operation,
                catch=str,
                retry=RetryPolicy.constant(
                    times=1,
                    should_retry=predicate,
                ),
                cancel_token=token,
            )

    asyncio.run(run())


@given(
    result=st.one_of(
        st.builds(Ok, st.integers()),
        st.builds(Err, st.text()),
    ),
)
def test_result_map_err_identity_and_composition_laws(
    result: Result[int, str],
) -> None:
    assert result.map_err(lambda error: error) == result
    assert result.map_err(str.upper).map_err(
        lambda error: f"[{error}]",
    ) == result.map_err(lambda error: f"[{error.upper()}]")


@given(
    result=st.one_of(
        st.builds(Ok, st.integers()),
        st.builds(Err, st.text()),
    ),
)
def test_result_and_then_is_associative(result: Result[int, str]) -> None:
    def first(value: int) -> Result[int, str]:
        if value % 3 == 0:
            return Err("first")
        return Ok(value + 1)

    def second(value: int) -> Result[int, str]:
        if value % 5 == 0:
            return Err("second")
        return Ok(value * 2)

    left = result.and_then(first).and_then(second)
    right = result.and_then(lambda value: first(value).and_then(second))
    assert left == right


@given(
    result=st.one_of(
        st.builds(Ok, st.integers()),
        st.builds(Err, st.text()),
    ),
)
def test_result_or_else_is_associative(result: Result[int, str]) -> None:
    def recover(error: str) -> Result[int, str]:
        if len(error) % 2 == 0:
            return Ok(len(error))
        return Err(f"recover:{error}")

    def fallback(error: str) -> Result[int, str]:
        return Ok(-len(error))

    left = result.or_else(recover).or_else(fallback)
    right = result.or_else(lambda error: recover(error).or_else(fallback))
    assert left == right


@settings(max_examples=100)
@given(
    items=st.lists(
        st.one_of(
            st.builds(Ok, st.integers()),
            st.builds(Err, st.text()),
        ),
        max_size=30,
    ),
)
def test_all_results_stops_consuming_after_the_first_error(
    items: list[Result[int, str]],
) -> None:
    consumed: list[int] = []

    def source() -> Iterator[Result[int, str]]:
        for index, item in enumerate(items):
            consumed.append(index)
            yield item

    all_results(source())

    first_error = next(
        (index for index, item in enumerate(items) if isinstance(item, Err)),
        len(items),
    )
    expected_consumed = first_error + 1 if first_error < len(items) else len(items)
    assert consumed == list(range(expected_consumed))


@settings(max_examples=100)
@given(
    times=st.integers(min_value=0, max_value=12),
    attempt=st.integers(min_value=1, max_value=20),
    predicate_result=st.booleans(),
)
def test_retry_policy_decide_only_evaluates_retry_callbacks_when_needed(
    times: int,
    attempt: int,
    *,
    predicate_result: bool,
) -> None:
    context = RetryContext(error="temporary", attempt=attempt)
    seen_schedule_contexts: list[RetryContext[str]] = []
    seen_predicate_contexts: list[RetryContext[str]] = []

    def delay(context: RetryContext[str]) -> float:
        seen_schedule_contexts.append(context)
        return context.attempt / 10

    def should_retry(context: RetryContext[str]) -> bool:
        seen_predicate_contexts.append(context)
        return predicate_result

    policy = RetryPolicy(
        times=times,
        schedule=DynamicDelay(delay),
        should_retry=should_retry,
    )
    decision = policy.decide(context)

    within_retry_bound = attempt <= times
    if within_retry_bound and predicate_result:
        assert decision == RetryAfter(attempt / 10)
        assert seen_schedule_contexts == [context]
    else:
        assert isinstance(decision, StopRetry)
        assert seen_schedule_contexts == []

    assert seen_predicate_contexts == ([context] if within_retry_bound else [])


@settings(max_examples=100)
@given(
    failures=st.integers(min_value=0, max_value=12),
    times=st.integers(min_value=0, max_value=12),
    stop_at=st.integers(min_value=1, max_value=14),
)
def test_try_result_retry_predicates_control_exact_attempt_count(
    failures: int,
    times: int,
    stop_at: int,
) -> None:
    attempts: list[int] = []

    def operation(context: TryContext) -> str:
        attempts.append(context.attempt)
        if len(attempts) <= failures:
            message = "temporary"
            raise RuntimeError(message)
        return "ok"

    result = try_result(
        operation,
        catch=str,
        retry=RetryPolicy.constant(
            times=times,
            should_retry=lambda context: context.attempt < stop_at,
        ),
    )

    allowed_retries = min(times, stop_at - 1)
    expected_attempt_count = min(failures, allowed_retries) + 1
    assert attempts == list(range(1, expected_attempt_count + 1))
    if failures <= allowed_retries:
        assert result == Ok("ok")
    else:
        assert result == Err("temporary")


@settings(max_examples=50, deadline=None)
@given(
    failures=st.integers(min_value=0, max_value=8),
    retries=st.integers(min_value=0, max_value=8),
)
def test_sync_and_async_retry_contracts_are_equivalent(
    failures: int,
    retries: int,
) -> None:
    sync_attempts: list[int] = []

    def sync_operation(context: TryContext) -> str:
        sync_attempts.append(context.attempt)
        if len(sync_attempts) <= failures:
            message = "temporary"
            raise RuntimeError(message)
        return "ok"

    sync_result = try_result(
        sync_operation,
        catch=str,
        retry=retries,
    )

    async_attempts: list[int] = []

    async def async_operation(context: TryContext) -> str:
        async_attempts.append(context.attempt)
        if len(async_attempts) <= failures:
            message = "temporary"
            raise RuntimeError(message)
        return "ok"

    async def run_async() -> Result[str, str | Exception]:
        return await try_async(
            async_operation,
            catch=str,
            retry=retries,
        )

    async_result = asyncio.run(run_async())
    assert sync_result == async_result
    assert sync_attempts == async_attempts


invalid_delay_values = st.one_of(
    st.floats(max_value=-1, allow_nan=False, allow_infinity=False),
    st.just(float("nan")),
    st.just(float("inf")),
    st.just(float("-inf")),
    st.booleans(),
)


@settings(max_examples=75)
@given(delay=invalid_delay_values)
def test_retry_delay_validation_rejects_generated_invalid_values(
    *,
    delay: float | bool,
) -> None:
    with pytest.raises(ValueError, match="delay"):
        ConstantDelay(delay)
    with pytest.raises(ValueError, match="delay"):
        RetryAfter(delay)
    with pytest.raises(ValueError, match="delay"):
        DynamicDelay(lambda _: delay).delay_for(
            RetryContext(error="temporary", attempt=1),
        )


invalid_backoff_factors = st.one_of(
    st.floats(max_value=0.999999, allow_nan=False, allow_infinity=False),
    st.just(float("nan")),
    st.just(float("inf")),
    st.just(float("-inf")),
    st.booleans(),
)


@settings(max_examples=75)
@given(factor=invalid_backoff_factors)
def test_backoff_factor_validation_rejects_generated_invalid_values(
    *,
    factor: float | bool,
) -> None:
    with pytest.raises(ValueError, match="factor"):
        ExponentialBackoff(1, factor=factor)


invalid_jitter_factors = st.one_of(
    st.floats(max_value=-1e-9, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.000001, allow_nan=False, allow_infinity=False),
    st.just(float("nan")),
    st.just(float("inf")),
    st.just(float("-inf")),
    st.booleans(),
)


@settings(max_examples=75)
@given(factor=invalid_jitter_factors)
def test_jitter_factor_validation_rejects_generated_invalid_values(
    *,
    factor: float | bool,
) -> None:
    with pytest.raises(ValueError, match="jitter"):
        Jittered(ConstantDelay(1), factor=factor)


@given(
    is_ok=st.booleans(),
    value=st.integers(),
    error=st.text(),
)
def test_identity_codec_round_trips_generated_results(
    *,
    is_ok: bool,
    value: int,
    error: str,
) -> None:
    result: Result[int, str] = Ok(value) if is_ok else Err(error)
    result_codec = codec(
        serialize_ok=lambda item: item,
        serialize_err=lambda item: item,
        deserialize_ok=lambda item: item,
        deserialize_err=lambda item: item,
    )

    wire = result_codec.serialize_unsafe(result)
    expected_wire = (
        {"status": "ok", "value": value}
        if is_ok
        else {"status": "error", "error": error}
    )
    assert wire == expected_wire
    assert result_codec.deserialize_unsafe(wire) == result


@given(
    is_ok=st.booleans(),
    value=st.integers(),
    error=st.text(),
)
def test_identity_async_codec_round_trips_generated_results(
    *,
    is_ok: bool,
    value: int,
    error: str,
) -> None:
    result: Result[int, str] = Ok(value) if is_ok else Err(error)

    async def identity(item: object) -> object:
        return item

    async def run() -> tuple[object, Result[object, object]]:
        result_codec = async_codec(
            serialize_ok=identity,
            serialize_err=identity,
            deserialize_ok=identity,
            deserialize_err=identity,
        )
        wire = await result_codec.serialize_unsafe(result)
        decoded = await result_codec.deserialize_unsafe(wire)
        return wire, decoded

    wire, decoded = asyncio.run(run())
    expected_wire = (
        {"status": "ok", "value": value}
        if is_ok
        else {"status": "error", "error": error}
    )
    assert wire == expected_wire
    assert decoded == result


@given(
    status=st.one_of(
        st.none(),
        st.integers(),
        st.lists(st.integers(), max_size=3),
        st.text().filter(lambda value: value not in {"ok", "error"}),
    ),
)
def test_codec_rejects_generated_invalid_envelopes(status: object) -> None:
    result_codec = codec(
        serialize_ok=lambda item: item,
        serialize_err=lambda item: item,
        deserialize_ok=lambda item: item,
        deserialize_err=lambda item: item,
    )
    value = {"status": status}

    decoded = result_codec.deserialize(value)
    assert isinstance(decoded, Err)
    assert isinstance(decoded.err_value, ResultDeserializationError)
    assert decoded.err_value.value == value
    assert decoded.err_value.issues is None
