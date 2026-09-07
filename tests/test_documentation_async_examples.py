"""Coroutine-based equivalents of the better-result async examples."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Never

import pytest

from better_result.collections import all_results_async, partition_async
from better_result.combinators import and_then, and_then_async
from better_result.core import Err, Ok, Panic, Result, err, ok
from better_result.error import TaggedError
from better_result.retry import AsyncRetryConfig, TryAsyncContext, try_async


class NetworkError(TaggedError, tag="NetworkError"):
    url: str

    def __init__(self, url: str, cause: BaseException) -> None:
        super().__init__(message="Network request failed", url=url, cause=cause)


class HttpResponseError(TaggedError, tag="HttpResponseError"):
    status: int
    url: str

    def __init__(self, status: int, url: str) -> None:
        super().__init__(
            message=f"Request failed with status {status}",
            status=status,
            url=url,
        )


class ParseError(TaggedError, tag="ParseError"):
    def __init__(self, cause: BaseException) -> None:
        super().__init__(message="Could not parse JSON", cause=cause)


class RateLimited(TaggedError, tag="RateLimited"):
    retry_after_ms: float

    def __init__(self, retry_after_ms: float = 0.0) -> None:
        super().__init__(message="Rate limited", retry_after_ms=retry_after_ms)


@dataclass(frozen=True)
class Response:
    status: int
    url: str
    body: str

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


def parse_json_value(text: str) -> object:
    value = json.loads(text)
    if value is None or isinstance(value, (bool, float, int, str, list, dict)):
        return value
    raise TypeError("JSON decoder returned an unsupported value")


async def parse_json(text: str) -> Result[object, ParseError]:
    try:
        return Ok[object, ParseError](parse_json_value(text))
    except json.JSONDecodeError as cause:
        return Err[object, ParseError](ParseError(cause))


async def fetch_response(
    response: Response | BaseException,
    url: str,
) -> Result[Response, NetworkError]:
    async def operation(_context: TryAsyncContext) -> Response:
        if isinstance(response, BaseException):
            raise response
        return response

    return await try_async(
        operation,
        lambda cause: NetworkError(url, cause),
    )


async def get_user_profile(
    response: Response,
) -> Result[dict[str, object], TaggedError]:
    fetched = await fetch_response(response, response.url)
    if isinstance(fetched, Err):
        return Err[dict[str, object], TaggedError](fetched.error)

    checked = and_then(
        fetched,
        lambda selected: (
            ok(selected)
            if selected.ok
            else Err[Response, HttpResponseError](
                HttpResponseError(selected.status, selected.url),
            )
        ),
    )
    if isinstance(checked, Err):
        return Err[dict[str, object], TaggedError](checked.error)

    parsed = await parse_json(checked.value.body)
    if isinstance(parsed, Err):
        return Err[dict[str, object], TaggedError](parsed.error)
    if not isinstance(parsed.value, dict):
        return Err[dict[str, object], TaggedError](
            ParseError(TypeError("object expected")),
        )
    return Ok[dict[str, object], TaggedError](parsed.value)


@pytest.mark.asyncio
async def test_async_workflow_uses_await_and_explicit_short_circuiting() -> None:
    success = await get_user_profile(
        Response(200, "https://example.test/user/1", '{"id": 1, "name": "Alice"}'),
    )
    assert isinstance(success, Ok)
    assert success.value["name"] == "Alice"

    http_failure = await get_user_profile(
        Response(503, "https://example.test/user/1", "unreachable"),
    )
    assert isinstance(http_failure, Err)
    assert isinstance(http_failure.error, HttpResponseError)
    assert http_failure.error.status == 503

    parse_failure = await get_user_profile(
        Response(200, "https://example.test/user/1", "not json"),
    )
    assert isinstance(parse_failure, Err)
    assert isinstance(parse_failure.error, ParseError)


@pytest.mark.asyncio
async def test_async_combinator_pipeline_matches_promise_result_examples() -> None:
    async def fetch_posts(user: dict[str, object]) -> Result[list[str], NetworkError]:
        user_id = user["id"]
        return Ok[list[str], NetworkError]([f"post-for-{user_id}"])

    user = Ok[dict[str, object], NetworkError]({"id": 1, "name": "Alice"})
    result = await and_then_async(user, fetch_posts)
    assert isinstance(result, Ok)
    assert result.value == ["post-for-1"]

    posts_later = and_then_async(fetch_posts)
    result_later = await posts_later(user)
    assert isinstance(result_later, Ok)
    assert result_later.value == ["post-for-1"]


@pytest.mark.asyncio
async def test_async_retry_examples_are_bounded_and_deterministic() -> None:
    attempts: list[int] = []

    async def call_api(context: TryAsyncContext) -> str:
        attempts.append(context.attempt)
        if context.attempt < 3:
            raise RuntimeError("temporary")
        return "ready"

    result = await try_async(
        call_api,
        lambda _cause: RateLimited(),
        AsyncRetryConfig[RateLimited](
            times=3,
            delay_ms=0,
            backoff="exponential",
            should_retry=lambda error, context: (
                error.retry_after_ms == 0 and context.attempt < 3
            ),
        ),
    )

    assert isinstance(result, Ok)
    assert result.value == "ready"
    assert attempts == [1, 2, 3]


@pytest.mark.asyncio
async def test_async_cancellation_stops_retry_scheduling() -> None:
    cancel_event = asyncio.Event()
    attempts: list[int] = []

    async def call_api(context: TryAsyncContext) -> str:
        attempts.append(context.attempt)
        raise RuntimeError("temporary")

    def stop_retry(_error: RateLimited, _context: TryAsyncContext) -> bool:
        cancel_event.set()
        return True

    result = await try_async(
        call_api,
        lambda _cause: RateLimited(),
        AsyncRetryConfig[RateLimited](
            times=3,
            delay_ms=100,
            should_retry=stop_retry,
            cancel_event=cancel_event,
        ),
    )

    assert isinstance(result, Err)
    assert isinstance(result.error, RateLimited)
    assert attempts == [1]


@pytest.mark.asyncio
async def test_async_collections_collect_and_partition_without_generators() -> None:
    async def load(value: int) -> Result[int, Never]:
        await asyncio.sleep(0)
        return ok(value)

    collected = await all_results_async([load(1), load(2), ok(3)])
    assert isinstance(collected, Ok)
    assert collected.value == [1, 2, 3]

    partitioned = await partition_async([load(1), err("missing"), load(3)])
    assert partitioned == ([1, 3], ["missing"])

    async def rejected() -> Result[int, str]:
        raise RuntimeError("broken promise")

    with pytest.raises(Panic, match="input awaitable rejected"):
        await all_results_async([rejected()])
