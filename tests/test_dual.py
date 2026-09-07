"""Tests for data-first/data-last function adaptation."""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_type, cast

import pytest

from better_result.dual import dual


def test_dual_supports_data_first_and_data_last_calls() -> None:
    def add_body(left: int, right: int) -> int:
        return left + right

    add = dual(2, add_body)

    assert_type(add(2, 3), int)
    assert add(2, 3) == 5
    assert_type(add(3), Callable[[int], int])
    assert add(3)(2) == 5


def test_dual_supports_three_and_four_argument_functions() -> None:
    def join_body(first: str, second: str, third: str) -> str:
        return f"{first}-{second}-{third}"

    def make_url_body(scheme: str, host: str, path: str, query: str) -> str:
        return f"{scheme}://{host}/{path}?{query}"

    join = dual(3, join_body)
    make_url = dual(4, make_url_body)

    assert_type(join("a", "b", "c"), str)
    assert join("a", "b", "c") == "a-b-c"
    join_from_two = join("b", "c")
    assert_type(join_from_two, Callable[[str], str])
    assert join_from_two("a") == "a-b-c"
    assert (
        make_url("https", "example.com", "search", "q=1")
        == "https://example.com/search?q=1"
    )
    make_url_from_three = make_url("example.com", "search", "q=1")
    assert_type(make_url_from_three, Callable[[str], str])
    assert make_url_from_three("https") == "https://example.com/search?q=1"


def test_dual_rejects_non_positive_arities() -> None:
    with pytest.raises(ValueError, match="arity must be positive"):
        dual(0, lambda value: value)


def test_dual_supports_all_declared_partial_forms() -> None:
    make_url = dual(
        4,
        lambda scheme, host, path, query: f"{scheme}://{host}/{path}?{query}",
    )

    assert make_url("example.com", "search", "q=1")("https") == (
        "https://example.com/search?q=1"
    )
    assert make_url("search", "q=1")("https", "example.com") == (
        "https://example.com/search?q=1"
    )
    assert make_url("q=1")("https", "example.com", "search") == (
        "https://example.com/search?q=1"
    )


def test_dual_rejects_excess_arguments() -> None:
    add = dual(2, lambda left, right: left + right)

    with pytest.raises(TypeError, match="expected at most 2 arguments"):
        cast("Callable[..., object]", add)(1, 2, 3)
