"""
Static type contracts for the supported Result API.

These assertions are checked by ``ty`` as part of the project checks. The
runtime test body also exercises the same callbacks so the examples remain
executable. Named callbacks intentionally pin the input types that ``ty``
cannot infer for lambdas passed through the ``Ok | Err`` union.
"""

from __future__ import annotations

from typing import assert_type

from better_result import Err, Ok, Result, TaggedError


class InvalidPortError(TaggedError, tag="TypeContractInvalidPort"):
    raw: str

    def __init__(self, raw: str) -> None:
        super().__init__(message="invalid port", raw=raw)


def parse_port(raw: str) -> Result[int, InvalidPortError]:
    try:
        port = int(raw)
    except ValueError:
        return Err(InvalidPortError(raw))
    return Ok(port) if 1 <= port <= 65_535 else Err(InvalidPortError(raw))


def format_address(port: int) -> str:
    assert_type(port, int)
    return f"http://localhost:{port}"


def format_error(error: InvalidPortError) -> str:
    assert_type(error, InvalidPortError)
    assert_type(error.raw, str)
    assert_type(error.message, str)
    return f"Invalid input {error.raw!r}: {error.message}"


def test_map_and_match_preserve_their_result_types() -> None:
    parsed = parse_port("8080")
    assert_type(parsed, Result[int, InvalidPortError])

    mapped = parsed.map(format_address)
    assert_type(mapped, Result[str, InvalidPortError])
    assert mapped == Ok("http://localhost:8080")

    message = mapped.match(str, format_error)
    assert_type(message, str)
    assert message == "http://localhost:8080"


def test_error_branch_is_unchanged_by_map_and_consumed_by_match() -> None:
    failed: Result[int, InvalidPortError] = Err(InvalidPortError("nope"))

    mapped = failed.map(format_address)
    assert_type(mapped, Err[InvalidPortError])
    assert mapped == failed

    message = mapped.match(str, format_error)
    assert_type(message, str)
    assert message == "Invalid input 'nope': invalid port"
