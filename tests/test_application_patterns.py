"""
Regression tests for application-boundary error patterns.

The application-patterns guide says that strings should not be the error
vocabulary at a module boundary.  These tests model a small boundary and
exercise the public tagged-error API callers use instead.
"""

from __future__ import annotations

from typing import assert_type

from better_result.core import Err, Ok, Result, err
from better_result.error import TaggedError, match_error


class InvalidPort(TaggedError, tag="InvalidPort"):
    """The caller supplied a value that is not a valid port."""

    input: str

    def __init__(self, input_value: str) -> None:
        super().__init__(message="Expected a port from 1 to 65535", input=input_value)


class PortUnavailable(TaggedError, tag="PortUnavailable"):
    """The port source could not be reached."""

    cause: BaseException

    def __init__(self, cause: BaseException) -> None:
        super().__init__(message="Port source unavailable", cause=cause)


type LoadPortError = InvalidPort | PortUnavailable


def parse_port(input_value: str) -> Result[int, InvalidPort]:
    """Parse untrusted input and return a tagged boundary error."""
    try:
        port = int(input_value)
    except ValueError:
        return err(InvalidPort(input_value))
    if not 1 <= port <= 65_535:
        return err(InvalidPort(input_value))
    return Ok[int, InvalidPort](port)


def load_port(
    input_value: str,
    *,
    source_fails: bool = False,
) -> Result[int, LoadPortError]:
    """Compose parsing with a second typed failure at the application edge."""
    parsed = parse_port(input_value)
    if isinstance(parsed, Err):
        return Err[int, LoadPortError](parsed.error)
    if source_fails:
        return Err[int, LoadPortError](
            PortUnavailable(ConnectionError("port source is down")),
        )
    return Ok[int, LoadPortError](parsed.value)


def test_application_boundaries_do_not_return_string_errors() -> None:
    invalid = parse_port("not-a-port")
    unavailable = load_port("8080", source_fails=True)

    assert_type(invalid, Result[int, InvalidPort])
    assert_type(unavailable, Result[int, LoadPortError])

    assert isinstance(invalid, Err)
    assert isinstance(invalid.error, InvalidPort)
    assert not isinstance(invalid.error, str)
    assert invalid.error.input == "not-a-port"

    assert isinstance(unavailable, Err)
    assert isinstance(unavailable.error, PortUnavailable)
    assert not isinstance(unavailable.error, str)
    assert isinstance(unavailable.error.cause, ConnectionError)


def test_tagged_boundary_errors_are_exhaustively_matchable() -> None:
    invalid = parse_port("nope")
    unavailable = load_port("8080", source_fails=True)

    assert isinstance(invalid, Err)
    assert isinstance(unavailable, Err)

    handlers = {
        "InvalidPort": lambda error: f"invalid input: {error.input}",
        "PortUnavailable": lambda error: f"unavailable: {error.message}",
    }

    assert match_error(invalid.error, handlers) == "invalid input: nope"
    assert (
        match_error(unavailable.error, handlers)
        == "unavailable: Port source unavailable"
    )


def test_tagged_boundary_errors_preserve_context_when_serialized() -> None:
    invalid = parse_port("70000")
    unavailable = load_port("8080", source_fails=True)

    assert isinstance(invalid, Err)
    assert isinstance(unavailable, Err)

    invalid_payload = invalid.error.to_json()
    unavailable_payload = unavailable.error.to_json()

    assert invalid_payload["_tag"] == "InvalidPort"
    assert invalid_payload["input"] == "70000"
    assert invalid_payload["message"] == "Expected a port from 1 to 65535"
    assert unavailable_payload["_tag"] == "PortUnavailable"
    assert unavailable_payload["message"] == "Port source unavailable"
    cause_payload = unavailable_payload["cause"]
    assert isinstance(cause_payload, dict)
    assert cause_payload["name"] == "ConnectionError"
    assert cause_payload["message"] == "port source is down"
    assert isinstance(cause_payload["stack"], str)


def test_successful_boundary_values_remain_values_not_errors() -> None:
    result = load_port("8080")

    assert isinstance(result, Ok)
    assert result.value == 8080
