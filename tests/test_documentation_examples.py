"""Executable examples for the supported root API."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, assert_type

import pytest

from better_result import Err, Ok, PanicError, Result, TaggedError

if TYPE_CHECKING:
    from collections.abc import Mapping


class MissingEnvError(TaggedError, tag="MissingEnv"):
    env_name: str

    def __init__(self, name: str) -> None:
        super().__init__(message=f"{name} is required", env_name=name)


class InvalidPortError(TaggedError, tag="InvalidPort"):
    input: str

    def __init__(self, input_value: str) -> None:
        super().__init__(message="Expected a port from 1 to 65535", input=input_value)


def read_env(environment: Mapping[str, str], name: str) -> Result[str, MissingEnvError]:
    value = environment.get(name)
    if value is None:
        return Err(MissingEnvError(name))
    return Ok(value)


def parse_port(input_value: str) -> Result[int, InvalidPortError]:
    try:
        port = int(input_value)
    except ValueError:
        return Err(InvalidPortError(input_value))
    if not 1 <= port <= 65_535:
        return Err(InvalidPortError(input_value))
    return Ok(port)


def read_server_address(
    environment: Mapping[str, str],
) -> Result[str, MissingEnvError | InvalidPortError]:
    return read_env(environment, "HOST").and_then(
        lambda host: read_env(environment, "PORT").and_then(
            lambda port_text: parse_port(port_text).map(
                lambda port: f"http://{host}:{port}",
            ),
        ),
    )


def test_quickstart_workflow_uses_one_root_import_and_composition_style() -> None:
    result = read_server_address({"HOST": "localhost", "PORT": "8080"})

    assert_type(result, Result[str, MissingEnvError | InvalidPortError])
    assert (
        result.match(
            lambda address: address,
            lambda error: error.message,
        )
        == "http://localhost:8080"
    )

    missing_host = read_server_address({"PORT": "8080"})
    invalid_port = read_server_address({"HOST": "localhost", "PORT": "nope"})

    assert isinstance(missing_host, Err)
    assert isinstance(missing_host.error, MissingEnvError)
    assert isinstance(invalid_port, Err)
    assert isinstance(invalid_port.error, InvalidPortError)


def test_result_matching_is_the_only_branch_consumer_in_examples() -> None:
    result: Result[int, str] = Ok(42)
    assert result.match(lambda value: f"Success: {value}", str) == "Success: 42"
    assert Err("failed").match(str, lambda error: f"Error: {error}") == "Error: failed"
    assert Err("failed").unwrap_or(0) == 0


def test_transforming_uses_map_and_and_then() -> None:
    parsed: Result[int, InvalidPortError] = Ok(21)
    assert parsed.map(lambda value: value * 2) == Ok(42)

    failure: Result[int, InvalidPortError] = Err(InvalidPortError("x"))
    translated = failure.map_error(lambda error: error.message)
    assert isinstance(translated, Err)
    assert translated.error == "Expected a port from 1 to 65535"

    chained = parsed.and_then(lambda value: Ok(str(value)))
    assert chained == Ok("21")


def test_callbacks_fail_fast_as_panics() -> None:
    with pytest.raises(PanicError, match="map callback threw"):
        Ok(1).map(lambda _value: (_ for _ in ()).throw(ZeroDivisionError("bug")))


def test_tagged_error_transport_serialization_is_safe() -> None:
    error = InvalidPortError("not-a-port")
    payload = error.to_safe_json()
    json.dumps(payload)
    assert payload["_tag"] == "InvalidPort"
    assert "stack" not in payload
