"""Executable Python equivalents of the public better-result examples."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, assert_type

from better_result.collections import all_results, partition
from better_result.retry import try_result

if TYPE_CHECKING:
    from collections.abc import Mapping

import pytest

from better_result.core import Err, Ok, PanicError, Result, is_err, is_ok
from better_result.error import TaggedError


class MissingEnvError(TaggedError, tag="MissingEnv"):
    env_name: str

    def __init__(self, name: str) -> None:
        super().__init__(message=f"{name} is required", env_name=name)


class InvalidPortError(TaggedError, tag="InvalidPort"):
    input: str

    def __init__(self, input_value: str) -> None:
        super().__init__(message="Expected a port from 1 to 65535", input=input_value)


class InvalidJsonError(TaggedError, tag="InvalidJson"):
    input: str

    def __init__(self, input_value: str, cause: BaseException) -> None:
        super().__init__(
            message="Input is not valid JSON",
            input=input_value,
            cause=cause,
        )


class NotFoundError(TaggedError, tag="NotFound"):
    def __init__(self, item_id: str) -> None:
        super().__init__(message=f"{item_id} was not found", item_id=item_id)


class DatabaseUnavailableError(TaggedError, tag="DatabaseUnavailable"):
    def __init__(self, message: str = "database unavailable") -> None:
        super().__init__(message=message)


class ValidationError(TaggedError, tag="ValidationError"):
    def __init__(self, field: str) -> None:
        super().__init__(message=f"Invalid {field}", field=field)


class LoadUserFailedError(TaggedError, tag="LoadUserFailed"):
    def __init__(self, cause: TaggedError) -> None:
        super().__init__(message="Could not load user", cause=cause)


class InvalidInputError(TaggedError, tag="InvalidInput"):
    def __init__(self, value: str) -> None:
        super().__init__(message="Invalid input", value=value)


class GuestUser:
    name = "Guest"


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


def test_quickstart_workflow_translates_to_explicit_python_composition() -> None:
    result = read_server_address({"HOST": "localhost", "PORT": "8080"})

    assert_type(result, Result[str, MissingEnvError | InvalidPortError])
    assert isinstance(result, Ok)
    assert result.value == "http://localhost:8080"

    missing_host = read_server_address({"PORT": "8080"})
    invalid_port = read_server_address({"HOST": "localhost", "PORT": "nope"})

    assert isinstance(missing_host, Err)
    assert isinstance(missing_host.error, MissingEnvError)
    assert missing_host.error.env_name == "HOST"
    assert isinstance(invalid_port, Err)
    assert isinstance(invalid_port.error, InvalidPortError)
    assert invalid_port.error.input == "nope"


def test_quickstart_result_and_tagged_error_matching() -> None:
    result = read_server_address({"HOST": "localhost", "PORT": "bad"})

    exit_code = result.match(
        lambda address: 0 if address else 1,
        lambda error: error.match(
            {
                "MissingEnv": lambda missing: len(missing.env_name),
                "InvalidPort": lambda invalid: len(invalid.input),
            },
        ),
    )

    assert exit_code == 3


def parse_json_value(input_value: str) -> object:
    value = json.loads(input_value)
    if value is None or isinstance(value, (bool, float, int, str, list, dict)):
        return value
    msg = "JSON decoder returned an unsupported value"
    raise TypeError(msg)


def parse_json(input_value: str) -> Result[object, InvalidJsonError]:
    return try_result(
        lambda _: parse_json_value(input_value),
        lambda cause: InvalidJsonError(input_value, cause),
    )


def test_creating_results_wraps_sync_exceptions_and_customizes_errors() -> None:
    parsed = parse_json('{"name": "Alice"}')
    failed = parse_json("not json")

    assert isinstance(parsed, Ok)
    assert parsed.value == {"name": "Alice"}
    assert isinstance(failed, Err)
    assert isinstance(failed.error, InvalidJsonError)
    assert failed.error.input == "not json"
    assert isinstance(failed.error.cause, json.JSONDecodeError)

    ok_result = Ok(42)
    assert ok_result.value == 42
    failure = Err("Something went wrong")
    assert failure.error == "Something went wrong"


def test_narrowing_and_matching_examples() -> None:
    result: Result[int, str] = Ok(42)

    if is_ok(result):
        assert result.value == 42
    else:
        msg = "unexpected error"
        raise AssertionError(msg)

    assert is_ok(result)
    assert result.value == 42
    assert not is_err(result)

    message = result.match(
        lambda value: f"Success: {value}",
        lambda error: f"Error: {error}",
    )
    assert message == "Success: 42"

    assert result.match(lambda value: value, lambda _error: 0) == 42


def test_transforming_chaining_and_recovery() -> None:
    parsed: Result[int, InvalidInputError] = Ok(21)
    doubled = parsed.map(lambda value: value * 2)
    assert isinstance(doubled, Ok)
    assert doubled.value == 42

    failure: Result[int, InvalidInputError] = Err(InvalidInputError("x"))
    translated = failure.map_error(LoadUserFailedError)
    assert isinstance(translated, Err)
    assert isinstance(translated.error, LoadUserFailedError)

    chained = parsed.and_then(
        lambda value: Ok(str(value)) if value > 0 else Err(ValidationError("value")),
    )
    assert isinstance(chained, Ok)
    assert chained.value == "21"

    recovered = Err(NotFoundError("missing")).try_recover(
        lambda _error: Ok(GuestUser()),
    )
    assert isinstance(recovered, Ok)
    assert isinstance(recovered.value, GuestUser)

    assert isinstance(Ok(1).map(lambda value: value + 1), Ok)
    assert Err("failed").map_error(str.upper).error == "FAILED"


def test_observing_and_extracting_values() -> None:
    seen: list[int] = []
    result = Ok(2)
    observed = result.tap(seen.append)

    assert observed is result
    assert seen == [2]
    tapped_error = Err("failed").tap(seen.append)
    assert isinstance(tapped_error, Err)
    assert tapped_error.error == "failed"

    assert Err("failed").unwrap_or(0) == 0
    assert Ok(42).unwrap() == 42
    with pytest.raises(PanicError):
        Err("failed").unwrap()


def test_collections_and_flatten_examples() -> None:
    collected = all_results([Ok(1), Ok(2), Ok(3)])
    assert isinstance(collected, Ok)
    assert collected.value == [1, 2, 3]

    first_error = all_results(
        [Ok(1), Err("failed"), Ok(3)],
    )
    assert isinstance(first_error, Err)
    assert first_error.error == "failed"

    values, errors = partition(
        [Ok(1), Err("a"), Ok(2), Err("b")],
    )
    assert values == [1, 2]
    assert errors == ["a", "b"]

    flattened = Ok(Ok(42)).flatten()
    assert isinstance(flattened, Ok)
    assert flattened.value == 42


def test_matching_errors_supports_exhaustive_and_partial_forms() -> None:
    error = NotFoundError("user-123")
    message = error.match(
        {
            "NotFound": lambda selected: f"No user {selected.to_dict()['item_id']}",
            "DatabaseUnavailable": lambda _selected: "Try again",
        },
    )
    assert message == "No user user-123"

    assert (
        error.match(
            {
                "NotFound": lambda selected: f"No user {selected.to_dict()['item_id']}",
                "DatabaseUnavailable": lambda _selected: "Try again",
            },
        )
        == "No user user-123"
    )

    partial = error.match_partial(
        {"NotFound": lambda selected: selected.to_dict()["item_id"]},
    )
    assert partial == "user-123"
    untouched = DatabaseUnavailableError().match_partial(
        {"NotFound": lambda selected: selected.to_dict()["item_id"]},
    )
    assert isinstance(untouched, DatabaseUnavailableError)


def test_callback_defects_are_panics_not_expected_errors() -> None:
    def broken_map(_value: int) -> int:
        msg = "bug"
        raise ZeroDivisionError(msg)

    def broken_match(_error: TaggedError) -> int:
        msg = "bug"
        raise ZeroDivisionError(msg)

    with pytest.raises(PanicError, match="map callback threw"):
        Ok(1).map(broken_map)

    with pytest.raises(PanicError, match=r"TaggedError\.match handler threw"):
        NotFoundError("x").match({"NotFound": broken_match})
