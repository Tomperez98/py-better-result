"""Executable Python equivalents of the public better-result examples."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, assert_type

if TYPE_CHECKING:
    from collections.abc import Mapping

import pytest

from better_result.core import Err, Ok, Panic, Result, is_err, is_ok
from better_result.error import TaggedError, match_error, match_error_partial
from better_result.result import (
    all as all_results,
    and_then,
    flatten,
    map as map_result,
    map_error,
    match,
    partition,
    tap,
    try_recover,
    try_result,
    unwrap,
    unwrap_or,
)


class MissingEnv(TaggedError, tag="MissingEnv"):
    env_name: str

    def __init__(self, name: str) -> None:
        super().__init__(message=f"{name} is required", env_name=name)


class InvalidPort(TaggedError, tag="InvalidPort"):
    input: str

    def __init__(self, input_value: str) -> None:
        super().__init__(message="Expected a port from 1 to 65535", input=input_value)


class InvalidJson(TaggedError, tag="InvalidJson"):
    input: str

    def __init__(self, input_value: str, cause: BaseException) -> None:
        super().__init__(
            message="Input is not valid JSON",
            input=input_value,
            cause=cause,
        )


class NotFound(TaggedError, tag="NotFound"):
    def __init__(self, item_id: str) -> None:
        super().__init__(message=f"{item_id} was not found", item_id=item_id)


class DatabaseUnavailable(TaggedError, tag="DatabaseUnavailable"):
    def __init__(self, message: str = "database unavailable") -> None:
        super().__init__(message=message)


class ValidationError(TaggedError, tag="ValidationError"):
    def __init__(self, field: str) -> None:
        super().__init__(message=f"Invalid {field}", field=field)


class LoadUserFailed(TaggedError, tag="LoadUserFailed"):
    def __init__(self, cause: TaggedError) -> None:
        super().__init__(message="Could not load user", cause=cause)


class InvalidInput(TaggedError, tag="InvalidInput"):
    def __init__(self, value: str) -> None:
        super().__init__(message="Invalid input", value=value)


class GuestUser:
    name = "Guest"


def read_env(environment: Mapping[str, str], name: str) -> Result[str, MissingEnv]:
    value = environment.get(name)
    if value is None:
        return Err[str, MissingEnv](MissingEnv(name))
    return Ok[str, MissingEnv](value)


def parse_port(input_value: str) -> Result[int, InvalidPort]:
    try:
        port = int(input_value)
    except ValueError:
        return Err[int, InvalidPort](InvalidPort(input_value))
    if not 1 <= port <= 65_535:
        return Err[int, InvalidPort](InvalidPort(input_value))
    return Ok[int, InvalidPort](port)


def read_server_address(
    environment: Mapping[str, str],
) -> Result[str, MissingEnv | InvalidPort]:
    return read_env(environment, "HOST").and_then(
        lambda host: read_env(environment, "PORT").and_then(
            lambda port_text: parse_port(port_text).map(
                lambda port: f"http://{host}:{port}",
            ),
        ),
    )


def test_quickstart_workflow_translates_to_explicit_python_composition() -> None:
    result = read_server_address({"HOST": "localhost", "PORT": "8080"})

    assert_type(result, Result[str, MissingEnv | InvalidPort])
    assert isinstance(result, Ok)
    assert result.value == "http://localhost:8080"

    missing_host = read_server_address({"PORT": "8080"})
    invalid_port = read_server_address({"HOST": "localhost", "PORT": "nope"})

    assert isinstance(missing_host, Err)
    assert isinstance(missing_host.error, MissingEnv)
    assert missing_host.error.env_name == "HOST"
    assert isinstance(invalid_port, Err)
    assert isinstance(invalid_port.error, InvalidPort)
    assert invalid_port.error.input == "nope"


def test_quickstart_result_and_tagged_error_matching() -> None:
    result = read_server_address({"HOST": "localhost", "PORT": "bad"})

    exit_code = result.match(
        {
            "ok": lambda address: 0 if address else 1,
            "err": lambda error: error.match(
                {
                    "MissingEnv": lambda missing: len(missing.env_name),
                    "InvalidPort": lambda invalid: len(invalid.input),
                },
            ),
        },
    )

    assert exit_code == 3


def parse_json_value(input_value: str) -> object:
    value = json.loads(input_value)
    if value is None or isinstance(value, (bool, float, int, str, list, dict)):
        return value
    raise TypeError("JSON decoder returned an unsupported value")


def parse_json(input_value: str) -> Result[object, InvalidJson]:
    return try_result(
        lambda _context: parse_json_value(input_value),
        lambda cause: InvalidJson(input_value, cause),
    )


def test_creating_results_wraps_sync_exceptions_and_customizes_errors() -> None:
    parsed = parse_json('{"name": "Alice"}')
    failed = parse_json("not json")

    assert isinstance(parsed, Ok)
    assert parsed.value == {"name": "Alice"}
    assert isinstance(failed, Err)
    assert isinstance(failed.error, InvalidJson)
    assert failed.error.input == "not json"
    assert isinstance(failed.error.cause, json.JSONDecodeError)

    ok_result = Ok[int, str](42)
    assert ok_result.status == "ok"
    assert ok_result.value == 42
    failure = Err[int, str]("Something went wrong")
    assert failure.status == "error"
    assert failure.error == "Something went wrong"


def test_narrowing_and_matching_examples() -> None:
    result: Result[int, str] = Ok[int, str](42)

    if result.status == "ok":
        assert result.value == 42
    else:
        raise AssertionError("unexpected error")

    assert is_ok(result)
    assert result.value == 42
    assert not is_err(result)

    message = match(
        result,
        {
            "ok": lambda value: f"Success: {value}",
            "err": lambda error: f"Error: {error}",
        },
    )
    assert message == "Success: 42"

    reusable = match({"ok": lambda value: value, "err": lambda _error: 0})
    assert reusable(result) == 42


def test_transforming_chaining_recovery_and_data_last_forms() -> None:
    parsed: Result[int, InvalidInput] = Ok[int, InvalidInput](21)
    doubled = map_result(parsed, lambda value: value * 2)
    assert isinstance(doubled, Ok)
    assert doubled.value == 42

    failure: Result[int, InvalidInput] = Err[int, InvalidInput](InvalidInput("x"))
    translated = map_error(failure, LoadUserFailed)
    assert isinstance(translated, Err)
    assert isinstance(translated.error, LoadUserFailed)

    chained = and_then(
        parsed,
        lambda value: (
            Ok[str, ValidationError](str(value))
            if value > 0
            else Err[str, ValidationError](ValidationError("value"))
        ),
    )
    assert isinstance(chained, Ok)
    assert chained.value == "21"

    recovered = try_recover(
        Err[int, NotFound](NotFound("missing")),
        lambda _error: Ok[GuestUser, DatabaseUnavailable](GuestUser()),
    )
    assert isinstance(recovered, Ok)
    assert isinstance(recovered.value, GuestUser)

    mapped_later = map_result(lambda value: value + 1)(Ok[int, str](1))
    assert isinstance(mapped_later, Ok)
    assert mapped_later.value == 2
    mapped_error_later = map_error(str.upper)(Err[int, str]("failed"))
    assert isinstance(mapped_error_later, Err)
    assert mapped_error_later.error == "FAILED"


def test_observing_and_extracting_values() -> None:
    seen: list[int] = []
    result = Ok[int, str](2)
    observed = tap(result, seen.append)

    assert observed is result
    assert seen == [2]
    tapped_error = tap(Err[int, str]("failed"), seen.append)
    assert isinstance(tapped_error, Err)
    assert tapped_error.error == "failed"

    assert unwrap_or(Err[int, str]("failed"), 0) == 0
    assert unwrap(Ok[int, str](42)) == 42
    with pytest.raises(Panic):
        unwrap(Err[int, str]("failed"))


def test_collections_and_flatten_examples() -> None:
    collected = all_results([Ok[int, str](1), Ok[int, str](2), Ok[int, str](3)])
    assert isinstance(collected, Ok)
    assert collected.value == [1, 2, 3]

    first_error = all_results(
        [Ok[int, str](1), Err[int, str]("failed"), Ok[int, str](3)],
    )
    assert isinstance(first_error, Err)
    assert first_error.error == "failed"

    values, errors = partition(
        [Ok[int, str](1), Err[int, str]("a"), Ok[int, str](2), Err[int, str]("b")],
    )
    assert values == [1, 2]
    assert errors == ["a", "b"]

    flattened = flatten(Ok[Ok[int, str], str](Ok[int, str](42)))
    assert isinstance(flattened, Ok)
    assert flattened.value == 42


def test_matching_errors_supports_exhaustive_and_partial_forms() -> None:
    error = NotFound("user-123")
    message = match_error(
        error,
        {
            "NotFound": lambda selected: f"No user {selected.item_id}",
            "DatabaseUnavailable": lambda _selected: "Try again",
        },
    )
    assert message == "No user user-123"

    to_message = match_error(
        {
            "NotFound": lambda selected: f"No user {selected.item_id}",
            "DatabaseUnavailable": lambda _selected: "Try again",
        },
    )
    assert to_message(error) == "No user user-123"

    partial = match_error_partial(
        error,
        {"NotFound": lambda selected: selected.item_id},
    )
    assert partial == "user-123"
    untouched = match_error_partial(
        DatabaseUnavailable(),
        {"NotFound": lambda selected: selected.item_id},
    )
    assert isinstance(untouched, DatabaseUnavailable)


def test_callback_defects_are_panics_not_expected_errors() -> None:
    def broken_map(_value: int) -> int:
        raise ZeroDivisionError("bug")

    def broken_match(_error: NotFound) -> int:
        raise ZeroDivisionError("bug")

    with pytest.raises(Panic, match="map callback threw"):
        Ok[int, str](1).map(broken_map)

    with pytest.raises(Panic, match="match_error handler threw"):
        match_error(NotFound("x"), {"NotFound": broken_match})
