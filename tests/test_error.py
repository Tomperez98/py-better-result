"""Tests for tagged errors and error matching."""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_type

import pytest

from better_result.core import Err, Panic, err
from better_result.error import (
    ResultDeserializationError,
    ResultSerializationError,
    TaggedError,
    UnhandledException,
    is_tagged_error,
    match_error,
    match_error_partial,
    tagged_error,
)


class NotFoundError(TaggedError, tag="NotFoundError"):
    def __init__(self, item_id: str) -> None:
        super().__init__(message=f"Not found: {item_id}", item_id=item_id)


class ValidationError(TaggedError, tag="ValidationError"):
    def __init__(self, field: str) -> None:
        super().__init__(message=f"Invalid: {field}", field=field)


def test_tagged_error_supports_computed_messages_causes_and_reserved_names() -> None:
    class RequestFailed(TaggedError, tag="RequestFailed"):
        url: str
        status_code: int

        def __init__(self, url: str, status_code: int) -> None:
            super().__init__(
                message=f"Request to {url} failed with {status_code}",
                url=url,
                status_code=status_code,
            )

    cause = ValueError("timeout")
    error = RequestFailed("https://example.test", 503)
    with_cause = TaggedError(cause=cause, message="failed")

    assert error.name == "RequestFailed"
    assert error.message == "Request to https://example.test failed with 503"
    assert error.url == "https://example.test"
    assert error.status_code == 503
    assert with_cause.__cause__ is cause
    cause_json = with_cause.to_json()["cause"]
    assert isinstance(cause_json, dict)
    assert cause_json["message"] == "timeout"

    with pytest.raises(TypeError, match="reserved"):
        TaggedError({"match": "shadowing the method"})


def test_tagged_error_preserves_properties_and_serializes() -> None:
    error = NotFoundError("123")

    assert error.name == "NotFoundError"
    assert error.message == "Not found: 123"
    assert error._tag == "NotFoundError"
    assert error.to_dict()["item_id"] == "123"
    assert error.to_json()["_tag"] == "NotFoundError"
    assert error.to_json()["item_id"] == "123"
    assert is_tagged_error(error)
    assert TaggedError.is_tagged_error(error)
    assert NotFoundError.is_(error)
    assert not NotFoundError.is_(ValidationError("email"))
    assert not is_tagged_error(ValueError("plain"))


def test_tagged_error_requires_a_class_header_tag() -> None:
    with pytest.raises(TypeError, match="tag"):
        type("MissingTag", (TaggedError,), {})

    with pytest.raises(ValueError, match="must not be empty"):

        class EmptyTag(TaggedError, tag=""):
            pass


def test_dynamic_tagged_error_factory() -> None:
    dynamic_error = tagged_error("DynamicError")
    error = dynamic_error({"message": "dynamic", "value": 42})

    assert error._tag == "DynamicError"
    assert error.message == "dynamic"
    assert error.to_dict()["value"] == 42
    assert dynamic_error.is_(error)


def test_match_error_supports_data_first_and_data_last_forms() -> None:
    handlers = {
        "NotFoundError": lambda error: f"missing {error.item_id}",
        "ValidationError": lambda error: f"invalid {error.field}",
    }
    not_found = NotFoundError("123")
    validation = ValidationError("email")

    assert match_error(not_found, handlers) == "missing 123"
    matcher = match_error(handlers)
    assert_type(matcher, Callable[[TaggedError], str])
    assert matcher(validation) == "invalid email"


def test_match_error_rejects_missing_handlers_as_panics() -> None:
    with pytest.raises(Panic, match="match_error handler threw"):
        match_error(NotFoundError("123"), {"ValidationError": lambda _: "wrong"})


def test_match_error_wraps_handler_failures_in_panic() -> None:
    error = NotFoundError("123")

    with pytest.raises(Panic, match="match_error handler threw") as raised:
        match_error(error, {"NotFoundError": lambda _: (_ for _ in ()).throw(error)})

    assert raised.value.cause is error


def test_match_error_partial_preserves_or_transforms_unhandled_errors() -> None:
    handled = NotFoundError("123")
    unhandled = ValidationError("email")
    handlers = {"NotFoundError": lambda error: str(error.item_id)}

    assert match_error_partial(handled, handlers) == "123"
    assert match_error_partial(unhandled, handlers) is unhandled
    assert (
        match_error_partial(
            unhandled, handlers, lambda error: f"fallback: {error._tag}"
        )
        == "fallback: ValidationError"
    )

    matcher = match_error_partial(handlers)
    assert_type(matcher, Callable[[TaggedError], str | TaggedError])
    assert matcher(handled) == "123"
    assert matcher(unhandled) is unhandled

    matcher_with_fallback = match_error_partial(
        handlers, lambda error: f"fallback: {error._tag}"
    )
    assert_type(matcher_with_fallback, Callable[[TaggedError], str])
    assert matcher_with_fallback(unhandled) == "fallback: ValidationError"


def test_partial_match_callback_defects_are_panics() -> None:
    def explode(_error: NotFoundError) -> int:
        raise RuntimeError("broken handler")

    with pytest.raises(Panic, match="match_error_partial handler threw"):
        match_error_partial(NotFoundError("123"), {"NotFoundError": explode})


def test_match_error_partial_can_wrap_unhandled_errors_as_err() -> None:
    error = ValidationError("email")
    matcher = match_error_partial(
        {"NotFoundError": lambda selected: str(selected.item_id)}, err
    )

    result = matcher(error)
    assert isinstance(result, Err)
    assert result.error is error


def test_builtin_tagged_errors_keep_their_values() -> None:
    cause = ValueError("bad input")
    unhandled = UnhandledException(cause)
    deserialization = ResultDeserializationError({"raw": "bad"})
    serialization = ResultSerializationError({"raw": object()})

    assert unhandled.message == "Unhandled exception: bad input"
    assert unhandled.cause is cause
    assert deserialization.to_dict()["value"] == {"raw": "bad"}
    assert serialization.message == "Failed to serialize Result payload"
    assert ResultDeserializationError.is_(deserialization)
    assert ResultSerializationError.is_(serialization)


def test_tagged_errors_are_yieldable_as_result_errors() -> None:
    iterator = iter(NotFoundError("123"))
    yielded = next(iterator)

    assert isinstance(yielded, Err)
    assert yielded.error._tag == "NotFoundError"
    with pytest.raises(Panic, match="Unreachable"):
        next(iterator)
