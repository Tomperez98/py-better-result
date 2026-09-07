"""Tests for tagged errors and error matching."""

from __future__ import annotations

import json
from typing import cast

import pytest

from better_result.core import Err, PanicError
from better_result.error import (
    ResultDeserializationError,
    ResultSerializationError,
    TaggedError,
    UnhandledError,
    is_tagged_error,
    tagged_error,
)


class NotFoundError(TaggedError, tag="NotFoundError"):
    def __init__(self, item_id: str) -> None:
        super().__init__(message=f"Not found: {item_id}", item_id=item_id)


class ValidationError(TaggedError, tag="ValidationError"):
    def __init__(self, field: str) -> None:
        super().__init__(message=f"Invalid: {field}", field=field)


def test_tagged_error_supports_computed_messages_causes_and_reserved_names() -> None:
    class RequestFailedError(TaggedError, tag="RequestFailed"):
        url: str
        status_code: int

        def __init__(self, url: str, status_code: int) -> None:
            super().__init__(
                message=f"Request to {url} failed with {status_code}",
                url=url,
                status_code=status_code,
            )

    cause = ValueError("timeout")
    error = RequestFailedError("https://example.test", 503)
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
        TaggedError(match="shadowing the method")


def test_tagged_error_preserves_properties_and_serializes() -> None:
    error = NotFoundError("123")

    assert error.name == "NotFoundError"
    assert error.message == "Not found: 123"
    assert error._tag == "NotFoundError"
    assert error.to_dict()["item_id"] == "123"
    assert error.to_json()["_tag"] == "NotFoundError"
    assert error.to_json()["item_id"] == "123"
    assert is_tagged_error(error)
    assert NotFoundError.is_(error)
    assert not NotFoundError.is_(ValidationError("email"))
    assert not is_tagged_error(ValueError("plain"))


def test_tagged_error_requires_a_class_header_tag() -> None:
    with pytest.raises(TypeError, match="tag"):
        type("MissingTag", (TaggedError,), {})

    with pytest.raises(ValueError, match="must not be empty"):

        class EmptyTagError(TaggedError, tag=""):
            pass


def test_dynamic_tagged_error_factory() -> None:
    dynamic_error = tagged_error("DynamicError")
    error = dynamic_error(message="dynamic", value=42)

    assert error._tag == "DynamicError"
    assert error.message == "dynamic"
    assert error.to_dict()["value"] == 42
    assert dynamic_error.is_(error)


def test_tagged_error_match_dispatches_exhaustively() -> None:
    handlers = {
        "NotFoundError": lambda error: f"missing {error.item_id}",
        "ValidationError": lambda error: f"invalid {error.field}",
    }
    not_found = NotFoundError("123")
    validation = ValidationError("email")

    assert not_found.match(handlers) == "missing 123"
    assert validation.match(handlers) == "invalid email"


def test_tagged_error_match_rejects_missing_handlers_as_panics() -> None:
    with pytest.raises(PanicError, match=r"TaggedError\.match handler threw"):
        NotFoundError("123").match(
            {"ValidationError": lambda _: "wrong"},
        )


def test_tagged_error_match_wraps_handler_failures_in_panic() -> None:
    error = NotFoundError("123")

    with pytest.raises(PanicError, match=r"TaggedError\.match handler threw") as raised:
        error.match({"NotFoundError": lambda _: (_ for _ in ()).throw(error)})

    assert raised.value.cause is error


def test_tagged_error_match_partial_preserves_or_transforms_unhandled_errors() -> None:
    handled = NotFoundError("123")
    unhandled = ValidationError("email")
    handlers = {"NotFoundError": lambda error: str(error.item_id)}

    assert handled.match_partial(handlers) == "123"
    assert unhandled.match_partial(handlers) is unhandled
    assert (
        unhandled.match_partial(
            handlers,
            lambda error: f"fallback: {error._tag}",
        )
        == "fallback: ValidationError"
    )


def test_partial_match_callback_defects_are_panics() -> None:
    def explode(_error: TaggedError) -> int:
        msg = "broken handler"
        raise RuntimeError(msg)

    with pytest.raises(PanicError, match=r"TaggedError\.match_partial handler threw"):
        NotFoundError("123").match_partial({"NotFoundError": explode})


def test_tagged_error_match_partial_can_wrap_unhandled_errors_as_result() -> None:
    error = ValidationError("email")
    result = error.match_partial(
        {"NotFoundError": lambda selected: str(selected.to_dict()["item_id"])},
        Err,
    )

    assert isinstance(result, Err)
    assert result.error is error


def test_tagged_error_match_partial_preserves_falsey_fallback_callables() -> None:
    error = ValidationError("email")
    calls: list[str] = []

    class FalseyFallback:
        def __bool__(self) -> bool:
            return False

        def __call__(self, selected: TaggedError) -> str:
            calls.append(selected._tag)
            return "handled"

    result = error.match_partial({}, FalseyFallback())

    assert result == "handled"
    assert calls == ["ValidationError"]


def test_safe_error_serialization_omits_diagnostic_stack() -> None:
    error = NotFoundError("123")

    assert "stack" in error.to_json()
    safe = error.to_safe_json()
    assert "stack" not in safe
    assert safe["_tag"] == "NotFoundError"
    assert safe["item_id"] == "123"


def test_builtin_tagged_errors_keep_their_values() -> None:
    cause = ValueError("bad input")
    unhandled = UnhandledError(cause)
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
    with pytest.raises(PanicError, match="Unreachable"):
        next(iterator)


def test_tagged_error_to_json_is_json_compatible() -> None:
    payload = TaggedError(cause=object(), nested={"items": {1, 2}}).to_json()

    json.dumps(payload)
    assert isinstance(payload["cause"], str)
    nested = cast("dict[str, object]", payload["nested"])
    assert isinstance(nested["items"], list)


def test_tagged_error_detection_is_nominal() -> None:
    class FakeTaggedError(Exception):
        _tag = "Fake"

        def to_json(self) -> dict[str, object]:
            return {}

    assert not is_tagged_error(FakeTaggedError())
