"""Tests for concrete tagged errors."""

from __future__ import annotations

import json
from typing import cast

import pytest

from better_result import TaggedError
from better_result._error import (
    ResultDeserializationError,
    ResultSerializationError,
    UnhandledError,
)


class NotFoundError(TaggedError, tag="NotFoundError"):
    item_id: str

    def __init__(self, item_id: str) -> None:
        super().__init__(message=f"Not found: {item_id}", item_id=item_id)


class ValidationError(TaggedError, tag="ValidationError"):
    field: str

    def __init__(self, field: str) -> None:
        super().__init__(message=f"Invalid: {field}", field=field)


def test_concrete_tagged_errors_preserve_typed_properties() -> None:
    error = NotFoundError("123")

    assert error.name == "NotFoundError"
    assert error._tag == "NotFoundError"
    assert error.message == "Not found: 123"
    assert error.item_id == "123"
    assert isinstance(error, TaggedError)


def test_subclasses_require_valid_string_tags() -> None:
    with pytest.raises(TypeError, match="tag"):
        type("MissingTag", (TaggedError,), {})

    with pytest.raises(TypeError, match="string"):
        type("BadTag", (TaggedError,), {}, tag=123)

    with pytest.raises(ValueError, match="must not be empty"):
        type("EmptyTag", (TaggedError,), {}, tag="   ")

    with pytest.raises(TypeError, match="must not define _tag"):

        class InvalidTagError(TaggedError, tag="InvalidTag"):
            _tag = "shadowed"


def test_messages_and_properties_are_validated() -> None:
    with pytest.raises(TypeError, match="message"):
        TaggedError(message=cast("str", object()))

    for reserved in ("to_json", "to_dict", "match", "name", "stack"):
        with pytest.raises(TypeError, match="reserved"):
            TaggedError(**{reserved: "bad"})


def test_serialization_has_diagnostic_and_safe_forms() -> None:
    cause = ValueError("timeout")
    error = TaggedError(message="failed", cause=cause, nested={"items": {1, 2}})
    cause_only = TaggedError(cause=cause)

    payload = error.to_json()
    json.dumps(payload)
    assert payload["message"] == "failed"
    assert cause_only.message == ""
    assert "stack" in payload
    assert "stack" not in error.to_safe_json()
    assert isinstance(payload["cause"], dict)
    assert isinstance(payload["nested"], dict)


def test_builtin_errors_are_concrete_tagged_errors() -> None:
    cause = ValueError("bad input")
    unhandled = UnhandledError(cause)
    deserialization = ResultDeserializationError({"raw": "bad"})
    serialization = ResultSerializationError({"raw": object()})

    assert unhandled.cause is cause
    assert unhandled.message == "Unhandled exception: bad input"
    assert deserialization.value == {"raw": "bad"}
    assert serialization.message == "Failed to serialize Result payload"
