"""
Static type contracts for Result codecs.

These assertions are checked by ``ty``; pytest only executes the runtime bodies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, assert_type, cast

import pytest

from better_result import (
    AsyncResultCodec,
    CodecIssue,
    Err,
    Ok,
    Result,
    ResultCodec,
    ResultDeserializationError,
    ResultSerializationError,
    SchemaFailure,
    SerializedResult,
    async_codec,
    codec,
)


@dataclass(frozen=True)
class User:
    user_id: int


@dataclass(frozen=True)
class UserError:
    code: str


class UserWire(TypedDict):
    user_id: int


class UserErrorWire(TypedDict):
    code: str


def user_to_wire(user: User) -> UserWire:
    return {"user_id": user.user_id}


def error_to_wire(error: UserError) -> UserErrorWire:
    return {"code": error.code}


def user_from_wire(value: UserWire) -> User:
    return User(value["user_id"])


def error_from_wire(value: UserErrorWire) -> UserError:
    return UserError(value["code"])


async def async_user_to_wire(user: User) -> UserWire:
    return user_to_wire(user)


async def async_error_to_wire(error: UserError) -> UserErrorWire:
    return error_to_wire(error)


async def async_user_from_wire(value: UserWire) -> User:
    return user_from_wire(value)


async def async_error_from_wire(value: UserErrorWire) -> UserError:
    return error_from_wire(value)


def test_sync_codec_type_contract() -> None:
    result_codec = codec(
        serialize_ok=user_to_wire,
        serialize_err=error_to_wire,
        deserialize_ok=user_from_wire,
        deserialize_err=error_from_wire,
    )

    assert_type(
        result_codec,
        ResultCodec[User, UserError, UserWire, UserErrorWire, User, UserError],
    )
    assert_type(
        result_codec.serialize(Ok(User(1))),
        Result[SerializedResult[UserWire, UserErrorWire], ResultSerializationError],
    )
    assert_type(
        result_codec.serialize(Err(UserError("missing"))),
        Result[SerializedResult[UserWire, UserErrorWire], ResultSerializationError],
    )
    assert_type(
        result_codec.deserialize({"status": "ok", "value": {"user_id": 1}}),
        Result[User, UserError | ResultDeserializationError],
    )
    assert_type(
        result_codec.deserialize_unsafe({"status": "ok", "value": {"user_id": 1}}),
        Result[User, UserError],
    )

    encoded = result_codec.serialize(Ok(User(1)))
    if isinstance(encoded, Ok):
        assert_type(encoded.ok_value, SerializedResult[UserWire, UserErrorWire])
    elif isinstance(encoded, Err):
        assert_type(encoded.err_value, ResultSerializationError)
    else:
        pytest.fail("unknown Result variant")

    decoded = result_codec.deserialize(
        {"status": "error", "error": {"code": "missing"}}
    )
    if isinstance(decoded, Ok):
        assert_type(decoded.ok_value, User)
    elif isinstance(decoded, Err):
        assert_type(decoded.err_value, UserError | ResultDeserializationError)
    else:
        pytest.fail("unknown Result variant")


def test_async_codec_type_contract() -> None:
    result_codec = async_codec(
        serialize_ok=async_user_to_wire,
        serialize_err=async_error_to_wire,
        deserialize_ok=async_user_from_wire,
        deserialize_err=async_error_from_wire,
    )

    assert_type(
        result_codec,
        AsyncResultCodec[User, UserError, UserWire, UserErrorWire, User, UserError],
    )
    # The awaited return type is checked in the async contract test below.


@pytest.mark.asyncio
async def test_async_codec_awaited_type_contract() -> None:
    result_codec = async_codec(
        serialize_ok=async_user_to_wire,
        serialize_err=async_error_to_wire,
        deserialize_ok=async_user_from_wire,
        deserialize_err=async_error_from_wire,
    )

    encoded = await result_codec.serialize(Ok(User(1)))
    assert_type(
        encoded,
        Result[SerializedResult[UserWire, UserErrorWire], ResultSerializationError],
    )

    decoded = await result_codec.deserialize(
        {"status": "ok", "value": {"user_id": 1}},
    )
    assert_type(decoded, Result[User, UserError | ResultDeserializationError])


def test_schema_failure_has_typed_issues() -> None:
    issue = cast("CodecIssue", {"message": "bad user"})
    failure = SchemaFailure([issue])
    assert_type(failure.issues, Sequence[CodecIssue])

    if TYPE_CHECKING:
        # The schema result is explicitly part of the callable's output type.
        def rejecting_schema(_: User) -> UserWire | SchemaFailure:
            return failure

        assert_type(rejecting_schema(User(1)), UserWire | SchemaFailure)
