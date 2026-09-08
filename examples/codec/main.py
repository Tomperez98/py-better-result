"""Encode and decode Result values at a JSON-like application boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from better_result import (
    Err,
    Ok,
    ResultDeserializationError,
    SchemaFailure,
    codec,
    is_err,
    is_ok,
)


@dataclass(frozen=True, slots=True)
class User:
    user_id: int
    name: str


def serialize_user(user: User) -> dict[str, object]:
    return {"id": user.user_id, "name": user.name}


def serialize_error(error: str) -> dict[str, str]:
    return {"code": error}


def deserialize_user(value: object) -> User | SchemaFailure:
    if not isinstance(value, Mapping):
        return SchemaFailure([{"message": "user must be an object", "path": []}])

    user_id = value.get("id")
    name = value.get("name")
    if not isinstance(user_id, int) or not isinstance(name, str):
        return SchemaFailure(
            [{"message": "user needs an integer id and string name", "path": []}]
        )
    return User(user_id, name)


def deserialize_error(value: object) -> str | SchemaFailure:
    if not isinstance(value, Mapping):
        return SchemaFailure([{"message": "error needs a string code", "path": []}])

    code = value.get("code")
    if not isinstance(code, str):
        return SchemaFailure([{"message": "error needs a string code", "path": []}])
    return code


def main() -> None:
    user_codec = codec(
        serialize_ok=serialize_user,
        serialize_err=serialize_error,
        deserialize_ok=deserialize_user,
        deserialize_err=deserialize_error,
    )

    encoded = user_codec.serialize(Ok(User(42, "Ada")))
    assert is_ok(encoded)
    assert encoded == Ok({"status": "ok", "value": {"id": 42, "name": "Ada"}})
    print(encoded)

    decoded = user_codec.deserialize(encoded.ok_value)
    assert decoded == Ok(User(42, "Ada"))
    print(decoded)

    # Wire-level errors remain Err values after decoding.
    decoded_error = user_codec.deserialize(
        {"status": "error", "error": {"code": "not_found"}}
    )
    assert decoded_error == Err("not_found")
    print(decoded_error)

    # Schema failures are returned as ResultDeserializationError values.
    schema_failure = user_codec.deserialize({"status": "ok", "value": {"id": "wrong"}})
    assert is_err(schema_failure)
    assert isinstance(schema_failure.err_value, ResultDeserializationError)
    assert schema_failure.err_value.value == {
        "status": "ok",
        "value": {"id": "wrong"},
    }
    print(schema_failure)

    # Serializing an Err produces an error envelope.
    serialized_error = user_codec.serialize(Err("not_found"))
    assert is_ok(serialized_error)
    assert serialized_error == Ok({"status": "error", "error": {"code": "not_found"}})
    print(serialized_error)


if __name__ == "__main__":
    main()
