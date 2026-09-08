"""Encode and decode Result values at a JSON-like application boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from better_result import Err, Ok, SchemaFailure, codec


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
    print(encoded)  # noqa: T201

    if isinstance(encoded, Ok):
        decoded = user_codec.deserialize(encoded.ok_value)
        print(decoded)  # noqa: T201

    # Wire-level errors remain Err values after decoding.
    print(user_codec.deserialize({"status": "error", "error": {"code": "not_found"}}))  # noqa: T201

    # Schema failures are returned as ResultDeserializationError values.
    print(user_codec.deserialize({"status": "ok", "value": {"id": "wrong"}}))  # noqa: T201

    # Serialization errors use the same Result shape.
    print(user_codec.serialize(Err("not_found")))  # noqa: T201


if __name__ == "__main__":
    main()
