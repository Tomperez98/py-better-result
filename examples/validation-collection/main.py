"""Accumulate independent validation errors before building a command."""

from __future__ import annotations

from dataclasses import dataclass

from better_result import Err, Ok, Result, collect_results


@dataclass(frozen=True, slots=True)
class CreateUser:
    name: str
    email: str
    age: int


def validate_name(value: object) -> Result[str, str]:
    if not isinstance(value, str) or not value.strip():
        return Err("name is required")
    return Ok(value.strip())


def validate_email(value: object) -> Result[str, str]:
    if not isinstance(value, str) or "@" not in value:
        return Err("email must contain @")
    return Ok(value)


def validate_age(value: object) -> Result[int, str]:
    if not isinstance(value, int) or isinstance(value, bool):
        return Err("age must be an integer")
    if value < 18:
        return Err("age must be at least 18")
    return Ok(value)


def parse_create_user(value: object) -> Result[CreateUser, tuple[str, ...]]:
    if not isinstance(value, dict):
        return Err(("input must be an object",))

    validated = collect_results(
        (
            validate_name(value.get("name")),
            validate_email(value.get("email")),
            validate_age(value.get("age")),
        )
    )
    if isinstance(validated, Err):
        return Err(validated.err_value)
    assert isinstance(validated, Ok)

    name, email, age = validated.ok_value
    return Ok(CreateUser(name, email, age))


def main() -> None:
    for value in (
        {"name": "", "email": "invalid", "age": 16},
        {"name": "Ada", "email": "ada@example.com", "age": 36},
    ):
        print(parse_create_user(value))


if __name__ == "__main__":
    main()
