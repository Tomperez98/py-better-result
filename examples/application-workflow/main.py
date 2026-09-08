"""Compose a registration workflow with domain error translation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import assert_never

from better_result import Err, Ok, Result, is_err, is_ok


@dataclass(frozen=True, slots=True)
class CreateUser:
    email: str


@dataclass(frozen=True, slots=True)
class User:
    user_id: int
    email: str


@dataclass(frozen=True, slots=True)
class InvalidCreateUser:
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EmailTaken:
    email: str


@dataclass(frozen=True, slots=True)
class UserStoreUnavailable:
    cause: object


@dataclass(frozen=True, slots=True)
class PublishFailed:
    user_id: int


type RegisterUserError = (
    InvalidCreateUser | EmailTaken | UserStoreUnavailable | PublishFailed
)


def parse_create_user(value: object) -> Result[CreateUser, InvalidCreateUser]:
    """Parse untrusted input once at the application edge."""
    if not isinstance(value, Mapping):
        return Err(InvalidCreateUser(("input must be an object",)))

    email = value.get("email")
    if not isinstance(email, str) or "@" not in email:
        return Err(InvalidCreateUser(("email must be valid",)))

    return Ok(CreateUser(email))


def ensure_email_available(command: CreateUser) -> Result[None, EmailTaken]:
    if command.email == "taken@example.com":
        return Err(EmailTaken(command.email))
    return Ok(None)


def insert_user(command: CreateUser) -> Result[User, UserStoreUnavailable]:
    if command.email == "store-down@example.com":
        return Err(UserStoreUnavailable("database unavailable"))
    return Ok(User(user_id=42, email=command.email))


def publish_user_registered(user: User) -> Result[None, PublishFailed]:
    if user.email == "publish-fails@example.com":
        return Err(PublishFailed(user.user_id))
    return Ok(None)


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    body: object


def register_user(value: object) -> Result[User, RegisterUserError]:
    """Compose a workflow while keeping every expected failure in its type."""
    return parse_create_user(value).and_then(
        lambda command: ensure_email_available(command).and_then(
            lambda _: insert_user(command).and_then(
                lambda user: publish_user_registered(user).map(lambda _: user)
            )
        )
    )


def to_http_response(result: Result[User, RegisterUserError]) -> HttpResponse:
    """Map the domain error union to an HTTP response once at the edge."""
    if is_ok(result):
        return HttpResponse(status=201, body={"id": result.ok_value.user_id})

    if is_err(result):
        error = result.err_value
    else:
        message = "expected a concrete Result variant"
        raise TypeError(message)

    match error:
        case InvalidCreateUser():
            return HttpResponse(status=400, body={"message": "invalid user"})
        case EmailTaken():
            return HttpResponse(status=409, body={"message": "email is taken"})
        case UserStoreUnavailable():
            return HttpResponse(status=503, body={"message": "try again"})
        case PublishFailed():
            return HttpResponse(status=503, body={"message": "try again"})
        case _:
            assert_never(error)


def main() -> None:
    for raw_input, label in [
        ({"email": "ada@example.com"}, "success"),
        ({"email": "not-an-email"}, "invalid input"),
        ({"email": "taken@example.com"}, "email taken"),
        ({"email": "store-down@example.com"}, "store down"),
        ({"email": "publish-fails@example.com"}, "publish fails"),
    ]:
        response = to_http_response(register_user(raw_input))
        expected_status = {
            "success": 201,
            "invalid input": 400,
            "email taken": 409,
            "store down": 503,
            "publish fails": 503,
        }
        assert response.status == expected_status[label]
        print(f"{label}: status={response.status}")


if __name__ == "__main__":
    main()
