"""Application-level patterns for composing typed Result workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import assert_never, assert_type

from better_result import Err, Ok, Result


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


@dataclass(frozen=True, slots=True)
class UserRow:
    user_id: int
    email: str


@dataclass(frozen=True, slots=True)
class NoSuchUser:
    user_id: int


@dataclass(frozen=True, slots=True)
class DatabaseUnavailable:
    cause: object


type DriverError = NoSuchUser | DatabaseUnavailable


@dataclass(frozen=True, slots=True)
class UserNotFound:
    user_id: int


type FindUserError = UserNotFound | UserStoreUnavailable


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    body: object


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


def register_user(value: object) -> Result[User, RegisterUserError]:
    """Compose a workflow while keeping every expected failure in its type."""
    return parse_create_user(value).and_then(
        lambda command: ensure_email_available(command).and_then(
            lambda _: insert_user(command).and_then(
                lambda user: publish_user_registered(user).map(lambda _: user)
            )
        )
    )


def query_user_row(user_id: int) -> Result[UserRow, DriverError]:
    if user_id == 404:
        return Err(NoSuchUser(user_id))
    if user_id == 503:
        return Err(DatabaseUnavailable("database unavailable"))
    return Ok(UserRow(user_id=user_id, email="user@example.com"))


def user_from_row(row: UserRow) -> User:
    return User(row.user_id, row.email)


def find_user(user_id: int) -> Result[User, FindUserError]:
    """Translate driver errors once at the repository seam."""

    def translate(error: DriverError) -> FindUserError:
        match error:
            case NoSuchUser(user_id=user_id):
                return UserNotFound(user_id)
            case DatabaseUnavailable(cause=cause):
                return UserStoreUnavailable(cause)
            case _:
                assert_never(error)

    return query_user_row(user_id).map(user_from_row).map_err(translate)


def to_http_response(result: Result[User, RegisterUserError]) -> HttpResponse:
    """Make the transport decision at one edge of the application."""
    if isinstance(result, Ok):
        return HttpResponse(status=201, body={"id": result.ok_value.user_id})
    if not isinstance(result, Err):
        message = "expected a concrete Result variant"
        raise TypeError(message)

    error = result.err_value
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


def test_parse_at_the_edge_returns_domain_input_or_a_named_failure() -> None:
    assert parse_create_user({"email": "ada@example.com"}) == Ok(
        CreateUser("ada@example.com")
    )
    assert parse_create_user({"email": "not-an-email"}) == Err(
        InvalidCreateUser(("email must be valid",))
    )
    assert parse_create_user(None) == Err(
        InvalidCreateUser(("input must be an object",))
    )


def test_repository_translates_driver_errors_at_its_boundary() -> None:
    assert find_user(1) == Ok(User(1, "user@example.com"))
    assert find_user(404) == Err(UserNotFound(404))
    assert find_user(503) == Err(UserStoreUnavailable("database unavailable"))


def test_workflow_short_circuits_and_preserves_the_failure_vocabulary() -> None:
    assert register_user({"email": "ada@example.com"}) == Ok(
        User(42, "ada@example.com")
    )
    assert register_user({"email": "not-an-email"}) == Err(
        InvalidCreateUser(("email must be valid",))
    )
    assert register_user({"email": "taken@example.com"}) == Err(
        EmailTaken("taken@example.com")
    )
    assert register_user({"email": "store-down@example.com"}) == Err(
        UserStoreUnavailable("database unavailable")
    )
    assert register_user({"email": "publish-fails@example.com"}) == Err(
        PublishFailed(42)
    )


def test_transport_mapping_handles_success_and_each_expected_error() -> None:
    assert to_http_response(Ok(User(42, "ada@example.com"))) == HttpResponse(
        201,
        {"id": 42},
    )
    assert to_http_response(Err(InvalidCreateUser(("bad",)))) == HttpResponse(
        400,
        {"message": "invalid user"},
    )
    assert to_http_response(Err(EmailTaken("ada@example.com"))) == HttpResponse(
        409,
        {"message": "email is taken"},
    )
    assert to_http_response(Err(UserStoreUnavailable("down"))) == HttpResponse(
        503,
        {"message": "try again"},
    )
    assert to_http_response(Err(PublishFailed(42))) == HttpResponse(
        503,
        {"message": "try again"},
    )


def test_result_signatures_keep_the_application_error_union_visible() -> None:
    parsed = parse_create_user({"email": "ada@example.com"})
    assert_type(parsed, Result[CreateUser, InvalidCreateUser])

    registered = register_user({"email": "ada@example.com"})
    assert_type(registered, Result[User, RegisterUserError])

    found = find_user(1)
    assert_type(found, Result[User, FindUserError])
