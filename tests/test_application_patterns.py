"""
Application-boundary patterns from the better-result guide.

These tests exercise the public API at application seams rather than testing
private implementation details.  The examples deliberately use Python's
explicit ``and_then`` chain instead of hiding control flow in a generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_type

from better_result import Err, Ok, Result, TaggedError

# ---------------------------------------------------------------------------
# Parse at the edge


@dataclass(frozen=True)
class CreateUser:
    email: str
    display_name: str


class InvalidCreateUser(TaggedError, tag="InvalidCreateUser"):  # noqa: N818
    """The transport input cannot become a valid application command."""

    input_value: object

    def __init__(self, input_value: object) -> None:
        super().__init__(
            message="Invalid user",
            input_value=input_value,
        )


def parse_create_user(input_value: object) -> Result[CreateUser, InvalidCreateUser]:
    """Turn untrusted input into a domain command exactly once."""
    if not isinstance(input_value, dict):
        return Err(InvalidCreateUser(input_value))
    email = input_value.get("email")
    display_name = input_value.get("display_name")
    if (
        not isinstance(email, str)
        or "@" not in email
        or not isinstance(display_name, str)
        or not display_name.strip()
    ):
        return Err(InvalidCreateUser(input_value))
    return Ok(CreateUser(email=email, display_name=display_name.strip()))


# ---------------------------------------------------------------------------
# One error vocabulary per boundary


class NoRows(Exception):  # noqa: N818
    """Database-specific indication that a record was absent."""


class DriverUnavailable(Exception):  # noqa: N818
    """Database-specific connectivity failure."""


@dataclass(frozen=True)
class User:
    user_id: int
    email: str
    display_name: str


@dataclass(frozen=True)
class UserRow:
    user_id: int
    email: str
    display_name: str


class UserNotFound(TaggedError, tag="UserNotFound"):  # noqa: N818
    user_id: int

    def __init__(self, user_id: int) -> None:
        super().__init__(message="User not found", user_id=user_id)


class UserStoreUnavailable(TaggedError, tag="UserStoreUnavailable"):  # noqa: N818
    cause: BaseException

    def __init__(self, cause: BaseException) -> None:
        super().__init__(message="User store unavailable", cause=cause)


type FindUserError = UserNotFound | UserStoreUnavailable
type StoreError = NoRows | DriverUnavailable


def query_user_row(user_id: int) -> Result[UserRow, StoreError]:
    """A fake adapter whose errors are deliberately not application errors."""
    if user_id == 404:
        return Err(NoRows("no matching row"))
    if user_id == 503:
        return Err(DriverUnavailable("database is offline"))
    return Ok(UserRow(user_id, "alice@example.com", "Alice"))


def translate_store_error(error: StoreError) -> FindUserError:
    if isinstance(error, NoRows):
        return UserNotFound(404)
    return UserStoreUnavailable(error)


def find_user(user_id: int) -> Result[User, FindUserError]:
    """Translate driver failures once at the repository boundary."""
    return (
        query_user_row(user_id)
        .map(
            lambda row: User(row.user_id, row.email, row.display_name),
        )
        .map_error(translate_store_error)
    )


# ---------------------------------------------------------------------------
# Compose in an application workflow


class EmailTaken(TaggedError, tag="EmailTaken"):  # noqa: N818
    email: str

    def __init__(self, email: str) -> None:
        super().__init__(message="Email is already registered", email=email)


class PublishFailed(TaggedError, tag="PublishFailed"):  # noqa: N818
    cause: BaseException

    def __init__(self, cause: BaseException) -> None:
        super().__init__(message="Could not publish user event", cause=cause)


type RegisterUserError = (
    InvalidCreateUser | EmailTaken | UserStoreUnavailable | PublishFailed
)


def ensure_email_available(
    command: CreateUser,
    trace: list[str],
) -> Result[CreateUser, EmailTaken]:
    trace.append("check-email")
    if command.email == "taken@example.com":
        return Err(EmailTaken(command.email))
    return Ok(command)


def insert_user(
    command: CreateUser,
    trace: list[str],
) -> Result[User, UserStoreUnavailable]:
    trace.append("insert-user")
    if command.email == "offline@example.com":
        return Err(UserStoreUnavailable(ConnectionError("database is offline")))
    return Ok(User(7, command.email, command.display_name))


def publish_user_registered(
    user: User,
    trace: list[str],
) -> Result[User, PublishFailed]:
    trace.append("publish-event")
    if user.email == "events@example.com":
        return Err(PublishFailed(RuntimeError("broker rejected event")))
    return Ok(user)


def register_user(
    input_value: object,
    trace: list[str],
) -> Result[User, RegisterUserError]:
    """Compose expected failures without exceptions as ordinary control flow."""
    return (
        parse_create_user(input_value)
        .and_then(
            lambda command: ensure_email_available(command, trace),
        )
        .and_then(
            lambda command: insert_user(command, trace),
        )
        .and_then(
            lambda user: publish_user_registered(user, trace),
        )
    )


# ---------------------------------------------------------------------------
# Map to transport once


@dataclass(frozen=True)
class Response:
    status: int
    body: dict[str, object]


def to_error_response(error: RegisterUserError) -> Response:
    """Apply the transport policy in one place, after the workflow finishes."""
    if isinstance(error, InvalidCreateUser):
        return Response(400, error.to_safe_json())
    if isinstance(error, EmailTaken):
        return Response(409, error.to_safe_json())
    if isinstance(error, (UserStoreUnavailable, PublishFailed)):
        return Response(503, {"message": "Try again"})
    raise AssertionError


def to_http_response(result: Result[User, RegisterUserError]) -> Response:
    return result.match(
        lambda user: Response(
            201,
            {"id": user.user_id, "email": user.email},
        ),
        to_error_response,
    )


def test_parse_at_the_edge_returns_domain_values_or_typed_errors() -> None:
    invalid = parse_create_user({"email": "not-an-email", "display_name": ""})
    valid = parse_create_user({"email": "alice@example.com", "display_name": " Alice "})

    assert_type(invalid, Result[CreateUser, InvalidCreateUser])
    assert isinstance(invalid, Err)
    assert isinstance(invalid.error, InvalidCreateUser)
    assert invalid.error.input_value == {
        "email": "not-an-email",
        "display_name": "",
    }

    assert isinstance(valid, Ok)
    assert valid.value == CreateUser("alice@example.com", "Alice")


def test_repository_translates_adapter_errors_once() -> None:
    missing = find_user(404)
    unavailable = find_user(503)
    found = find_user(7)

    assert_type(missing, Result[User, FindUserError])
    assert isinstance(missing, Err)
    assert isinstance(missing.error, UserNotFound)
    assert missing.error.user_id == 404

    assert isinstance(unavailable, Err)
    assert isinstance(unavailable.error, UserStoreUnavailable)
    assert isinstance(unavailable.error.cause, DriverUnavailable)

    assert isinstance(found, Ok)
    assert found.value == User(7, "alice@example.com", "Alice")


def test_map_error_leaves_success_values_untouched() -> None:
    result = Ok(UserRow(7, "alice@example.com", "Alice"))
    translated = result.map_error(translate_store_error)

    assert translated is result
    assert isinstance(translated, Ok)
    assert translated.value.user_id == 7


def test_workflow_short_circuits_and_preserves_expected_error_vocabulary() -> None:
    cases = [
        (
            {"email": "bad", "display_name": "Alice"},
            [],
            InvalidCreateUser,
        ),
        (
            {"email": "taken@example.com", "display_name": "Alice"},
            ["check-email"],
            EmailTaken,
        ),
        (
            {"email": "offline@example.com", "display_name": "Alice"},
            ["check-email", "insert-user"],
            UserStoreUnavailable,
        ),
        (
            {"email": "events@example.com", "display_name": "Alice"},
            ["check-email", "insert-user", "publish-event"],
            PublishFailed,
        ),
    ]

    for input_value, expected_trace, expected_error in cases:
        trace: list[str] = []
        result = register_user(input_value, trace)

        assert_type(result, Result[User, RegisterUserError])
        assert isinstance(result, Err)
        assert isinstance(result.error, expected_error)
        assert trace == expected_trace

    success_trace: list[str] = []
    success = register_user(
        {"email": "alice@example.com", "display_name": "Alice"},
        success_trace,
    )
    assert isinstance(success, Ok)
    assert success.value == User(7, "alice@example.com", "Alice")
    assert success_trace == ["check-email", "insert-user", "publish-event"]


def test_transport_mapping_consumes_the_workflow_once_and_keeps_payloads_safe() -> None:
    invalid_response = to_http_response(
        register_user({"email": "bad", "display_name": "Alice"}, []),
    )
    conflict_response = to_http_response(
        register_user(
            {"email": "taken@example.com", "display_name": "Alice"},
            [],
        ),
    )
    unavailable_response = to_http_response(
        register_user(
            {"email": "offline@example.com", "display_name": "Alice"},
            [],
        ),
    )
    created_response = to_http_response(
        register_user(
            {"email": "alice@example.com", "display_name": "Alice"},
            [],
        ),
    )

    assert invalid_response.status == 400
    assert invalid_response.body["_tag"] == "InvalidCreateUser"
    assert "stack" not in invalid_response.body

    assert conflict_response.status == 409
    assert conflict_response.body["_tag"] == "EmailTaken"

    assert unavailable_response == Response(503, {"message": "Try again"})
    assert created_response == Response(
        201,
        {"id": 7, "email": "alice@example.com"},
    )


def test_transport_mapping_handles_each_typed_failure_without_string_dispatch() -> None:
    errors: list[tuple[RegisterUserError, int]] = [
        (InvalidCreateUser({"email": "bad"}), 400),
        (EmailTaken("taken@example.com"), 409),
        (UserStoreUnavailable(ConnectionError("offline")), 503),
        (PublishFailed(RuntimeError("broker")), 503),
    ]

    for error, expected_status in errors:
        response = to_http_response(Err(error))
        assert response.status == expected_status
        assert isinstance(response.body, dict)
