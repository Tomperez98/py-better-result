"""Recover from an expected failure and translate errors at a boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from better_result import Err, Ok, Result


@dataclass(frozen=True, slots=True)
class User:
    user_id: int
    name: str


@dataclass(frozen=True, slots=True)
class DriverUnavailable:
    cause: str


@dataclass(frozen=True, slots=True)
class DriverNotFound:
    user_id: int


type DriverError = DriverUnavailable | DriverNotFound


@dataclass(frozen=True, slots=True)
class UserStoreUnavailable:
    cause: str


@dataclass(frozen=True, slots=True)
class UserNotFound:
    user_id: int


type UserStoreError = UserStoreUnavailable | UserNotFound


def primary_user(user_id: int) -> Result[User, DriverError]:
    if user_id == 503:
        return Err(DriverUnavailable("primary database unavailable"))
    if user_id == 404:
        return Err(DriverNotFound(user_id))
    return Ok(User(user_id, "Ada"))


def cached_user(user_id: int) -> Result[User, UserStoreError]:
    if user_id == 503:
        return Ok(User(user_id, "Ada (cached)"))
    return Err(UserNotFound(user_id))


def translate_driver_error(error: DriverError) -> UserStoreError:
    match error:
        case DriverUnavailable(cause=cause):
            return UserStoreUnavailable(cause)
        case DriverNotFound(user_id=user_id):
            return UserNotFound(user_id)
        case _:
            assert_never(error)


def recover_from_store_error(
    user_id: int,
    error: UserStoreError,
) -> Result[User, UserStoreError]:
    if isinstance(error, UserStoreUnavailable):
        return cached_user(user_id)
    return Err(error)


def load_user(user_id: int) -> Result[User, UserStoreError]:
    """Translate the driver error, then recover only from store outages."""
    translated = primary_user(user_id).map_err(translate_driver_error)
    return translated.or_else(lambda error: recover_from_store_error(user_id, error))


def describe_error(error: UserStoreError) -> str:
    match error:
        case UserStoreUnavailable(cause=cause):
            return cause
        case UserNotFound(user_id=user_id):
            return f"user {user_id} was not found"
        case _:
            assert_never(error)


def main() -> None:
    for user_id in (1, 503, 404):
        result = load_user(user_id)
        result = result.inspect(lambda user: print(f"loaded {user.name}")).inspect_err(
            lambda error: print(f"load failed: {describe_error(error)}")
        )
        print(f"{user_id}: {result}")


if __name__ == "__main__":
    main()
