"""Compose a small workflow with Ok, Err, map, and and_then."""

from __future__ import annotations

from dataclasses import dataclass

from better_result import Err, Ok, Result


@dataclass(frozen=True, slots=True)
class User:
    name: str


def parse_user_id(raw: str) -> Result[int, str]:
    """Validate input at the edge of the application."""
    if not raw.isdecimal():
        return Err("user id must be a number")
    return Ok(int(raw))


def load_user(user_id: int) -> Result[User, str]:
    """Return an expected domain error instead of raising for a missing user."""
    users = {42: User("Ada"), 7: User("Grace")}
    user = users.get(user_id)
    if user is None:
        return Err("user not found")
    return Ok(user)


def main() -> None:
    for raw_id in ("42", "not-an-id", "404"):
        result = parse_user_id(raw_id).and_then(load_user)
        expected = {
            "42": Ok(User("Ada")),
            "not-an-id": Err("user id must be a number"),
            "404": Err("user not found"),
        }
        assert result == expected[raw_id]

        # Only the callback for the active branch runs.
        greeting = result.map(lambda user: f"Hello, {user.name}!")

        match greeting:
            case Ok(message):
                print(message)
            case Err(message):
                print(f"Could not load user: {message}")

    # Use a fallback when the caller does not need to preserve the error.
    fallback = Err("service unavailable").unwrap_or_else(
        lambda error: f"using cached data ({error})"
    )
    assert fallback == "using cached data (service unavailable)"
    print(fallback)


if __name__ == "__main__":
    main()
