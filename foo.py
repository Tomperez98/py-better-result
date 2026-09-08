from __future__ import annotations

from better_result import Err, Ok, Result, TaggedError


class InvalidPort(TaggedError, tag="InvalidPort"):
    raw: str

    def __init__(self, raw: str) -> None:
        super().__init__(message="port must be between 1 and 65535", raw=raw)


def parse_port(raw: str) -> Result[int, InvalidPort]:
    try:
        port = int(raw)
    except ValueError:
        return Err(InvalidPort(raw))
    return Ok(port) if 1 <= port <= 65_535 else Err(InvalidPort(raw))


def format_address(port: int) -> str:
    return f"http://localhost:{port}"


def address_for(raw: str) -> str:
    return (
        parse_port(raw)
        .map(format_address)
        .match(
            str,
            lambda error: f"Invalid input {error.raw!r}: {error.message}",
        )
    )


print(address_for("8080"))
print(address_for("nope"))
