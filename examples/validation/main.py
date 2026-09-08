"""Turn an expected parsing exception into a typed Result error."""

from __future__ import annotations

from better_result import Err, Ok, Result, TryContext, try_result


def parse_port(raw: str) -> Result[int, str]:
    """Parse and validate a TCP port without a broad try/except in the caller."""

    def convert(_context: TryContext) -> int:
        return int(raw)

    parsed = try_result(convert, catch=str)
    return parsed.and_then(validate_port)


def validate_port(port: int) -> Result[int, str]:
    if not 1 <= port <= 65_535:
        return Err("port must be between 1 and 65535")
    return Ok(port)


def main() -> None:
    for raw_port in ("8080", "not-a-port", "70000"):
        result = parse_port(raw_port)
        print(f"{raw_port!r} -> {result}")


if __name__ == "__main__":
    main()
