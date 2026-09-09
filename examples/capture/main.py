"""Convert exception-based operations into typed Results."""

from __future__ import annotations

import asyncio

from better_result import Err, Ok, Result, capture, capture_async


def parse_port(raw: str) -> int:
    return int(raw)


def parse_port_result(raw: str) -> Result[int, str]:
    return capture(lambda: parse_port(raw), catch=lambda e: f"invalid port: {e}")


async def fetch_config() -> str:
    message = "configuration service timed out"
    raise TimeoutError(message)


async def main_async() -> None:
    result = await capture_async(fetch_config, catch=str)
    assert result == Err("configuration service timed out")
    print(f"async failure: {result}")


def main() -> None:
    valid = parse_port_result("8080")
    invalid = parse_port_result("not-a-port")

    assert valid == Ok(8080)
    assert invalid == Err(
        "invalid port: invalid literal for int() with base 10: 'not-a-port'"
    )
    print(f"success: {valid}")
    print(f"mapped failure: {invalid}")

    asyncio.run(main_async())


if __name__ == "__main__":
    main()
