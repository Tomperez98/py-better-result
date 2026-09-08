"""Traverse async operations with input ordering and bounded concurrency."""

from __future__ import annotations

import asyncio

from better_result import Err, Ok, Result, traverse_async


async def fetch_user(user_id: int) -> Result[str, str]:
    await asyncio.sleep(0.01)
    if user_id == 404:
        return Err(f"user {user_id} not found")
    return Ok(f"user-{user_id}")


async def main() -> None:
    result = await traverse_async(
        (1, 2, 404, 3),
        fetch_user,
        max_concurrency=2,
    )
    assert result == Err("user 404 not found")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
