"""Compose asynchronous Result operations and collect them concurrently."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from better_result import Err, Ok, Result, all_results_async


@dataclass(frozen=True, slots=True)
class User:
    user_id: int
    name: str


async def fetch_user(user_id: int) -> Result[User, str]:
    """Pretend to call an async data source."""
    await asyncio.sleep(0)
    if user_id == 404:
        return Err(f"user {user_id} not found")
    return Ok(User(user_id, f"user-{user_id}"))


async def fetch_display_name(user_id: int) -> Result[str, str]:
    user = await Ok(user_id).and_then_async(fetch_user)

    async def title_name(value: User) -> str:
        await asyncio.sleep(0)
        return value.name.title()

    return await user.map_async(title_name)


async def main() -> None:
    # Awaitables passed to all_results_async run concurrently and keep input order.
    names = await all_results_async(
        (
            fetch_display_name(1),
            fetch_display_name(2),
            fetch_display_name(404),
        )
    )
    print(names)


if __name__ == "__main__":
    asyncio.run(main())
