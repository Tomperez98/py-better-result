"""Result transformations and side-effect combinators."""

# ``map`` intentionally mirrors the Result API, despite shadowing a builtin.

from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

from .core import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping


def _is_data_last_call(
    value: object,
    *,
    supplied: bool,
    name: str,
    argument: str = "callback",
) -> bool:
    """Validate a dual-form call and report whether it is data-last."""
    if not supplied:
        if isinstance(value, (Ok, Err)):
            message = f"{name} data-last form requires a {argument}"
            raise TypeError(message)
        return True
    if not isinstance(value, (Ok, Err)):
        message = f"{name} data-first form requires a Result"
        raise TypeError(message)
    return False


@overload
def map_result[A, B, E](result: Result[A, E], fn: Callable[[A], B]) -> Result[B, E]: ...


@overload
def map_result[A, B, E](
    fn: Callable[[A], B],
) -> Callable[[Result[A, E]], Result[B, E]]: ...


def map_result[A, B, E](
    result_or_fn: Result[A, E] | Callable[[A], B],
    fn: Callable[[A], B] | None = None,
) -> Result[B, E] | Callable[[Result[A, E]], Result[B, E]]:
    """Map a success value in data-first or data-last form."""
    if _is_data_last_call(result_or_fn, supplied=fn is not None, name="map"):
        callback = cast("Callable[[A], B]", result_or_fn)
        return lambda result: result.map(callback)
    result = cast("Result[A, E]", result_or_fn)
    callback = cast("Callable[[A], B]", fn)
    return result.map(callback)


@overload
def map_error[A, E, E2](
    result: Result[A, E],
    fn: Callable[[E], E2],
) -> Result[A, E2]: ...


@overload
def map_error[A, E, E2](
    fn: Callable[[E], E2],
) -> Callable[[Result[A, E]], Result[A, E2]]: ...


def map_error[A, E, E2](
    result_or_fn: Result[A, E] | Callable[[E], E2],
    fn: Callable[[E], E2] | None = None,
) -> Result[A, E2] | Callable[[Result[A, E]], Result[A, E2]]:
    """Map an error value in data-first or data-last form."""
    if _is_data_last_call(result_or_fn, supplied=fn is not None, name="map_error"):
        callback = cast("Callable[[E], E2]", result_or_fn)
        return lambda result: result.map_error(callback)
    result = cast("Result[A, E]", result_or_fn)
    callback = cast("Callable[[E], E2]", fn)
    return result.map_error(callback)


@overload
def try_recover[A, E, B, E2](
    result: Result[A, E],
    fn: Callable[[E], Result[B, E2]],
) -> Result[A | B, E2]: ...


@overload
def try_recover[A, E, B, E2](
    fn: Callable[[E], Result[B, E2]],
) -> Callable[[Result[A, E]], Result[A | B, E2]]: ...


def try_recover[A, E, B, E2](
    result_or_fn: Result[A, E] | Callable[[E], Result[B, E2]],
    fn: Callable[[E], Result[B, E2]] | None = None,
) -> Result[A | B, E2] | Callable[[Result[A, E]], Result[A | B, E2]]:
    """Recover an error in data-first or data-last form."""
    if _is_data_last_call(result_or_fn, supplied=fn is not None, name="try_recover"):
        callback = cast("Callable[[E], Result[B, E2]]", result_or_fn)
        return lambda result: cast("Result[A | B, E2]", result.try_recover(callback))
    result = cast("Result[A, E]", result_or_fn)
    callback = cast("Callable[[E], Result[B, E2]]", fn)
    return cast("Result[A | B, E2]", result.try_recover(callback))


@overload
def try_recover_async[A, E, B, E2](
    result: Result[A, E],
    fn: Callable[[E], Awaitable[Result[B, E2]]],
) -> Awaitable[Result[A | B, E2]]: ...


@overload
def try_recover_async[A, E, B, E2](
    fn: Callable[[E], Awaitable[Result[B, E2]]],
) -> Callable[[Result[A, E]], Awaitable[Result[A | B, E2]]]: ...


def try_recover_async[A, E, B, E2](
    result_or_fn: Result[A, E] | Callable[[E], Awaitable[Result[B, E2]]],
    fn: Callable[[E], Awaitable[Result[B, E2]]] | None = None,
) -> (
    Awaitable[Result[A | B, E2]]
    | Callable[[Result[A, E]], Awaitable[Result[A | B, E2]]]
):
    """Async recovery in data-first or data-last form."""
    if _is_data_last_call(
        result_or_fn, supplied=fn is not None, name="try_recover_async"
    ):
        callback = cast(
            "Callable[[E], Awaitable[Result[B, E2]]]",
            result_or_fn,
        )

        async def recover_later(result: Result[A, E]) -> Result[A | B, E2]:
            return cast("Result[A | B, E2]", await result.try_recover_async(callback))

        return recover_later
    result = cast("Result[A, E]", result_or_fn)
    callback = cast("Callable[[E], Awaitable[Result[B, E2]]]", fn)
    return cast("Awaitable[Result[A | B, E2]]", result.try_recover_async(callback))


@overload
def and_then[A, E, B, E2](
    result: Result[A, E],
    fn: Callable[[A], Result[B, E2]],
) -> Result[B, E | E2]: ...


@overload
def and_then[A, E, B, E2](
    fn: Callable[[A], Result[B, E2]],
) -> Callable[[Result[A, E]], Result[B, E | E2]]: ...


def and_then[A, E, B, E2](
    result_or_fn: Result[A, E] | Callable[[A], Result[B, E2]],
    fn: Callable[[A], Result[B, E2]] | None = None,
) -> Result[B, E | E2] | Callable[[Result[A, E]], Result[B, E | E2]]:
    """Chain a Result-returning callback in data-first or data-last form."""
    if _is_data_last_call(result_or_fn, supplied=fn is not None, name="and_then"):
        callback = cast("Callable[[A], Result[B, E2]]", result_or_fn)
        return lambda result: result.and_then(callback)
    result = cast("Result[A, E]", result_or_fn)
    callback = cast("Callable[[A], Result[B, E2]]", fn)
    return result.and_then(callback)


@overload
def and_then_async[A, E, B, E2](
    result: Result[A, E],
    fn: Callable[[A], Awaitable[Result[B, E2]]],
) -> Awaitable[Result[B, E | E2]]: ...


@overload
def and_then_async[A, E, B, E2](
    fn: Callable[[A], Awaitable[Result[B, E2]]],
) -> Callable[[Result[A, E]], Awaitable[Result[B, E | E2]]]: ...


def and_then_async[A, E, B, E2](
    result_or_fn: Result[A, E] | Callable[[A], Awaitable[Result[B, E2]]],
    fn: Callable[[A], Awaitable[Result[B, E2]]] | None = None,
) -> (
    Awaitable[Result[B, E | E2]]
    | Callable[[Result[A, E]], Awaitable[Result[B, E | E2]]]
):
    """Async Result chaining in data-first or data-last form."""
    if _is_data_last_call(result_or_fn, supplied=fn is not None, name="and_then_async"):
        callback = cast(
            "Callable[[A], Awaitable[Result[B, E2]]]",
            result_or_fn,
        )

        async def chain_later(result: Result[A, E]) -> Result[B, E | E2]:
            return await result.and_then_async(callback)

        return chain_later
    result = cast("Result[A, E]", result_or_fn)
    callback = cast("Callable[[A], Awaitable[Result[B, E2]]]", fn)
    return result.and_then_async(callback)


@overload
def match[A, E, T](
    result: Result[A, E],
    handlers: Mapping[str, Callable[..., T]],
) -> T: ...


@overload
def match[A, E, T](
    handlers: Mapping[str, Callable[..., T]],
) -> Callable[[Result[A, E]], T]: ...


def match[A, E, T](
    result_or_handlers: Result[A, E] | Mapping[str, Callable[..., T]],
    handlers: Mapping[str, Callable[..., T]] | None = None,
) -> T | Callable[[Result[A, E]], T]:
    """Match a Result in data-first or data-last form."""
    if _is_data_last_call(
        result_or_handlers,
        supplied=handlers is not None,
        name="match",
        argument="handlers",
    ):
        callback = cast("Mapping[str, Callable[..., T]]", result_or_handlers)
        return lambda result: result.match(callback)
    result = cast("Result[A, E]", result_or_handlers)
    return result.match(cast("Mapping[str, Callable[..., T]]", handlers))


@overload
def tap[A, E](result: Result[A, E], fn: Callable[[A], object]) -> Result[A, E]: ...


@overload
def tap[A, E](fn: Callable[[A], object]) -> Callable[[Result[A, E]], Result[A, E]]: ...


def tap[A, E](
    result_or_fn: Result[A, E] | Callable[[A], object],
    fn: Callable[[A], object] | None = None,
) -> Result[A, E] | Callable[[Result[A, E]], Result[A, E]]:
    """Run a success side effect in data-first or data-last form."""
    if _is_data_last_call(result_or_fn, supplied=fn is not None, name="tap"):
        callback = cast("Callable[[A], object]", result_or_fn)
        return lambda result: result.tap(callback)
    result = cast("Result[A, E]", result_or_fn)
    callback = cast("Callable[[A], object]", fn)
    return result.tap(callback)


@overload
def tap_async[A, E](
    result: Result[A, E],
    fn: Callable[[A], Awaitable[object]],
) -> Awaitable[Result[A, E]]: ...


@overload
def tap_async[A, E](
    fn: Callable[[A], Awaitable[object]],
) -> Callable[[Result[A, E]], Awaitable[Result[A, E]]]: ...


def tap_async[A, E](
    result_or_fn: Result[A, E] | Callable[[A], Awaitable[object]],
    fn: Callable[[A], Awaitable[object]] | None = None,
) -> Awaitable[Result[A, E]] | Callable[[Result[A, E]], Awaitable[Result[A, E]]]:
    """Run an async success side effect in data-first or data-last form."""
    if _is_data_last_call(result_or_fn, supplied=fn is not None, name="tap_async"):
        callback = cast("Callable[[A], Awaitable[object]]", result_or_fn)

        async def tap_later(result: Result[A, E]) -> Result[A, E]:
            return await result.tap_async(callback)

        return tap_later
    result = cast("Result[A, E]", result_or_fn)
    callback = cast("Callable[[A], Awaitable[object]]", fn)
    return result.tap_async(callback)


@overload
def tap_error[A, E](
    result: Result[A, E],
    fn: Callable[[E], object],
) -> Result[A, E]: ...


@overload
def tap_error[A, E](
    fn: Callable[[E], object],
) -> Callable[[Result[A, E]], Result[A, E]]: ...


def tap_error[A, E](
    result_or_fn: Result[A, E] | Callable[[E], object],
    fn: Callable[[E], object] | None = None,
) -> Result[A, E] | Callable[[Result[A, E]], Result[A, E]]:
    """Run an error side effect in data-first or data-last form."""
    if _is_data_last_call(result_or_fn, supplied=fn is not None, name="tap_error"):
        callback = cast("Callable[[E], object]", result_or_fn)
        return lambda result: result.tap_error(callback)
    result = cast("Result[A, E]", result_or_fn)
    callback = cast("Callable[[E], object]", fn)
    return result.tap_error(callback)


@overload
def tap_error_async[A, E](
    result: Result[A, E],
    fn: Callable[[E], Awaitable[object]],
) -> Awaitable[Result[A, E]]: ...


@overload
def tap_error_async[A, E](
    fn: Callable[[E], Awaitable[object]],
) -> Callable[[Result[A, E]], Awaitable[Result[A, E]]]: ...


def tap_error_async[A, E](
    result_or_fn: Result[A, E] | Callable[[E], Awaitable[object]],
    fn: Callable[[E], Awaitable[object]] | None = None,
) -> Awaitable[Result[A, E]] | Callable[[Result[A, E]], Awaitable[Result[A, E]]]:
    """Run an async error side effect in data-first or data-last form."""
    if _is_data_last_call(
        result_or_fn, supplied=fn is not None, name="tap_error_async"
    ):
        callback = cast("Callable[[E], Awaitable[object]]", result_or_fn)

        async def tap_error_later(result: Result[A, E]) -> Result[A, E]:
            return await result.tap_error_async(callback)

        return tap_error_later
    result = cast("Result[A, E]", result_or_fn)
    callback = cast("Callable[[E], Awaitable[object]]", fn)
    return result.tap_error_async(callback)


@overload
def tap_both[A, E](
    result: Result[A, E],
    handlers: Mapping[str, Callable[..., object]],
) -> Result[A, E]: ...


@overload
def tap_both[A, E](
    handlers: Mapping[str, Callable[..., object]],
) -> Callable[[Result[A, E]], Result[A, E]]: ...


def tap_both[A, E](
    result_or_handlers: Result[A, E] | Mapping[str, Callable[..., object]],
    handlers: Mapping[str, Callable[..., object]] | None = None,
) -> Result[A, E] | Callable[[Result[A, E]], Result[A, E]]:
    """Run the active side effect in data-first or data-last form."""
    if _is_data_last_call(
        result_or_handlers,
        supplied=handlers is not None,
        name="tap_both",
        argument="handlers",
    ):
        callback = cast("Mapping[str, Callable[..., object]]", result_or_handlers)
        return lambda result: result.tap_both(callback)
    result = cast("Result[A, E]", result_or_handlers)
    callback = cast("Mapping[str, Callable[..., object]]", handlers)
    return result.tap_both(callback)


@overload
def tap_both_async[A, E](
    result: Result[A, E],
    handlers: Mapping[str, Callable[..., Awaitable[object]]],
) -> Awaitable[Result[A, E]]: ...


@overload
def tap_both_async[A, E](
    handlers: Mapping[str, Callable[..., Awaitable[object]]],
) -> Callable[[Result[A, E]], Awaitable[Result[A, E]]]: ...


def tap_both_async[A, E](
    result_or_handlers: Result[A, E] | Mapping[str, Callable[..., Awaitable[object]]],
    handlers: Mapping[str, Callable[..., Awaitable[object]]] | None = None,
) -> Awaitable[Result[A, E]] | Callable[[Result[A, E]], Awaitable[Result[A, E]]]:
    """Run the active async side effect in data-first or data-last form."""
    if _is_data_last_call(
        result_or_handlers,
        supplied=handlers is not None,
        name="tap_both_async",
        argument="handlers",
    ):
        callback = cast(
            "Mapping[str, Callable[..., Awaitable[object]]]",
            result_or_handlers,
        )

        async def tap_both_later(result: Result[A, E]) -> Result[A, E]:
            return await result.tap_both_async(callback)

        return tap_both_later
    result = cast("Result[A, E]", result_or_handlers)
    callback = cast("Mapping[str, Callable[..., Awaitable[object]]]", handlers)
    return result.tap_both_async(callback)


_MISSING = object()


def unwrap[A, E](result: Result[A, E], message: str | None = None) -> A:
    """Extract a success value or raise ``Panic`` for an error result."""
    return result.unwrap(message)


@overload
def unwrap_or[A, E, B](result: Result[A, E], fallback: B) -> A | B: ...


@overload
def unwrap_or[A, E, B](fallback: B) -> Callable[[Result[A, E]], A | B]: ...


def unwrap_or[A, E, B](
    result_or_fallback: Result[A, E] | B,
    fallback: B | object = _MISSING,
) -> A | B | Callable[[Result[A, E]], A | B]:
    """Extract a success value or return a fallback in either call form."""
    if _is_data_last_call(
        result_or_fallback,
        supplied=fallback is not _MISSING,
        name="unwrap_or",
        argument="fallback",
    ):
        fallback_value = cast("B", result_or_fallback)
        return lambda result: result.unwrap_or(fallback_value)
    result = cast("Result[A, E]", result_or_fallback)
    return cast("A | B", result.unwrap_or(fallback))
