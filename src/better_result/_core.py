"""Internal implementation of the narrow Result API."""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import Mapping
from typing import TYPE_CHECKING, Generic, Literal, Never, TypeVar, cast, override

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

E_co = TypeVar("E_co", covariant=True)
E2 = TypeVar("E2")
T_co = TypeVar("T_co", covariant=True)
U = TypeVar("U")


class PanicError(BaseException):
    """An unrecoverable programmer or invariant failure."""

    _tag: Literal["Panic"] = "Panic"
    name: Literal["Panic"] = "Panic"

    def __init__(self, message: str, cause: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause
        if isinstance(cause, BaseException):
            self.__cause__ = cause
        self.stack = "".join(traceback.format_stack()[:-1])

    def to_dict(self) -> dict[str, object | None]:
        """Return diagnostic metadata as Python values."""
        return {
            "_tag": self._tag,
            "name": self.name,
            "message": self.message,
            "cause": _serialize_cause(self.cause),
            "stack": self.stack,
        }

    def to_json(self) -> dict[str, object | None]:
        """Return recursively JSON-compatible diagnostic metadata."""
        return cast("dict[str, object | None]", _json_safe(self.to_dict()))

    def to_safe_dict(self) -> dict[str, object | None]:
        """Return diagnostic metadata without stack traces."""
        return cast(
            "dict[str, object | None]",
            _without_diagnostic_stack(self.to_dict()),
        )

    def to_safe_json(self) -> dict[str, object | None]:
        """Return a JSON-compatible transport payload."""
        return cast("dict[str, object | None]", _json_safe(self.to_safe_dict()))


def _serialize_cause(cause: object | None) -> object | None:
    if isinstance(cause, BaseException):
        return {
            "name": type(cause).__name__,
            "message": str(cause),
            "stack": "".join(
                traceback.format_exception(type(cause), cause, cause.__traceback__),
            ),
        }
    return cause


def _without_diagnostic_stack(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _without_diagnostic_stack(item)
            for key, item in value.items()
            if key != "stack"
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_without_diagnostic_stack(item) for item in value]
    return value


def _json_safe(value: object, seen: set[int] | None = None) -> object:
    """Convert arbitrary values into JSON-compatible primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseException):
        return _json_safe(_serialize_cause(value), seen)

    active = seen if seen is not None else set()
    identity = id(value)
    if identity in active:
        return "<cycle>"

    if isinstance(value, Mapping):
        active.add(identity)
        try:
            return {str(key): _json_safe(item, active) for key, item in value.items()}
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple, set, frozenset)):
        active.add(identity)
        try:
            return [_json_safe(item, active) for item in value]
        finally:
            active.remove(identity)
    return repr(value)


def panic(message: str, cause: object | None = None) -> Never:
    """Raise an internal unrecoverable failure."""
    raise PanicError(message, cause)


def _try_or_panic[C](fn: Callable[[], C], message: str) -> C:
    try:
        return fn()
    except PanicError:
        raise
    except Exception as cause:
        raise PanicError(message, cause) from cause


async def _try_or_panic_async[C](fn: Callable[[], Awaitable[C]], message: str) -> C:
    try:
        return await fn()
    except asyncio.CancelledError:
        raise
    except PanicError:
        raise
    except Exception as cause:
        raise PanicError(message, cause) from cause


class Ok(Generic[T_co]):  # noqa: UP046
    """Successful Result variant."""

    __slots__ = ("value",)
    __hash__ = None
    value: T_co

    def __init__(self, value: T_co) -> None:
        object.__setattr__(self, "value", value)

    @override
    def __setattr__(self, _name: str, _value: object) -> Never:
        message = "Ok is immutable"
        raise AttributeError(message)

    @override
    def __repr__(self) -> str:
        return f"Ok({self.value!r})"

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Ok)
            and type(self) is type(other)
            and self.value == other.value
        )

    def map(self, fn: Callable[[T_co], U]) -> Ok[U]:
        """Transform the success value."""
        return _try_or_panic(lambda: Ok(fn(self.value)), "map callback threw")

    def map_error(self, _fn: Callable[[Never], E2]) -> Ok[T_co]:
        """Leave a successful result unchanged."""
        return self

    def and_then(self, fn: Callable[[T_co], Result[U, E2]]) -> Result[U, E2]:
        """Sequence a Result-returning operation."""

        def callback() -> Result[U, E2]:
            return cast(
                "Result[U, E2]",
                _require_result(
                    fn(self.value), "and_then callback must return a Result"
                ),
            )

        return _try_or_panic(callback, "and_then callback threw")

    async def and_then_async(
        self,
        fn: Callable[[T_co], Awaitable[Result[U, E2]]],
    ) -> Result[U, E2]:
        """Async version of :meth:`and_then`."""

        async def callback() -> Result[U, E2]:
            return cast(
                "Result[U, E2]",
                _require_result(
                    await fn(self.value),
                    "and_then_async callback must return a Result",
                ),
            )

        return await _try_or_panic_async(callback, "and_then_async callback threw")

    def match(self, on_ok: Callable[[T_co], U], on_err: Callable[[Never], U]) -> U:
        """Consume the successful branch with explicit branch callbacks."""
        if not callable(on_ok) or not callable(on_err):
            panic("match callbacks must be callable")
        return _try_or_panic(lambda: on_ok(self.value), "match ok handler threw")

    def unwrap_or(self, _fallback: U) -> T_co:
        """Return the success value."""
        return self.value


class Err(Generic[E_co]):  # noqa: UP046
    """Failed Result variant."""

    __slots__ = ("error",)
    __hash__ = None
    error: E_co

    def __init__(self, error: E_co) -> None:
        object.__setattr__(self, "error", error)

    @override
    def __setattr__(self, _name: str, _value: object) -> Never:
        message = "Err is immutable"
        raise AttributeError(message)

    @override
    def __repr__(self) -> str:
        return f"Err({self.error!r})"

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Err)
            and type(self) is type(other)
            and self.error == other.error
        )

    def map(self, _fn: Callable[[Never], U]) -> Err[E_co]:
        """Leave a failed result unchanged."""
        return self

    def map_error(self, fn: Callable[[E_co], E2]) -> Err[E2]:
        """Transform the failure value."""
        return _try_or_panic(lambda: Err(fn(self.error)), "map_error callback threw")

    def and_then(self, _fn: Callable[[Never], Result[U, E2]]) -> Err[E_co]:
        """Leave a failed result unchanged."""
        return self

    async def and_then_async(
        self,
        _fn: Callable[[Never], Awaitable[Result[U, E2]]],
    ) -> Err[E_co]:
        """Async no-op for a failed result."""
        return self

    def match(self, on_ok: Callable[[Never], U], on_err: Callable[[E_co], U]) -> U:
        """Consume the failed branch with explicit branch callbacks."""
        if not callable(on_ok) or not callable(on_err):
            panic("match callbacks must be callable")
        return _try_or_panic(lambda: on_err(self.error), "match err handler threw")

    def unwrap_or(self, fallback: U) -> U:
        """Return the fallback value."""
        return fallback


type Result[A, E] = Ok[A] | Err[E]


def _require_result(value: object, message: str) -> Result[object, object]:
    """Fail fast when a callback violates the Result return contract."""
    if not isinstance(value, (Ok, Err)):
        raise PanicError(message, value)
    return cast("Result[object, object]", value)
