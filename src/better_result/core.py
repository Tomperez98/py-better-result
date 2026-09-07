"""
Core result types and fail-fast panic handling.

The implementation mirrors ``core.ts`` while following Python's naming and
typing conventions.  ``Ok`` and ``Err`` are deliberately small immutable-in-
practice containers: their payloads are references, just like the TypeScript
implementation, so mutating a mutable payload remains visible to callers.
"""

# Python 3.12 cannot use the newer ``type``/generic syntax that these rules
# recommend; the typing aliases below are intentional.
from __future__ import annotations

import asyncio
import traceback
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Generic,
    Literal,
    Never,
    TypedDict,
    TypeGuard,
    TypeVar,
    cast,
    overload,
    override,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator

A = TypeVar("A")
B = TypeVar("B")
E = TypeVar("E")
E_co = TypeVar("E_co", covariant=True)
E2 = TypeVar("E2")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
U = TypeVar("U")


class Panic(Exception):
    """
    An unrecoverable failure raised by a Result operation.

    Callback exceptions are wrapped in ``Panic`` so that programmer errors do
    not accidentally become ordinary ``Err`` values.  ``cause`` is retained
    as the original value and, when it is an exception, is also attached to
    Python's exception chain.
    """

    _tag: Literal["Panic"] = "Panic"
    name: Literal["Panic"] = "Panic"

    def __init__(self, message: str, cause: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause

        # Python normally sets __cause__ only for ``raise ... from ...``.  A
        # directly-created Panic still represents the same causal relation as
        # the TypeScript Error's ``cause`` property.
        if isinstance(cause, BaseException):
            self.__cause__ = cause

        self.stack = "".join(traceback.format_stack()[:-1])

    @staticmethod
    def is_panic(value: object) -> TypeGuard[Panic]:
        """Return whether *value* is a Panic, including subclasses."""
        return isinstance(value, Panic)

    def to_dict(self) -> dict[str, object | None]:
        """Return this panic as a Python dictionary."""
        return {
            "_tag": self._tag,
            "name": self.name,
            "message": self.message,
            "cause": _serialize_cause(self.cause),
            "stack": self.stack,
        }

    def to_json(self) -> dict[str, object | None]:
        """Return a recursively JSON-compatible diagnostic dictionary."""
        return cast("dict[str, object | None]", _json_safe(self.to_dict()))

    def to_safe_dict(self) -> dict[str, object | None]:
        """Return panic metadata without diagnostic stack traces."""
        return cast(
            "dict[str, object | None]", _without_diagnostic_stack(self.to_dict())
        )

    def to_safe_json(self) -> dict[str, object | None]:
        """Return a JSON-compatible panic payload safe for transport."""
        return cast("dict[str, object | None]", _json_safe(self.to_safe_dict()))

    def __iter__(self) -> Generator[Err[Panic], None, Never]:
        """Yield this panic as an Err, then fail if iteration continues."""
        yield err(self)
        panic("Unreachable: Err yielded in Panic but generator continued", self)


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
    """Remove stack fields recursively from a diagnostic payload."""
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


def is_panic(value: object) -> TypeGuard[Panic]:
    """Return whether *value* is a Panic."""
    return isinstance(value, Panic)


def panic(message: str, cause: object | None = None) -> Never:
    """Raise an unrecoverable :class:`Panic`."""
    raise Panic(message, cause)


C = TypeVar("C")


def _try_or_panic[C](fn: Callable[[], C], message: str) -> C:
    """Execute a callback and wrap ordinary callback failures in Panic."""
    try:
        return fn()
    except Panic:
        raise
    except Exception as cause:
        raise Panic(message, cause) from cause


async def _try_or_panic_async[C](fn: Callable[[], Awaitable[C]], message: str) -> C:
    """Async version of :func:`_try_or_panic` that preserves cancellation."""
    try:
        return await fn()
    except asyncio.CancelledError:
        raise
    except Panic:
        raise
    except Exception as cause:
        raise Panic(message, cause) from cause


type Handlers[T] = Mapping[str, Callable[..., T]]


class ResultHandlers[T, E, U](TypedDict):
    """Required handlers for the two fixed Result branches."""

    ok: Callable[[T], U]
    err: Callable[[E], U]


class TapHandlers[T, E](TypedDict):
    """Required synchronous observers for both Result branches."""

    ok: Callable[[T], object]
    err: Callable[[E], object]


class AsyncTapHandlers[T, E](TypedDict):
    """Required asynchronous observers for both Result branches."""

    ok: Callable[[T], Awaitable[object]]
    err: Callable[[E], Awaitable[object]]


class Ok(Generic[T_co]):  # noqa: UP046
    """Successful immutable Result variant."""

    __slots__ = ("value",)
    __hash__ = None
    status: ClassVar[Literal["ok"]] = "ok"
    value: T_co

    def __init__(self, value: T_co) -> None:
        object.__setattr__(self, "value", value)

    @override
    def __setattr__(self, _name: str, _value: object) -> Never:
        raise AttributeError("Ok is immutable")

    @override
    def __repr__(self) -> str:
        """Return a concise representation of the successful result."""
        return f"Ok({self.value!r})"

    @override
    def __eq__(self, other: object) -> bool:
        """Compare successful results by concrete variant and value."""
        if not isinstance(other, Ok):
            return False
        return type(self) is type(other) and self.value == other.value

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def map(self, fn: Callable[[T_co], U]) -> Ok[U]:
        """Transform the success value, panicking if the callback fails."""
        return _try_or_panic(lambda: Ok(fn(self.value)), "map callback threw")

    def map_error(self, _fn: Callable[[Never], E2]) -> Ok[T_co]:
        """No-op on Ok; the error type changes only at the type level."""
        return self

    @overload
    def try_recover(self, _fn: Callable[[Never], Ok[U]]) -> Ok[T_co]: ...

    @overload
    def try_recover(self, _fn: Callable[[Never], Err[E2]]) -> Ok[T_co]: ...

    @overload
    def try_recover(self, _fn: Callable[[Never], Result[U, E2]]) -> Ok[T_co]: ...

    def try_recover(self, _fn: Callable[[Never], Result[U, E2]]) -> Ok[T_co]:
        """No-op on Ok; never invokes the recovery callback."""
        return self

    async def try_recover_async(
        self,
        _fn: Callable[[Never], Awaitable[Result[U, E2]]],
    ) -> Ok[T_co]:
        """Async no-op on Ok; never invokes the recovery callback."""
        return self

    def and_then(self, fn: Callable[[T_co], Result[U, E2]]) -> Result[U, E2]:
        """Run and validate a Result-returning callback on success."""

        def callback() -> Result[U, E2]:
            return cast(
                "Result[U, E2]",
                _require_result(
                    fn(self.value),
                    "and_then callback must return a Result",
                ),
            )

        return _try_or_panic(callback, "and_then callback threw")

    async def and_then_async(
        self,
        fn: Callable[[T_co], Awaitable[Result[U, E2]]],
    ) -> Result[U, E2]:
        """Async version of :meth:`and_then` with Result validation."""

        async def callback() -> Result[U, E2]:
            return cast(
                "Result[U, E2]",
                _require_result(
                    await fn(self.value),
                    "and_then_async callback must return a Result",
                ),
            )

        return await _try_or_panic_async(callback, "and_then_async callback threw")

    @overload
    def match(self, handlers: ResultHandlers[T_co, Never, U]) -> U: ...

    @overload
    def match(self, handlers: Handlers[U]) -> U: ...

    def match(self, handlers: ResultHandlers[T_co, Never, U] | Handlers[U]) -> U:
        """Call the ``ok`` branch of a pair of match handlers."""
        handler = cast("Callable[[T_co], U]", handlers["ok"])
        return _try_or_panic(lambda: handler(self.value), "match ok handler threw")

    def unwrap(self, _message: str | None = None) -> T_co:
        """Extract the success value."""
        return self.value

    def unwrap_or(self, _fallback: U) -> T_co:
        """Extract the value, ignoring the fallback."""
        return self.value

    def tap(self, fn: Callable[[T_co], object]) -> Ok[T_co]:
        """Run a success side effect and return this result."""

        def callback() -> Ok[T_co]:
            fn(self.value)
            return self

        return _try_or_panic(callback, "tap callback threw")

    async def tap_async(self, fn: Callable[[T_co], Awaitable[object]]) -> Ok[T_co]:
        """Run an async success side effect and return this result."""

        async def callback() -> Ok[T_co]:
            await fn(self.value)
            return self

        return await _try_or_panic_async(callback, "tap_async callback threw")

    def tap_error(self, _fn: Callable[[Never], object]) -> Ok[T_co]:
        """No-op on Ok."""
        return self

    async def tap_error_async(
        self,
        _fn: Callable[[Never], Awaitable[object]],
    ) -> Ok[T_co]:
        """Async no-op on Ok."""
        return self

    @overload
    def tap_both(self, handlers: TapHandlers[T_co, Never]) -> Ok[T_co]: ...

    @overload
    def tap_both(self, handlers: Handlers[object]) -> Ok[T_co]: ...

    def tap_both(
        self, handlers: TapHandlers[T_co, Never] | Handlers[object]
    ) -> Ok[T_co]:
        """Run only the ``ok`` side effect and return this result."""
        handler = cast("Callable[[T_co], object]", handlers["ok"])

        def callback() -> Ok[T_co]:
            handler(self.value)
            return self

        return _try_or_panic(callback, "tap_both ok callback threw")

    @overload
    async def tap_both_async(
        self,
        handlers: AsyncTapHandlers[T_co, Never],
    ) -> Ok[T_co]: ...

    @overload
    async def tap_both_async(
        self,
        handlers: Handlers[Awaitable[object]],
    ) -> Ok[T_co]: ...

    async def tap_both_async(
        self,
        handlers: AsyncTapHandlers[T_co, Never] | Handlers[Awaitable[object]],
    ) -> Ok[T_co]:
        """Async version of :meth:`tap_both`."""
        handler = cast("Callable[[T_co], Awaitable[object]]", handlers["ok"])

        async def callback() -> Ok[T_co]:
            await handler(self.value)
            return self

        return await _try_or_panic_async(callback, "tap_both_async ok callback threw")

    def __iter__(self) -> Generator[object, None, T_co]:
        """Return the value through ``yield from`` without yielding it."""
        if False:  # Make this a generator so StopIteration carries the value.
            yield self.value
        return self.value


class Err(Generic[E_co]):  # noqa: UP046
    """Failed immutable Result variant."""

    __slots__ = ("error",)
    __hash__ = None
    status: ClassVar[Literal["error"]] = "error"
    error: E_co

    def __init__(self, error: E_co) -> None:
        object.__setattr__(self, "error", error)

    @override
    def __setattr__(self, _name: str, _value: object) -> Never:
        raise AttributeError("Err is immutable")

    @override
    def __repr__(self) -> str:
        """Return a concise representation of the failed result."""
        return f"Err({self.error!r})"

    @override
    def __eq__(self, other: object) -> bool:
        """Compare failed results by concrete variant and error."""
        if not isinstance(other, Err):
            return False
        return type(self) is type(other) and self.error == other.error

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def map(self, _fn: Callable[[Never], U]) -> Err[E_co]:
        """No-op on Err; the success type changes only at the type level."""
        return self

    def map_error(self, fn: Callable[[E_co], E2]) -> Err[E2]:
        """Transform the error value, panicking if the callback fails."""
        return _try_or_panic(lambda: Err(fn(self.error)), "map_error callback threw")

    def try_recover(self, fn: Callable[[E_co], Result[U, E2]]) -> Result[U, E2]:
        """Attempt and validate recovery through a Result callback."""

        def callback() -> Result[U, E2]:
            return cast(
                "Result[U, E2]",
                _require_result(
                    fn(self.error),
                    "try_recover callback must return a Result",
                ),
            )

        return _try_or_panic(callback, "try_recover callback threw")

    async def try_recover_async(
        self,
        fn: Callable[[E_co], Awaitable[Result[U, E2]]],
    ) -> Result[U, E2]:
        """Async version of :meth:`try_recover` with Result validation."""

        async def callback() -> Result[U, E2]:
            return cast(
                "Result[U, E2]",
                _require_result(
                    await fn(self.error),
                    "try_recover_async callback must return a Result",
                ),
            )

        return await _try_or_panic_async(callback, "try_recover_async callback threw")

    def and_then(self, _fn: Callable[[Never], Result[U, E2]]) -> Result[U, E_co | E2]:
        """No-op on Err; never invokes the callback."""
        return self

    async def and_then_async(
        self,
        _fn: Callable[[Never], Awaitable[Result[U, E2]]],
    ) -> Result[U, E_co | E2]:
        """Async no-op on Err; never invokes the callback."""
        return self

    @overload
    def match(self, handlers: ResultHandlers[Never, E_co, U]) -> U: ...

    @overload
    def match(self, handlers: Handlers[U]) -> U: ...

    def match(self, handlers: ResultHandlers[Never, E_co, U] | Handlers[U]) -> U:
        """Call the ``err`` branch of a pair of match handlers."""
        handler = cast("Callable[[E_co], U]", handlers["err"])
        return _try_or_panic(lambda: handler(self.error), "match err handler threw")

    def unwrap(self, message: str | None = None) -> Never:
        """Raise a Panic because this result contains an error."""
        panic(
            message if message is not None else f"Unwrap called on Err: {self.error}",
            self.error,
        )

    def unwrap_or(self, fallback: U) -> U:
        """Return the fallback value."""
        return fallback

    def tap(self, _fn: Callable[[Never], object]) -> Err[E_co]:
        """No-op on Err."""
        return self

    def tap_error(self, fn: Callable[[E_co], object]) -> Err[E_co]:
        """Run an error side effect and return this result."""

        def callback() -> Err[E_co]:
            fn(self.error)
            return self

        return _try_or_panic(callback, "tap_error callback threw")

    async def tap_async(self, _fn: Callable[[Never], Awaitable[object]]) -> Err[E_co]:
        """Async no-op on Err."""
        return self

    async def tap_error_async(
        self, fn: Callable[[E_co], Awaitable[object]]
    ) -> Err[E_co]:
        """Run an async error side effect and return this result."""

        async def callback() -> Err[E_co]:
            await fn(self.error)
            return self

        return await _try_or_panic_async(callback, "tap_error_async callback threw")

    @overload
    def tap_both(self, handlers: TapHandlers[Never, E_co]) -> Err[E_co]: ...

    @overload
    def tap_both(self, handlers: Handlers[object]) -> Err[E_co]: ...

    def tap_both(
        self, handlers: TapHandlers[Never, E_co] | Handlers[object]
    ) -> Err[E_co]:
        """Run only the ``err`` side effect and return this result."""
        handler = cast("Callable[[E_co], object]", handlers["err"])

        def callback() -> Err[E_co]:
            handler(self.error)
            return self

        return _try_or_panic(callback, "tap_both err callback threw")

    @overload
    async def tap_both_async(
        self,
        handlers: AsyncTapHandlers[Never, E_co],
    ) -> Err[E_co]: ...

    @overload
    async def tap_both_async(
        self,
        handlers: Handlers[Awaitable[object]],
    ) -> Err[E_co]: ...

    async def tap_both_async(
        self,
        handlers: AsyncTapHandlers[Never, E_co] | Handlers[Awaitable[object]],
    ) -> Err[E_co]:
        """Async version of :meth:`tap_both`."""
        handler = cast("Callable[[E_co], Awaitable[object]]", handlers["err"])

        async def callback() -> Err[E_co]:
            await handler(self.error)
            return self

        return await _try_or_panic_async(callback, "tap_both_async err callback threw")

    def __iter__(self) -> Generator[Err[E_co], None, Never]:
        """Yield this Err once, then panic if iteration continues."""
        yield self
        panic(
            "Unreachable: Err yielded in Result.gen but generator continued",
            self.error,
        )


type Result[A, E] = Ok[A] | Err[E]
type AnyResult = Ok[object] | Err[object]


def _require_result(value: object, message: str) -> Result[object, object]:
    """Fail fast when a Result callback violates its return contract."""
    if not isinstance(value, (Ok, Err)):
        raise Panic(message, value)
    return cast("Result[object, object]", value)


@overload
def ok() -> Ok[None]: ...


@overload
def ok[A](value: A) -> Ok[A]: ...


def ok[A](value: A | None = None) -> Ok[A | None]:
    """Construct an Ok result; ``ok()`` stores ``None``."""
    return Ok(value)


def err[E](error: E) -> Err[E]:
    """Construct an Err result."""
    return Err(error)


def is_ok[A, E](result: Result[A, E]) -> TypeGuard[Ok[A]]:
    """Return whether a result is Ok."""
    return isinstance(result, Ok)


def is_err[A, E](result: Result[A, E]) -> TypeGuard[Err[E]]:
    """Return whether a result is Err."""
    return isinstance(result, Err)


def is_error[A, E](result: Result[A, E]) -> TypeGuard[Err[E]]:
    """Alias for :func:`is_err` matching the TypeScript status vocabulary."""
    return is_err(result)


def assert_ok(result: object, expected_value: object) -> None:
    """
    Assert that a result is Ok and contains the expected value.

    This small helper is useful in tests and mirrors the helper used by the
    TypeScript port's test suite.
    """
    assert isinstance(result, Ok), f"Expected Ok, got {type(result).__name__}"
    assert result.value == expected_value


def assert_err(result: object, expected_error: object) -> None:
    """Assert that a result is Err and contains the expected error."""
    assert isinstance(result, Err), f"Expected Err, got {type(result).__name__}"
    assert result.error == expected_error


def assert_panic_raised(
    fn: Callable[[], object],
    message_contains: str | None = None,
) -> Panic:
    """Call *fn*, assert it raises Panic, and return the Panic."""
    try:
        fn()
    except Panic as panic_error:
        if message_contains is not None and message_contains not in panic_error.message:
            assertion_message = (
                f"Expected {message_contains!r} in {panic_error.message!r}"
            )
            raise AssertionError(assertion_message) from None
        return panic_error
    raise AssertionError("Expected Panic to be raised")
