"""Core success and failure variants for the fresh Result API."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Never,
    NoReturn,
    TypeVar,
    cast,
    override,
)

from typing_extensions import TypeIs

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

U = TypeVar("U")
F = TypeVar("F")
TBE = TypeVar("TBE", bound=BaseException)


class Ok[T]:
    __match_args__ = ("ok_value",)
    __slots__ = ("_value",)
    __hash__ = None

    _value: T

    def __init__(self, value: T) -> None:
        object.__setattr__(self, "_value", value)

    @override
    def __setattr__(self, _name: str, _value: object) -> NoReturn:
        message = "Ok is immutable"
        raise AttributeError(message)

    @override
    def __repr__(self) -> str:
        return f"Ok({self._value!r})"

    @override
    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other) or not isinstance(other, Ok):
            return False
        return bool(self._value == other._value)

    def is_ok(self) -> Literal[True]:
        return True

    def is_err(self) -> Literal[False]:
        return False

    def ok(self) -> T:
        return self._value

    def err(self) -> None:
        return None

    @property
    def ok_value(self) -> T:
        return self._value

    def expect(self, _message: str) -> T:
        return self._value

    def expect_err(self, message: str) -> NoReturn:
        raise UnwrapError(self, message)

    def unwrap(self) -> T:
        return self._value

    def unwrap_err(self) -> NoReturn:
        raise UnwrapError(self, "Called `Result.unwrap_err()` on an `Ok` value")

    def unwrap_or(self, _default: U) -> T:
        return self._value

    def unwrap_or_else(self, _op: Callable[[Never], U]) -> T:
        return self._value

    def unwrap_or_raise(self, _error: object) -> T:
        return self._value

    def map(self, op: Callable[[T], U]) -> Ok[U]:
        return Ok(op(self._value))

    async def map_async(self, op: Callable[[T], Awaitable[U]]) -> Ok[U]:
        return Ok(await op(self._value))

    def map_or(self, _default: U, op: Callable[[T], U]) -> U:
        return op(self._value)

    def map_or_else(self, _default_op: Callable[[], U], op: Callable[[T], U]) -> U:
        return op(self._value)

    def map_err(self, _op: Callable[[Never], F]) -> Ok[T]:
        return self

    def and_then[U, E](self, op: Callable[[T], Result[U, E]]) -> Result[U, E]:
        result = op(self._value)
        return cast("Result[U, E]", _require_result(result))

    async def and_then_async[U, E](
        self, op: Callable[[T], Awaitable[Result[U, E]]]
    ) -> Result[U, E]:
        result = await op(self._value)
        return cast("Result[U, E]", _require_result(result))

    def or_else[U, F](self, _op: Callable[[Never], Result[U, F]]) -> Ok[T]:
        return self

    def inspect(self, op: Callable[[T], Any]) -> Ok[T]:
        op(self._value)
        return self

    def inspect_err(self, _op: Callable[[Never], Any]) -> Ok[T]:
        return self


class Err[E]:
    __match_args__ = ("err_value",)
    __slots__ = ("_value",)
    __hash__ = None

    _value: E

    def __init__(self, value: E) -> None:
        object.__setattr__(self, "_value", value)

    @override
    def __setattr__(self, _name: str, _value: object) -> NoReturn:
        message = "Err is immutable"
        raise AttributeError(message)

    @override
    def __repr__(self) -> str:
        return f"Err({self._value!r})"

    @override
    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other) or not isinstance(other, Err):
            return False
        return bool(self._value == other._value)

    def is_ok(self) -> Literal[False]:
        return False

    def is_err(self) -> Literal[True]:
        return True

    def ok(self) -> None:
        return None

    def err(self) -> E:
        return self._value

    @property
    def err_value(self) -> E:
        return self._value

    def expect(self, message: str) -> NoReturn:
        exc = UnwrapError(self, f"{message}: {self._value!r}")
        if isinstance(self._value, BaseException):
            raise exc from self._value
        raise exc

    def expect_err(self, _message: str) -> E:
        return self._value

    def unwrap(self) -> NoReturn:
        exc = UnwrapError(
            self,
            f"Called `Result.unwrap()` on an `Err` value: {self._value!r}",
        )
        if isinstance(self._value, BaseException):
            raise exc from self._value
        raise exc

    def unwrap_err(self) -> E:
        return self._value

    def unwrap_or(self, default: U) -> U:
        return default

    def unwrap_or_else[T](self, op: Callable[[E], T]) -> T:
        return op(self._value)

    def unwrap_or_raise(self, error: type[TBE]) -> NoReturn:
        raise error(self._value)

    def map(self, _op: Callable[[Never], U]) -> Err[E]:
        return self

    async def map_async(self, _op: Callable[[Never], Awaitable[U]]) -> Err[E]:
        return self

    def map_or(self, default: U, _op: Callable[[Never], U]) -> U:
        return default

    def map_or_else(self, default_op: Callable[[], U], _op: Callable[[Never], U]) -> U:
        return default_op()

    def map_err(self, op: Callable[[E], F]) -> Err[F]:
        return Err(op(self._value))

    def and_then[U, F](self, _op: Callable[[Never], Result[U, F]]) -> Err[E]:
        return self

    async def and_then_async[U, F](
        self, _op: Callable[[Never], Awaitable[Result[U, F]]]
    ) -> Err[E]:
        return self

    def or_else[T, F](self, op: Callable[[E], Result[T, F]]) -> Result[T, F]:
        return op(self._value)

    def inspect(self, _op: Callable[[Never], Any]) -> Err[E]:
        return self

    def inspect_err(self, op: Callable[[E], Any]) -> Err[E]:
        op(self._value)
        return self


type Result[T, E] = Ok[T] | Err[E]


OkErr: Final = (Ok, Err)


class UnwrapError(Exception):
    """Raised when an unwrap or expect operation selects the wrong variant."""

    _result: Result[Any, Any]

    def __init__(self, result: Result[Any, Any], message: str) -> None:
        self._result = result
        super().__init__(message)

    @property
    def result(self) -> Result[Any, Any]:
        return self._result


def _require_result(value: object) -> Result[object, object]:
    if not isinstance(value, OkErr):
        message = "and_then callback must return a Result"
        raise TypeError(message)
    return value


def is_ok[T, E](result: Result[T, E]) -> TypeIs[Ok[T]]:
    return result.is_ok()


def is_err[T, E](result: Result[T, E]) -> TypeIs[Err[E]]:
    return result.is_err()
