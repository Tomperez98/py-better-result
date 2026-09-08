"""Core success and failure variants for the fresh Result API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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


class Result[T, E](ABC):
    """
    Abstract common interface for successful and failed Results.

    Keeping the operations on a generic common base gives type checkers one
    ``T`` and ``E`` context when a callback is passed to a Result workflow.
    ``Ok`` and ``Err`` remain the concrete runtime variants.
    """

    @abstractmethod
    def is_ok(self) -> bool: ...

    @abstractmethod
    def is_err(self) -> bool: ...

    @abstractmethod
    def ok(self) -> T | None: ...

    @abstractmethod
    def err(self) -> E | None: ...

    @abstractmethod
    def expect(self, message: str) -> T: ...

    @abstractmethod
    def expect_err(self, message: str) -> E: ...

    @abstractmethod
    def unwrap(self) -> T: ...

    @abstractmethod
    def unwrap_err(self) -> E: ...

    @abstractmethod
    def unwrap_or(self, default: U) -> T | U: ...

    @abstractmethod
    def unwrap_or_else(self, op: Callable[[E], U]) -> T | U: ...

    @abstractmethod
    def unwrap_or_raise(self, error: type[TBE]) -> T: ...

    @abstractmethod
    def map(self, op: Callable[[T], U]) -> Result[U, E]: ...

    @abstractmethod
    async def map_async(self, op: Callable[[T], Awaitable[U]]) -> Result[U, E]: ...

    @abstractmethod
    def map_or(self, default: U, op: Callable[[T], U]) -> U: ...

    @abstractmethod
    def map_or_else(self, default_op: Callable[[], U], op: Callable[[T], U]) -> U: ...

    @abstractmethod
    def map_err(self, op: Callable[[E], F]) -> Result[T, F]: ...

    @abstractmethod
    def and_then[U, F](self, op: Callable[[T], Result[U, F]]) -> Result[U, E | F]: ...

    @abstractmethod
    async def and_then_async[U, F](
        self, op: Callable[[T], Awaitable[Result[U, F]]]
    ) -> Result[U, E | F]: ...

    @abstractmethod
    def or_else[U, F](self, op: Callable[[E], Result[U, F]]) -> Result[T | U, F]: ...

    @abstractmethod
    def inspect(self, op: Callable[[T], Any]) -> Result[T, E]: ...

    @abstractmethod
    def inspect_err(self, op: Callable[[E], Any]) -> Result[T, E]: ...


@dataclass(frozen=True, slots=True)
class Ok[T](Result[T, Never]):
    __match_args__ = ("ok_value",)
    __hash__ = None

    value: T

    @override
    def is_ok(self) -> Literal[True]:
        return True

    @override
    def is_err(self) -> Literal[False]:
        return False

    @override
    def ok(self) -> T:
        return self.value

    @override
    def err(self) -> None:
        return None

    @property
    def ok_value(self) -> T:
        return self.value

    @override
    def expect(self, message: str) -> T:
        return self.value

    @override
    def expect_err(self, message: str) -> NoReturn:
        raise UnwrapError(self, message)

    @override
    def unwrap(self) -> T:
        return self.value

    @override
    def unwrap_err(self) -> NoReturn:
        raise UnwrapError(self, "Called `Result.unwrap_err()` on an `Ok` value")

    @override
    def unwrap_or(self, default: U) -> T:
        return self.value

    @override
    def unwrap_or_else(self, op: Callable[[Never], U]) -> T:
        return self.value

    @override
    def unwrap_or_raise(self, error: type[TBE]) -> T:
        return self.value

    @override
    def map(self, op: Callable[[T], U]) -> Ok[U]:
        return Ok(op(self.value))

    @override
    async def map_async(self, op: Callable[[T], Awaitable[U]]) -> Ok[U]:
        return Ok(await op(self.value))

    @override
    def map_or(self, default: U, op: Callable[[T], U]) -> U:
        return op(self.value)

    @override
    def map_or_else(self, default_op: Callable[[], U], op: Callable[[T], U]) -> U:
        return op(self.value)

    @override
    def map_err(self, op: Callable[[Never], F]) -> Ok[T]:
        return self

    @override
    def and_then[U, E](self, op: Callable[[T], Result[U, E]]) -> Result[U, E]:
        result = op(self.value)
        return cast("Result[U, E]", _require_result(result))

    @override
    async def and_then_async[U, E](
        self, op: Callable[[T], Awaitable[Result[U, E]]]
    ) -> Result[U, E]:
        result = await op(self.value)
        return cast("Result[U, E]", _require_result(result))

    @override
    def or_else[U, F](self, op: Callable[[Never], Result[U, F]]) -> Ok[T]:
        return self

    @override
    def inspect(self, op: Callable[[T], Any]) -> Ok[T]:
        op(self.value)
        return self

    @override
    def inspect_err(self, op: Callable[[Never], Any]) -> Ok[T]:
        return self


@dataclass(frozen=True, slots=True)
class Err[E](Result[Never, E]):
    __match_args__ = ("err_value",)
    __hash__ = None

    value: E

    @override
    def is_ok(self) -> Literal[False]:
        return False

    @override
    def is_err(self) -> Literal[True]:
        return True

    @override
    def ok(self) -> None:
        return None

    @override
    def err(self) -> E:
        return self.value

    @property
    def err_value(self) -> E:
        return self.value

    @override
    def expect(self, message: str) -> NoReturn:
        exc = UnwrapError(self, f"{message}: {self.value!r}")
        if isinstance(self.value, BaseException):
            raise exc from self.value
        raise exc

    @override
    def expect_err(self, message: str) -> E:
        return self.value

    @override
    def unwrap(self) -> NoReturn:
        exc = UnwrapError(
            self,
            f"Called `Result.unwrap()` on an `Err` value: {self.value!r}",
        )
        if isinstance(self.value, BaseException):
            raise exc from self.value
        raise exc

    @override
    def unwrap_err(self) -> E:
        return self.value

    @override
    def unwrap_or(self, default: U) -> U:
        return default

    @override
    def unwrap_or_else[T](self, op: Callable[[E], T]) -> T:
        return op(self.value)

    @override
    def unwrap_or_raise(self, error: type[TBE]) -> NoReturn:
        raise error(self.value)

    @override
    def map(self, op: Callable[[Never], U]) -> Err[E]:
        return self

    @override
    async def map_async(self, op: Callable[[Never], Awaitable[U]]) -> Err[E]:
        return self

    @override
    def map_or(self, default: U, op: Callable[[Never], U]) -> U:
        return default

    @override
    def map_or_else(self, default_op: Callable[[], U], op: Callable[[Never], U]) -> U:
        return default_op()

    @override
    def map_err(self, op: Callable[[E], F]) -> Err[F]:
        return Err(op(self.value))

    @override
    def and_then[U, F](self, op: Callable[[Never], Result[U, F]]) -> Err[E]:
        return self

    @override
    async def and_then_async[U, F](
        self, op: Callable[[Never], Awaitable[Result[U, F]]]
    ) -> Err[E]:
        return self

    @override
    def or_else[T, F](self, op: Callable[[E], Result[T, F]]) -> Result[T, F]:
        return op(self.value)

    @override
    def inspect(self, op: Callable[[Never], Any]) -> Err[E]:
        return self

    @override
    def inspect_err(self, op: Callable[[E], Any]) -> Err[E]:
        op(self.value)
        return self


OkErr: Final = (Ok, Err)


class UnwrapError(BaseException):
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
