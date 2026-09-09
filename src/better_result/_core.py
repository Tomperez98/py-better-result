"""A small Python implementation of Rust's ``std::result::Result``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Never, TypeVar, override

if TYPE_CHECKING:
    from collections.abc import Callable

T_co = TypeVar("T_co", covariant=True)
E_co = TypeVar("E_co", covariant=True)


class Result[T_co, E_co](ABC):
    """Common interface for successful and failed results."""

    @abstractmethod
    def is_ok(self) -> bool: ...

    @abstractmethod
    def is_err(self) -> bool: ...

    @abstractmethod
    def is_ok_and(self, predicate: Callable[[T_co], bool]) -> bool: ...

    @abstractmethod
    def is_err_and(self, predicate: Callable[[E_co], bool]) -> bool: ...

    @abstractmethod
    def ok(self) -> T_co | None: ...

    @abstractmethod
    def err(self) -> E_co | None: ...

    @abstractmethod
    def map[U](self, operation: Callable[[T_co], U]) -> Result[U, E_co]: ...

    @abstractmethod
    def map_or[U](self, default: U, operation: Callable[[T_co], U]) -> U: ...

    @abstractmethod
    def map_or_else[U](
        self,
        default_operation: Callable[[], U],
        operation: Callable[[T_co], U],
    ) -> U: ...

    @abstractmethod
    def map_err[F](self, operation: Callable[[E_co], F]) -> Result[T_co, F]: ...

    @abstractmethod
    def and_[U, F](self, result: Result[U, F]) -> Result[U, E_co | F]: ...

    @abstractmethod
    def and_then[U, F](
        self, operation: Callable[[T_co], Result[U, F]]
    ) -> Result[U, E_co | F]: ...

    @abstractmethod
    def or_else[U, F](
        self, operation: Callable[[E_co], Result[U, F]]
    ) -> Result[T_co | U, F]: ...

    @abstractmethod
    def inspect(self, operation: Callable[[T_co], object]) -> Result[T_co, E_co]: ...

    @abstractmethod
    def inspect_err(
        self, operation: Callable[[E_co], object]
    ) -> Result[T_co, E_co]: ...

    @abstractmethod
    def unwrap(self) -> T_co: ...

    @abstractmethod
    def expect(self, message: str) -> T_co: ...

    @abstractmethod
    def unwrap_err(self) -> E_co: ...

    @abstractmethod
    def expect_err(self, message: str) -> E_co: ...

    @abstractmethod
    def unwrap_or[U](self, default: U) -> T_co | U: ...

    @abstractmethod
    def unwrap_or_else[U](self, operation: Callable[[E_co], U]) -> T_co | U: ...

    @abstractmethod
    def unwrap_or_default(self) -> T_co | None: ...


@dataclass(frozen=True, slots=True)
class Ok[T](Result[T, Never]):
    """Successful result variant."""

    __match_args__ = ("value",)
    __hash__ = None

    value: T

    @override
    def is_ok(self) -> bool:
        return True

    @override
    def is_err(self) -> bool:
        return False

    @override
    def is_ok_and(self, predicate: Callable[[T], bool]) -> bool:
        return predicate(self.value)

    @override
    def is_err_and(self, predicate: Callable[[Never], bool]) -> bool:
        del predicate
        return False

    @override
    def ok(self) -> T:
        return self.value

    @override
    def err(self) -> None:
        return None

    @override
    def map[U](self, operation: Callable[[T], U]) -> Ok[U]:
        return Ok(operation(self.value))

    @override
    def map_or[U](self, default: U, operation: Callable[[T], U]) -> U:
        del default
        return operation(self.value)

    @override
    def map_or_else[U](
        self,
        default_operation: Callable[[], U],
        operation: Callable[[T], U],
    ) -> U:
        del default_operation
        return operation(self.value)

    @override
    def map_err[F](self, operation: Callable[[Never], F]) -> Ok[T]:
        del operation
        return self

    @override
    def and_[U, F](self, result: Result[U, F]) -> Result[U, F]:
        return _require_result(result)

    @override
    def and_then[U, F](self, operation: Callable[[T], Result[U, F]]) -> Result[U, F]:
        return _require_result(operation(self.value))

    def or_[F](self, result: Result[T, F]) -> Ok[T]:
        _require_result(result)
        return self

    @override
    def or_else[U, F](self, operation: Callable[[Never], Result[U, F]]) -> Ok[T]:
        del operation
        return self

    @override
    def inspect(self, operation: Callable[[T], object]) -> Ok[T]:
        operation(self.value)
        return self

    @override
    def inspect_err(self, operation: Callable[[Never], object]) -> Ok[T]:
        del operation
        return self

    @override
    def unwrap(self) -> T:
        return self.value

    @override
    def expect(self, message: str) -> T:
        del message
        return self.value

    @override
    def unwrap_err(self) -> Never:
        return os.abort()

    @override
    def expect_err(self, message: str) -> Never:
        return os.abort()

    @override
    def unwrap_or[U](self, default: U) -> T:
        del default
        return self.value

    @override
    def unwrap_or_else[U](self, operation: Callable[[Never], U]) -> T:
        del operation
        return self.value

    @override
    def unwrap_or_default(self) -> T:
        return self.value


@dataclass(frozen=True, slots=True)
class Err[E](Result[Never, E]):
    """Failed result variant."""

    __match_args__ = ("value",)
    __hash__ = None

    value: E

    @override
    def is_ok(self) -> bool:
        return False

    @override
    def is_err(self) -> bool:
        return True

    @override
    def is_ok_and(self, predicate: Callable[[Never], bool]) -> bool:
        del predicate
        return False

    @override
    def is_err_and(self, predicate: Callable[[E], bool]) -> bool:
        return predicate(self.value)

    @override
    def ok(self) -> None:
        return None

    @override
    def err(self) -> E:
        return self.value

    @override
    def map[U](self, operation: Callable[[Never], U]) -> Err[E]:
        del operation
        return self

    @override
    def map_or[U](self, default: U, operation: Callable[[Never], U]) -> U:
        del operation
        return default

    @override
    def map_or_else[U](
        self,
        default_operation: Callable[[], U],
        operation: Callable[[Never], U],
    ) -> U:
        del operation
        return default_operation()

    @override
    def map_err[F](self, operation: Callable[[E], F]) -> Err[F]:
        return Err(operation(self.value))

    @override
    def and_[U, F](self, result: Result[U, F]) -> Err[E]:
        _require_result(result)
        return self

    @override
    def and_then[U, F](self, operation: Callable[[Never], Result[U, F]]) -> Err[E]:
        del operation
        return self

    def or_[U, F](self, result: Result[U, F]) -> Result[U, F]:
        return _require_result(result)

    @override
    def or_else[U, F](self, operation: Callable[[E], Result[U, F]]) -> Result[U, F]:
        return _require_result(operation(self.value))

    @override
    def inspect(self, operation: Callable[[Never], object]) -> Err[E]:
        del operation
        return self

    @override
    def inspect_err(self, operation: Callable[[E], object]) -> Err[E]:
        operation(self.value)
        return self

    @override
    def unwrap(self) -> Never:
        return os.abort()

    @override
    def expect(self, message: str) -> Never:
        return os.abort()

    @override
    def unwrap_err(self) -> E:
        return self.value

    @override
    def expect_err(self, message: str) -> E:
        del message
        return self.value

    @override
    def unwrap_or[U](self, default: U) -> U:
        return default

    @override
    def unwrap_or_else[U](self, operation: Callable[[E], U]) -> U:
        return operation(self.value)

    @override
    def unwrap_or_default(self) -> None:
        return None


def _require_result[T, E](value: Result[T, E]) -> Result[T, E]:
    if not isinstance(value, (Ok, Err)):
        message = "operation must return a Result"
        raise TypeError(message)
    return value
