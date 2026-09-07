"""Helpers for functions supporting data-first and data-last calls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, overload

if TYPE_CHECKING:
    from collections.abc import Callable

A_contra = TypeVar("A_contra", contravariant=True)
B_contra = TypeVar("B_contra", contravariant=True)
C_contra = TypeVar("C_contra", contravariant=True)
D_contra = TypeVar("D_contra", contravariant=True)
R_co = TypeVar("R_co", covariant=True)


class Dual1(Protocol[A_contra, R_co]):
    """A one-argument function with a data-last form."""

    @overload
    def __call__(self, value: A_contra) -> R_co: ...

    @overload
    def __call__(self) -> Callable[[A_contra], R_co]: ...


class Dual2(Protocol[A_contra, B_contra, R_co]):
    """A two-argument function with a data-last form."""

    @overload
    def __call__(self, first: A_contra, second: B_contra) -> R_co: ...

    @overload
    def __call__(self, second: B_contra) -> Callable[[A_contra], R_co]: ...


class Dual3(Protocol[A_contra, B_contra, C_contra, R_co]):
    """A three-argument function with a data-last form."""

    @overload
    def __call__(self, first: A_contra, second: B_contra, third: C_contra) -> R_co: ...

    @overload
    def __call__(
        self, second: B_contra, third: C_contra
    ) -> Callable[[A_contra], R_co]: ...


class Dual4(Protocol[A_contra, B_contra, C_contra, D_contra, R_co]):
    """A four-argument function with a data-last form."""

    @overload
    def __call__(
        self, first: A_contra, second: B_contra, third: C_contra, fourth: D_contra
    ) -> R_co: ...

    @overload
    def __call__(
        self, second: B_contra, third: C_contra, fourth: D_contra
    ) -> Callable[[A_contra], R_co]: ...

    @overload
    def __call__(
        self, third: C_contra, fourth: D_contra
    ) -> Callable[[A_contra, B_contra], R_co]: ...

    @overload
    def __call__(
        self, fourth: D_contra
    ) -> Callable[[A_contra, B_contra, C_contra], R_co]: ...


@overload
def dual[A, R](arity: Literal[1], body: Callable[[A], R]) -> Dual1[A, R]: ...


@overload
def dual[A, B, R](arity: Literal[2], body: Callable[[A, B], R]) -> Dual2[A, B, R]: ...


@overload
def dual[A, B, C, R](
    arity: Literal[3], body: Callable[[A, B, C], R]
) -> Dual3[A, B, C, R]: ...


@overload
def dual[A, B, C, D, R](
    arity: Literal[4], body: Callable[[A, B, C, D], R]
) -> Dual4[A, B, C, D, R]: ...


@overload
def dual[R](arity: int, body: Callable[..., R]) -> Callable[..., object]: ...


def dual[R](arity: int, body: Callable[..., R]) -> Callable[..., object]:
    """
    Create a function supporting both data-first and data-last forms.

    For ``arity=2`` and ``body(a, b)``, the returned function accepts either
    ``function(a, b)`` or ``function(b)(a)``. The same convention works for
    any arity greater than one. Arity-specific overloads preserve the body
    parameter and return types for the common forms; the fallback overload
    remains available for a runtime arity that is not known statically.
    """
    if arity < 1:
        message = f"arity must be positive, got {arity}"
        raise ValueError(message)

    def wrapper(*args: object) -> object:
        if len(args) >= arity:
            return body(*args[:arity])

        def data_last(first: object) -> R:
            return body(first, *args)

        return data_last

    return wrapper


__all__ = ["Dual1", "Dual2", "Dual3", "Dual4", "dual"]
