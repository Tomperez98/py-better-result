"""
Tagged errors and exhaustive error matching.

This module translates the tagged-error portion of better-result into Python's
idioms: subclass :class:`TaggedError` for fixed error types, or use
:func:`tagged_error` when a dynamic tagged class is useful.
"""

from __future__ import annotations

import traceback
import types
from collections.abc import Callable, Generator, Mapping, Sequence
from typing import Any, ClassVar, Never, TypeGuard, TypeVar, overload

from .core import Err, err, panic

TaggedErrorLike = TypeVar("TaggedErrorLike", bound="TaggedError")
HandlerResult = TypeVar("HandlerResult")


class TaggedError(Exception):
    """
    An error carrying a string discriminator and serializable properties.

    Subclasses must declare ``tag=...`` in the class header and pass their
    public properties to ``super().__init__``. For example::

        class NotFoundError(TaggedError, tag="NotFoundError"):
            def __init__(self, item_id: str) -> None:
                super().__init__(message=f"Not found: {item_id}", item_id=item_id)
    """

    _tag: ClassVar[str] = "TaggedError"
    name: str

    def __init_subclass__(cls, *, tag: str, **kwargs: object) -> None:
        """Require subclasses to declare their discriminator in the class header."""
        super().__init_subclass__(**kwargs)
        declared_tag = cls.__dict__.get("_tag")
        if not tag:
            raise ValueError("TaggedError tag must not be empty")
        if declared_tag is not None:
            raise TypeError("TaggedError subclasses must not define _tag; use tag=...")
        cls._tag = tag

    def __init__(
        self,
        properties: Mapping[str, object] | None = None,
        *,
        message: str | None = None,
        cause: object | None = None,
        **extra: object,
    ) -> None:
        values = dict(properties or {})
        values.update(extra)
        if "match" in values:
            raise TypeError("'match' is reserved by TaggedError")
        if "message" not in values and message is not None:
            values["message"] = message
        if "cause" not in values and cause is not None:
            values["cause"] = cause

        message_value = values.get("message")
        message_text = message_value if isinstance(message_value, str) else ""
        cause_value = values.get("cause")

        super().__init__(message_text)
        self.message = message_text
        self.cause = cause_value
        self.name = self._tag
        self._properties = values

        for key, value in values.items():
            if key not in {"_tag", "name", "stack"}:
                setattr(self, key, value)

        if isinstance(cause_value, BaseException):
            self.__cause__ = cause_value

        self.stack = "".join(traceback.format_stack()[:-1])
        if isinstance(cause_value, BaseException):
            cause_stack = "".join(
                traceback.format_exception(
                    type(cause_value),
                    cause_value,
                    cause_value.__traceback__,
                ),
            ).replace("\n", "\n  ")
            self.stack = f"{self.stack}\nCaused by: {cause_stack}"

    @classmethod
    def is_(cls, value: object) -> TypeGuard[TaggedErrorLike]:
        """Return whether *value* is an instance of this tagged class."""
        return isinstance(value, cls)

    @staticmethod
    def is_tagged_error(value: object) -> TypeGuard[TaggedError]:
        """Return whether *value* is any TaggedError instance."""
        return isinstance(value, TaggedError)

    def to_dict(self) -> dict[str, object | None]:
        """Return the enumerable error properties plus reserved metadata."""
        return {
            **self._properties,
            "_tag": self._tag,
            "name": self.name,
            "message": self.message,
            "cause": _serialize_cause(self.cause),
            "stack": self.stack,
        }

    def to_json(self) -> dict[str, object | None]:
        """Return a JSON-compatible representation of this error."""
        return self.to_dict()

    def match(
        self,
        handlers: Mapping[str, Callable[[Any], HandlerResult]],
    ) -> HandlerResult:
        """Exhaustively dispatch to the handler for this error's tag."""
        return match_error(self, handlers)

    def __iter__(self) -> Generator[Err[Never, TaggedError], None, Never]:
        """Yield this error as an Err, then panic if iteration continues."""
        yield err(self)
        panic("Unreachable: Err yielded in TaggedError but generator continued", self)


def tagged_error(tag: str) -> type[TaggedError]:
    """Create a dynamic TaggedError subclass for *tag*."""
    if not tag:
        raise ValueError("tag must not be empty")
    return types.new_class(tag, (TaggedError,), {"tag": tag})


def is_tagged_error(value: object) -> TypeGuard[TaggedError]:
    """Return whether *value* has the TaggedError protocol shape."""
    return (
        isinstance(value, BaseException)
        and isinstance(getattr(value, "_tag", None), str)
        and callable(getattr(value, "to_json", None))
    )


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


def _tag_of(error: object) -> str:
    tag = getattr(error, "_tag", None)
    if not isinstance(tag, str):
        raise TypeError("error must have a string _tag")
    return tag


def _match_error[HandlerResult](
    error: object,
    handlers: Mapping[str, Callable[[Any], HandlerResult]],
) -> HandlerResult:
    try:
        handler = handlers[_tag_of(error)]
        return handler(error)
    except BaseException as cause:  # noqa: BLE001
        panic("match_error handler threw", cause)


@overload
def match_error[MatchResult](
    handlers: Mapping[str, Callable[[Any], MatchResult]],
) -> Callable[[TaggedError], MatchResult]: ...


@overload
def match_error[MatchResult](
    error: TaggedError,
    handlers: Mapping[str, Callable[[Any], MatchResult]],
) -> MatchResult: ...


def match_error(
    error_or_handlers: TaggedError | Mapping[str, Callable[[Any], object]],
    handlers: Mapping[str, Callable[[Any], object]] | None = None,
) -> object | Callable[[TaggedError], object]:
    """Dispatch to a tagged error handler in data-first or data-last form."""
    if handlers is None:
        if not isinstance(error_or_handlers, Mapping):
            raise TypeError("handlers must be a mapping")
        return lambda error: _match_error(error, error_or_handlers)
    if not isinstance(error_or_handlers, TaggedError):
        raise TypeError("error must be a TaggedError")
    return _match_error(error_or_handlers, handlers)


def _identity(error: TaggedError) -> TaggedError:
    return error


def _apply_partial[HandlerResult](
    error: TaggedError,
    handlers: Mapping[str, Callable[[Any], HandlerResult]],
    on_unhandled: Callable[[TaggedError], object],
) -> object:
    try:
        tag = _tag_of(error)
        if tag in handlers:
            return handlers[tag](error)
        return on_unhandled(error)
    except BaseException as cause:  # noqa: BLE001
        panic("match_error_partial handler threw", cause)


@overload
def match_error_partial[MatchResult](
    handlers: Mapping[str, Callable[[Any], MatchResult]],
) -> Callable[[TaggedError], MatchResult | TaggedError]: ...


@overload
def match_error_partial[MatchResult, UnhandledResult](
    handlers: Mapping[str, Callable[[Any], MatchResult]],
    on_unhandled: Callable[[TaggedError], UnhandledResult],
) -> Callable[[TaggedError], MatchResult | UnhandledResult]: ...


@overload
def match_error_partial[MatchResult](
    error: TaggedError,
    handlers: Mapping[str, Callable[[Any], MatchResult]],
) -> MatchResult | TaggedError: ...


@overload
def match_error_partial[MatchResult, UnhandledResult](
    error: TaggedError,
    handlers: Mapping[str, Callable[[Any], MatchResult]],
    on_unhandled: Callable[[TaggedError], UnhandledResult],
) -> MatchResult | UnhandledResult: ...


def match_error_partial[HandlerResult](
    error_or_handlers: TaggedError | Mapping[str, Callable[[Any], HandlerResult]],
    handlers_or_on_unhandled: Mapping[str, Callable[[Any], HandlerResult]]
    | Callable[[TaggedError], object]
    | None = None,
    on_unhandled: Callable[[TaggedError], object] | None = None,
) -> object | Callable[[TaggedError], object]:
    """
    Partially match tagged errors, preserving unhandled errors by default.

    Supports data-first calls::

        match_error_partial(error, handlers)
        match_error_partial(error, handlers, fallback)

    and data-last calls::

        match_error_partial(handlers)
        match_error_partial(handlers, fallback)
    """
    if isinstance(error_or_handlers, Mapping):
        handlers = error_or_handlers
        if handlers_or_on_unhandled is None:
            fallback: Callable[[TaggedError], object] = _identity
        elif isinstance(handlers_or_on_unhandled, Mapping):
            raise TypeError("on_unhandled must be callable")
        else:
            fallback = handlers_or_on_unhandled
        return lambda error: _apply_partial(error, handlers, fallback)

    error = error_or_handlers
    if not isinstance(handlers_or_on_unhandled, Mapping):
        raise TypeError("handlers must be a mapping")
    fallback: Callable[[TaggedError], object] = on_unhandled or _identity
    return _apply_partial(error, handlers_or_on_unhandled, fallback)


class UnhandledException(TaggedError, tag="UnhandledException"):
    """Wrap an exception or other value caught by a Result operation."""

    def __init__(self, cause: object) -> None:
        super().__init__(
            message=f"Unhandled exception: {_stringify(cause)}",
            cause=cause,
        )


type ResultCodecIssue = Mapping[str, object]


class ResultDeserializationError(TaggedError, tag="ResultDeserializationError"):
    """A value could not be decoded as a Result envelope."""

    value: object
    issues: Sequence[ResultCodecIssue] | None

    def __init__(
        self,
        value: object,
        issues: Sequence[ResultCodecIssue] | None = None,
    ) -> None:
        message = (
            "Failed to deserialize Result payload"
            if issues is not None
            else 'Failed to deserialize value as Result: expected { status: "ok", value } or { status: "error", error }'
        )
        super().__init__(message=message, value=value, issues=issues)


class ResultSerializationError(TaggedError, tag="ResultSerializationError"):
    """A Result payload could not be serialized."""

    value: object
    issues: Sequence[ResultCodecIssue] | None

    def __init__(
        self,
        value: object,
        issues: Sequence[ResultCodecIssue] | None = None,
    ) -> None:
        super().__init__(
            message="Failed to serialize Result payload",
            value=value,
            issues=issues,
        )


def _stringify(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


__all__ = [
    "ResultCodecIssue",
    "ResultDeserializationError",
    "ResultSerializationError",
    "TaggedError",
    "UnhandledException",
    "is_tagged_error",
    "match_error",
    "match_error_partial",
    "tagged_error",
]
