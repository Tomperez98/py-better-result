"""
Tagged errors and exhaustive error matching.

This module translates the tagged-error portion of better-result into Python's
idioms: subclass :class:`TaggedError` for fixed error types, or use
:func:`tagged_error` when a dynamic tagged class is useful.
"""

from __future__ import annotations

import traceback
import types
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Never,
    TypedDict,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence

from better_result.core import (
    Err,
    PanicError,
    _json_safe,
    _without_diagnostic_stack,
    panic,
)

TaggedErrorLike = TypeVar("TaggedErrorLike", bound="TaggedError")


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
            msg = "TaggedError tag must not be empty"
            raise ValueError(msg)
        if declared_tag is not None:
            msg = "TaggedError subclasses must not define _tag; use tag=..."
            raise TypeError(msg)
        cls._tag = tag

    def __init__(
        self,
        *,
        message: str | None = None,
        cause: object | None = None,
        **properties: object,
    ) -> None:
        values = dict(properties)
        if "match" in values:
            msg = "'match' is reserved by TaggedError"
            raise TypeError(msg)
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
        """Return a recursively JSON-compatible diagnostic representation."""
        return cast("dict[str, object | None]", _json_safe(self.to_dict()))

    def to_safe_dict(self) -> dict[str, object | None]:
        """Return error metadata without diagnostic stack traces."""
        return cast(
            "dict[str, object | None]",
            _without_diagnostic_stack(self.to_dict()),
        )

    def to_safe_json(self) -> dict[str, object | None]:
        """Return a JSON-compatible error payload safe for transport."""
        return cast("dict[str, object | None]", _json_safe(self.to_safe_dict()))

    def match[HandlerResult](
        self,
        handlers: Mapping[str, Callable[[TaggedError], HandlerResult]],
    ) -> HandlerResult:
        """Exhaustively dispatch to the handler for this error's tag."""
        try:
            handler = handlers[self._tag]
            return handler(self)
        except PanicError:
            raise
        except Exception as cause:
            panic("TaggedError.match handler threw", cause)

    @overload
    def match_partial[HandlerResult](
        self,
        handlers: Mapping[str, Callable[[TaggedError], HandlerResult]],
        on_unhandled: None = None,
    ) -> HandlerResult | TaggedError: ...

    @overload
    def match_partial[HandlerResult, UnhandledResult](
        self,
        handlers: Mapping[str, Callable[[TaggedError], HandlerResult]],
        on_unhandled: Callable[[TaggedError], UnhandledResult],
    ) -> HandlerResult | UnhandledResult: ...

    def match_partial[HandlerResult, UnhandledResult](
        self,
        handlers: Mapping[str, Callable[[TaggedError], HandlerResult]],
        on_unhandled: Callable[[TaggedError], UnhandledResult] | None = None,
    ) -> HandlerResult | UnhandledResult | TaggedError:
        """Dispatch a tagged handler, preserving or transforming unknown tags."""
        try:
            if self._tag in handlers:
                return handlers[self._tag](self)
            if on_unhandled is None:
                return self
            return on_unhandled(self)
        except PanicError:
            raise
        except Exception as cause:
            panic("TaggedError.match_partial handler threw", cause)

    def __iter__(self) -> Generator[Err[TaggedError], None, Never]:
        """Yield this error as an Err, then panic if iteration continues."""
        yield Err(self)
        panic("Unreachable: Err yielded in TaggedError but generator continued", self)


def tagged_error(tag: str) -> type[TaggedError]:
    """Create a dynamic TaggedError subclass for *tag*."""
    if not tag:
        msg = "tag must not be empty"
        raise ValueError(msg)
    return types.new_class(tag, (TaggedError,), {"tag": tag})


def is_tagged_error(value: object) -> TypeGuard[TaggedError]:
    """Return whether *value* is a TaggedError instance."""
    return isinstance(value, TaggedError)


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


class UnhandledError(TaggedError, tag="UnhandledException"):
    """Wrap an exception or other value caught by a Result operation."""

    def __init__(self, cause: object) -> None:
        super().__init__(
            message=f"Unhandled exception: {_stringify(cause)}",
            cause=cause,
        )


class ResultCodecIssue(TypedDict, total=False):
    """A structured issue returned by a Result codec schema."""

    message: str
    value: object


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
