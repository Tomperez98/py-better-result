"""Internal tagged-error implementation."""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, ClassVar, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

from better_result._core import (
    _json_safe,
    _without_diagnostic_stack,
)


class TaggedError(Exception):
    """
    Base class for application-defined, string-tagged errors.

    Applications define concrete subclasses and pass typed public properties to
    this constructor. Dynamic tagged classes and string-handler dispatch are
    deliberately not part of the supported API.
    """

    _tag: ClassVar[str] = "TaggedError"
    name: str

    _reserved_properties = frozenset(
        {
            "_tag",
            "_properties",
            "args",
            "cause",
            "message",
            "name",
            "stack",
            "__cause__",
            "__context__",
            "__dict__",
            "__init__",
            "__notes__",
            "__repr__",
            "__str__",
            "__suppress_context__",
            "__traceback__",
            "add_note",
            "is_",
            "match",
            "match_partial",
            "to_dict",
            "to_json",
            "to_safe_dict",
            "to_safe_json",
            "with_traceback",
        },
    )

    def __init_subclass__(cls, *, tag: str, **kwargs: object) -> None:
        """Require every concrete subclass to declare a valid tag."""
        if not isinstance(tag, str):
            message = "TaggedError tag must be a string"
            raise TypeError(message)
        if not tag.strip():
            message = "TaggedError tag must not be empty"
            raise ValueError(message)
        if "_tag" in cls.__dict__:
            message = "TaggedError subclasses must not define _tag; use tag=..."
            raise TypeError(message)
        super().__init_subclass__(**kwargs)
        cls._tag = tag

    def __init__(
        self,
        *,
        message: str | None = None,
        cause: object | None = None,
        **properties: object,
    ) -> None:
        if message is not None and not isinstance(message, str):
            error_message = "TaggedError message must be a string or None"
            raise TypeError(error_message)
        collisions = self._reserved_properties.intersection(properties)
        if collisions:
            names = ", ".join(sorted(collisions))
            error_message = f"TaggedError properties are reserved: {names}"
            raise TypeError(error_message)

        values = dict(properties)
        if message is not None:
            values["message"] = message
        if cause is not None:
            values["cause"] = cause

        message_text = message if message is not None else ""
        super().__init__(message_text)
        self.message = message_text
        self.cause = cause
        self.name = self._tag
        self._properties = values

        for key, value in properties.items():
            setattr(self, key, value)

        if isinstance(cause, BaseException):
            self.__cause__ = cause

        self.stack = "".join(traceback.format_stack()[:-1])
        if isinstance(cause, BaseException):
            cause_stack = "".join(
                traceback.format_exception(type(cause), cause, cause.__traceback__),
            ).replace("\n", "\n  ")
            self.stack = f"{self.stack}\nCaused by: {cause_stack}"

    def to_dict(self) -> dict[str, object | None]:
        """Return the error properties plus reserved metadata."""
        return {
            **self._properties,
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
        """Return metadata without diagnostic stack traces."""
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


class UnhandledError(TaggedError, tag="UnhandledException"):
    """Wrap an exception or other value caught by an internal operation."""

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
