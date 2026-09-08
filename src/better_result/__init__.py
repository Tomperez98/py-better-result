"""The supported public API for better-result."""

from __future__ import annotations

from better_result._core import Err, Ok, PanicError, Result
from better_result._error import TaggedError

__all__ = ["Err", "Ok", "PanicError", "Result", "TaggedError"]
del annotations
