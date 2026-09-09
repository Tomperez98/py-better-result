"""A small Python port of the Rust retry-result crate."""

from __future__ import annotations

from better_result._core import Err, Ok, Result, UnwrapError
from better_result._operations import (
    AlwaysRetry,
    ExponentialBackoff,
    FixedBackoff,
    InvalidBackoffMultiplier,
    InvalidJitterFactor,
    Jittered,
    LinearBackoff,
    RetryAfter,
    RetryContext,
    RetryDecision,
    RetryPolicy,
    RetryPredicate,
    RetrySchedule,
    StopRetry,
    retry,
    retry_async,
)

__all__ = [
    "AlwaysRetry",
    "Err",
    "ExponentialBackoff",
    "FixedBackoff",
    "InvalidBackoffMultiplier",
    "InvalidJitterFactor",
    "Jittered",
    "LinearBackoff",
    "Ok",
    "Result",
    "RetryAfter",
    "RetryContext",
    "RetryDecision",
    "RetryPolicy",
    "RetryPredicate",
    "RetrySchedule",
    "StopRetry",
    "UnwrapError",
    "retry",
    "retry_async",
]
