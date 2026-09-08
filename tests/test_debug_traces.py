"""Tests for useful diagnostic traces at Result failure boundaries."""

from __future__ import annotations

import json

import pytest

from better_result import Ok, PanicError, TaggedError
from better_result._core import panic


def test_panic_trace_keeps_user_call_site_and_callback_cause() -> None:
    def user_pipeline() -> object:
        return Ok("input").map(_raise_callback_error)

    def _raise_callback_error(_value: str) -> str:
        message = "invalid callback state"
        raise ValueError(message)

    with pytest.raises(PanicError, match="map callback threw") as raised:
        user_pipeline()

    error = raised.value
    assert isinstance(error.cause, ValueError)
    assert error.__cause__ is error.cause
    assert "user_pipeline" in error.stack
    payload = error.to_json()
    json.dumps(payload)
    assert payload["message"] == "map callback threw"
    assert isinstance(payload["cause"], dict)


def test_tagged_error_trace_includes_a_nested_cause_trace() -> None:
    cause = LookupError("dependency unavailable")
    error = TaggedError(message="request failed", cause=cause)

    assert error.__cause__ is cause
    assert "Caused by:" in error.stack
    payload = error.to_json()
    json.dumps(payload)
    assert isinstance(payload["cause"], dict)
    assert "stack" in payload["cause"]


def test_safe_debug_payload_removes_all_stack_traces() -> None:
    with pytest.raises(PanicError) as raised:
        panic("invalid state", cause=ValueError("root cause"))

    payload = raised.value.to_safe_json()
    assert "stack" not in payload
    assert isinstance(payload["cause"], dict)
    assert "stack" not in payload["cause"]
    assert payload["cause"]["message"] == "root cause"
