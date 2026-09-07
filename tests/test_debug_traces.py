"""Tests for useful diagnostic traces at Result failure boundaries."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from better_result.core import Ok, PanicError, panic
from better_result.error import TaggedError

GOLDEN_DIR = Path(__file__).with_name("golden") / "diagnostics"


def _golden_json(filename: str) -> object:
    return json.loads((GOLDEN_DIR / filename).read_text(encoding="utf-8"))


def _normalize_stack(value: str) -> str:
    value = re.sub(r'File ".*[/\\]([^"/\\]+)"', r'File "<path>/\1"', value)
    value = re.sub(r", line \d+", ", line <line>", value)

    relevant_files = {"core.py", "error.py", "test_debug_traces.py"}
    filtered: list[str] = []
    keep_frame = False
    for line in value.splitlines():
        if line.lstrip().startswith("File "):
            keep_frame = any(
                f"<path>/{filename}" in line for filename in relevant_files
            )
            if keep_frame:
                filtered.append(line)
        elif line.startswith(("    ", "      ")):
            if keep_frame:
                filtered.append(line)
        else:
            filtered.append(line)
    return "\n".join(filtered) + "\n"


def _normalize_diagnostic(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: (
                _normalize_stack(item)
                if key == "stack" and isinstance(item, str)
                else _normalize_diagnostic(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_diagnostic(item) for item in value]
    return value


def _assert_matches_golden(filename: str, payload: object) -> None:
    assert _normalize_diagnostic(payload) == _golden_json(filename)


def _raise_dependency_error() -> None:
    msg = "dependency unavailable"
    raise LookupError(msg)


def test_panic_trace_keeps_the_call_site_and_callback_cause() -> None:
    def user_pipeline() -> object:
        return Ok("input").map(_raise_callback_error)

    def _raise_callback_error(_value: str) -> str:
        msg = "invalid callback state"
        raise ValueError(msg)

    with pytest.raises(PanicError, match="map callback threw") as raised:
        user_pipeline()

    error = raised.value
    assert isinstance(error.cause, ValueError)
    assert error.__cause__ is error.cause
    assert "user_pipeline" in error.stack

    payload = error.to_json()
    _assert_matches_golden("panic-callback.json", payload)
    assert payload["message"] == "map callback threw"
    cause = payload["cause"]
    assert isinstance(cause, dict)
    assert cause["name"] == "ValueError"
    assert cause["message"] == "invalid callback state"
    assert isinstance(cause["stack"], str)
    assert "_raise_callback_error" in cause["stack"]
    json.dumps(payload)


def test_tagged_error_trace_includes_a_nested_cause_trace() -> None:
    def build_error() -> TaggedError:
        try:
            _raise_dependency_error()
        except LookupError as cause:
            return TaggedError(message="request failed", cause=cause)
        msg = "the dependency error should have been raised"
        raise AssertionError(msg)

    error = build_error()

    assert error.__cause__ is error.cause
    assert "build_error" in error.stack
    assert "Caused by:" in error.stack

    payload = error.to_json()
    _assert_matches_golden("tagged-cause.json", payload)
    cause = payload["cause"]
    assert isinstance(cause, dict)
    assert cause["name"] == "LookupError"
    assert "_raise_dependency_error" in cause["stack"]
    json.dumps(payload)


def test_safe_debug_payload_removes_all_stack_traces() -> None:
    def user_boundary() -> None:
        panic("invalid state", cause=ValueError("root cause"))

    with pytest.raises(PanicError) as raised:
        user_boundary()

    payload = raised.value.to_safe_json()
    _assert_matches_golden("safe-panic.json", payload)
    assert "stack" not in payload
    cause = payload["cause"]
    assert isinstance(cause, dict)
    assert "stack" not in cause
    assert cause["message"] == "root cause"
    json.dumps(payload)
