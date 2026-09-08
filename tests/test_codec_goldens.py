"""Golden protocol coverage for richer codec edge payloads."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from better_result import Err, Ok, TaggedError
from better_result._codec import ResultCodec, SchemaFailure, async_codec, codec
from better_result._error import ResultCodecIssue, ResultDeserializationError


class UserNotFoundError(TaggedError, tag="UserNotFound"):
    """A requested user does not exist."""

    user_id: str

    def __init__(self, user_id: str) -> None:
        super().__init__(message=f"User {user_id} was not found", user_id=user_id)


class ValidationRejectedError(TaggedError, tag="ValidationRejected"):
    """A payload contains multiple validation failures."""

    field: str
    issues: list[str]

    def __init__(self, field: str, issues: list[str]) -> None:
        super().__init__(
            message=f"Validation failed for {field}",
            field=field,
            issues=issues,
        )


class ServiceUnavailableError(TaggedError, tag="ServiceUnavailable"):
    """A dependency could not be reached."""

    service: str

    def __init__(self, service: str, cause: BaseException) -> None:
        super().__init__(
            message=f"Service {service} is unavailable",
            service=service,
            cause=cause,
        )


GOLDEN_DIR = Path(__file__).with_name("golden") / "codec"
SUPPORTED_TAGS = {"UserNotFound", "ValidationRejected", "ServiceUnavailable"}


def _golden_json(filename: str) -> object:
    return json.loads((GOLDEN_DIR / filename).read_text(encoding="utf-8"))


def _normalize_traces(value: object) -> object:
    """Keep trace presence in goldens without pinning source locations."""
    if isinstance(value, dict):
        return {
            key: "<trace>" if key == "stack" else _normalize_traces(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_traces(item) for item in value]
    return value


def _trace_count(value: object) -> int:
    if isinstance(value, dict):
        own = int("stack" in value and isinstance(value["stack"], str))
        return own + sum(_trace_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_trace_count(item) for item in value)
    return 0


def _stable_error(error: ResultDeserializationError) -> dict[str, object | None]:
    payload = error.to_json()
    stable = {
        key: payload[key]
        for key in ("_tag", "name", "message", "value", "issues")
        if key in payload
    }
    return cast("dict[str, object | None]", json.loads(json.dumps(stable)))


def encode_record(value: Mapping[str, object]) -> dict[str, object]:
    return dict(value)


def encode_tagged_error(error: TaggedError) -> dict[str, object | None]:
    return error.to_json()


def decode_record(value: object) -> dict[str, object] | SchemaFailure:
    if not isinstance(value, Mapping):
        issue: ResultCodecIssue = {"message": "expected an object"}
        return SchemaFailure((issue,))
    return dict(value)


def _decode_user_not_found(value: Mapping[str, object]) -> TaggedError | SchemaFailure:
    user_id = value.get("user_id")
    if not isinstance(user_id, str):
        issue: ResultCodecIssue = {"message": "invalid user_id"}
        return SchemaFailure((issue,))
    return UserNotFoundError(user_id)


def _decode_validation_rejected(
    value: Mapping[str, object],
) -> TaggedError | SchemaFailure:
    field = value.get("field")
    issues = value.get("issues")
    if (
        not isinstance(field, str)
        or not isinstance(issues, list)
        or not all(isinstance(issue, str) for issue in issues)
    ):
        issue: ResultCodecIssue = {"message": "invalid validation error"}
        return SchemaFailure((issue,))
    return ValidationRejectedError(field, issues)


def _decode_service_unavailable(
    value: Mapping[str, object],
) -> TaggedError | SchemaFailure:
    service = value.get("service")
    if not isinstance(service, str):
        issue: ResultCodecIssue = {"message": "invalid service"}
        return SchemaFailure((issue,))
    return ServiceUnavailableError(service, RuntimeError("decoded cause"))


def decode_tagged_error(value: object) -> TaggedError | SchemaFailure:
    if not isinstance(value, Mapping):
        issue: ResultCodecIssue = {"message": "expected a tagged error object"}
        return SchemaFailure((issue,))
    decoders = {
        "UserNotFound": _decode_user_not_found,
        "ValidationRejected": _decode_validation_rejected,
        "ServiceUnavailable": _decode_service_unavailable,
    }
    tag = value.get("_tag")
    decoder = decoders.get(tag) if isinstance(tag, str) else None
    if decoder is None:
        issue: ResultCodecIssue = {"message": "unsupported error tag"}
        return SchemaFailure((issue,))
    return decoder(value)


def make_codec() -> ResultCodec[
    Mapping[str, object],
    TaggedError,
    dict[str, object],
    dict[str, object | None],
    dict[str, object],
    TaggedError,
]:
    return codec(
        serialize_ok=encode_record,
        serialize_err=encode_tagged_error,
        deserialize_ok=decode_record,
        deserialize_err=decode_tagged_error,
    )


def test_tagged_error_tags_properties_causes_and_traces_match_golden() -> None:
    result_codec = make_codec()
    examples = {
        "user-not-found": UserNotFoundError("user-42"),
        "validation-rejected": ValidationRejectedError(
            "email",
            ["required", "must contain @"],
        ),
        "service-unavailable": ServiceUnavailableError(
            "billing",
            TimeoutError("upstream timed out"),
        ),
    }
    expected = cast("dict[str, object]", _golden_json("tagged-errors.json"))

    for name, error in examples.items():
        encoded = result_codec.serialize(Err(error))
        assert isinstance(encoded, Ok)
        assert _trace_count(encoded.value) == (
            1 if name != "service-unavailable" else 2
        )
        assert _normalize_traces(encoded.value) == expected[name]


def test_rich_ok_payload_and_inbound_tagged_errors_match_golden() -> None:
    result_codec = make_codec()
    rich_payload = cast("dict[str, object]", _golden_json("rich-ok-payload.json"))

    encoded = result_codec.serialize(
        Ok(rich_payload),
    )
    assert isinstance(encoded, Ok)
    assert encoded.value == _golden_json("rich-ok-envelope.json")

    for name in ("user-not-found", "validation-rejected", "service-unavailable"):
        envelope = expected_tagged_envelope(name)
        decoded = result_codec.deserialize(envelope)
        assert isinstance(decoded, Err)
        assert isinstance(decoded.error, TaggedError)
        error_payload = cast("dict[str, object]", envelope["error"])
        assert decoded.error._tag == error_payload["_tag"]


def expected_tagged_envelope(name: str) -> dict[str, object]:
    expected = cast("dict[str, object]", _golden_json("tagged-errors.json"))
    return cast("dict[str, object]", expected[name])


def test_unknown_tag_is_a_typed_codec_error_pinned_by_golden() -> None:
    result_codec = make_codec()
    decoded = result_codec.deserialize(_golden_json("unknown-tag.json"))

    assert isinstance(decoded, Err)
    assert isinstance(decoded.error, ResultDeserializationError)
    assert _stable_error(decoded.error) == _golden_json("unknown-tag-error.json")


@pytest.mark.asyncio
async def test_async_rich_codec_uses_the_same_tagged_protocol_goldens() -> None:
    async def async_encode_record(value: Mapping[str, object]) -> dict[str, object]:
        await asyncio.sleep(0)
        return encode_record(value)

    async def async_encode_error(error: TaggedError) -> dict[str, object | None]:
        await asyncio.sleep(0)
        return encode_tagged_error(error)

    result_codec = async_codec(
        serialize_ok=async_encode_record,
        serialize_err=async_encode_error,
        deserialize_ok=decode_record,
        deserialize_err=decode_tagged_error,
    )
    error = ServiceUnavailableError("billing", TimeoutError("upstream timed out"))

    encoded = await result_codec.serialize(
        Err(error),
    )
    assert isinstance(encoded, Ok)
    assert (
        _normalize_traces(encoded.value)
        == cast("dict[str, object]", _golden_json("tagged-errors.json"))[
            "service-unavailable"
        ]
    )
