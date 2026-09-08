"""The package exposes one curated public entrypoint."""

from __future__ import annotations

import importlib

import pytest

import better_result


def test_root_exports_only_the_supported_names() -> None:
    assert better_result.__all__ == [
        "Err",
        "Ok",
        "PanicError",
        "Result",
        "TaggedError",
    ]
    assert {name for name in vars(better_result) if not name.startswith("_")} == {
        "Err",
        "Ok",
        "PanicError",
        "Result",
        "TaggedError",
    }


def test_root_exports_are_the_canonical_implementations() -> None:
    assert better_result.Ok.__module__ == "better_result._core"
    assert better_result.Err.__module__ == "better_result._core"
    assert better_result.TaggedError.__module__ == "better_result._error"


@pytest.mark.parametrize("module", ["core", "error", "codec", "collections", "retry"])
def test_legacy_module_entrypoints_are_not_supported(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"better_result.{module}")
