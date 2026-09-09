"""Behavioral tests for the reduced example runner."""

# ruff: noqa: INP001

from __future__ import annotations

from pathlib import Path

from better_result_example.cli import (
    _discover_examples,
    _find_examples_dir,
    _resolve_example,
    main,
)
from click.testing import CliRunner

from better_result import Err, Ok

REPOSITORY_ROOT = Path(__file__).parents[3]
EXPECTED_EXAMPLES = ("async-retry", "basic", "sync-retry")


def test_find_examples_dir_returns_a_result(tmp_path: Path) -> None:
    assert _find_examples_dir(tmp_path) == Err(
        "run better-result-example from the repository root"
    )
    assert _find_examples_dir(REPOSITORY_ROOT) == Ok(REPOSITORY_ROOT / "examples")


def test_discover_examples_matches_the_rust_workspace() -> None:
    examples_dir = REPOSITORY_ROOT / "examples"
    assert _discover_examples(examples_dir) == EXPECTED_EXAMPLES


def test_resolve_example_returns_script_or_a_selection_error() -> None:
    examples_dir = REPOSITORY_ROOT / "examples"

    resolved = _resolve_example(examples_dir, "basic")
    assert resolved == Ok(examples_dir / "basic" / "main.py")

    missing = _resolve_example(examples_dir, "does-not-exist")
    assert isinstance(missing, Err)
    assert "does-not-exist" in missing.err()
    assert "basic" in missing.err()


def test_cli_lists_examples() -> None:
    result = CliRunner().invoke(main, ["--list"], catch_exceptions=False)
    assert result.output.splitlines() == list(EXPECTED_EXAMPLES)


def test_cli_runs_a_named_example() -> None:
    result = CliRunner().invoke(main, ["basic"], catch_exceptions=False)
    assert "loaded from the service" in result.output


def test_cli_runs_all_examples_in_sorted_order() -> None:
    result = CliRunner().invoke(main, ["--all"], catch_exceptions=False)
    assert result.output.count("==> ") == len(EXPECTED_EXAMPLES)
    header_positions = [
        result.output.index(f"==> {example}") for example in EXPECTED_EXAMPLES
    ]
    assert header_positions == sorted(header_positions)
    assert "loaded from the service" in result.output
    assert "async response" in result.output


def test_cli_reports_selection_and_usage_errors() -> None:
    runner = CliRunner()

    missing_name = runner.invoke(main, [])
    assert missing_name.exit_code == 2
    assert "provide an example name" in missing_name.output

    unknown = runner.invoke(main, ["does-not-exist"])
    assert unknown.exit_code == 1
    assert "unknown example 'does-not-exist'" in unknown.output

    all_with_name = runner.invoke(main, ["--all", "basic"])
    assert all_with_name.exit_code == 2
    assert "--all cannot be combined" in all_with_name.output

    all_with_list = runner.invoke(main, ["--all", "--list"])
    assert all_with_list.exit_code == 2
    assert "--list and --all cannot be combined" in all_with_list.output
