"""Run repository examples through a small Click command."""

from __future__ import annotations

import runpy
from pathlib import Path

import click

from better_result import Err, Ok, Result, is_err


def _find_examples_dir(working_directory: Path) -> Result[Path, str]:
    examples_dir = working_directory / "examples"
    if not examples_dir.is_dir():
        return Err("run better-result-example from the repository root")
    return Ok(examples_dir)


def _discover_examples(examples_dir: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.name
            for path in examples_dir.iterdir()
            if path.is_dir() and (path / "main.py").is_file()
        )
    )


def _resolve_example(
    examples_dir: Path,
    name: str,
) -> Result[Path, str]:
    available = _discover_examples(examples_dir)
    if name not in available:
        choices = ", ".join(available)
        return Err(f"unknown example {name!r}; choose one of: {choices}")

    script = examples_dir / name / "main.py"
    assert script.is_file(), f"discovered example has no script: {script}"
    return Ok(script)


@click.command()
@click.argument("name", required=False)
@click.option(
    "--list",
    "list_examples",
    is_flag=True,
    help="List runnable examples.",
)
@click.option(
    "--all",
    "run_all",
    is_flag=True,
    help="Run every example in sorted order.",
)
def main(
    name: str | None,
    *,
    list_examples: bool,
    run_all: bool,
) -> None:
    """Run one of the repository's runnable examples."""
    examples_result = _find_examples_dir(Path.cwd())
    if is_err(examples_result):
        raise click.ClickException(examples_result.err_value)
    examples_dir = examples_result.unwrap()

    available = _discover_examples(examples_dir)
    if list_examples and run_all:
        message = "--list and --all cannot be combined"
        raise click.UsageError(message)
    if run_all and name is not None:
        message = "--all cannot be combined with an example name"
        raise click.UsageError(message)

    if list_examples:
        for example in available:
            click.echo(example)
        return

    if run_all:
        for example in available:
            click.echo(f"==> {example}")
            _run_example(examples_dir, example)
        return

    if name is None:
        message = "provide an example name, --all, or use --list"
        raise click.UsageError(message)

    _run_example(examples_dir, name)


def _run_example(examples_dir: Path, name: str) -> None:
    script_result = _resolve_example(examples_dir, name)
    if is_err(script_result):
        raise click.ClickException(script_result.err_value)
    runpy.run_path(str(script_result.unwrap()), run_name="__main__")
