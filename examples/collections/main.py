"""Collect several Results or split them into successes and errors."""

from __future__ import annotations

from better_result import (
    Err,
    Ok,
    Result,
    all_results,
    collect_results,
    flatten_result,
    partition_results,
)


def main() -> None:
    results: list[Result[int, str]] = [Ok(1), Err("bad input"), Ok(3)]

    # all_results short-circuits at the first error.
    first_error = all_results(results)
    assert first_error == Err("bad input")
    print("all_results:", first_error)

    # collect_results accumulates every error instead of short-circuiting.
    all_errors = collect_results(results)
    assert all_errors == Err(("bad input",))
    print("collect_results:", all_errors)

    # partition_results keeps both sides and preserves their relative order.
    values, errors = partition_results(results)
    assert values == [1, 3]
    assert errors == ["bad input"]
    print(f"partition_results: values={values}, errors={errors}")

    # flatten_result removes one nested Result layer.
    nested: Result[Result[int, str], str] = Ok(Ok(42))
    flattened = flatten_result(nested)
    assert flattened == Ok(42)
    print("flatten_result:", flattened)


if __name__ == "__main__":
    main()
