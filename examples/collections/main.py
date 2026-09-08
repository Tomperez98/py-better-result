"""Collect several Results or split them into successes and errors."""

from __future__ import annotations

from better_result import (
    Err,
    Ok,
    Result,
    all_results,
    flatten_result,
    partition_results,
)


def main() -> None:
    results: list[Result[int, str]] = [Ok(1), Err("bad input"), Ok(3)]

    # all_results short-circuits at the first error.
    print(all_results(results))

    # partition_results keeps both sides and preserves their relative order.
    values, errors = partition_results(results)
    print(f"values={values}, errors={errors}")

    nested: Result[Result[int, str], str] = Ok(Ok(42))
    print(flatten_result(nested))


if __name__ == "__main__":
    main()
