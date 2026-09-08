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
    print("all_results:", all_results(results))

    # collect_results accumulates every error instead of short-circuiting.
    print("collect_results:", collect_results(results))

    # partition_results keeps both sides and preserves their relative order.
    values, errors = partition_results(results)
    print(f"partition_results: values={values}, errors={errors}")

    # flatten_result removes one nested Result layer.
    nested: Result[Result[int, str], str] = Ok(Ok(42))
    print("flatten_result:", flatten_result(nested))


if __name__ == "__main__":
    main()
