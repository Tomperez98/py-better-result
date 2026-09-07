"""Workflow tests using explicit Result combinators instead of generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_type

import pytest

from better_result.core import Err, Ok, Result
from better_result.error import TaggedError


class InvalidAmountError(TaggedError, tag="InvalidAmount"):
    """The checkout amount is not valid."""

    input: str

    def __init__(self, input_value: str) -> None:
        super().__init__(message="Amount must be a positive integer", input=input_value)


class InsufficientFundsError(TaggedError, tag="InsufficientFunds"):
    """The account cannot cover the checkout amount."""

    requested: int
    balance: int

    def __init__(self, requested: int, balance: int) -> None:
        super().__init__(
            message="Insufficient funds",
            requested=requested,
            balance=balance,
        )


class GatewayUnavailableError(TaggedError, tag="GatewayUnavailable"):
    """The selected payment gateway is unavailable."""

    channel: str

    def __init__(self, channel: str) -> None:
        super().__init__(message="Payment gateway unavailable", channel=channel)


@dataclass(frozen=True)
class Receipt:
    amount: int
    channel: str


type CheckoutError = (
    InvalidAmountError | InsufficientFundsError | GatewayUnavailableError
)


def parse_amount(raw: str) -> Result[int, InvalidAmountError]:
    try:
        amount = int(raw)
    except ValueError:
        return Err(InvalidAmountError(raw))
    if amount <= 0:
        return Err(InvalidAmountError(raw))
    return Ok(amount)


def checkout(
    raw_amount: str,
    balance: int,
    *,
    gateway_available: bool = True,
    trace: list[str] | None = None,
) -> Result[Receipt, CheckoutError]:
    """Compose parsing, authorization, routing, and charging with and_then."""
    events = trace if trace is not None else []

    def authorize(amount: int) -> Result[int, InsufficientFundsError]:
        events.append("authorize")
        if amount > balance:
            return Err(InsufficientFundsError(amount, balance))
        return Ok(amount)

    def charge(amount: int) -> Result[Receipt, GatewayUnavailableError]:
        channel = "card" if amount <= 100 else "bank"
        events.append(f"charge:{channel}")
        if not gateway_available:
            return Err(GatewayUnavailableError(channel))
        return Ok(Receipt(amount, channel))

    return parse_amount(raw_amount).and_then(authorize).and_then(charge)


async def checkout_async(
    raw_amount: str,
    balance: int,
    *,
    gateway_available: bool = True,
    trace: list[str] | None = None,
) -> Result[Receipt, CheckoutError]:
    """Compose the same workflow through async and_then operations."""
    events = trace if trace is not None else []
    parsed = parse_amount(raw_amount)

    async def authorize(amount: int) -> Result[int, CheckoutError]:
        events.append("authorize")
        if amount > balance:
            return Err(InsufficientFundsError(amount, balance))
        return Ok(amount)

    async def charge(amount: int) -> Result[Receipt, CheckoutError]:
        channel = "card" if amount <= 100 else "bank"
        events.append(f"charge:{channel}")
        if not gateway_available:
            return Err(GatewayUnavailableError(channel))
        return Ok(Receipt(amount, channel))

    authorized = await parsed.and_then_async(authorize)
    assert_type(authorized, Result[int, InvalidAmountError | CheckoutError])
    charged = await authorized.and_then_async(charge)
    assert_type(charged, Result[Receipt, InvalidAmountError | CheckoutError])
    return charged


def test_and_then_composes_success_and_value_based_branches() -> None:
    small_trace: list[str] = []
    small = checkout("50", 100, trace=small_trace)
    large_trace: list[str] = []
    large = checkout("250", 500, trace=large_trace)

    assert_type(small, Result[Receipt, CheckoutError])
    assert_type(large, Result[Receipt, CheckoutError])
    assert isinstance(small, Ok)
    assert small.value == Receipt(50, "card")
    assert small_trace == ["authorize", "charge:card"]
    assert isinstance(large, Ok)
    assert large.value == Receipt(250, "bank")
    assert large_trace == ["authorize", "charge:bank"]


def test_and_then_short_circuits_each_expected_error() -> None:
    invalid_trace: list[str] = []
    invalid = checkout("not-an-amount", 500, trace=invalid_trace)
    assert isinstance(invalid, Err)
    assert isinstance(invalid.error, InvalidAmountError)
    assert invalid_trace == []

    funds_trace: list[str] = []
    funds = checkout("250", 100, trace=funds_trace)
    assert isinstance(funds, Err)
    assert isinstance(funds.error, InsufficientFundsError)
    assert funds_trace == ["authorize"]

    gateway_trace: list[str] = []
    gateway = checkout("250", 500, gateway_available=False, trace=gateway_trace)
    assert isinstance(gateway, Err)
    assert isinstance(gateway.error, GatewayUnavailableError)
    assert gateway.error.channel == "bank"
    assert gateway_trace == ["authorize", "charge:bank"]


@pytest.mark.asyncio
async def test_and_then_async_preserves_the_full_error_union() -> None:
    async def authorize(amount: int) -> Result[int, InsufficientFundsError]:
        return Ok(amount)

    async def charge(amount: int) -> Result[Receipt, GatewayUnavailableError]:
        return Ok(Receipt(amount, "card"))

    async def authorize_chain(
        result: Result[int, InvalidAmountError],
    ) -> Result[int, InvalidAmountError | InsufficientFundsError]:
        return await result.and_then_async(authorize)

    async def charge_chain(
        result: Result[int, InvalidAmountError | InsufficientFundsError],
    ) -> Result[
        Receipt,
        InvalidAmountError | InsufficientFundsError | GatewayUnavailableError,
    ]:
        return await result.and_then_async(charge)

    initial: Result[int, InvalidAmountError] = Ok(5)
    authorized = await authorize_chain(initial)
    assert_type(authorized, Result[int, InvalidAmountError | InsufficientFundsError])
    charged = await charge_chain(authorized)
    assert_type(
        charged,
        Result[
            Receipt,
            InvalidAmountError | InsufficientFundsError | GatewayUnavailableError,
        ],
    )
    assert isinstance(charged, Ok)
    assert charged.value == Receipt(5, "card")


@pytest.mark.asyncio
async def test_and_then_async_composes_and_short_circuits_the_same_workflow() -> None:
    success_trace: list[str] = []
    success = await checkout_async("75", 100, trace=success_trace)
    assert isinstance(success, Ok)
    assert success.value == Receipt(75, "card")
    assert success_trace == ["authorize", "charge:card"]

    failed_trace: list[str] = []
    failed = await checkout_async("250", 100, trace=failed_trace)
    assert isinstance(failed, Err)
    assert isinstance(failed.error, InsufficientFundsError)
    assert failed_trace == ["authorize"]
