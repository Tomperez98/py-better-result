"""Workflow tests using explicit Result combinators instead of generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_type

import pytest

from better_result.combinators import and_then_async
from better_result.core import Err, Ok, Result, err
from better_result.error import TaggedError


class InvalidAmount(TaggedError, tag="InvalidAmount"):
    """The checkout amount is not valid."""

    input: str

    def __init__(self, input_value: str) -> None:
        super().__init__(message="Amount must be a positive integer", input=input_value)


class InsufficientFunds(TaggedError, tag="InsufficientFunds"):
    """The account cannot cover the checkout amount."""

    requested: int
    balance: int

    def __init__(self, requested: int, balance: int) -> None:
        super().__init__(
            message="Insufficient funds",
            requested=requested,
            balance=balance,
        )


class GatewayUnavailable(TaggedError, tag="GatewayUnavailable"):
    """The selected payment gateway is unavailable."""

    channel: str

    def __init__(self, channel: str) -> None:
        super().__init__(message="Payment gateway unavailable", channel=channel)


@dataclass(frozen=True)
class Receipt:
    amount: int
    channel: str


type CheckoutError = InvalidAmount | InsufficientFunds | GatewayUnavailable


def parse_amount(raw: str) -> Result[int, InvalidAmount]:
    try:
        amount = int(raw)
    except ValueError:
        return err(InvalidAmount(raw))
    if amount <= 0:
        return err(InvalidAmount(raw))
    return Ok[int, InvalidAmount](amount)


def checkout(
    raw_amount: str,
    balance: int,
    *,
    gateway_available: bool = True,
    trace: list[str] | None = None,
) -> Result[Receipt, CheckoutError]:
    """Compose parsing, authorization, routing, and charging with and_then."""
    events = trace if trace is not None else []

    def authorize(amount: int) -> Result[int, InsufficientFunds]:
        events.append("authorize")
        if amount > balance:
            return Err[int, InsufficientFunds](InsufficientFunds(amount, balance))
        return Ok[int, InsufficientFunds](amount)

    def charge(amount: int) -> Result[Receipt, GatewayUnavailable]:
        channel = "card" if amount <= 100 else "bank"
        events.append(f"charge:{channel}")
        if not gateway_available:
            return Err[Receipt, GatewayUnavailable](GatewayUnavailable(channel))
        return Ok[Receipt, GatewayUnavailable](Receipt(amount, channel))

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
            return Err[int, CheckoutError](InsufficientFunds(amount, balance))
        return Ok[int, CheckoutError](amount)

    async def charge(amount: int) -> Result[Receipt, CheckoutError]:
        channel = "card" if amount <= 100 else "bank"
        events.append(f"charge:{channel}")
        if not gateway_available:
            return Err[Receipt, CheckoutError](GatewayUnavailable(channel))
        return Ok[Receipt, CheckoutError](Receipt(amount, channel))

    authorized = await and_then_async(parsed, authorize)
    return await and_then_async(authorized, charge)


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
    assert isinstance(invalid.error, InvalidAmount)
    assert invalid_trace == []

    funds_trace: list[str] = []
    funds = checkout("250", 100, trace=funds_trace)
    assert isinstance(funds, Err)
    assert isinstance(funds.error, InsufficientFunds)
    assert funds_trace == ["authorize"]

    gateway_trace: list[str] = []
    gateway = checkout("250", 500, gateway_available=False, trace=gateway_trace)
    assert isinstance(gateway, Err)
    assert isinstance(gateway.error, GatewayUnavailable)
    assert gateway.error.channel == "bank"
    assert gateway_trace == ["authorize", "charge:bank"]


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
    assert isinstance(failed.error, InsufficientFunds)
    assert failed_trace == ["authorize"]
