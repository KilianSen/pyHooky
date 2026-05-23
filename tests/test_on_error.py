"""Tests for the ON_ERROR hook kind: on_error fires when the wrapped
function (or anything inside the AROUND chain) raises, but cannot swallow
the exception. AFTER must not fire on exception."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from pyhooky import (
    HookKind,
    HookPoint,
    HookRegistry,
    after,
    around,
    hook,
    on_error,
)


@pytest.fixture
def registry() -> HookRegistry:
    return HookRegistry()


def test_on_error_fires_when_body_raises(registry: HookRegistry) -> None:
    seen: list[tuple[type[BaseException], int]] = []

    @on_error("t", registry=registry)
    def capture(exc: Exception, x: int) -> None:
        seen.append((type(exc), x))

    @hook("t", registry=registry)
    def body(x: int) -> int:
        raise ValueError(f"boom:{x}")

    with pytest.raises(ValueError, match="boom:3"):
        body(3)
    assert seen == [(ValueError, 3)]


def test_after_does_not_fire_on_exception(registry: HookRegistry) -> None:
    fired: list[str] = []

    @after("t", registry=registry)
    def post(result: object, x: int) -> None:
        fired.append("after")

    @on_error("t", registry=registry)
    def err(exc: Exception, x: int) -> None:
        fired.append("on_error")

    @hook("t", registry=registry)
    def body(x: int) -> int:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        body(1)
    assert fired == ["on_error"]


def test_on_error_does_not_swallow(registry: HookRegistry) -> None:
    @on_error("t", registry=registry)
    def ignore(exc: Exception) -> str:
        return "ignored"  # return value is discarded

    @hook("t", registry=registry)
    def body() -> int:
        raise ValueError("still raises")

    with pytest.raises(ValueError, match="still raises"):
        body()


def test_on_error_fires_for_around_chain_exceptions(registry: HookRegistry) -> None:
    seen: list[str] = []

    @around("t", registry=registry)
    def guard(inner, x: int) -> int:
        # Raises before reaching the inner body
        raise RuntimeError(f"around-rejected:{x}")

    @on_error("t", registry=registry)
    def err(exc: Exception, x: int) -> None:
        seen.append(str(exc))

    @hook("t", registry=registry)
    def body(x: int) -> int:
        return x

    with pytest.raises(RuntimeError, match="around-rejected:7"):
        body(7)
    assert seen == ["around-rejected:7"]


def test_around_can_swallow_and_on_error_does_not_fire(registry: HookRegistry) -> None:
    seen: list[str] = []

    @around("t", registry=registry)
    def swallow(inner, x: int) -> int:
        try:
            return inner(x)
        except ValueError:
            return -1

    @on_error("t", registry=registry)
    def err(exc: Exception, x: int) -> None:
        seen.append("err")

    @hook("t", registry=registry)
    def body(x: int) -> int:
        raise ValueError("oops")

    assert body(5) == -1
    assert seen == []


def test_on_error_priority_order(registry: HookRegistry) -> None:
    calls: list[str] = []

    @on_error("t", priority=1, registry=registry)
    def low(exc: Exception) -> None:
        calls.append("low")

    @on_error("t", priority=10, registry=registry)
    def high(exc: Exception) -> None:
        calls.append("high")

    @hook("t", registry=registry)
    def body() -> int:
        raise RuntimeError

    with pytest.raises(RuntimeError):
        body()
    assert calls == ["high", "low"]


def test_on_error_with_tag_for_plugin_teardown(registry: HookRegistry) -> None:
    @on_error("t", tag="audit", registry=registry)
    def err(exc: Exception) -> None:
        pass

    pairs = registry.hooks_by_tag("audit")
    assert len(pairs) == 1
    assert pairs[0][1].kind == HookKind.ON_ERROR


async def test_async_on_error(registry: HookRegistry) -> None:
    seen: list[str] = []

    @on_error("t", registry=registry)
    async def err(exc: Exception, x: int) -> None:
        await asyncio.sleep(0)
        seen.append(str(exc))

    @hook("t", registry=registry)
    async def body(x: int) -> int:
        raise ValueError(f"async-boom:{x}")

    with pytest.raises(ValueError, match="async-boom:2"):
        await body(2)
    assert seen == ["async-boom:2"]


def test_sync_target_async_on_error_rejected(registry: HookRegistry) -> None:
    @on_error("t", registry=registry)
    async def err(exc: Exception) -> None:
        pass

    @hook("t", registry=registry)
    def body() -> int:
        raise ValueError

    with pytest.raises(RuntimeError, match="Async on_error"):
        body()


def test_hookpoint_on_error(registry: HookRegistry) -> None:
    class Order(BaseModel):
        id: int

    point: HookPoint[Order] = HookPoint("place_order", Order, registry=registry)
    seen: list[tuple[type[BaseException], int]] = []

    @point.on_error
    def err(exc: Exception, order: Order) -> None:
        seen.append((type(exc), order.id))

    @point.wrap
    def place_order(order: Order) -> int:
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError, match="fail"):
        place_order(id=99)
    assert seen == [(RuntimeError, 99)]
