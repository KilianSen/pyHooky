from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, ValidationError

from pyhooky import HookRegistry, before
from pyhooky.typed import HookPoint


class CheckoutStep(BaseModel):
    step: str
    cart_id: int


class Order(BaseModel):
    id: int
    items: list[str]


@pytest.fixture
def registry() -> HookRegistry:
    return HookRegistry()


def test_trigger_with_kwargs_validates_and_dispatches(registry: HookRegistry) -> None:
    point: HookPoint[CheckoutStep] = HookPoint("checkout:step", CheckoutStep, registry=registry)
    seen: list[CheckoutStep] = []

    @point.listen
    def audit(event: CheckoutStep) -> str:
        seen.append(event)
        return f"{event.step}:{event.cart_id}"

    results = point.trigger(step="validate", cart_id=42)

    assert results == ["validate:42"]
    assert seen[0].step == "validate"
    assert seen[0].cart_id == 42


def test_trigger_with_model_instance(registry: HookRegistry) -> None:
    point: HookPoint[CheckoutStep] = HookPoint("step", CheckoutStep, registry=registry)
    point.listen(lambda e: e.cart_id * 2)

    payload = CheckoutStep(step="charge", cart_id=7)
    assert point.trigger(payload) == [14]


def test_trigger_with_dict(registry: HookRegistry) -> None:
    point: HookPoint[CheckoutStep] = HookPoint("step", CheckoutStep, registry=registry)
    point.listen(lambda e: e.step)

    assert point.trigger({"step": "ship", "cart_id": 1}) == ["ship"]


def test_invalid_payload_raises_validation_error(registry: HookRegistry) -> None:
    point: HookPoint[CheckoutStep] = HookPoint("step", CheckoutStep, registry=registry)
    point.listen(lambda e: None)

    with pytest.raises(ValidationError):
        point.trigger(step="charge", cart_id="not-an-int")


def test_mixing_payload_and_kwargs_rejected(registry: HookRegistry) -> None:
    point: HookPoint[CheckoutStep] = HookPoint("step", CheckoutStep, registry=registry)
    payload = CheckoutStep(step="x", cart_id=1)

    with pytest.raises(TypeError):
        point.trigger(payload, cart_id=2)  # type: ignore[call-overload]


def test_listen_non_decorator(registry: HookRegistry) -> None:
    point: HookPoint[CheckoutStep] = HookPoint("step", CheckoutStep, registry=registry)

    def handler(event: CheckoutStep) -> int:
        return event.cart_id

    point.listen(handler)
    assert point.trigger(step="x", cart_id=99) == [99]


def test_priority_order(registry: HookRegistry) -> None:
    point: HookPoint[CheckoutStep] = HookPoint("step", CheckoutStep, registry=registry)
    calls: list[str] = []

    point.listen(lambda e: calls.append("low"), priority=1)
    point.listen(lambda e: calls.append("high"), priority=10)

    point.trigger(step="x", cart_id=1)
    assert calls == ["high", "low"]


def test_remove_and_clear(registry: HookRegistry) -> None:
    point: HookPoint[CheckoutStep] = HookPoint("step", CheckoutStep, registry=registry)

    def h(e: CheckoutStep) -> None:
        raise AssertionError("should not fire")

    point.listen(h)
    assert point.remove(h) is True
    point.trigger(step="x", cart_id=1)

    point.listen(h)
    point.clear()
    point.trigger(step="x", cart_id=1)


# ---------- typed wrap + before/after/around (#14) ----------


def test_wrap_validates_kwargs_and_calls_fn(registry: HookRegistry) -> None:
    point: HookPoint[Order] = HookPoint("place_order", Order, registry=registry)

    @point.wrap
    def place_order(order: Order) -> int:
        return order.id * 10

    assert place_order(id=4, items=["a", "b"]) == 40


def test_wrap_accepts_dict_and_model_passthrough(registry: HookRegistry) -> None:
    point: HookPoint[Order] = HookPoint("place_order", Order, registry=registry)

    @point.wrap
    def place_order(order: Order) -> Order:
        return order

    assert place_order({"id": 1, "items": ["a"]}).id == 1
    payload = Order(id=2, items=["b"])
    assert place_order(payload) is payload


def test_wrap_raises_validation_error_before_hooks_fire(registry: HookRegistry) -> None:
    point: HookPoint[Order] = HookPoint("place_order", Order, registry=registry)
    fired: list[str] = []

    @point.before
    def hook(order: Order) -> None:
        fired.append("before")

    @point.wrap
    def place_order(order: Order) -> int:
        fired.append("body")
        return order.id

    with pytest.raises(ValidationError):
        place_order(id="not-an-int", items=["a"])  # type: ignore[arg-type]

    assert fired == []


def test_typed_before_receives_model(registry: HookRegistry) -> None:
    point: HookPoint[Order] = HookPoint("place_order", Order, registry=registry)
    seen: list[Order] = []

    @point.before
    def hook(order: Order) -> None:
        seen.append(order)

    @point.wrap
    def place_order(order: Order) -> int:
        return order.id

    place_order(id=7, items=["x"])
    assert seen[0].id == 7
    assert seen[0].items == ["x"]


def test_typed_after_receives_result_and_model(registry: HookRegistry) -> None:
    point: HookPoint[Order] = HookPoint("place_order", Order, registry=registry)
    seen: list[tuple[int, Order]] = []

    @point.after
    def hook(result: int, order: Order) -> None:
        seen.append((result, order))

    @point.wrap
    def place_order(order: Order) -> int:
        return order.id * 2

    place_order(id=5, items=["a"])
    assert seen[0][0] == 10
    assert seen[0][1].id == 5


def test_typed_around_can_short_circuit(registry: HookRegistry) -> None:
    point: HookPoint[Order] = HookPoint("place_order", Order, registry=registry)

    @point.around
    def guard(inner, order: Order) -> int:
        if order.id < 0:
            return -1
        return inner(order)

    @point.wrap
    def place_order(order: Order) -> int:
        return order.id

    assert place_order(id=5, items=["a"]) == 5
    assert place_order(id=-1, items=[]) == -1


def test_typed_around_calls_inner_with_model(registry: HookRegistry) -> None:
    point: HookPoint[Order] = HookPoint("place_order", Order, registry=registry)
    saw: list[Order] = []

    @point.around
    def wrap_it(inner, order: Order) -> int:
        saw.append(order)
        return inner(order) + 100

    @point.wrap
    def place_order(order: Order) -> int:
        return order.id

    assert place_order(id=3, items=["a"]) == 103
    assert saw[0].id == 3


async def test_async_wrap_with_async_hooks(registry: HookRegistry) -> None:
    point: HookPoint[Order] = HookPoint("place_order", Order, registry=registry)
    trail: list[str] = []

    @point.before
    async def b(order: Order) -> None:
        await asyncio.sleep(0)
        trail.append(f"before:{order.id}")

    @point.after
    async def a(result: int, order: Order) -> None:
        await asyncio.sleep(0)
        trail.append(f"after:{result}")

    @point.around
    async def ar(inner, order: Order) -> int:
        trail.append("around:in")
        result = await inner(order)
        trail.append("around:out")
        return result + 1

    @point.wrap
    async def place_order(order: Order) -> int:
        await asyncio.sleep(0)
        trail.append("body")
        return order.id

    result = await place_order(id=9, items=["x"])
    assert result == 10
    assert trail == ["before:9", "around:in", "body", "around:out", "after:10"]


def test_listener_and_wrap_coexist_on_same_point(registry: HookRegistry) -> None:
    point: HookPoint[Order] = HookPoint("place_order", Order, registry=registry)
    fired: list[str] = []

    @point.listen
    def evt(order: Order) -> None:
        fired.append(f"listen:{order.id}")

    @point.before
    def b(order: Order) -> None:
        fired.append(f"before:{order.id}")

    @point.wrap
    def place_order(order: Order) -> int:
        fired.append("body")
        return order.id

    # trigger fires only the listener
    point.trigger(id=1, items=["a"])
    assert fired == ["listen:1"]

    # calling the wrapped fn fires before+body but NOT the listener
    fired.clear()
    place_order(id=2, items=["b"])
    assert fired == ["before:2", "body"]


def test_typed_hook_target_attr_set(registry: HookRegistry) -> None:
    point: HookPoint[Order] = HookPoint("place_order", Order, registry=registry)

    @point.wrap
    def place_order(order: Order) -> int:
        return order.id

    assert place_order.__hook_target__ == "place_order"  # type: ignore[attr-defined]

    # Function-reference form to attach more hooks
    seen: list[Order] = []
    before(place_order, lambda order: seen.append(order), registry=registry)

    place_order(id=11, items=["a"])
    assert seen[0].id == 11


def test_typed_before_registered_before_wrap_still_fires(registry: HookRegistry) -> None:
    point: HookPoint[Order] = HookPoint("place_order", Order, registry=registry)
    seen: list[int] = []

    @point.before
    def b(order: Order) -> None:
        seen.append(order.id)

    @point.wrap
    def place_order(order: Order) -> int:
        return order.id

    place_order(id=4, items=["a"])
    assert seen == [4]


def test_typed_wrap_with_tag_clears_via_clear_tag(registry: HookRegistry) -> None:
    point: HookPoint[Order] = HookPoint("place_order", Order, registry=registry)
    fired: list[str] = []

    @point.before(tag="audit")
    def b(order: Order) -> None:
        fired.append("before")

    @point.after(tag="audit")
    def a(result: int, order: Order) -> None:
        fired.append("after")

    @point.wrap
    def place_order(order: Order) -> int:
        fired.append("body")
        return order.id

    place_order(id=1, items=["a"])
    assert fired == ["before", "body", "after"]

    fired.clear()
    removed = registry.clear_tag("audit")
    assert len(removed) == 2

    place_order(id=2, items=["b"])
    assert fired == ["body"]


def test_typed_wrap_priority_order(registry: HookRegistry) -> None:
    point: HookPoint[Order] = HookPoint("place_order", Order, registry=registry)
    calls: list[str] = []

    @point.before(priority=1)
    def low(order: Order) -> None:
        calls.append("low")

    @point.before(priority=10)
    def high(order: Order) -> None:
        calls.append("high")

    @point.wrap
    def place_order(order: Order) -> int:
        return order.id

    place_order(id=1, items=["a"])
    assert calls == ["high", "low"]
