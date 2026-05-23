from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from pyhooky import (
    HookPoint,
    HookRegistry,
    PriorityBoost,
    PriorityDemote,
    RunAfter,
    RunBefore,
    SetPriority,
    after,
    around,
    atrigger,
    before,
    hook,
    on,
    trigger,
)


@pytest.fixture
def registry() -> HookRegistry:
    return HookRegistry()


def test_priority_boost_reorders_for_next_dispatch(registry: HookRegistry) -> None:
    calls: list[str] = []

    @before("t", priority=10, registry=registry)
    def high() -> None:
        calls.append("high")

    @before("t", priority=1, registry=registry)
    def low() -> None:
        calls.append("low")
        if len(calls) == 2:  # only boost on first dispatch
            raise PriorityBoost(by=100)

    @hook("t", registry=registry)
    def target() -> None:
        return None

    target()
    assert calls == ["high", "low"]

    target()
    assert calls == ["high", "low", "low", "high"]


def test_priority_demote(registry: HookRegistry) -> None:
    calls: list[str] = []

    @before("t", priority=10, registry=registry)
    def high() -> None:
        calls.append("high")
        if len(calls) == 1:
            raise PriorityDemote(by=20)

    @before("t", priority=5, registry=registry)
    def mid() -> None:
        calls.append("mid")

    @hook("t", registry=registry)
    def target() -> None:
        return None

    target()
    assert calls == ["high", "mid"]

    target()
    assert calls == ["high", "mid", "mid", "high"]


def test_set_priority_absolute(registry: HookRegistry) -> None:
    calls: list[str] = []
    boosted = [False]

    @before("t", priority=1, registry=registry)
    def a() -> None:
        calls.append("a")
        if not boosted[0]:
            boosted[0] = True
            raise SetPriority(99)

    @before("t", priority=50, registry=registry)
    def b() -> None:
        calls.append("b")

    @hook("t", registry=registry)
    def target() -> None:
        return None

    target()
    target()
    assert calls == ["b", "a", "a", "b"]


def test_run_before_ensures_order(registry: HookRegistry) -> None:
    calls: list[str] = []

    def hook_a() -> None:
        calls.append("a")

    def hook_b() -> None:
        calls.append("b")
        if len(calls) <= 2:
            raise RunBefore(hook_a)

    registry.add_before("t", hook_a, priority=10)
    registry.add_before("t", hook_b, priority=1)

    @hook("t", registry=registry)
    def target() -> None:
        return None

    target()
    assert calls == ["a", "b"]

    target()
    assert calls == ["a", "b", "b", "a"]


def test_run_before_does_not_demote_already_ordered(registry: HookRegistry) -> None:
    def hook_a() -> None:
        pass

    def hook_b() -> None:
        raise RunBefore(hook_a)

    registry.add_before("t", hook_a, priority=5)
    registry.add_before("t", hook_b, priority=100)

    @hook("t", registry=registry)
    def target() -> None:
        return None

    target()  # hook_b raises, but priority already > hook_a; no change
    assert registry._priority_of("t", hook_b) == 100


def test_run_after_ensures_order(registry: HookRegistry) -> None:
    calls: list[str] = []

    def hook_a() -> None:
        calls.append("a")

    def hook_b() -> None:
        calls.append("b")
        if len(calls) <= 2:
            raise RunAfter(hook_a)

    registry.add_before("t", hook_a, priority=1)
    registry.add_before("t", hook_b, priority=10)

    @hook("t", registry=registry)
    def target() -> None:
        return None

    target()
    assert calls == ["b", "a"]

    target()
    assert calls == ["b", "a", "a", "b"]


def test_run_before_unknown_function_raises_runtime_error(registry: HookRegistry) -> None:
    def stranger() -> None:
        pass

    @before("t", registry=registry)
    def hook_b() -> None:
        raise RunBefore(stranger)

    @hook("t", registry=registry)
    def target() -> None:
        return None

    # The LookupError from _priority_of is wrapped in a RuntimeError that
    # mentions both the PriorityChange and the unknown function.
    with pytest.raises(RuntimeError, match="not registered"):
        target()


def test_resort_behavior_reorders_within_same_dispatch(registry: HookRegistry) -> None:
    calls: list[str] = []

    def first() -> None:
        calls.append("first")
        raise PriorityBoost(by=0)  # no-op boost, but with resort behavior

    def second() -> None:
        calls.append("second")

    def third() -> None:
        calls.append("third")

    registry.add_before("t", first, priority=10, on_priority_change="resort")
    registry.add_before("t", second, priority=5)
    registry.add_before("t", third, priority=1)

    @hook("t", registry=registry)
    def target() -> None:
        return None

    target()
    # first fires, raises with resort — but already-fired (first) doesn't fire again
    # remaining sorted by current priorities: second (5), third (1)
    assert calls == ["first", "second", "third"]


def test_resort_behavior_with_actual_reorder(registry: HookRegistry) -> None:
    calls: list[str] = []
    boosted = [False]

    def low() -> None:
        calls.append("low")
        if not boosted[0]:
            boosted[0] = True
            raise PriorityBoost(by=100)

    def mid() -> None:
        calls.append("mid")

    def high() -> None:
        calls.append("high")

    registry.add_before("t", low, priority=1, on_priority_change="resort")
    registry.add_before("t", mid, priority=5)
    registry.add_before("t", high, priority=10)

    @hook("t", registry=registry)
    def target() -> None:
        return None

    target()
    # initial order: high, mid, low
    # high fires, mid fires, low fires & boosts itself to 101 (above high=10)
    # resort: remaining = [] (high and mid already fired)
    assert calls == ["high", "mid", "low"]

    target()
    # new order: low (101), high (10), mid (5)
    assert calls == ["high", "mid", "low", "low", "high", "mid"]


def test_propagate_behavior_re_raises(registry: HookRegistry) -> None:
    @before("t", on_priority_change="propagate", registry=registry)
    def boost() -> None:
        raise PriorityBoost(by=5)

    @hook("t", registry=registry)
    def target() -> None:
        return None

    with pytest.raises(PriorityBoost):
        target()

    # priority was still updated before the re-raise
    assert registry._priority_of("t", boost) == 5


def test_registry_default_behavior_applies(registry: HookRegistry = None) -> None:  # type: ignore[assignment]
    reg = HookRegistry(on_priority_change="propagate")

    @before("t", registry=reg)
    def boost() -> None:
        raise PriorityBoost(by=1)

    @hook("t", registry=reg)
    def target() -> None:
        return None

    with pytest.raises(PriorityBoost):
        target()


def test_per_hook_overrides_registry_default() -> None:
    reg = HookRegistry(on_priority_change="propagate")

    @before("t", on_priority_change="continue", registry=reg)
    def boost() -> None:
        raise PriorityBoost(by=1)

    @hook("t", registry=reg)
    def target() -> None:
        return None

    target()  # should not raise — per-hook override wins
    assert reg._priority_of("t", boost) == 1


def test_priority_change_in_after_hook(registry: HookRegistry) -> None:
    calls: list[str] = []

    @after("t", priority=1, registry=registry)
    def low(result: int) -> None:
        calls.append("low")
        if len(calls) == 2:
            raise PriorityBoost(by=100)

    @after("t", priority=10, registry=registry)
    def high(result: int) -> None:
        calls.append("high")

    @hook("t", registry=registry)
    def target() -> int:
        return 1

    target()
    assert calls == ["high", "low"]
    target()
    assert calls == ["high", "low", "low", "high"]


def test_priority_change_in_trigger_listener(registry: HookRegistry) -> None:
    calls: list[str] = []

    @on("evt", priority=1, registry=registry)
    def low() -> str:
        calls.append("low")
        if len(calls) == 2:
            raise PriorityBoost(by=100)
        return "low"

    @on("evt", priority=10, registry=registry)
    def high() -> str:
        calls.append("high")
        return "high"

    results = trigger("evt", registry=registry)
    assert calls == ["high", "low"]
    # low raised PriorityChange instead of returning, so its result isn't collected
    assert results == ["high"]

    results = trigger("evt", registry=registry)
    assert calls == ["high", "low", "low", "high"]
    assert results == ["low", "high"]


async def test_priority_change_in_async_listener(registry: HookRegistry) -> None:
    calls: list[str] = []

    @on("evt", priority=1, registry=registry)
    async def low() -> str:
        await asyncio.sleep(0)
        calls.append("low")
        if len(calls) == 2:
            raise PriorityBoost(by=100)
        return "low"

    @on("evt", priority=10, registry=registry)
    async def high() -> str:
        calls.append("high")
        return "high"

    await atrigger("evt", registry=registry)
    assert calls == ["high", "low"]

    await atrigger("evt", registry=registry)
    assert calls == ["high", "low", "low", "high"]


async def test_priority_change_in_hookpoint_atrigger(registry: HookRegistry) -> None:
    class Event(BaseModel):
        n: int

    point: HookPoint[Event] = HookPoint("e", Event, registry=registry)
    calls: list[str] = []

    @point.listen(priority=1)
    async def low(event: Event) -> int:
        calls.append("low")
        if len(calls) == 2:
            raise PriorityBoost(by=100)
        return event.n

    @point.listen(priority=10)
    async def high(event: Event) -> int:
        calls.append("high")
        return event.n * 2

    await point.atrigger(n=1)
    assert calls == ["high", "low"]
    await point.atrigger(n=1)
    assert calls == ["high", "low", "low", "high"]


def test_priority_change_in_around_raises_runtime_error(registry: HookRegistry) -> None:
    @around("t", registry=registry)
    def guard(inner) -> int:
        raise PriorityBoost(by=5)

    @hook("t", registry=registry)
    def target() -> int:
        return 1

    # around hooks can't reorder themselves; raising PriorityChange from an
    # around now surfaces as a RuntimeError (chained from the original) that
    # explains why AFTER didn't run.
    with pytest.raises(RuntimeError, match="around hook") as info:
        target()
    assert isinstance(info.value.__cause__, PriorityBoost)
