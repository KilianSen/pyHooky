from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from pyhooky import (
    HookPoint,
    HookRegistry,
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


async def test_async_wrapped_function_awaits_async_before_and_after(registry: HookRegistry) -> None:
    calls: list[str] = []

    async def log_before(x: int) -> None:
        await asyncio.sleep(0)
        calls.append(f"before:{x}")

    async def log_after(result: int, x: int) -> None:
        await asyncio.sleep(0)
        calls.append(f"after:{result}:{x}")

    @hook("compute", registry=registry)
    async def compute(x: int) -> int:
        await asyncio.sleep(0)
        calls.append(f"body:{x}")
        return x * 2

    before("compute", log_before, registry=registry)
    after("compute", log_after, registry=registry)

    assert await compute(5) == 10
    assert calls == ["before:5", "body:5", "after:10:5"]


async def test_async_target_mixes_sync_and_async_hooks(registry: HookRegistry) -> None:
    calls: list[str] = []

    def sync_before(x: int) -> None:
        calls.append("sync_before")

    async def async_before(x: int) -> None:
        calls.append("async_before")

    @hook("m", registry=registry)
    async def fn(x: int) -> int:
        calls.append("body")
        return x

    before("m", sync_before, priority=10, registry=registry)
    before("m", async_before, priority=1, registry=registry)

    await fn(1)
    assert calls == ["sync_before", "async_before", "body"]


async def test_async_around_can_await_inner(registry: HookRegistry) -> None:
    @around("guarded", registry=registry)
    async def guard(inner, x: int) -> int:
        if x < 0:
            return -1
        return (await inner(x)) + 100

    @hook("guarded", registry=registry)
    async def work(x: int) -> int:
        return x * 2

    assert await work(5) == 110
    assert await work(-1) == -1


async def test_atrigger_awaits_async_listeners(registry: HookRegistry) -> None:
    @on("evt", registry=registry)
    async def a(x: int) -> int:
        await asyncio.sleep(0)
        return x + 1

    @on("evt", registry=registry)
    def b(x: int) -> int:
        return x + 10

    results = await atrigger("evt", 5, registry=registry)
    # priority order is registration order at equal priority
    assert sorted(results) == [6, 15]


async def test_atrigger_returns_in_priority_order(registry: HookRegistry) -> None:
    @on("evt", priority=1, registry=registry)
    async def low() -> str:
        return "low"

    @on("evt", priority=10, registry=registry)
    async def high() -> str:
        return "high"

    assert await atrigger("evt", registry=registry) == ["high", "low"]


def test_trigger_raises_on_async_listener(registry: HookRegistry) -> None:
    @on("evt", registry=registry)
    async def listener() -> None:
        pass

    with pytest.raises(RuntimeError, match="async"):
        trigger("evt", registry=registry)


def test_sync_wrapper_raises_on_async_before_hook(registry: HookRegistry) -> None:
    @hook("s", registry=registry)
    def fn() -> int:
        return 1

    @before("s", registry=registry)
    async def bad() -> None:
        pass

    with pytest.raises(RuntimeError, match="Async before hook"):
        fn()


def test_sync_wrapper_raises_on_async_after_hook(registry: HookRegistry) -> None:
    @hook("s", registry=registry)
    def fn() -> int:
        return 1

    @after("s", registry=registry)
    async def bad(result: int) -> None:
        pass

    with pytest.raises(RuntimeError, match="Async after hook"):
        fn()


def test_sync_wrapper_raises_on_async_around_hook(registry: HookRegistry) -> None:
    @hook("s", registry=registry)
    def fn() -> int:
        return 1

    @around("s", registry=registry)
    async def bad(inner) -> int:
        return inner()

    with pytest.raises(RuntimeError, match="Async around"):
        fn()


async def test_async_wrapper_raises_on_sync_around_hook(registry: HookRegistry) -> None:
    @hook("a", registry=registry)
    async def fn() -> int:
        return 1

    @around("a", registry=registry)
    def bad(inner) -> int:
        return 1

    with pytest.raises(RuntimeError, match="Sync around"):
        await fn()


async def test_hookpoint_atrigger_validates_and_awaits(registry: HookRegistry) -> None:
    class Event(BaseModel):
        n: int

    point: HookPoint[Event] = HookPoint("e", Event, registry=registry)

    @point.listen
    async def listener(event: Event) -> int:
        await asyncio.sleep(0)
        return event.n * 3

    assert await point.atrigger(n=4) == [12]
    assert await point.atrigger({"n": 5}) == [15]
    assert await point.atrigger(Event(n=6)) == [18]


async def test_async_function_reference_target(registry: HookRegistry) -> None:
    @hook(registry=registry)
    async def workflow(n: int) -> int:
        return n + 1

    seen: list[int] = []

    @on(workflow, registry=registry)
    async def listener(value: int) -> None:
        seen.append(value)

    await atrigger(workflow, 42, registry=registry)
    assert seen == [42]
