"""Tests for the v0.2 fixes: around priority order, duplicate rejection, tags,
introspection, registry scoping, and repr.
"""

from __future__ import annotations

import asyncio

import pytest

from pyhooky import (
    Hook,
    HookKind,
    HookRegistry,
    around,
    before,
    get_default_registry,
    hook,
    on,
    set_default_registry,
    use_registry,
)


@pytest.fixture
def registry() -> HookRegistry:
    return HookRegistry()


# ---------- #1: around-hook priority order ----------


def test_around_high_priority_fires_first(registry: HookRegistry) -> None:
    calls: list[str] = []

    @around("t", priority=10, registry=registry)
    def high(inner, x: int) -> int:
        calls.append("high:in")
        result = inner(x)
        calls.append("high:out")
        return result

    @around("t", priority=1, registry=registry)
    def low(inner, x: int) -> int:
        calls.append("low:in")
        result = inner(x)
        calls.append("low:out")
        return result

    @hook("t", registry=registry)
    def body(x: int) -> int:
        calls.append("body")
        return x

    body(1)
    # High priority is outermost — fires first, exits last.
    assert calls == ["high:in", "low:in", "body", "low:out", "high:out"]


async def test_around_high_priority_fires_first_async(registry: HookRegistry) -> None:
    calls: list[str] = []

    @around("t", priority=10, registry=registry)
    async def high(inner, x: int) -> int:
        calls.append("high:in")
        result = await inner(x)
        calls.append("high:out")
        return result

    @around("t", priority=1, registry=registry)
    async def low(inner, x: int) -> int:
        calls.append("low:in")
        result = await inner(x)
        calls.append("low:out")
        return result

    @hook("t", registry=registry)
    async def body(x: int) -> int:
        calls.append("body")
        return x

    await body(1)
    assert calls == ["high:in", "low:in", "body", "low:out", "high:out"]


# ---------- #2: duplicate rejection ----------


def test_duplicate_registration_rejected(registry: HookRegistry) -> None:
    def h() -> None:
        pass

    registry.add_before("t", h)
    with pytest.raises(ValueError, match="already registered"):
        registry.add_before("t", h)


def test_duplicate_decorator_rejected(registry: HookRegistry) -> None:
    def h() -> None:
        pass

    before("t", h, registry=registry)
    with pytest.raises(ValueError):
        before("t", h, registry=registry)


def test_same_fn_different_kinds_ok(registry: HookRegistry) -> None:
    def h(*args, **kwargs) -> None:
        pass

    registry.add_before("t", h)
    registry.add_after("t", h)  # different kind — allowed


def test_same_fn_different_targets_ok(registry: HookRegistry) -> None:
    def h() -> None:
        pass

    registry.add_before("a", h)
    registry.add_before("b", h)  # different target — allowed


# ---------- #4: RunBefore unknown → RuntimeError ----------
#   (covered in test_priority.py — verified there)


# ---------- #5: snapshot per phase ----------


def test_before_can_register_after_for_same_dispatch(registry: HookRegistry) -> None:
    seen: list[int] = []

    def late_after(result: int) -> None:
        seen.append(result)

    @before("t", registry=registry)
    def setup() -> None:
        registry.add_after("t", late_after)

    @hook("t", registry=registry)
    def body() -> int:
        return 42

    body()
    # late_after was registered by setup() during BEFORE; AFTER takes a fresh
    # snapshot so it picks up the new registration.
    assert seen == [42]


# ---------- #7: introspection ----------


def test_targets_and_all_hooks(registry: HookRegistry) -> None:
    def a() -> None: ...
    def b() -> None: ...

    registry.add_before("t1", a)
    registry.add_after("t1", b)
    registry.add_listener("t2", a)

    assert set(registry.targets()) == {"t1", "t2"}
    all_pairs = registry.all_hooks()
    assert len(all_pairs) == 3
    assert all(isinstance(h, Hook) for _, h in all_pairs)


def test_registry_repr(registry: HookRegistry) -> None:
    def a() -> None: ...

    registry.add_before("t", a)
    text = repr(registry)
    assert "HookRegistry" in text
    assert "targets=1" in text
    assert "hooks=1" in text


def test_registry_repr_with_custom_name() -> None:
    reg = HookRegistry(name="plugin-host")
    assert "plugin-host" in repr(reg)


# ---------- #8: tags ----------


def test_clear_tag_removes_across_targets(registry: HookRegistry) -> None:
    def a() -> None: ...
    def b() -> None: ...
    def c() -> None: ...

    registry.add_before("t1", a, tag="plugin-x")
    registry.add_before("t2", b, tag="plugin-x")
    registry.add_before("t1", c, tag="other")

    removed = registry.clear_tag("plugin-x")
    assert len(removed) == 2
    assert {(t, h.fn) for t, h in removed} == {("t1", a), ("t2", b)}

    remaining = {(t, h.fn) for t, h in registry.all_hooks()}
    assert remaining == {("t1", c)}


def test_hooks_by_tag(registry: HookRegistry) -> None:
    def a() -> None: ...
    def b() -> None: ...

    registry.add_before("t", a, tag="plugin-x")
    registry.add_after("t", b, tag="plugin-x")
    registry.add_listener("evt", a, tag="other")

    hooks_x = registry.hooks_by_tag("plugin-x")
    assert {(t, h.kind) for t, h in hooks_x} == {("t", HookKind.BEFORE), ("t", HookKind.AFTER)}


def test_tag_via_module_helper(registry: HookRegistry) -> None:
    @before("t", tag="audit", registry=registry)
    def h() -> None: ...

    assert len(registry.hooks_by_tag("audit")) == 1


def test_tag_round_trips_through_priority_change(registry: HookRegistry) -> None:
    from pyhooky import PriorityBoost

    @before("t", priority=1, tag="audit", registry=registry)
    def h() -> None:
        if not getattr(h, "_done", False):
            h._done = True  # type: ignore[attr-defined]
            raise PriorityBoost(by=10)

    @hook("t", registry=registry)
    def body() -> None:
        pass

    body()
    # tag must survive the priority update (which replaces the Hook instance)
    assert registry.hooks_by_tag("audit")[0][1].priority == 11


# ---------- #9: registry scoping ----------


def test_use_registry_scopes_module_helpers() -> None:
    scoped = HookRegistry(name="scoped")
    calls: list[str] = []

    with use_registry(scoped):

        @on("evt")
        def listener(x: int) -> None:
            calls.append(f"scoped:{x}")

    # Inside scope, the helper hit `scoped` — verify it's not on the default
    assert len(scoped.hooks_by_tag("__never__")) == 0
    assert len(scoped.all_hooks()) == 1
    assert listener not in [h.fn for _, h in get_default_registry().all_hooks()]


def test_use_registry_unwinds_on_exit() -> None:
    scoped = HookRegistry()
    with use_registry(scoped):
        assert get_default_registry() is scoped
    assert get_default_registry() is not scoped


def test_set_default_registry_swaps_process_wide() -> None:
    new = HookRegistry(name="swapped")
    previous = set_default_registry(new)
    try:
        assert get_default_registry() is new
    finally:
        set_default_registry(previous)
    assert get_default_registry() is previous


async def test_use_registry_works_across_asyncio_tasks() -> None:
    scoped = HookRegistry()
    seen_scoped: list[bool] = []

    async def child() -> None:
        # ContextVar should propagate to the child task.
        seen_scoped.append(get_default_registry() is scoped)

    with use_registry(scoped):
        await asyncio.gather(child(), child())
    assert seen_scoped == [True, True]


# ---------- #11: composed-around wraps don't pollute __wrapped__ chain ----------


def test_wrapped_attribute_points_to_original(registry: HookRegistry) -> None:
    @around("t", registry=registry)
    def outer(inner, *a, **kw):
        return inner(*a, **kw)

    @around("t", registry=registry)
    def inner_around(inner, *a, **kw):
        return inner(*a, **kw)

    def body() -> int:
        return 7

    body.__name__ = "body"
    wrapped = registry.wrap("t", body)
    # The wrapper carries body's metadata directly — the composition layers
    # no longer @wraps inside, so `wrapped.__wrapped__` points at body.
    assert wrapped.__wrapped__ is body  # type: ignore[attr-defined]
