"""Tests covering the fixes/additions from the v0.1 review pass.

Each block corresponds to one of the issues tagged in the review plan.
"""

from __future__ import annotations

import asyncio
import warnings

import pytest

from pyhooky import (
    HookKind,
    HookRegistry,
    PriorityBoost,
    after,
    around,
    before,
    disabled,
    hook,
    on,
    set_default_registry,
    use_registry,
)


@pytest.fixture
def registry() -> HookRegistry:
    return HookRegistry()


# ---------- BUG-1: sync dispatch must not drop awaitables ----------


def test_sync_before_rejects_callable_returning_coroutine(registry: HookRegistry) -> None:
    async def _impl(x: int) -> None:
        pass

    class Wrapper:
        # iscoroutinefunction() returns False for an instance whose __call__
        # is a sync method that just returns a coroutine — pre-fix the sync
        # phase would call this and silently drop the coroutine.
        def __call__(self, x: int):
            return _impl(x)

    registry.add_before("t", Wrapper())

    @hook("t", registry=registry)
    def body(x: int) -> int:
        return x

    with pytest.raises(RuntimeError, match="awaitable"):
        body(1)


def test_sync_after_rejects_callable_returning_coroutine(registry: HookRegistry) -> None:
    async def _impl(result: int) -> None:
        pass

    class CallableReturningCoroutine:
        def __call__(self, result: int):
            return _impl(result)

    registry.add_after("t", CallableReturningCoroutine())

    @hook("t", registry=registry)
    def body() -> int:
        return 1

    with pytest.raises(RuntimeError, match="awaitable"):
        body()


# ---------- BUG-2: remove() with kind disambiguates ----------


def test_remove_with_kind_targets_specific_registration(registry: HookRegistry) -> None:
    fired: list[str] = []

    def both(*args, **kwargs) -> None:
        fired.append("called")

    registry.add_before("t", both)
    registry.add_after("t", both)

    assert registry.remove("t", both, kind=HookKind.BEFORE) is True
    # Only BEFORE removed; AFTER still there.
    remaining = registry.get("t")
    assert len(remaining) == 1
    assert remaining[0].kind == HookKind.AFTER


def test_remove_without_kind_keeps_old_first_match_behavior(registry: HookRegistry) -> None:
    def fn() -> None:
        pass

    registry.add_before("t", fn)
    registry.add_after("t", fn)
    assert registry.remove("t", fn) is True
    # Exactly one of the two remains.
    assert len(registry.get("t")) == 1


# ---------- FOOTGUN-1: HookPoint resolves registry lazily ----------


def test_hookpoint_follows_use_registry_after_construction() -> None:
    from pydantic import BaseModel

    from pyhooky import HookPoint

    class E(BaseModel):
        n: int

    point: HookPoint[E] = HookPoint("e", E)  # no explicit registry

    scoped = HookRegistry(name="scoped")
    seen: list[int] = []

    with use_registry(scoped):

        @point.listen
        def listener(event: E) -> None:
            seen.append(event.n)

        point.trigger(n=42)

    assert seen == [42]
    assert len(scoped.all_hooks()) == 1


# ---------- FOOTGUN-2: @hook resolves registry lazily ----------


def test_hook_follows_use_registry_per_call() -> None:
    @hook
    def fn(x: int) -> int:
        return x

    scoped = HookRegistry(name="scoped")
    seen: list[int] = []

    with use_registry(scoped):
        before(fn, lambda x: seen.append(x))
        assert fn(7) == 7

    assert seen == [7]
    # Default registry stayed clean
    from pyhooky import get_default_registry

    assert all(t != fn.__hook_target__ for t, _ in get_default_registry().all_hooks())  # type: ignore[attr-defined]


# ---------- FOOTGUN-3: cross-registry mismatch detection ----------


def test_register_on_wrong_registry_raises(registry: HookRegistry) -> None:
    other = HookRegistry(name="other")

    @hook("t", registry=registry)
    def body() -> None:
        pass

    with pytest.raises(ValueError, match="bound to"):
        before(body, lambda: None, registry=other)


def test_register_with_no_explicit_registry_routes_to_target(registry: HookRegistry) -> None:
    @hook("t", registry=registry)
    def body() -> int:
        return 1

    seen: list[int] = []
    # No registry= kwarg — should auto-route to the registry body is bound to.
    before(body, lambda: seen.append(1))

    body()
    assert seen == [1]


# ---------- FOOTGUN-5/6: set_default_registry safety ----------


def test_set_default_registry_warns_inside_use_registry() -> None:
    scoped = HookRegistry()
    new = HookRegistry()

    with use_registry(scoped), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        previous = set_default_registry(new)
        set_default_registry(previous)  # restore

    msgs = [str(w.message) for w in caught]
    assert any("use_registry" in m for m in msgs)


# ---------- POLISH-1: around raising PriorityChange yields a clear error ----------


def test_around_priority_change_raises_runtime_error(registry: HookRegistry) -> None:
    @around("t", registry=registry)
    def guard(inner) -> None:
        raise PriorityBoost(by=5)

    @hook("t", registry=registry)
    def body() -> None:
        return None

    with pytest.raises(RuntimeError, match="around hook"):
        body()


# ---------- POLISH-2: clear_tag returns the removed pairs ----------


def test_clear_tag_returns_removed_hooks(registry: HookRegistry) -> None:
    def a() -> None: ...
    def b() -> None: ...

    registry.add_before("t1", a, tag="x")
    registry.add_after("t2", b, tag="x")

    removed = registry.clear_tag("x")
    assert {(t, h.kind, h.fn) for t, h in removed} == {
        ("t1", HookKind.BEFORE, a),
        ("t2", HookKind.AFTER, b),
    }


# ---------- POLISH-3: disabled(tag) hides matching hooks during dispatch ----------


def test_disabled_hides_tagged_hooks(registry: HookRegistry) -> None:
    fired: list[str] = []

    @before("t", tag="audit", registry=registry)
    def audit() -> None:
        fired.append("audit")

    @hook("t", registry=registry)
    def body() -> None:
        fired.append("body")

    with disabled("audit"):
        body()
    assert fired == ["body"]

    fired.clear()
    body()
    assert fired == ["audit", "body"]


def test_disabled_multiple_tags(registry: HookRegistry) -> None:
    fired: list[str] = []

    before("t", lambda: fired.append("a"), tag="a", registry=registry)
    before("t", lambda: fired.append("b"), tag="b", registry=registry)
    before("t", lambda: fired.append("c"), tag="c", registry=registry)

    @hook("t", registry=registry)
    def body() -> None:
        pass

    with disabled("a", "c"):
        body()
    assert fired == ["b"]


def test_disabled_listener_via_trigger(registry: HookRegistry) -> None:
    fired: list[str] = []

    @on("evt", tag="plugin", registry=registry)
    def listener() -> None:
        fired.append("plugin")

    @on("evt", registry=registry)
    def other() -> None:
        fired.append("other")

    from pyhooky import trigger

    with disabled("plugin"):
        trigger("evt", registry=registry)
    assert fired == ["other"]


async def test_disabled_propagates_across_asyncio_tasks(registry: HookRegistry) -> None:
    fired: list[str] = []

    @before("t", tag="x", registry=registry)
    async def audit() -> None:
        fired.append("audit")

    @hook("t", registry=registry)
    async def body() -> None:
        fired.append("body")

    async def child() -> None:
        await body()

    with disabled("x"):
        await asyncio.gather(child(), child())

    # Both child tasks should have seen the contextvar.
    assert fired == ["body", "body"]


# ---------- POLISH-6: better error for unhooked callable target ----------


def test_unhooked_callable_error_mentions_string_form() -> None:
    def not_yet_hooked() -> None:
        pass

    with pytest.raises(TypeError, match="string"):
        before(not_yet_hooked, lambda: None)


# ---------- tags() / dump_target() ----------


def test_tags_returns_unique_sorted(registry: HookRegistry) -> None:
    before("t", lambda: None, tag="b", registry=registry)
    before("t", lambda: None, tag="a", registry=registry)
    after("t", lambda r: None, tag="a", registry=registry)
    registry.add_listener("evt", lambda: None)  # untagged

    assert registry.tags() == ["a", "b"]


def test_dump_target_returns_priority_order(registry: HookRegistry) -> None:
    def low() -> None: ...
    def high() -> None: ...

    before("t", low, priority=1, registry=registry)
    before("t", high, priority=10, registry=registry)

    dumped = registry.dump_target("t")
    assert [h.fn for h in dumped] == [high, low]


# ---------- POLISH-4 (regression): registration order at equal priority ----------


def test_equal_priority_registration_order_preserved(registry: HookRegistry) -> None:
    """bisect-based insort must keep the same equal-priority stability as
    the previous full-sort approach."""
    calls: list[str] = []

    @before("t", registry=registry)
    def first() -> None:
        calls.append("first")

    @before("t", registry=registry)
    def second() -> None:
        calls.append("second")

    @before("t", registry=registry)
    def third() -> None:
        calls.append("third")

    @hook("t", registry=registry)
    def body() -> None:
        return None

    body()
    assert calls == ["first", "second", "third"]
