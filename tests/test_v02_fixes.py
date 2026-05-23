"""Tests for the v0.2 fixes:
- tag_scope context manager
- on_hook_error policy ('raise' vs 'log')
- lazy @hook binds to its registration registry on first attach
- priority_of public method
- set_default_registry validates type
- _auto_target_name uniques lambdas
- HookPoint.listen accepts tag / on_priority_change
- HookPoint.remove accepts kind=
- AROUND raising PriorityChange yields RuntimeError mentioning the around fn
  (and body-raised PriorityChange is NOT misreported as around)
"""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel

from pyhooky import (
    HookKind,
    HookPoint,
    HookRegistry,
    PriorityBoost,
    after,
    around,
    before,
    get_default_registry,
    hook,
    on,
    set_default_registry,
    tag_scope,
    use_registry,
)


@pytest.fixture
def registry() -> HookRegistry:
    return HookRegistry()


# ---------- tag_scope ----------


def test_tag_scope_auto_tags_registrations(registry: HookRegistry) -> None:
    with tag_scope("plugin-x"):

        @before("t", registry=registry)
        def b() -> None: ...

        @after("t", registry=registry)
        def a(result: object) -> None: ...

        @on("evt", registry=registry)
        def listener() -> None: ...

    hooks = registry.hooks_by_tag("plugin-x")
    assert len(hooks) == 3
    assert {h.kind for _, h in hooks} == {HookKind.BEFORE, HookKind.AFTER, HookKind.LISTENER}


def test_tag_scope_explicit_tag_wins(registry: HookRegistry) -> None:
    with tag_scope("outer"):

        @before("t", tag="inner-explicit", registry=registry)
        def b() -> None: ...

    assert registry.hooks_by_tag("inner-explicit")
    assert not registry.hooks_by_tag("outer")


def test_tag_scope_nested_uses_innermost(registry: HookRegistry) -> None:
    with tag_scope("outer"):

        @before("t", registry=registry)
        def outer() -> None: ...

        with tag_scope("inner"):

            @before("t", registry=registry)
            def inner() -> None: ...

        @before("t2", registry=registry)
        def outer2() -> None: ...

    outer_pairs = registry.hooks_by_tag("outer")
    inner_pairs = registry.hooks_by_tag("inner")
    assert {h.fn for _, h in outer_pairs} == {outer, outer2}
    assert {h.fn for _, h in inner_pairs} == {inner}


def test_tag_scope_clears_all_with_clear_tag(registry: HookRegistry) -> None:
    with tag_scope("my-plugin"):

        @before("t1", registry=registry)
        def b() -> None: ...
        @after("t2", registry=registry)
        def a(r: object) -> None: ...

    removed = registry.clear_tag("my-plugin")
    assert len(removed) == 2
    assert registry.all_hooks() == []


# ---------- on_hook_error ----------


def test_on_hook_error_raise_default(registry: HookRegistry) -> None:
    @before("t", registry=registry)
    def b() -> None:
        raise ValueError("buggy before")

    @hook("t", registry=registry)
    def body() -> int:
        return 1

    with pytest.raises(ValueError, match="buggy before"):
        body()


def test_on_hook_error_log_continues(caplog: pytest.LogCaptureFixture) -> None:
    reg = HookRegistry(on_hook_error="log")
    fired: list[str] = []

    @before("t", registry=reg)
    def buggy() -> None:
        raise ValueError("nope")

    @before("t", priority=-1, registry=reg)
    def ok() -> None:
        fired.append("ok")

    @hook("t", registry=reg)
    def body() -> int:
        fired.append("body")
        return 1

    with caplog.at_level(logging.ERROR, logger="pyhooky"):
        assert body() == 1

    assert fired == ["ok", "body"]
    assert any(
        "ValueError" in rec.getMessage() or "ValueError" in str(rec.exc_info)
        for rec in caplog.records
    )


def test_on_hook_error_log_applies_to_after(caplog: pytest.LogCaptureFixture) -> None:
    reg = HookRegistry(on_hook_error="log")

    @after("t", registry=reg)
    def buggy(result: int) -> None:
        raise ValueError("after fail")

    @hook("t", registry=reg)
    def body() -> int:
        return 42

    with caplog.at_level(logging.ERROR, logger="pyhooky"):
        assert body() == 42


def test_on_hook_error_invalid_value_rejected() -> None:
    with pytest.raises(ValueError, match="on_hook_error"):
        HookRegistry(on_hook_error="bogus")  # type: ignore[arg-type]


# ---------- lazy @hook binding ----------


def test_lazy_hook_binds_on_first_attach_outside_scope() -> None:
    """Hook registered outside `use_registry` must still fire when the wrapped
    fn is called inside `use_registry(scoped)`."""
    reg_a = HookRegistry(name="A")
    reg_b = HookRegistry(name="B")
    seen: list[str] = []

    with use_registry(reg_a):

        @hook
        def fn() -> int:
            return 1

    # Attach hook outside any scope — should land in default? No:
    # _maybe_bind_lazy_wrapper hasn't bound yet (no hooks attached), and the
    # active registry now is default. So before() registers on default.
    # The wrapper is then bound to default.
    from pyhooky import get_default_registry as gdr

    default = gdr()

    before(fn, lambda: seen.append("first"))
    assert fn.__hook_registry__ is default  # type: ignore[attr-defined]

    # Now switch context — the wrapper should still dispatch via default.
    with use_registry(reg_b):
        fn()

    assert seen == ["first"]
    default.clear(fn)


def test_lazy_hook_second_registration_to_different_registry_rejected() -> None:
    @hook
    def fn() -> int:
        return 1

    other = HookRegistry(name="other")
    before(fn, lambda: None)  # binds fn to default

    with pytest.raises(ValueError, match="bound to"):
        before(fn, lambda: None, registry=other)

    get_default_registry().clear(fn)


# ---------- priority_of public ----------


def test_priority_of_public_method(registry: HookRegistry) -> None:
    @before("t", priority=42, registry=registry)
    def h() -> None: ...

    assert registry.priority_of("t", h) == 42

    with pytest.raises(LookupError):
        registry.priority_of("t", lambda: None)


def test_priority_of_works_with_callable_target(registry: HookRegistry) -> None:
    @hook("t", registry=registry)
    def body() -> int:
        return 1

    def h() -> None: ...

    before(body, h, priority=7)

    assert registry.priority_of(body, h) == 7


# ---------- set_default_registry validation ----------


def test_set_default_registry_rejects_non_registry() -> None:
    with pytest.raises(TypeError, match="HookRegistry"):
        set_default_registry(None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="HookRegistry"):
        set_default_registry("not a registry")  # type: ignore[arg-type]


# ---------- lambda uniqueness ----------


def test_auto_target_lambdas_get_unique_names() -> None:
    h1 = hook(lambda x: x + 1)
    h2 = hook(lambda x: x + 2)

    assert h1.__hook_target__ != h2.__hook_target__  # type: ignore[attr-defined]
    assert "<lambda>" in h1.__hook_target__  # type: ignore[attr-defined]
    assert "#" in h1.__hook_target__  # type: ignore[attr-defined]


# ---------- HookPoint.listen kwargs ----------


def test_hookpoint_listen_accepts_tag(registry: HookRegistry) -> None:
    class E(BaseModel):
        n: int

    point: HookPoint[E] = HookPoint("e", E, registry=registry)

    @point.listen(tag="audit")
    def lst(event: E) -> int:
        return event.n

    assert len(registry.hooks_by_tag("audit")) == 1


def test_hookpoint_listen_accepts_on_priority_change(registry: HookRegistry) -> None:
    class E(BaseModel):
        n: int

    point: HookPoint[E] = HookPoint("e", E, registry=registry)

    @point.listen(on_priority_change="propagate")
    def lst(event: E) -> int:
        raise PriorityBoost(by=1)

    with pytest.raises(PriorityBoost):
        point.trigger(n=1)


# ---------- HookPoint.remove with kind ----------


def test_hookpoint_remove_with_kind(registry: HookRegistry) -> None:
    class E(BaseModel):
        n: int

    point: HookPoint[E] = HookPoint("e", E, registry=registry)

    def handler(*args: object, **kwargs: object) -> None: ...

    point.listen(handler)
    point.before(handler)

    assert point.remove(handler, kind=HookKind.LISTENER) is True
    remaining = registry.get("e")
    assert len(remaining) == 1
    assert remaining[0].kind == HookKind.BEFORE


# ---------- body-raised PriorityChange not misreported ----------


def test_body_raised_priority_change_not_caught_as_around(registry: HookRegistry) -> None:
    """A PriorityChange escaping from the wrapped body (no around in chain)
    should NOT be wrapped in the 'around hook ...' RuntimeError — that error
    is for around hooks that misuse priority. The body's exception bubbles
    through the ON_ERROR phase like any other exception."""

    seen: list[type[BaseException]] = []

    @hook("t", registry=registry)
    def body() -> int:
        raise PriorityBoost(by=1)

    from pyhooky import on_error

    @on_error("t", registry=registry)
    def err(exc: BaseException) -> None:
        seen.append(type(exc))

    with pytest.raises(PriorityBoost):
        body()
    # ON_ERROR observed the actual PriorityBoost, not a wrapping RuntimeError
    assert seen == [PriorityBoost]


def test_around_priority_change_message_names_the_hook(registry: HookRegistry) -> None:
    @around("t", registry=registry)
    def my_guard(inner) -> int:
        raise PriorityBoost(by=1)

    @hook("t", registry=registry)
    def body() -> int:
        return 1

    with pytest.raises(RuntimeError) as info:
        body()
    assert "around hook" in str(info.value)
    assert isinstance(info.value.__cause__, PriorityBoost)
