from __future__ import annotations

import pytest

from pyhooky import HookRegistry, after, around, before, hook, on, trigger


@pytest.fixture
def registry() -> HookRegistry:
    return HookRegistry()


def test_before_hook_fires(registry: HookRegistry) -> None:
    calls: list[str] = []

    @before("greet", registry=registry)
    def log(name: str) -> None:
        calls.append(f"before:{name}")

    @hook("greet", registry=registry)
    def greet(name: str) -> str:
        calls.append(f"greet:{name}")
        return f"hi {name}"

    assert greet("ada") == "hi ada"
    assert calls == ["before:ada", "greet:ada"]


def test_after_hook_receives_result(registry: HookRegistry) -> None:
    seen: list[object] = []

    @after("compute", registry=registry)
    def capture(result: int, x: int) -> None:
        seen.append((result, x))

    @hook("compute", registry=registry)
    def compute(x: int) -> int:
        return x * 2

    compute(21)
    assert seen == [(42, 21)]


def test_around_can_short_circuit(registry: HookRegistry) -> None:
    @around("maybe", registry=registry)
    def guard(inner, x: int) -> int:
        if x < 0:
            return 0
        return inner(x)

    @hook("maybe", registry=registry)
    def maybe(x: int) -> int:
        return x + 1

    assert maybe(5) == 6
    assert maybe(-1) == 0


def test_priority_order(registry: HookRegistry) -> None:
    calls: list[str] = []

    @before("t", priority=1, registry=registry)
    def low() -> None:
        calls.append("low")

    @before("t", priority=10, registry=registry)
    def high() -> None:
        calls.append("high")

    @hook("t", registry=registry)
    def target() -> None:
        return None

    target()
    assert calls == ["high", "low"]


def test_clear(registry: HookRegistry) -> None:
    @before("x", registry=registry)
    def h() -> None:
        raise AssertionError("should not fire")

    registry.clear("x")

    @hook("x", registry=registry)
    def fn() -> str:
        return "ok"

    assert fn() == "ok"


def test_non_decorator_module_level(registry: HookRegistry) -> None:
    calls: list[str] = []

    def log(name: str) -> None:
        calls.append(f"before:{name}")

    def greet(name: str) -> str:
        calls.append(f"greet:{name}")
        return f"hi {name}"

    before("greet", log, registry=registry)
    wrapped = hook("greet", greet, registry=registry)

    assert wrapped("ada") == "hi ada"
    assert calls == ["before:ada", "greet:ada"]


def test_non_decorator_registry_methods(registry: HookRegistry) -> None:
    seen: list[object] = []

    def capture(result: int, x: int) -> None:
        seen.append((result, x))

    def guard(inner, x: int) -> int:
        return 0 if x < 0 else inner(x)

    def compute(x: int) -> int:
        return x * 2

    registry.add_after("compute", capture)
    registry.add_around("compute", guard)
    wrapped = registry.wrap("compute", compute)

    assert wrapped(5) == 10
    assert wrapped(-3) == 0
    assert seen == [(10, 5), (0, -3)]


def test_trigger_fires_listeners(registry: HookRegistry) -> None:
    seen: list[tuple[str, int]] = []

    @on("step", registry=registry)
    def listener(name: str, n: int) -> str:
        seen.append((name, n))
        return f"{name}:{n}"

    def workflow(n: int) -> int:
        trigger("step", "start", n, registry=registry)
        result = n + 1
        trigger("step", "done", result, registry=registry)
        return result

    assert workflow(10) == 11
    assert seen == [("start", 10), ("done", 11)]


def test_trigger_returns_listener_results(registry: HookRegistry) -> None:
    registry.add_listener("calc", lambda x: x * 2)
    registry.add_listener("calc", lambda x: x + 100, priority=-1)

    assert registry.trigger("calc", 5) == [10, 105]


def test_trigger_ignores_wrapping_hooks(registry: HookRegistry) -> None:
    calls: list[str] = []

    @before("evt", registry=registry)
    def b() -> None:
        calls.append("before")

    @on("evt", registry=registry)
    def listener() -> None:
        calls.append("listener")

    registry.trigger("evt")
    assert calls == ["listener"]


def test_on_non_decorator(registry: HookRegistry) -> None:
    def fn(x: int) -> int:
        return x * 3

    on("mul", fn, registry=registry)
    assert registry.trigger("mul", 4) == [12]


def test_hook_bare_decorator_auto_target() -> None:
    # @hook (bare) and the matching before() both target the default registry,
    # so the registration round-trips even without an explicit name.
    from pyhooky import get_default_registry

    default = get_default_registry()

    @hook
    def greet_bare(name: str) -> str:
        return f"hi {name}"

    target_name: str = greet_bare.__hook_target__  # type: ignore[attr-defined]
    assert target_name.endswith("greet_bare")

    calls: list[str] = []
    before(greet_bare, lambda name: calls.append(f"b:{name}"))

    try:
        assert greet_bare("ada") == "hi ada"
        assert calls == ["b:ada"]
    finally:
        default.clear(greet_bare)


def test_hook_called_no_args_auto_target() -> None:
    from pyhooky import get_default_registry

    @hook()
    def fn(x: int) -> int:
        return x + 1

    calls: list[int] = []
    after(fn, lambda result, x: calls.append(result))

    try:
        assert fn(10) == 11
        assert calls == [11]
    finally:
        get_default_registry().clear(fn)


def test_target_by_function_reference(registry: HookRegistry) -> None:
    @hook("checkout", registry=registry)
    def checkout(cart: str) -> str:
        return f"ok:{cart}"

    seen: list[str] = []

    # Pass the wrapped function instead of the name string
    @before(checkout, registry=registry)
    def log(cart: str) -> None:
        seen.append(cart)

    checkout("cart1")
    assert seen == ["cart1"]


def test_resolve_target_rejects_unhooked_callable() -> None:
    def random_fn() -> None:
        pass

    with pytest.raises(TypeError, match="__hook_target__"):
        before(random_fn, lambda: None)


def test_trigger_via_function_reference(registry: HookRegistry) -> None:
    @hook(registry=registry)
    def workflow(n: int) -> int:
        trigger(workflow, "step", n, registry=registry)
        return n

    seen: list[tuple[str, int]] = []
    on(workflow, lambda step, n: seen.append((step, n)), registry=registry)

    workflow(7)
    assert seen == [("step", 7)]


def test_remove(registry: HookRegistry) -> None:
    calls: list[str] = []

    def h1() -> None:
        calls.append("h1")

    def h2() -> None:
        calls.append("h2")

    registry.add_before("t", h1)
    registry.add_before("t", h2)

    assert registry.remove("t", h1) is True
    assert registry.remove("t", h1) is False

    wrapped = registry.wrap("t", lambda: None)
    wrapped()
    assert calls == ["h2"]
