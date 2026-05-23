from __future__ import annotations

import contextlib
import logging
import warnings
from bisect import insort
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import wraps
from inspect import isawaitable, iscoroutinefunction
from threading import Lock, RLock
from typing import Any, Literal, ParamSpec, TypeVar, overload

from pyhooky.exceptions import PriorityChange, PriorityChangeBehavior

P = ParamSpec("P")
R = TypeVar("R")

HookFn = Callable[..., Any]
Decorator = Callable[[HookFn], HookFn]
TargetRef = str | Callable[..., Any]
OnHookError = Literal["raise", "log"]

HOOK_TARGET_ATTR = "__hook_target__"
HOOK_REGISTRY_ATTR = "__hook_registry__"

_logger = logging.getLogger("pyhooky")


def _auto_target_name(fn: Callable[..., Any]) -> str:
    module = getattr(fn, "__module__", "") or ""
    qualname = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)
    name = f"{module}.{qualname}" if module else qualname
    # Lambdas all share qualname "<lambda>" (or "Outer.<locals>.<lambda>"), so
    # disambiguate per-object — otherwise two lambdas in the same scope collide
    # on the same target name.
    if qualname.endswith("<lambda>"):
        name = f"{name}#{id(fn):x}"
    return name


def _resolve_target(target: TargetRef) -> str:
    if isinstance(target, str):
        return target
    name = getattr(target, HOOK_TARGET_ATTR, None)
    if name is not None:
        return name
    if callable(target):
        raise TypeError(
            f"{target!r} has no {HOOK_TARGET_ATTR!r}. The function-reference form "
            "(e.g. @before(checkout)) only works after the target has been wrapped "
            "with @hook. If the wrapping happens later or in another module, pass "
            "the target name as a string instead "
            f"(e.g. @before({_auto_target_name(target)!r}))."
        )
    raise TypeError(f"target must be a str or hooked callable, got {type(target).__name__}")


def _target_registry(target: TargetRef) -> HookRegistry | None:
    """Return the registry a wrapped target is bound to, if any.

    Returns ``None`` for string targets, unwrapped callables, and wrappers
    built in lazy mode (no explicit ``registry=`` at decoration time).
    """
    if not callable(target):
        return None
    return getattr(target, HOOK_REGISTRY_ATTR, None)


class HookKind(StrEnum):
    BEFORE = "before"
    AFTER = "after"
    AROUND = "around"
    LISTENER = "listener"
    ON_ERROR = "on_error"


_ASYNC_ERROR_TEMPLATES: dict[HookKind, str] = {
    HookKind.BEFORE: "Async before hook {fn!r} attached to sync target {target!r}.",
    HookKind.AFTER: "Async after hook {fn!r} attached to sync target {target!r}.",
    HookKind.ON_ERROR: ("Async on_error hook {fn!r} attached to sync target {target!r}."),
    HookKind.LISTENER: (
        "Listener {fn!r} for {target!r} is async; use atrigger() from an async context."
    ),
}


_disabled_tags: ContextVar[frozenset[str]] = ContextVar(
    "pyhooky_disabled_tags", default=frozenset()
)
_current_tag_scope: ContextVar[str | None] = ContextVar("pyhooky_current_tag_scope", default=None)


@contextmanager
def disabled(*tags: str) -> Iterator[None]:
    """Temporarily filter hooks whose ``tag`` matches out of dispatch snapshots.

    Scope is the current async task / thread (contextvar-based). Already-loaded
    hooks aren't removed — just hidden from BEFORE/AROUND/AFTER/ON_ERROR and
    listener snapshots while the context is active.
    """
    if not tags:
        yield
        return
    current = _disabled_tags.get()
    token = _disabled_tags.set(current | frozenset(tags))
    try:
        yield
    finally:
        _disabled_tags.reset(token)


@contextmanager
def tag_scope(tag: str) -> Iterator[None]:
    """Auto-tag every registration inside the block with ``tag``.

    Intended for plugin authors — wrap a plugin's setup in
    ``with tag_scope("my-plugin"):`` and every ``@before`` / ``@after`` /
    ``@on`` / ``@on_error`` / ``add_*`` call inside picks up the tag, so the
    whole plugin can be torn down with a single ``registry.clear_tag(name)``.

    Registrations that pass an explicit ``tag=`` keep their explicit value —
    the scope only fills in missing tags. Nested scopes use the innermost tag.
    Implemented via :class:`contextvars.ContextVar`, so it propagates across
    asyncio tasks and threads.
    """
    token = _current_tag_scope.set(tag)
    try:
        yield
    finally:
        _current_tag_scope.reset(token)


def _insort_hook(bucket: list[Hook], hook: Hook) -> None:
    """Insert ``hook`` keeping ``bucket`` sorted by descending priority.

    ``bisect.insort`` with ``key=lambda h: -h.priority`` uses ``bisect_right``
    semantics, so equal-priority hooks insert after existing entries —
    preserving registration order, matching what stable sort gave previously.
    """
    insort(bucket, hook, key=lambda h: -h.priority)


@dataclass(frozen=True, slots=True)
class Hook:
    """A single registration. Snapshots are point-in-time — priority and tag
    reflect the values at the moment the snapshot was taken; later updates
    via :class:`PriorityChange` create a fresh ``Hook`` instance in the bucket
    and do not mutate references already handed out via :meth:`HookRegistry.get`.
    """

    kind: HookKind
    fn: HookFn
    priority: int = 0
    on_priority_change: PriorityChangeBehavior | None = None
    tag: str | None = None


class HookRegistry:
    __slots__ = ("_default_behavior", "_hooks", "_lock", "_on_hook_error", "name")

    def __init__(
        self,
        *,
        name: str = "registry",
        on_priority_change: PriorityChangeBehavior = "continue",
        on_hook_error: OnHookError = "raise",
    ) -> None:
        self._hooks: dict[str, list[Hook]] = {}
        self._lock = RLock()
        self._default_behavior: PriorityChangeBehavior = on_priority_change
        if on_hook_error not in ("raise", "log"):
            raise ValueError(f"on_hook_error must be 'raise' or 'log', got {on_hook_error!r}")
        self._on_hook_error: OnHookError = on_hook_error
        self.name = name

    def __repr__(self) -> str:
        with self._lock:
            target_count = len(self._hooks)
            hook_count = sum(len(b) for b in self._hooks.values())
        return f"HookRegistry(name={self.name!r}, targets={target_count}, hooks={hook_count})"

    # ---------- registration ----------

    def register(self, target: TargetRef, hook: Hook) -> None:
        """Insert ``hook`` into ``target``'s bucket.

        Raises :class:`ValueError` if a hook with the same ``(kind, fn)`` is
        already registered on this target, or if ``target`` is a wrapped
        callable bound to a different registry.

        If an outer :func:`tag_scope` is active and ``hook.tag`` is ``None``,
        the scope's tag is applied — so plugin authors get auto-tagging for
        free.
        """
        self._check_target_registry(target)
        if hook.tag is None:
            scoped_tag = _current_tag_scope.get()
            if scoped_tag is not None:
                hook = replace(hook, tag=scoped_tag)
        name = _resolve_target(target)
        with self._lock:
            bucket = self._hooks.setdefault(name, [])
            for existing in bucket:
                if existing.kind == hook.kind and existing.fn is hook.fn:
                    raise ValueError(
                        f"{hook.fn!r} is already registered as a "
                        f"{hook.kind.value!r} hook on target {name!r}"
                    )
            _insort_hook(bucket, hook)

    def _check_target_registry(self, target: TargetRef) -> None:
        bound = _target_registry(target)
        if bound is not None and bound is not self:
            raise ValueError(
                f"Cannot register on registry {self.name!r}: target {target!r} is "
                f"bound to a different registry ({bound.name!r}). Either pass "
                "the matching registry= argument, use a string target name, or "
                "rewrap the function on this registry."
            )

    def add_before(
        self,
        target: TargetRef,
        fn: HookFn,
        *,
        priority: int = 0,
        on_priority_change: PriorityChangeBehavior | None = None,
        tag: str | None = None,
    ) -> HookFn:
        self.register(target, Hook(HookKind.BEFORE, fn, priority, on_priority_change, tag))
        return fn

    def add_after(
        self,
        target: TargetRef,
        fn: HookFn,
        *,
        priority: int = 0,
        on_priority_change: PriorityChangeBehavior | None = None,
        tag: str | None = None,
    ) -> HookFn:
        self.register(target, Hook(HookKind.AFTER, fn, priority, on_priority_change, tag))
        return fn

    def add_around(
        self,
        target: TargetRef,
        fn: HookFn,
        *,
        priority: int = 0,
        on_priority_change: PriorityChangeBehavior | None = None,
        tag: str | None = None,
    ) -> HookFn:
        self.register(target, Hook(HookKind.AROUND, fn, priority, on_priority_change, tag))
        return fn

    def add_listener(
        self,
        target: TargetRef,
        fn: HookFn,
        *,
        priority: int = 0,
        on_priority_change: PriorityChangeBehavior | None = None,
        tag: str | None = None,
    ) -> HookFn:
        self.register(target, Hook(HookKind.LISTENER, fn, priority, on_priority_change, tag))
        return fn

    def add_on_error(
        self,
        target: TargetRef,
        fn: HookFn,
        *,
        priority: int = 0,
        on_priority_change: PriorityChangeBehavior | None = None,
        tag: str | None = None,
    ) -> HookFn:
        """Register an on_error hook for ``target``.

        On_error hooks fire when the wrapped function (or anything inside an
        AROUND chain) raises. Signature is ``(exc, *original_args,
        **original_kwargs)``. They are **observational only** — they cannot
        swallow the exception (use AROUND for that). After all on_error hooks
        run, the original exception re-raises.
        """
        self.register(target, Hook(HookKind.ON_ERROR, fn, priority, on_priority_change, tag))
        return fn

    # ---------- listener dispatch ----------

    def trigger(self, target: TargetRef, *args: Any, **kwargs: Any) -> list[Any]:
        """Fire all listeners registered for ``target`` and return their results.

        Listeners that raise :class:`PriorityChange` contribute no result, so
        ``len(returned) <= number_of_registered_listeners``.

        Raises :class:`RuntimeError` if any registered listener is an async
        function — call :meth:`atrigger` from an async context instead.
        """
        name = _resolve_target(target)
        snapshot = self._snapshot(name)
        return self._run_phase_sync(
            target=name,
            snapshot=snapshot,
            phase_kind=HookKind.LISTENER,
            args=args,
            kwargs=kwargs,
            collect=True,
        )

    async def atrigger(self, target: TargetRef, *args: Any, **kwargs: Any) -> list[Any]:
        """Async-aware version of :meth:`trigger`. See its docstring for the
        result-length caveat with :class:`PriorityChange`.
        """
        name = _resolve_target(target)
        snapshot = self._snapshot(name)
        return await self._run_phase_async(
            target=name,
            snapshot=snapshot,
            phase_kind=HookKind.LISTENER,
            args=args,
            kwargs=kwargs,
            collect=True,
        )

    # ---------- removal / introspection ----------

    def remove(
        self,
        target: TargetRef,
        fn: HookFn,
        *,
        kind: HookKind | None = None,
    ) -> bool:
        """Remove the first hook matching ``fn`` (and ``kind`` if given) on ``target``.

        Pass ``kind`` when the same callable is registered as multiple kinds
        (e.g. both before and after) on the same target — without it, the
        first match in priority order is removed regardless of kind.
        """
        name = _resolve_target(target)
        with self._lock:
            bucket = self._hooks.get(name)
            if not bucket:
                return False
            for i, h in enumerate(bucket):
                if h.fn is fn and (kind is None or h.kind == kind):
                    del bucket[i]
                    if not bucket:
                        del self._hooks[name]
                    return True
        return False

    def clear(self, target: TargetRef | None = None) -> None:
        with self._lock:
            if target is None:
                self._hooks.clear()
            else:
                self._hooks.pop(_resolve_target(target), None)

    def clear_tag(self, tag: str) -> list[tuple[str, Hook]]:
        """Remove every hook whose ``tag`` matches across all targets.

        Returns the ``(target, hook)`` pairs that were removed, so callers can
        log or otherwise verify exactly which registrations were torn down.
        """
        removed: list[tuple[str, Hook]] = []
        with self._lock:
            for target_name in list(self._hooks.keys()):
                bucket = self._hooks[target_name]
                kept: list[Hook] = []
                for h in bucket:
                    if h.tag == tag:
                        removed.append((target_name, h))
                    else:
                        kept.append(h)
                if kept:
                    self._hooks[target_name] = kept
                else:
                    del self._hooks[target_name]
        return removed

    def get(self, target: TargetRef) -> list[Hook]:
        with self._lock:
            return list(self._hooks.get(_resolve_target(target), ()))

    def targets(self) -> list[str]:
        """Return every target name that currently has at least one registration."""
        with self._lock:
            return list(self._hooks.keys())

    def tags(self) -> list[str]:
        """Return the sorted set of non-``None`` tags across all registrations."""
        seen: set[str] = set()
        with self._lock:
            for bucket in self._hooks.values():
                for h in bucket:
                    if h.tag is not None:
                        seen.add(h.tag)
        return sorted(seen)

    def all_hooks(self) -> list[tuple[str, Hook]]:
        """Return ``(target, hook)`` for every registration, snapshot-style."""
        with self._lock:
            return [(t, h) for t, bucket in self._hooks.items() for h in bucket]

    def hooks_by_tag(self, tag: str) -> list[tuple[str, Hook]]:
        with self._lock:
            return [(t, h) for t, bucket in self._hooks.items() for h in bucket if h.tag == tag]

    def dump_target(self, target: TargetRef) -> list[Hook]:
        """Return ``target``'s hooks in firing order (highest priority first).

        Identical to :meth:`get`; provided as a named affordance for the
        common debugging case of "what fires when I call this target?".
        """
        return self.get(target)

    def disabled(self, *tags: str) -> AbstractContextManager[None]:
        """Temporarily hide hooks whose ``tag`` matches from dispatch snapshots.

        Convenience wrapper around the module-level :func:`disabled`; uses the
        same contextvar so the effect crosses registries.
        """
        return disabled(*tags)

    # ---------- snapshot helpers ----------

    def _snapshot(self, target: str) -> tuple[Hook, ...]:
        disabled_tags = _disabled_tags.get()
        with self._lock:
            bucket = self._hooks.get(target, ())
            if not disabled_tags:
                return tuple(bucket)
            return tuple(h for h in bucket if h.tag not in disabled_tags)

    def priority_of(self, target: TargetRef, fn: HookFn) -> int:
        """Return the current priority of ``fn`` on ``target``.

        Raises :class:`LookupError` if ``fn`` is not registered on the
        resolved target. Useful for plugin coordination — e.g. picking a
        priority just above some other plugin's hook.
        """
        return self._priority_of(_resolve_target(target), fn)

    def _priority_of(self, target: str, fn: HookFn) -> int:
        with self._lock:
            bucket = self._hooks.get(target, ())
            for h in bucket:
                if h.fn is fn:
                    return h.priority
        raise LookupError(f"{fn!r} is not registered on target {target!r}")

    def _update_priority(self, target: str, fn: HookFn, new_priority: int) -> int | None:
        with self._lock:
            bucket = self._hooks.get(target)
            if not bucket:
                return None
            for i, h in enumerate(bucket):
                if h.fn is fn:
                    old = h.priority
                    bucket[i] = replace(h, priority=new_priority)
                    bucket.sort(key=lambda x: -x.priority)
                    return old
        return None

    def _apply_priority_change(
        self,
        exc: PriorityChange,
        hook: Hook,
        target: str,
    ) -> PriorityChangeBehavior:
        try:
            new_pri = exc.compute(hook.priority, registry=self, target=target, fn=hook.fn)
        except LookupError as inner:
            raise RuntimeError(
                f"{exc!r} referenced a function that is not registered on target "
                f"{target!r}: {inner}"
            ) from inner
        self._update_priority(target, hook.fn, new_pri)
        return hook.on_priority_change or self._default_behavior

    # ---------- phase runners ----------

    def _log_hook_error(self, exc: BaseException, hook: Hook, target: str) -> None:
        _logger.exception(
            "pyhooky: %s hook %r on target %r raised %s; continuing per "
            "on_hook_error='log' (registry=%r)",
            hook.kind.value,
            hook.fn,
            target,
            type(exc).__name__,
            self.name,
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    def _run_phase_sync(
        self,
        *,
        target: str,
        snapshot: tuple[Hook, ...],
        phase_kind: HookKind,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        collect: bool = False,
    ) -> list[Any]:
        template = _ASYNC_ERROR_TEMPLATES.get(phase_kind)
        fired: set[int] = set()
        remaining = [h for h in snapshot if h.kind == phase_kind]
        results: list[Any] = []
        while remaining:
            h = remaining.pop(0)
            fn_id = id(h.fn)
            if fn_id in fired:
                continue
            fired.add(fn_id)
            if iscoroutinefunction(h.fn):
                if template is None:
                    raise RuntimeError(
                        f"Async {phase_kind.value} hook {h.fn!r} attached to "
                        f"sync target {target!r}."
                    )
                raise RuntimeError(template.format(fn=h.fn, target=target))
            try:
                r = h.fn(*args, **kwargs)
            except PriorityChange as exc:
                behavior = self._apply_priority_change(exc, h, target)
                if behavior == "propagate":
                    raise
                if behavior == "resort":
                    fresh = self._snapshot(target)
                    remaining = [x for x in fresh if x.kind == phase_kind and id(x.fn) not in fired]
                continue
            except Exception as exc:
                if self._on_hook_error == "raise":
                    raise
                self._log_hook_error(exc, h, target)
                continue
            if isawaitable(r):
                close = getattr(r, "close", None)
                if callable(close):
                    close()
                raise RuntimeError(
                    f"{phase_kind.value} hook {h.fn!r} for target {target!r} returned "
                    "an awaitable from sync dispatch (the function isn't a coroutine "
                    "function but returned a coroutine — e.g. a callable whose "
                    "__call__ returns a coroutine, or a sync wrapper around an async "
                    "fn). The coroutine would never be awaited. Wrap the target with "
                    "@hook on an async def, or change the hook to not return a "
                    "coroutine."
                )
            if collect:
                results.append(r)
        return results

    async def _run_phase_async(
        self,
        *,
        target: str,
        snapshot: tuple[Hook, ...],
        phase_kind: HookKind,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        collect: bool = False,
    ) -> list[Any]:
        fired: set[int] = set()
        remaining = [h for h in snapshot if h.kind == phase_kind]
        results: list[Any] = []
        while remaining:
            h = remaining.pop(0)
            fn_id = id(h.fn)
            if fn_id in fired:
                continue
            fired.add(fn_id)
            try:
                r = h.fn(*args, **kwargs)
                if isawaitable(r):
                    r = await r
            except PriorityChange as exc:
                behavior = self._apply_priority_change(exc, h, target)
                if behavior == "propagate":
                    raise
                if behavior == "resort":
                    fresh = self._snapshot(target)
                    remaining = [x for x in fresh if x.kind == phase_kind and id(x.fn) not in fired]
                continue
            except Exception as exc:
                if self._on_hook_error == "raise":
                    raise
                self._log_hook_error(exc, h, target)
                continue
            if collect:
                results.append(r)
        return results

    # ---------- wrapping / dispatching ----------

    def _dispatch_sync(
        self,
        target: str,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        self._run_phase_sync(
            target=target,
            snapshot=self._snapshot(target),
            phase_kind=HookKind.BEFORE,
            args=args,
            kwargs=kwargs,
        )

        around_snapshot = self._snapshot(target)
        call: Callable[..., Any] = fn
        # Iterate low→high priority so high-priority around ends up outermost,
        # firing first — consistent with before/after/listener priority semantics.
        for h in reversed(around_snapshot):
            if h.kind == HookKind.AROUND:
                if iscoroutinefunction(h.fn):
                    raise RuntimeError(
                        f"Async around hook {h.fn!r} attached to sync target "
                        f"{target!r}; around hooks must match their target's "
                        "sync/async-ness."
                    )
                call = _compose_around_sync(h.fn, call, target)

        try:
            result = call(*args, **kwargs)
        except Exception as exc:
            # ON_ERROR fires on any exception bubbling out of the AROUND chain
            # or the wrapped body. Hooks observe — they cannot swallow.
            self._run_phase_sync(
                target=target,
                snapshot=self._snapshot(target),
                phase_kind=HookKind.ON_ERROR,
                args=(exc, *args),
                kwargs=kwargs,
            )
            raise

        if isawaitable(result):
            close = getattr(result, "close", None)
            if callable(close):
                close()
            raise RuntimeError(
                f"Sync target {target!r} returned an awaitable. Wrap the "
                "underlying function with @hook on an async def to get an async "
                "wrapper, or stop returning a coroutine from a sync target."
            )

        self._run_phase_sync(
            target=target,
            snapshot=self._snapshot(target),
            phase_kind=HookKind.AFTER,
            args=(result, *args),
            kwargs=kwargs,
        )
        return result

    async def _dispatch_async(
        self,
        target: str,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        await self._run_phase_async(
            target=target,
            snapshot=self._snapshot(target),
            phase_kind=HookKind.BEFORE,
            args=args,
            kwargs=kwargs,
        )

        around_snapshot = self._snapshot(target)
        call: Callable[..., Any] = fn
        for h in reversed(around_snapshot):
            if h.kind == HookKind.AROUND:
                if not iscoroutinefunction(h.fn):
                    raise RuntimeError(
                        f"Sync around hook {h.fn!r} attached to async target "
                        f"{target!r}; around hooks must match their target's "
                        "sync/async-ness."
                    )
                call = _compose_around_async(h.fn, call, target)

        try:
            result = await call(*args, **kwargs)
        except Exception as exc:
            await self._run_phase_async(
                target=target,
                snapshot=self._snapshot(target),
                phase_kind=HookKind.ON_ERROR,
                args=(exc, *args),
                kwargs=kwargs,
            )
            raise

        await self._run_phase_async(
            target=target,
            snapshot=self._snapshot(target),
            phase_kind=HookKind.AFTER,
            args=(result, *args),
            kwargs=kwargs,
        )
        return result

    def wrap(self, target: str, fn: Callable[P, R]) -> Callable[P, R]:
        if iscoroutinefunction(fn):
            return self._wrap_async(target, fn)  # type: ignore[return-value]
        return self._wrap_sync(target, fn)

    def _wrap_sync(self, target: str, fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return self._dispatch_sync(target, fn, args, kwargs)

        setattr(wrapper, HOOK_TARGET_ATTR, target)
        setattr(wrapper, HOOK_REGISTRY_ATTR, self)
        return wrapper

    def _wrap_async(self, target: str, fn: Callable[P, Any]) -> Callable[P, Any]:
        @wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            return await self._dispatch_async(target, fn, args, kwargs)

        setattr(wrapper, HOOK_TARGET_ATTR, target)
        setattr(wrapper, HOOK_REGISTRY_ATTR, self)
        return wrapper


def _around_priority_change_error(around_fn: Callable[..., Any], target: str) -> str:
    return (
        f"around hook {around_fn!r} for target {target!r} raised PriorityChange. "
        "around hooks are composed before dispatch starts, so they cannot "
        "reorder themselves — keep PriorityChange logic in before/after hooks. "
        "AFTER and ON_ERROR hooks were not run because the around chain "
        "aborted abnormally."
    )


def _compose_around_sync(
    around_fn: Callable[..., R],
    inner: Callable[P, R],
    target: str,
) -> Callable[P, R]:
    def composed(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return around_fn(inner, *args, **kwargs)
        except PriorityChange as exc:
            raise RuntimeError(_around_priority_change_error(around_fn, target)) from exc

    return composed


def _compose_around_async(
    around_fn: Callable[..., Any],
    inner: Callable[..., Any],
    target: str,
) -> Callable[..., Any]:
    async def composed(*args: Any, **kwargs: Any) -> Any:
        try:
            return await around_fn(inner, *args, **kwargs)
        except PriorityChange as exc:
            raise RuntimeError(_around_priority_change_error(around_fn, target)) from exc

    return composed


_default_registry = HookRegistry(name="default")
_default_registry_lock = Lock()
_current_registry: ContextVar[HookRegistry | None] = ContextVar(
    "pyhooky_current_registry", default=None
)


def get_default_registry() -> HookRegistry:
    """Return the active registry: the contextvar override if set, else the module default."""
    return _current_registry.get() or _default_registry


def set_default_registry(registry: HookRegistry) -> HookRegistry:
    """Replace the module-level default registry process-wide. Returns the previous default.

    Warns (but still applies) if a :func:`use_registry` context is currently
    active — under an active scope, ``get_default_registry()`` keeps returning
    the contextvar-bound registry until the scope exits, so the swap will only
    become observable afterwards.
    """
    if not isinstance(registry, HookRegistry):
        raise TypeError(
            f"set_default_registry expects a HookRegistry, got {type(registry).__name__}"
        )
    global _default_registry
    if _current_registry.get() is not None:
        warnings.warn(
            "set_default_registry called inside an active use_registry context — "
            "the change won't be visible until that context exits.",
            stacklevel=2,
        )
    with _default_registry_lock:
        previous = _default_registry
        _default_registry = registry
    return previous


@contextmanager
def use_registry(registry: HookRegistry) -> Iterator[HookRegistry]:
    """Scope ``registry`` as the default for module-level helpers within this context.

    Implemented via :class:`contextvars.ContextVar`, so it works correctly across
    threads and asyncio tasks::

        with use_registry(my_registry):
            @before("checkout")
            def hook(...): ...
    """
    token = _current_registry.set(registry)
    try:
        yield registry
    finally:
        _current_registry.reset(token)


def _active_registry(registry: HookRegistry | None) -> HookRegistry:
    if registry is not None:
        return registry
    return _current_registry.get() or _default_registry


def _resolve_registry(target: TargetRef, registry: HookRegistry | None) -> HookRegistry:
    """Pick the registry to use for a (target, registry=) call pair.

    - If ``registry`` is explicit and ``target`` is bound to a different
      registry, raises :class:`ValueError`.
    - If ``registry`` is None and ``target`` is bound to a registry, returns
      that registry (auto-routing — the registration follows the target).
    - Otherwise returns the active registry (contextvar or module default).
    """
    bound = _target_registry(target)
    if registry is not None:
        if bound is not None and bound is not registry:
            raise ValueError(
                f"target {target!r} is bound to registry {bound.name!r}, but "
                f"registry={registry.name!r} was passed. Either drop the registry= "
                "argument, pass the matching registry, or use a string target name."
            )
        return registry
    if bound is not None:
        return bound
    return _active_registry(None)


# ---------- module-level helpers ----------


def _make_dynamic_wrapper(
    target: str,
    fn: Callable[P, R],
    registry: HookRegistry | None,
) -> Callable[P, R]:
    """Build a wrapper for the ``@hook`` decorator.

    If ``registry`` is explicit, returns a sticky wrapper bound to that
    registry (``__hook_registry__`` set). If ``registry`` is None, the wrapper
    follows the active registry on every call until the first hook is
    attached against some registry R — at that point :func:`_register_or_decorate`
    locks the wrapper to R, so registration and dispatch stay consistent
    even if the caller later switches contexts via :func:`use_registry`.
    """
    if registry is not None:
        return registry.wrap(target, fn)

    if iscoroutinefunction(fn):

        @wraps(fn)
        async def awrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            reg = getattr(awrapper, HOOK_REGISTRY_ATTR, None) or _active_registry(None)
            return await reg._dispatch_async(target, fn, args, kwargs)

        setattr(awrapper, HOOK_TARGET_ATTR, target)
        return awrapper  # type: ignore[return-value]

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        reg = getattr(wrapper, HOOK_REGISTRY_ATTR, None) or _active_registry(None)
        return reg._dispatch_sync(target, fn, args, kwargs)

    setattr(wrapper, HOOK_TARGET_ATTR, target)
    return wrapper


@overload
def hook(target: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def hook(
    target: str,
    fn: Callable[P, R],
    *,
    registry: HookRegistry | None = ...,
) -> Callable[P, R]: ...
@overload
def hook(
    target: str | None = ...,
    fn: None = ...,
    *,
    registry: HookRegistry | None = ...,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
def hook(
    target: str | Callable[..., Any] | None = None,
    fn: Callable[P, R] | None = None,
    *,
    registry: HookRegistry | None = None,
) -> Any:
    """Wrap a function so registered hooks fire around it.

    Auto-detects sync vs async: async ``def`` targets get an async wrapper that
    awaits async before/after/around hooks. Supports four call styles::

        @hook                       # bare — auto-named from module.qualname
        @hook()                     # same as @hook
        @hook("custom-name")        # explicit target name
        wrapped = hook("name", fn)  # direct call, explicit name
        wrapped = hook(fn)          # direct call, auto-name

    When ``registry`` is omitted, the wrapper resolves the active registry on
    every call (so :func:`use_registry` / :func:`set_default_registry` keep
    working after decoration time). Passing ``registry=`` binds the wrapper
    permanently to that registry.
    """
    if callable(target) and not isinstance(target, str) and fn is None:
        f = target
        return _make_dynamic_wrapper(_auto_target_name(f), f, registry)

    if isinstance(target, str) and fn is not None:
        return _make_dynamic_wrapper(target, fn, registry)

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        name = target if isinstance(target, str) else _auto_target_name(f)
        return _make_dynamic_wrapper(name, f, registry)

    return decorator


@overload
def before(
    target: TargetRef,
    fn: HookFn,
    *,
    priority: int = ...,
    on_priority_change: PriorityChangeBehavior | None = ...,
    tag: str | None = ...,
    registry: HookRegistry | None = ...,
) -> HookFn: ...
@overload
def before(
    target: TargetRef,
    fn: None = ...,
    *,
    priority: int = ...,
    on_priority_change: PriorityChangeBehavior | None = ...,
    tag: str | None = ...,
    registry: HookRegistry | None = ...,
) -> Decorator: ...
def before(
    target: TargetRef,
    fn: HookFn | None = None,
    *,
    priority: int = 0,
    on_priority_change: PriorityChangeBehavior | None = None,
    tag: str | None = None,
    registry: HookRegistry | None = None,
) -> HookFn | Decorator:
    return _register_or_decorate(
        HookKind.BEFORE, target, fn, priority, on_priority_change, tag, registry
    )


@overload
def after(
    target: TargetRef,
    fn: HookFn,
    *,
    priority: int = ...,
    on_priority_change: PriorityChangeBehavior | None = ...,
    tag: str | None = ...,
    registry: HookRegistry | None = ...,
) -> HookFn: ...
@overload
def after(
    target: TargetRef,
    fn: None = ...,
    *,
    priority: int = ...,
    on_priority_change: PriorityChangeBehavior | None = ...,
    tag: str | None = ...,
    registry: HookRegistry | None = ...,
) -> Decorator: ...
def after(
    target: TargetRef,
    fn: HookFn | None = None,
    *,
    priority: int = 0,
    on_priority_change: PriorityChangeBehavior | None = None,
    tag: str | None = None,
    registry: HookRegistry | None = None,
) -> HookFn | Decorator:
    return _register_or_decorate(
        HookKind.AFTER, target, fn, priority, on_priority_change, tag, registry
    )


@overload
def around(
    target: TargetRef,
    fn: HookFn,
    *,
    priority: int = ...,
    on_priority_change: PriorityChangeBehavior | None = ...,
    tag: str | None = ...,
    registry: HookRegistry | None = ...,
) -> HookFn: ...
@overload
def around(
    target: TargetRef,
    fn: None = ...,
    *,
    priority: int = ...,
    on_priority_change: PriorityChangeBehavior | None = ...,
    tag: str | None = ...,
    registry: HookRegistry | None = ...,
) -> Decorator: ...
def around(
    target: TargetRef,
    fn: HookFn | None = None,
    *,
    priority: int = 0,
    on_priority_change: PriorityChangeBehavior | None = None,
    tag: str | None = None,
    registry: HookRegistry | None = None,
) -> HookFn | Decorator:
    return _register_or_decorate(
        HookKind.AROUND, target, fn, priority, on_priority_change, tag, registry
    )


@overload
def on(
    target: TargetRef,
    fn: HookFn,
    *,
    priority: int = ...,
    on_priority_change: PriorityChangeBehavior | None = ...,
    tag: str | None = ...,
    registry: HookRegistry | None = ...,
) -> HookFn: ...
@overload
def on(
    target: TargetRef,
    fn: None = ...,
    *,
    priority: int = ...,
    on_priority_change: PriorityChangeBehavior | None = ...,
    tag: str | None = ...,
    registry: HookRegistry | None = ...,
) -> Decorator: ...
def on(
    target: TargetRef,
    fn: HookFn | None = None,
    *,
    priority: int = 0,
    on_priority_change: PriorityChangeBehavior | None = None,
    tag: str | None = None,
    registry: HookRegistry | None = None,
) -> HookFn | Decorator:
    """Register a listener that fires when :func:`trigger` is called for ``target``."""
    return _register_or_decorate(
        HookKind.LISTENER, target, fn, priority, on_priority_change, tag, registry
    )


@overload
def on_error(
    target: TargetRef,
    fn: HookFn,
    *,
    priority: int = ...,
    on_priority_change: PriorityChangeBehavior | None = ...,
    tag: str | None = ...,
    registry: HookRegistry | None = ...,
) -> HookFn: ...
@overload
def on_error(
    target: TargetRef,
    fn: None = ...,
    *,
    priority: int = ...,
    on_priority_change: PriorityChangeBehavior | None = ...,
    tag: str | None = ...,
    registry: HookRegistry | None = ...,
) -> Decorator: ...
def on_error(
    target: TargetRef,
    fn: HookFn | None = None,
    *,
    priority: int = 0,
    on_priority_change: PriorityChangeBehavior | None = None,
    tag: str | None = None,
    registry: HookRegistry | None = None,
) -> HookFn | Decorator:
    """Register an on_error hook for ``target``.

    Fires when the wrapped function (or anything inside an AROUND chain)
    raises. Signature: ``(exc, *original_args, **original_kwargs)``. Cannot
    swallow the exception — for that, use an AROUND. After all on_error hooks
    fire, the original exception re-raises.
    """
    return _register_or_decorate(
        HookKind.ON_ERROR, target, fn, priority, on_priority_change, tag, registry
    )


def trigger(
    target: TargetRef,
    *args: Any,
    registry: HookRegistry | None = None,
    **kwargs: Any,
) -> list[Any]:
    """Fire all listeners registered for ``target`` (sync only).

    Routes to the target's bound registry when ``target`` is a wrapped
    callable; see :meth:`HookRegistry.trigger` for the result-length caveat.
    """
    return _resolve_registry(target, registry).trigger(target, *args, **kwargs)


async def atrigger(
    target: TargetRef,
    *args: Any,
    registry: HookRegistry | None = None,
    **kwargs: Any,
) -> list[Any]:
    """Async-aware version of :func:`trigger`."""
    return await _resolve_registry(target, registry).atrigger(target, *args, **kwargs)


def _maybe_bind_lazy_wrapper(target: TargetRef, reg: HookRegistry) -> None:
    """Lock a lazy ``@hook`` wrapper to ``reg`` on its first hook attach.

    Without this, a lazy wrapper (decorated without ``registry=``) would keep
    resolving the active registry per call — so if a user registered hooks
    against registry A but later called the wrapper under ``use_registry(B)``,
    A's hooks would silently not fire. Binding at first attach makes the
    wrapper's dispatch registry match the registration registry.
    """
    if not callable(target) or isinstance(target, str):
        return
    if getattr(target, HOOK_REGISTRY_ATTR, None) is not None:
        return
    if getattr(target, HOOK_TARGET_ATTR, None) is None:
        return
    with contextlib.suppress(AttributeError, TypeError):
        setattr(target, HOOK_REGISTRY_ATTR, reg)


def _register_or_decorate(
    kind: HookKind,
    target: TargetRef,
    fn: HookFn | None,
    priority: int,
    on_priority_change: PriorityChangeBehavior | None,
    tag: str | None,
    registry: HookRegistry | None,
) -> HookFn | Decorator:
    reg = _resolve_registry(target, registry)
    name = _resolve_target(target)
    _maybe_bind_lazy_wrapper(target, reg)
    if fn is not None:
        reg.register(
            name,
            Hook(
                kind=kind,
                fn=fn,
                priority=priority,
                on_priority_change=on_priority_change,
                tag=tag,
            ),
        )
        return fn

    def decorator(f: HookFn) -> HookFn:
        reg.register(
            name,
            Hook(
                kind=kind,
                fn=f,
                priority=priority,
                on_priority_change=on_priority_change,
                tag=tag,
            ),
        )
        return f

    return decorator
