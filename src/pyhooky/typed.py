"""Pydantic-typed hook points.

A :class:`HookPoint` binds a string event name to a pydantic schema. Triggering
the point validates the payload before dispatching to listeners, and listener
signatures can be statically typed against the schema model. The point can also
wrap a callable so that calling the wrapper validates the payload and runs
typed before/after/around hooks against the validated model.

Dispatch matrix — what fires from which entry point:

    point.trigger / point.atrigger   →  LISTENER hooks only (point.listen)
    wrapped_fn(...)                  →  BEFORE / AROUND / AFTER hooks only
                                        (point.before / point.around / point.after)

These two paths share the same target name but use disjoint :class:`HookKind`
values, so attaching ``@point.before`` won't fire on ``point.trigger()``, and
``@point.listen`` won't fire when the wrapped function is called.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, Generic, TypeVar, overload

try:
    from pydantic import BaseModel, ValidationError
except ImportError as exc:  # pragma: no cover - exercised when extra missing
    raise ImportError(
        "pyhooky.typed requires pydantic. Install with: pip install 'pyhooky[typed]'"
    ) from exc

from pyhooky._registry import (
    HOOK_REGISTRY_ATTR,
    HOOK_TARGET_ATTR,
    HookKind,
    HookRegistry,
    get_default_registry,
)
from pyhooky.exceptions import PriorityChangeBehavior

__all__ = ["HookPoint", "ValidationError"]

M = TypeVar("M", bound=BaseModel)
R = TypeVar("R")

Listener = Callable[[M], R]
BeforeFn = Callable[[M], Any]
AfterFn = Callable[[Any, M], Any]
AroundFn = Callable[..., Any]
OnErrorFn = Callable[..., Any]
WrapFn = Callable[[M], R]


class HookPoint(Generic[M]):
    """A typed hook point bound to a pydantic schema.

    Two modes, usable independently or together:

    **Event emission**::

        checkout_step = HookPoint("checkout:step", CheckoutStep)

        @checkout_step.listen
        def audit(event: CheckoutStep) -> None: ...

        checkout_step.trigger(step="validate", cart_id=42)

    **Function wrapping** — validates the payload, then runs typed
    before/after/around hooks with the validated model::

        place_order = HookPoint("place_order", Order)

        @place_order.wrap
        def place_order_impl(order: Order) -> int:
            return order.id

        @place_order.before
        def validate(order: Order) -> None: ...

        @place_order.after
        def audit(result: int, order: Order) -> None: ...

        @place_order.around
        def retry(inner, order: Order) -> int:
            return inner(order)

        place_order_impl(id=1, items=["a"])  # kwargs → Order → hooks fire → impl

    Registry resolution is lazy when no ``registry=`` is passed at
    construction: each call resolves via :func:`get_default_registry`, so
    :func:`use_registry` / :func:`set_default_registry` keep working after
    the ``HookPoint`` was created.
    """

    __slots__ = ("_explicit_registry", "name", "schema")

    def __init__(
        self,
        name: str,
        schema: type[M],
        *,
        registry: HookRegistry | None = None,
    ) -> None:
        self.name = name
        self.schema = schema
        self._explicit_registry = registry

    @property
    def _reg(self) -> HookRegistry:
        return self._explicit_registry or get_default_registry()

    # ---------- listeners (event emission) ----------

    @overload
    def listen(
        self,
        fn: Listener[M, R],
        *,
        priority: int = ...,
        on_priority_change: PriorityChangeBehavior | None = ...,
        tag: str | None = ...,
    ) -> Listener[M, R]: ...
    @overload
    def listen(
        self,
        fn: None = ...,
        *,
        priority: int = ...,
        on_priority_change: PriorityChangeBehavior | None = ...,
        tag: str | None = ...,
    ) -> Callable[[Listener[M, R]], Listener[M, R]]: ...
    def listen(
        self,
        fn: Listener[M, R] | None = None,
        *,
        priority: int = 0,
        on_priority_change: PriorityChangeBehavior | None = None,
        tag: str | None = None,
    ) -> Listener[M, R] | Callable[[Listener[M, R]], Listener[M, R]]:
        """Register a listener for this hook point. Usable as decorator or direct call.

        Fired only from :meth:`trigger` / :meth:`atrigger`; not from the
        wrapper produced by :meth:`wrap`. Accepts the same ``priority`` /
        ``on_priority_change`` / ``tag`` kwargs as the untyped helpers, so
        listeners can participate in tagged plugin teardown.
        """
        if fn is not None:
            self._reg.add_listener(
                self.name,
                fn,
                priority=priority,
                on_priority_change=on_priority_change,
                tag=tag,
            )
            return fn

        def decorator(f: Listener[M, R]) -> Listener[M, R]:
            self._reg.add_listener(
                self.name,
                f,
                priority=priority,
                on_priority_change=on_priority_change,
                tag=tag,
            )
            return f

        return decorator

    # ---------- typed function wrapping ----------

    @overload
    def wrap(self, fn: WrapFn[M, R], /) -> Callable[..., R]: ...
    @overload
    def wrap(self, fn: None = ..., /) -> Callable[[WrapFn[M, R]], Callable[..., R]]: ...
    def wrap(
        self,
        fn: WrapFn[M, R] | None = None,
        /,
    ) -> Callable[..., R] | Callable[[WrapFn[M, R]], Callable[..., R]]:
        """Wrap ``fn`` so calls validate the payload and dispatch through typed hooks.

        The returned wrapper accepts ``(payload | dict | None, /, **fields)`` — the
        same shape as :meth:`trigger` — and forwards the validated model to ``fn``.
        Typed ``before``/``after``/``around`` hooks attached to this point receive
        the model as their first argument (and ``(result, model)`` for ``after``).

        Sync vs async is auto-detected from ``fn`` — async functions get an
        ``async def`` wrapper that ``await``s. The wrapper resolves the active
        registry on every call (unless ``HookPoint`` was constructed with an
        explicit ``registry=``).
        """
        if fn is None:

            def decorator(f: WrapFn[M, R]) -> Callable[..., R]:
                return self.wrap(f)

            return decorator

        target = self.name

        if iscoroutinefunction(fn):

            @wraps(fn)
            async def avalidator(
                payload: M | dict[str, Any] | None = None,
                /,
                **fields: Any,
            ) -> R:
                event = self._coerce(payload, fields)
                return await self._reg._dispatch_async(target, fn, (event,), {})

            setattr(avalidator, HOOK_TARGET_ATTR, target)
            if self._explicit_registry is not None:
                setattr(avalidator, HOOK_REGISTRY_ATTR, self._explicit_registry)
            return avalidator  # type: ignore[return-value]

        @wraps(fn)
        def validator(
            payload: M | dict[str, Any] | None = None,
            /,
            **fields: Any,
        ) -> R:
            event = self._coerce(payload, fields)
            return self._reg._dispatch_sync(target, fn, (event,), {})

        setattr(validator, HOOK_TARGET_ATTR, target)
        if self._explicit_registry is not None:
            setattr(validator, HOOK_REGISTRY_ATTR, self._explicit_registry)
        return validator

    @overload
    def before(
        self,
        fn: BeforeFn[M],
        *,
        priority: int = ...,
        on_priority_change: PriorityChangeBehavior | None = ...,
        tag: str | None = ...,
    ) -> BeforeFn[M]: ...
    @overload
    def before(
        self,
        fn: None = ...,
        *,
        priority: int = ...,
        on_priority_change: PriorityChangeBehavior | None = ...,
        tag: str | None = ...,
    ) -> Callable[[BeforeFn[M]], BeforeFn[M]]: ...
    def before(
        self,
        fn: BeforeFn[M] | None = None,
        *,
        priority: int = 0,
        on_priority_change: PriorityChangeBehavior | None = None,
        tag: str | None = None,
    ) -> BeforeFn[M] | Callable[[BeforeFn[M]], BeforeFn[M]]:
        """Register a typed before-hook. Fires before the wrapped fn with the validated model."""
        if fn is not None:
            self._reg.add_before(
                self.name,
                fn,
                priority=priority,
                on_priority_change=on_priority_change,
                tag=tag,
            )
            return fn

        def decorator(f: BeforeFn[M]) -> BeforeFn[M]:
            self._reg.add_before(
                self.name,
                f,
                priority=priority,
                on_priority_change=on_priority_change,
                tag=tag,
            )
            return f

        return decorator

    @overload
    def after(
        self,
        fn: AfterFn[M],
        *,
        priority: int = ...,
        on_priority_change: PriorityChangeBehavior | None = ...,
        tag: str | None = ...,
    ) -> AfterFn[M]: ...
    @overload
    def after(
        self,
        fn: None = ...,
        *,
        priority: int = ...,
        on_priority_change: PriorityChangeBehavior | None = ...,
        tag: str | None = ...,
    ) -> Callable[[AfterFn[M]], AfterFn[M]]: ...
    def after(
        self,
        fn: AfterFn[M] | None = None,
        *,
        priority: int = 0,
        on_priority_change: PriorityChangeBehavior | None = None,
        tag: str | None = None,
    ) -> AfterFn[M] | Callable[[AfterFn[M]], AfterFn[M]]:
        """Register a typed after-hook. Receives ``(result, model)``."""
        if fn is not None:
            self._reg.add_after(
                self.name,
                fn,
                priority=priority,
                on_priority_change=on_priority_change,
                tag=tag,
            )
            return fn

        def decorator(f: AfterFn[M]) -> AfterFn[M]:
            self._reg.add_after(
                self.name,
                f,
                priority=priority,
                on_priority_change=on_priority_change,
                tag=tag,
            )
            return f

        return decorator

    @overload
    def around(
        self,
        fn: AroundFn,
        *,
        priority: int = ...,
        on_priority_change: PriorityChangeBehavior | None = ...,
        tag: str | None = ...,
    ) -> AroundFn: ...
    @overload
    def around(
        self,
        fn: None = ...,
        *,
        priority: int = ...,
        on_priority_change: PriorityChangeBehavior | None = ...,
        tag: str | None = ...,
    ) -> Callable[[AroundFn], AroundFn]: ...
    def around(
        self,
        fn: AroundFn | None = None,
        *,
        priority: int = 0,
        on_priority_change: PriorityChangeBehavior | None = None,
        tag: str | None = None,
    ) -> AroundFn | Callable[[AroundFn], AroundFn]:
        """Register a typed around-hook. Signature: ``(inner, model)``; call ``inner(model)``."""
        if fn is not None:
            self._reg.add_around(
                self.name,
                fn,
                priority=priority,
                on_priority_change=on_priority_change,
                tag=tag,
            )
            return fn

        def decorator(f: AroundFn) -> AroundFn:
            self._reg.add_around(
                self.name,
                f,
                priority=priority,
                on_priority_change=on_priority_change,
                tag=tag,
            )
            return f

        return decorator

    # ---------- on_error ----------

    @overload
    def on_error(
        self,
        fn: OnErrorFn,
        *,
        priority: int = ...,
        on_priority_change: PriorityChangeBehavior | None = ...,
        tag: str | None = ...,
    ) -> OnErrorFn: ...
    @overload
    def on_error(
        self,
        fn: None = ...,
        *,
        priority: int = ...,
        on_priority_change: PriorityChangeBehavior | None = ...,
        tag: str | None = ...,
    ) -> Callable[[OnErrorFn], OnErrorFn]: ...
    def on_error(
        self,
        fn: OnErrorFn | None = None,
        *,
        priority: int = 0,
        on_priority_change: PriorityChangeBehavior | None = None,
        tag: str | None = None,
    ) -> OnErrorFn | Callable[[OnErrorFn], OnErrorFn]:
        """Register an on_error hook for this point. Signature: ``(exc, model)``.

        Fires when the wrapped callable (or anything inside the AROUND chain)
        raises. Cannot swallow the exception — for that, use an AROUND.
        """
        if fn is not None:
            self._reg.add_on_error(
                self.name,
                fn,
                priority=priority,
                on_priority_change=on_priority_change,
                tag=tag,
            )
            return fn

        def decorator(f: OnErrorFn) -> OnErrorFn:
            self._reg.add_on_error(
                self.name,
                f,
                priority=priority,
                on_priority_change=on_priority_change,
                tag=tag,
            )
            return f

        return decorator

    # ---------- shared admin ----------

    def remove(self, fn: Callable[..., Any], *, kind: HookKind | None = None) -> bool:
        """Remove a hook attached to this point. Pass ``kind`` to disambiguate
        when the same callable is registered as multiple kinds on this point."""
        return self._reg.remove(self.name, fn, kind=kind)

    def clear(self) -> None:
        self._reg.clear(self.name)

    def _coerce(self, payload: M | dict[str, Any] | None, fields: dict[str, Any]) -> M:
        if payload is None:
            return self.schema(**fields)
        if isinstance(payload, self.schema):
            if fields:
                raise TypeError("Cannot mix a model instance with keyword fields")
            return payload
        if isinstance(payload, dict):
            if fields:
                raise TypeError("Cannot mix a dict payload with keyword fields")
            return self.schema.model_validate(payload)
        got = type(payload).__name__
        raise TypeError(f"Expected {self.schema.__name__}, dict, or keyword fields; got {got}")

    @overload
    def trigger(self, payload: M, /) -> list[Any]: ...
    @overload
    def trigger(self, payload: dict[str, Any], /) -> list[Any]: ...
    @overload
    def trigger(self, /, **fields: Any) -> list[Any]: ...
    def trigger(
        self,
        payload: M | dict[str, Any] | None = None,
        /,
        **fields: Any,
    ) -> list[Any]:
        """Validate ``payload`` (or ``**fields``) against the schema and dispatch (sync).

        Fires only LISTENER hooks attached via :meth:`listen`. Hooks attached
        via :meth:`before` / :meth:`after` / :meth:`around` fire from the
        :meth:`wrap` wrapper, not from here.

        Raises :class:`pydantic.ValidationError` if the payload doesn't fit, or
        :class:`RuntimeError` if any registered listener is async.
        """
        event = self._coerce(payload, fields)
        return self._reg.trigger(self.name, event)

    @overload
    async def atrigger(self, payload: M, /) -> list[Any]: ...
    @overload
    async def atrigger(self, payload: dict[str, Any], /) -> list[Any]: ...
    @overload
    async def atrigger(self, /, **fields: Any) -> list[Any]: ...
    async def atrigger(
        self,
        payload: M | dict[str, Any] | None = None,
        /,
        **fields: Any,
    ) -> list[Any]:
        """Async-aware version of :meth:`trigger`. Awaits async listeners."""
        event = self._coerce(payload, fields)
        return await self._reg.atrigger(self.name, event)

    def __repr__(self) -> str:
        return f"HookPoint(name={self.name!r}, schema={self.schema.__name__})"
