"""Exceptions a hook can raise to request priority reordering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyhooky._registry import HookRegistry

__all__ = [
    "PriorityBoost",
    "PriorityChange",
    "PriorityChangeBehavior",
    "PriorityDemote",
    "RunAfter",
    "RunBefore",
    "SetPriority",
]

PriorityChangeBehavior = Literal["continue", "resort", "propagate"]


class PriorityChange(Exception):
    """Base for exceptions a hook can raise to reorder itself.

    Subclasses implement :meth:`compute` to return the hook's new priority.
    The registry catches the exception, updates the priority, and then follows
    the hook's configured ``on_priority_change`` behavior (``continue``,
    ``resort``, or ``propagate``).
    """

    def compute(
        self,
        current: int,
        *,
        registry: HookRegistry,
        target: str,
        fn: Callable[..., object],
    ) -> int:
        raise NotImplementedError


class PriorityBoost(PriorityChange):
    """Raise to increase the firing hook's priority by ``by`` (default 1)."""

    def __init__(self, by: int = 1) -> None:
        super().__init__(f"PriorityBoost(by={by})")
        self.by = by

    def compute(
        self,
        current: int,
        *,
        registry: HookRegistry,
        target: str,
        fn: Callable[..., object],
    ) -> int:
        return current + self.by


class PriorityDemote(PriorityChange):
    """Raise to decrease the firing hook's priority by ``by`` (default 1)."""

    def __init__(self, by: int = 1) -> None:
        super().__init__(f"PriorityDemote(by={by})")
        self.by = by

    def compute(
        self,
        current: int,
        *,
        registry: HookRegistry,
        target: str,
        fn: Callable[..., object],
    ) -> int:
        return current - self.by


class SetPriority(PriorityChange):
    """Raise to set an absolute priority value."""

    def __init__(self, value: int) -> None:
        super().__init__(f"SetPriority(value={value})")
        self.value = value

    def compute(
        self,
        current: int,
        *,
        registry: HookRegistry,
        target: str,
        fn: Callable[..., object],
    ) -> int:
        return self.value


class RunBefore(PriorityChange):
    """Ensure the firing hook runs before ``other`` on the same target.

    "Ensure" semantics: priority is set to ``max(current, other.priority + 1)``,
    so a hook already correctly ordered above ``other`` keeps its priority.
    Raises :class:`LookupError` if ``other`` isn't registered on the target.
    """

    def __init__(self, other: Callable[..., object]) -> None:
        super().__init__(f"RunBefore({other!r})")
        self.other = other

    def compute(
        self,
        current: int,
        *,
        registry: HookRegistry,
        target: str,
        fn: Callable[..., object],
    ) -> int:
        other_priority = registry._priority_of(target, self.other)
        return max(current, other_priority + 1)


class RunAfter(PriorityChange):
    """Ensure the firing hook runs after ``other`` on the same target.

    Priority is set to ``min(current, other.priority - 1)`` (ensure semantics).
    """

    def __init__(self, other: Callable[..., object]) -> None:
        super().__init__(f"RunAfter({other!r})")
        self.other = other

    def compute(
        self,
        current: int,
        *,
        registry: HookRegistry,
        target: str,
        fn: Callable[..., object],
    ) -> int:
        other_priority = registry._priority_of(target, self.other)
        return min(current, other_priority - 1)
