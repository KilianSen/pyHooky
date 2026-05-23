"""pyHooky — lightweight before/after/around hooks for Python callables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyhooky._registry import (
    Hook,
    HookKind,
    HookRegistry,
    OnHookError,
    after,
    around,
    atrigger,
    before,
    disabled,
    get_default_registry,
    hook,
    on,
    on_error,
    set_default_registry,
    tag_scope,
    trigger,
    use_registry,
)
from pyhooky.exceptions import (
    PriorityBoost,
    PriorityChange,
    PriorityChangeBehavior,
    PriorityDemote,
    RunAfter,
    RunBefore,
    SetPriority,
)

if TYPE_CHECKING:
    from pyhooky.typed import HookPoint

__all__ = [
    "Hook",
    "HookKind",
    "HookPoint",
    "HookRegistry",
    "OnHookError",
    "PriorityBoost",
    "PriorityChange",
    "PriorityChangeBehavior",
    "PriorityDemote",
    "RunAfter",
    "RunBefore",
    "SetPriority",
    "after",
    "around",
    "atrigger",
    "before",
    "disabled",
    "get_default_registry",
    "hook",
    "on",
    "on_error",
    "set_default_registry",
    "tag_scope",
    "trigger",
    "use_registry",
]

__version__ = "0.2.0"


def __getattr__(name: str) -> Any:
    """Lazily import :class:`HookPoint` so the core package has no hard pydantic dep.

    Importing :class:`HookPoint` while pydantic is missing raises a clear
    ``ImportError`` pointing at the ``pyhooky[typed]`` extra.
    """
    if name == "HookPoint":
        from pyhooky.typed import HookPoint

        return HookPoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
