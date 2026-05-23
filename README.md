# pyHooky

Lightweight before/after/around hooks for Python callables.

## Install

```bash
pip install pyHooky

pip install pyhooky[typed]  # pydantic-typed hook points
```
or
```bash
uv add pyhooky            # core only — no runtime deps
uv add 'pyhooky[typed]'   # adds pydantic-typed hook points
```

## Usage

### As decorators

The target can be a string name, **or** omitted entirely — in which case it's derived from the wrapped function's `module.qualname`. The wrapped function exposes its target via `__hook_target__`, so other hooks can reference it directly:

```python
from pyhooky import before, after, around, hook

@hook                # bare decorator — target = "mymodule.checkout"
def checkout(cart):
    ...

@before(checkout)    # reference the function itself, no string needed
def log_start(cart):
    print("starting checkout for", cart)

@after(checkout)
def log_done(result, cart):
    print("checkout done:", result)

@around(checkout)
def retry(inner, cart):
    for _ in range(3):
        try:
            return inner(cart)
        except TransientError:
            continue
    raise
```

After hooks receive `(result, *original_args, **original_kwargs)` — `result` first, then everything the wrapped function was called with.

> **AFTER does not fire on exceptions.** If the wrapped body (or any `around`) raises, `after` hooks are skipped and the exception propagates. For "fired on success or failure" semantics, use [`on_error`](#error-hooks) for observability or `around` with `try`/`finally` if you need to mutate the outcome.

> **`kwargs` is passed by reference into BEFORE hooks**, so a BEFORE hook can rewrite the kwargs dict and the body will see the change. Positional `args` is a tuple — to rewrite positional arguments, use an `around` hook. (This asymmetry is intentional; if you need symmetric rewriting, always use `around`.)

> **Late-bound targets:** if the function isn't wrapped yet, pass the target *name* (a string) — the function-reference form (`@before(checkout)`) requires the target to already exist with `__hook_target__`. String targets work even when the wrapping `@hook` hasn't been imported yet.

Explicit names still work when you want them (cross-module hooks, stable wire names, etc.):

```python
@hook("checkout")
def checkout(cart): ...

@before("checkout")
def log_start(cart): ...
```

### Without decorators

Every decorator also works as a direct call — pass the function as the second argument:

```python
from pyhooky import HookRegistry, before, hook

# module-level helpers
before("checkout", log_start)
checkout = hook("checkout", checkout)

# or via an explicit registry
registry = HookRegistry()
registry.add_before("checkout", log_start)
registry.add_after("checkout", log_done)
registry.add_around("checkout", retry)
checkout = registry.wrap("checkout", checkout)

registry.remove("checkout", log_start)  # detach a single hook
registry.clear("checkout")               # detach all
```

### Hook points inside a function

Use `on` to register a listener and `trigger` to fire it from inside a function body — the classic event-emit pattern:

```python
from pyhooky import on, trigger

@on("checkout:step")
def audit(step, **ctx):
    print("step:", step, ctx)

def checkout(cart):
    trigger("checkout:step", "validate", cart=cart)
    validate(cart)
    trigger("checkout:step", "charge", cart=cart)
    charge(cart)
    trigger("checkout:step", "ship", cart=cart)

# trigger returns each listener's return value, in priority order
results = trigger("checkout:step", "validate", cart=cart)
```

`trigger` only fires `LISTENER`-kind hooks; `before`/`after`/`around`/`on_error` remain tied to wrapped callables.

<a id="error-hooks"></a>

### Error hooks

Use `on_error` to observe exceptions from the wrapped function or its `around` chain. The signature mirrors `before`: `(exc, *original_args, **original_kwargs)`. Error hooks **cannot swallow** the exception — they're observation only — so after they all fire, the original exception re-raises. To swallow or transform an exception, use an `around` with `try`/`except`.

```python
from pyhooky import hook, on_error

@on_error("checkout")
def report(exc, cart):
    metrics.increment("checkout.failed", tags={"err": type(exc).__name__})

@hook("checkout")
def checkout(cart):
    raise ValueError("payment declined")

checkout(cart)  # report() fires, then ValueError propagates
```

`on_error` fires for any exception bubbling out of the wrapped body or any `around` hook. If an `around` catches and recovers (returns a value or raises a different exception type and you handle it), `on_error` only fires for whatever exception ultimately escapes the chain. Typed hook points expose the same affordance via `@point.on_error` — the signature is `(exc, model)`.

Listeners that raise [`PriorityChange`](#priority-control) reorder themselves but contribute **no result** to the returned list — so `len(results)` can be less than the number of registered listeners. If you need a 1:1 mapping, keep priority-mutating logic out of value-returning listeners.

### Typed hook points (pydantic)

Bind a hook point to a pydantic schema. The payload is validated on every trigger; listeners receive the validated model. Requires the `[typed]` extra (`uv add 'pyhooky[typed]'`):

```python
from pydantic import BaseModel
from pyhooky import HookPoint

class CheckoutStep(BaseModel):
    step: str
    cart_id: int

checkout_step = HookPoint("checkout:step", CheckoutStep)

@checkout_step.listen
def audit(event: CheckoutStep) -> None:
    print(event.step, event.cart_id)

checkout_step.trigger(step="validate", cart_id=42)         # kwargs → validated
checkout_step.trigger({"step": "charge", "cart_id": 42})   # dict → validated
checkout_step.trigger(CheckoutStep(step="ship", cart_id=42))  # model → passthrough
```

Invalid payloads raise `pydantic.ValidationError` before any listener runs.

### Typed wrapping (validated before/after/around)

A `HookPoint` can also wrap a callable. The wrapper validates the payload, then dispatches typed `before`/`after`/`around` hooks that receive the validated model:

```python
from pyhooky import HookPoint
from pydantic import BaseModel

class Order(BaseModel):
    id: int
    items: list[str]

place_order = HookPoint("place_order", Order)

@place_order.wrap
def place_order_impl(order: Order) -> int:
    return order.id

@place_order.before
def validate(order: Order) -> None:
    assert order.id > 0

@place_order.after
def audit(result: int, order: Order) -> None:
    print("placed:", result, order.items)

@place_order.around
def retry(inner, order: Order) -> int:
    for _ in range(3):
        try:
            return inner(order)
        except TransientError:
            continue
    raise

place_order_impl(id=1, items=["a"])         # kwargs → Order → hooks fire
place_order_impl({"id": 2, "items": ["b"]}) # dict → Order
place_order_impl(Order(id=3, items=["c"]))  # passthrough
```

Hook signatures:

- `before(order: Order)` — receives the validated model.
- `after(result, order: Order)` — receives the return value and the model.
- `around(inner, order: Order)` — call `inner(order)` to invoke the next layer.

`@point.wrap` auto-detects `async def` and returns an async wrapper; typed `before`/`after`/`around` can be sync or async (subject to the same compatibility rules as `@hook`). `ValidationError` raises **before** any hook fires. `point.wrap`, `point.before`, `point.after`, `point.around` all accept the same `priority` / `on_priority_change` / `tag` kwargs as the untyped registry helpers.

**Dispatch matrix** — a single `HookPoint` exposes two parallel paths that share the target name but use disjoint hook kinds:

| Entry point                         | Fires                                          |
|-------------------------------------|------------------------------------------------|
| `point.trigger(...)` / `atrigger`   | listeners attached via `@point.listen`         |
| Calling the `@point.wrap` wrapper   | before / around / after attached via `@point.{before,around,after}` |

`@point.before` is **not** fired by `point.trigger(...)`, and `@point.listen` is **not** fired by calling the wrapped function. Pick the path that matches what you want to dispatch.

### Async

`hook` auto-detects `async def` targets and returns an async wrapper. Hooks themselves can be sync or async — the wrapper awaits whichever is awaitable.

```python
import asyncio
from pyhooky import hook, before, after, around, on, atrigger

@hook
async def checkout(cart):
    await asyncio.sleep(0)
    return charge(cart)

@before(checkout)             # sync hook on async target — fine
def log(cart): ...

@before(checkout)
async def audit(cart):        # async hook on async target — awaited
    await emit("audit", cart)

@around(checkout)             # MUST be async on async target
async def retry(inner, cart):
    for _ in range(3):
        try:
            return await inner(cart)
        except TransientError:
            continue
    raise

result = await checkout(my_cart)
```

For hook *points* (the `on`/`trigger` pair), use `atrigger` from async code — it awaits async listeners and calls sync ones directly:

```python
@on("step")
async def listener(name): ...

await atrigger("step", "validate")
```

`HookPoint.atrigger(...)` works the same way for pydantic-typed points.

**Compatibility rules** (enforced at call time with a clear `RuntimeError`):

| Target | Sync before/after/on_error | Async before/after/on_error | Sync around | Async around |
|---|---|---|---|---|
| `def` (sync) | ✓ | ✗ | ✓ | ✗ |
| `async def` | ✓ | ✓ | ✗ | ✓ |

`trigger` (sync) raises if any registered listener is async — use `atrigger` instead.

### Duplicate registrations

Registering the same `(kind, fn)` twice on the same target raises `ValueError`. This catches accidental double-decoration and the silent dispatch-time dedup that earlier versions did. To re-register with a different priority, call `remove()` first.

### Priority control

Higher priority fires first — for `before`, `after`, listeners, **and** `around` (the highest-priority around becomes the outermost wrapper).

A hook can raise a `PriorityChange` exception to adjust its own priority. The registry catches it, applies the change to the bucket, and continues. The reorder takes effect on the **next** dispatch by default.

```python
from pyhooky import (
    before, hook, PriorityBoost, PriorityDemote,
    SetPriority, RunBefore, RunAfter,
)

@before("checkout")
def validate(cart):
    if needs_higher_priority(cart):
        raise PriorityBoost(by=10)  # bump self by 10 for future dispatches

@before("checkout")
def audit(cart):
    raise RunBefore(validate)       # ensure I fire before `validate`
```

Exceptions:

| Exception | Effect |
|---|---|
| `PriorityBoost(by=1)` | `priority += by` |
| `PriorityDemote(by=1)` | `priority -= by` |
| `SetPriority(value)` | `priority = value` |
| `RunBefore(other)` | `priority = max(current, other.priority + 1)` — ensure-semantics |
| `RunAfter(other)` | `priority = min(current, other.priority - 1)` |

`RunBefore` / `RunAfter` resolve `other` against the **same target's bucket**; an unknown `other` raises `RuntimeError` from the firing dispatch, regardless of the configured `on_priority_change` behavior.

**Dispatch behavior** — what happens to the current dispatch when the exception is raised. Configurable per-registry (default) or per-hook (override):

```python
HookRegistry(on_priority_change="continue")           # registry default
before(target, fn, on_priority_change="resort")       # per-hook override
```

- `"continue"` *(default)* — catch, update priority, keep iterating in the original order. Reorder visible on next dispatch.
- `"resort"` — catch, update priority, re-sort the remaining hooks for this phase. Already-fired hooks don't fire again.
- `"propagate"` — update priority, then re-raise.

Applies to `before` / `after` / `on` (listener) hooks. **`around` hooks** are composed before dispatch starts, so a `PriorityChange` raised from `around` simply propagates — keep state-mutating logic in `before` if you want dynamic ordering.

### Snapshots per phase

Each dispatch takes a fresh snapshot **per phase** — once before BEFORE fires, once before AROUND composition, once before AFTER fires. So a BEFORE hook that registers a new AFTER hook (or around hook) sees its addition picked up later in the same call. `registry.get(target)` and `registry.all_hooks()` return point-in-time snapshots — priorities can change underneath via `PriorityChange`, but already-handed-out `Hook` instances stay immutable.

### Plugin systems — tags, introspection, and scoping

Tag your registrations with `tag="plugin-name"` and tear them all down with one call:

```python
from pyhooky import before, on, get_default_registry

@before("checkout", tag="audit-plugin")
def audit(cart): ...

@on("checkout:step", tag="audit-plugin")
def step_listener(step, **ctx): ...

# later — unload the plugin
removed = get_default_registry().clear_tag("audit-plugin")
print(f"unloaded {len(removed)} hooks")  # list of (target, Hook) pairs
```

**Auto-tagging via `tag_scope`** — wrap a plugin's setup in `tag_scope(name)` and every registration inside picks up the tag automatically. No more "I forgot to tag one of the 30 hooks":

```python
from pyhooky import tag_scope, before, after, on, on_error

with tag_scope("audit-plugin"):
    @before("checkout")
    def audit_before(cart): ...

    @after("checkout")
    def audit_after(result, cart): ...

    @on("checkout:step")
    def audit_step(step, **ctx): ...

    @on_error("checkout")
    def audit_error(exc, cart): ...

# All four are tagged "audit-plugin" — one call clears the whole plugin.
get_default_registry().clear_tag("audit-plugin")
```

Registrations that pass an explicit `tag=` keep their explicit value; the scope only fills in missing tags. Nested scopes use the innermost tag. Contextvar-based, so it propagates across asyncio tasks and threads.

Tags can also be hidden temporarily without unloading — useful for "disable this plugin for the next call" without losing its registrations:

```python
from pyhooky import disabled

with disabled("audit-plugin"):
    checkout(cart)   # audit-plugin hooks don't fire here
checkout(cart)       # ... but do fire here again
```

`disabled` is contextvar-scoped, so it propagates correctly across `asyncio` tasks and threads.

Introspection helpers:

```python
reg = get_default_registry()
reg.targets()              # ["checkout", "checkout:step", ...]
reg.tags()                 # ["audit-plugin", ...] — every non-None tag in use
reg.all_hooks()            # [(target, Hook), ...]
reg.hooks_by_tag("audit-plugin")
reg.dump_target("checkout")  # hooks in firing order (debugging)
repr(reg)                  # HookRegistry(name='default', targets=2, hooks=3)
```

To isolate a host or plugin to its own registry, either pass `registry=...` to every call, swap the process-wide default, or scope a context:

```python
from pyhooky import HookRegistry, set_default_registry, use_registry

plugin_registry = HookRegistry(name="plugin-host")

# Process-wide swap
previous = set_default_registry(plugin_registry)
# ... load plugins ...
set_default_registry(previous)  # restore

# Or scope it (contextvar-based, thread/asyncio-safe)
with use_registry(plugin_registry):
    load_plugin("auth")
```

> **Lazy `@hook` binding.** A wrapper produced by `@hook` (without `registry=`) starts unbound — it follows the active registry on every call. The **first** time any hook is attached to it (e.g. via `@before(fn)`), it locks to the registry that registration landed in. Registration and dispatch then stay consistent even if the caller later switches contexts. If you need different registries for the same function, declare separate wrappers (or pass `registry=` explicitly at decoration time).

`Hook` and `HookKind` are public — `registry.get(target)`, `registry.all_hooks()`, and `registry.hooks_by_tag(...)` all return immutable `Hook` snapshots whose fields (`kind`, `fn`, `priority`, `on_priority_change`, `tag`) you can read for diagnostics.

For plugin coordination by priority, `registry.priority_of(target, fn)` returns a hook's current priority so a later plugin can register at `priority_of(target, other_fn) + 1` for guaranteed ordering.

### Sandboxing hook failures — `on_hook_error`

By default, an exception raised by a `before` / `after` / `on_error` / listener hook propagates out of the dispatch, aborting the call. For plugin-host scenarios where one buggy plugin shouldn't take down the whole call, set the registry's `on_hook_error` policy:

```python
reg = HookRegistry(on_hook_error="log")

@before("checkout", registry=reg)
def buggy(cart):
    raise RuntimeError("plugin bug")

# Subsequent hooks and the wrapped body still run; the error is logged
# via logging.getLogger("pyhooky").
```

Policies:

| Value | Behavior |
|---|---|
| `"raise"` *(default)* | Hook errors propagate. Backwards compatible. |
| `"log"` | Hook errors are logged via `logging.getLogger("pyhooky")` and dispatch continues with the next hook. |

`PriorityChange` always flows through normal priority-change handling regardless of policy. `around` hooks are not covered — their exceptions are the wrapped function's outcome (and trigger `on_error`).

### Threads

The registry uses a re-entrant lock around register/remove/clear and the per-call snapshot, so concurrent registration and dispatch are safe. Hook bodies run **outside** the lock — user code is free to block, re-enter, or call back into the registry.

### Multiprocessing

Not supported. Each process has its own registry; runtime registrations don't cross the boundary. With `fork` start method on POSIX, child processes inherit the parent's registry state; with `spawn` (the Windows default and Python's default from 3.14 onward), workers re-import your code and re-run any module-level `@hook` / `@on` registrations — runtime-only registrations don't survive.

Use a private `HookRegistry()` instead of the module default when you need isolation (tests, plugins, multi-tenant code).

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run pyrefly check
```
