"""Faithful CombatState cloning.

``copy.deepcopy`` is not enough, and the way it fails is silent.

Monsters are built by factory functions that create a ``Creature`` and then
close over it::

    creature = Creature(max_hp=hp, monster_id=AXE_RUBY_RAIDER_ID)

    def swing(combat: CombatState) -> None:
        _deal_damage_to_player(combat, creature, swing_dmg)
        _gain_block(creature, swing_block, combat)   # <-- closure variable

``deepcopy`` treats a function as atomic: it returns *the same function object*,
closure cells and all. So the clone's ``MoveState.effect_fn`` is still bound to
the **original** combat's creature. The clone's monster then reads its strength
from the original, gains its block on the original, and fires its on-attack
hooks against the original. Both combats quietly corrupt each other.

Observed directly: clone a fight, drive both copies through the identical action
sequence, and on the twelfth action one enemy gains 10 block in the original and
0 in the clone -- and the amount changes depending on which copy you step first,
because they are fighting over one creature. Nothing raises. Nothing logs.

This matters far beyond search. Any deck or route evaluation built on cloned
states would have been measuring a fight whose monsters were partly somewhere
else, and it would have looked like ordinary variance.

THE FIX

Deep-copy with an explicit memo, then walk every copied object and rebuild any
function whose closure cells or defaults still point at something that was
copied, so it points at the copy instead. Rebuilding rather than mutating
matters: ``cell_contents`` is writable, and writing to it would rebind the
*original* monster's move as well and break the state we were cloning from.

Only cells referring to objects in the memo are touched. A closure over a
constant, a module, or a helper function is left exactly as it was.

WHY NOT FIX THE MONSTERS INSTEAD

Rewriting ~121 monster factories to look their creature up from ``combat``
rather than close over it is the deeper fix and would make the whole simulator
clone-safe by construction. It is also a large, risky change across the most
parity-sensitive code in the project. This is the containment: correct, tested,
and it does not touch a single monster definition. See docs/KNOWN_ISSUES.md.
"""

from __future__ import annotations

import copy
import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Annotation-only: eager import makes `import sts2_env.search` circular.
    from sts2_env.core.combat import CombatState


def clone_combat(combat: CombatState) -> CombatState:
    """Deep-copy a combat so the copy is genuinely independent of the original.

    Use this anywhere a combat is duplicated. Plain ``deepcopy`` leaves the
    copy's monsters acting on the original's creatures.
    """
    memo: dict[int, Any] = {}
    clone = copy.deepcopy(combat, memo)
    _rebind_copied_closures(memo)
    return clone


def _rebind_copied_closures(memo: dict[int, Any]) -> None:
    """Repoint functions held by copied objects at the copies.

    The deepcopy memo maps ``id(original) -> copy`` for everything that was
    copied, which is exactly the lookup table needed to decide whether a closure
    cell is stale.
    """
    rebuilt: dict[int, Any] = {}

    for holder in list(memo.values()):
        if isinstance(holder, dict):
            for key, value in list(holder.items()):
                rebound = _rebind(value, memo, rebuilt)
                if rebound is not value:
                    holder[key] = rebound
            continue

        if isinstance(holder, list):
            for index, value in enumerate(holder):
                rebound = _rebind(value, memo, rebuilt)
                if rebound is not value:
                    holder[index] = rebound
            continue

        for name, value in _attributes(holder):
            rebound = _rebind(value, memo, rebuilt)
            if rebound is not value:
                try:
                    setattr(holder, name, rebound)
                except (AttributeError, TypeError):
                    # Read-only attribute: nothing callable lives on one in this
                    # codebase, and failing loudly here would abort a clone over
                    # something that cannot hold a stale closure anyway.
                    pass


def _attributes(obj: Any):
    """Yield ``(name, value)`` for an object's own attributes, ``__slots__``
    included -- ``Creature`` uses slots, so ``vars()`` alone would miss it."""
    instance_dict = getattr(obj, "__dict__", None)
    if isinstance(instance_dict, dict):
        yield from list(instance_dict.items())

    for klass in type(obj).__mro__:
        for slot in getattr(klass, "__slots__", ()) or ():
            if slot == "__dict__":
                continue
            try:
                yield slot, getattr(obj, slot)
            except AttributeError:
                continue


def _rebind(value: Any, memo: dict[int, Any], rebuilt: dict[int, Any]) -> Any:
    """Return ``value`` with any references to copied objects repointed."""
    if isinstance(value, types.MethodType):
        bound_to = value.__self__
        replacement = memo.get(id(bound_to))
        if replacement is not None and replacement is not bound_to:
            return types.MethodType(value.__func__, replacement)
        return value

    if not isinstance(value, types.FunctionType):
        return value

    cached = rebuilt.get(id(value))
    if cached is not None:
        return cached

    new_closure, closure_changed = _rebind_closure(value.__closure__, memo)
    new_defaults, defaults_changed = _rebind_defaults(value.__defaults__, memo)
    if not closure_changed and not defaults_changed:
        return value

    replacement = types.FunctionType(
        value.__code__,
        value.__globals__,
        value.__name__,
        new_defaults,
        new_closure,
    )
    replacement.__qualname__ = value.__qualname__
    replacement.__doc__ = value.__doc__
    replacement.__dict__.update(value.__dict__)
    replacement.__kwdefaults__ = _rebind_kwdefaults(value.__kwdefaults__, memo)

    rebuilt[id(value)] = replacement
    # The memo owns a reference for the lifetime of the clone; without one the
    # original function could be collected and its id reused, which would make
    # `rebuilt` hand back the wrong replacement.
    memo.setdefault(id(memo), []).append(value)
    return replacement


def _rebind_closure(closure, memo: dict[int, Any]):
    if not closure:
        return closure, False

    changed = False
    cells = []
    for cell in closure:
        try:
            contents = cell.cell_contents
        except ValueError:
            # An empty cell: a recursive closure not yet filled in. Nothing to
            # repoint, and reading it again later is the original's business.
            cells.append(cell)
            continue
        replacement = memo.get(id(contents))
        if replacement is not None and replacement is not contents:
            cells.append(types.CellType(replacement))
            changed = True
        else:
            cells.append(cell)
    return tuple(cells), changed


def _rebind_defaults(defaults, memo: dict[int, Any]):
    if not defaults:
        return defaults, False

    changed = False
    values = []
    for default in defaults:
        replacement = memo.get(id(default))
        if replacement is not None and replacement is not default:
            values.append(replacement)
            changed = True
        else:
            values.append(default)
    return tuple(values), changed


def _rebind_kwdefaults(kwdefaults, memo: dict[int, Any]):
    if not kwdefaults:
        return kwdefaults
    return {
        key: memo.get(id(value), value) if memo.get(id(value)) is not None else value
        for key, value in kwdefaults.items()
    }
