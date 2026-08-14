"""Copying a combat so the search can play it out and throw it away.

`CombatState._pending_turn_setup` holds a callback closing over `self`, left
behind whenever turn setup or an enemy move stops on a pending choice -- "choose
cards to discard" and friends. `copy.deepcopy` returns functions BY REFERENCE,
so a naive copy carries a callback bound to the ORIGINAL combat. Resuming the
copy would then drive the real fight, invisibly.

WHY THIS FILE STOPPED REFUSING, AND WHAT IT COST TO FIND OUT
------------------------------------------------------------
It used to raise `CloneError` in that situation. That is safe and it was wrong,
because `SearchAgent` catches the error and ends the turn -- so a position the
search could not clone became a turn where the agent did nothing at all.

Live, 2026-08-14: floor 11, Punch Construct, an `?` room. The reconstruction was
perfect -- 61 HP, 3 energy, the right hand, the right enemy with Artifact -- and
the fight opened a discard choice. The journal recorded `searches: 9,
search_failures: 0` while `cards_played: 0` and the player took 61 damage across
8 turns and died. The search reported success while standing still, which is the
worst failure mode available: silent, and indistinguishable from working.

The same error had already killed two offline harness runs (841/1020 and
201/360) and was written off as an oddity.

THE FIX
-------
`CombatState` now records the pending callback as DATA alongside the lambda
(`_pending_turn_setup_spec`), so a clone can rebuild it bound to itself. The
spec is a plain tuple and survives deepcopy as a value.

  ("player", player_index, stage)  -> _continue_player_turn_setup(idx, stage)
  ("enemy", combat_id, index)      -> _finish_enemy_move_after_choice(e, index)

The enemy is stored by `combat_id` rather than by object, because the clone's
enemy is a different object and matching by identity would rebind the callback
to a creature in the original fight -- the very bug this exists to prevent.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sts2_env.core.combat import CombatState


class CloneError(RuntimeError):
    """A combat that cannot be safely copied."""


def clone_blockers(combat: "CombatState") -> list[str]:
    """Reasons this state cannot be cloned safely. Empty means it can.

    A pending turn-setup callback is no longer a blocker when it carries a spec
    to rebuild from. It still is when it does not, because silently copying a
    callback bound to the original is how a search mutates the live fight.
    """
    blockers = []
    if (getattr(combat, "_pending_turn_setup", None) is not None
            and getattr(combat, "_pending_turn_setup_spec", None) is None):
        blockers.append(
            "a turn-setup callback is pending with no spec to rebuild it from; "
            "deepcopy would share it with the original, so resuming the copy "
            "would mutate the real fight"
        )
    return blockers


def can_clone(combat: "CombatState") -> bool:
    return not clone_blockers(combat)


def _rebind_pending_turn_setup(clone: "CombatState") -> None:
    """Point the clone's pending callback at the CLONE, not the original."""
    spec = getattr(clone, "_pending_turn_setup_spec", None)
    if spec is None:
        return
    kind = spec[0]
    if kind == "player":
        _, player_index, stage = spec
        clone._pending_turn_setup = (
            lambda idx=player_index, st=stage:
            clone._continue_player_turn_setup(idx, st)
        )
        return
    if kind == "enemy":
        _, combat_id, index = spec
        enemy = next((e for e in clone.enemies
                      if getattr(e, "combat_id", None) == combat_id), None)
        if enemy is None:
            # The enemy is gone from the copy: there is nothing to resume, and
            # leaving the original's callback attached would be the sharing bug.
            clone._pending_turn_setup = None
            clone._pending_turn_setup_spec = None
            return
        clone._pending_turn_setup = (
            lambda e=enemy, i=index: clone._finish_enemy_move_after_choice(e, i)
        )
        return
    raise CloneError(f"Unknown pending turn-setup spec: {spec!r}")


def clone_combat(combat: "CombatState") -> "CombatState":
    """An independent copy of `combat`.

    Raises `CloneError` rather than returning a copy that shares state with the
    original, because the sharing is invisible afterwards.
    """
    blockers = clone_blockers(combat)
    if blockers:
        raise CloneError(
            "Cannot clone this combat: " + "; ".join(blockers) + "."
        )
    clone = copy.deepcopy(combat)
    _rebind_pending_turn_setup(clone)
    return clone
