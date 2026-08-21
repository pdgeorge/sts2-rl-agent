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


#: Run-level bookkeeping the SEARCH never reads. A turn of combat does not
#: enter a room, resolve an event, or roll a card reward, so none of this can
#: change during a search -- and copying it on every node was most of the cost.
#: Measured per subtree on a healthy clone, as a share of the whole copy:
#: visited_map_coords 12.0%, acts 10.5%, map_point_history 10.4%,
#: unknown_odds 8.0%, card_rarity_odds 7.5%, visited_event_ids 7.1%,
#: active_card_rewards 7.0%, current_act 6.5%.
#:
#: `player` and `players` are NOT here despite being the single biggest subtree
#: at 39.2%: the run player owns the deck, the combat's piles hold those very
#: card objects, and a search that mutated them would corrupt the real run.
#: The rule is that a shared object must be one the search cannot touch, not
#: merely one it usually does not.
_SHARED_RUN_STATE_ATTRS = (
    "map",
    "acts",
    "current_act",
    "map_point_history",
    "visited_map_coords",
    "visited_event_ids",
    "active_card_rewards",
    "card_rarity_odds",
    "unknown_odds",
)


def _shared_with_the_original(combat: "CombatState") -> list:
    """Objects the clone SHARES with the original instead of copying.

    Run-level state only -- the map, the act list, the run's history and its
    reward odds. The search never reads any of it and cannot reach a code path
    that writes it, because a searched turn never leaves the fight.

    THE MAP WAS THE ONE THAT MATTERED FOR CORRECTNESS, not just speed. Offline
    the combat graph reaches `RunState.map`, so a plain `deepcopy` duplicated
    the whole map on EVERY node of EVERY search, and the duplicates were
    retained -- each turn copying the previous turn's copies. Measured on seed
    372's Vantom fight: one clone cost 2.5 ms on turn 1 and 1,222 ms on turn 5,
    live MapPoint objects went 65 -> 1,040 -> 4,160 -> 16,640, and one A/B
    worker was OOM-killed at 15 GB.

    Live never had the problem: `CombatSituation.to_combat` builds a standalone
    `RunState`, and the live journals agree -- options per boss turn are flat at
    6.2-6.4 from turn 1 to turn 12.
    """
    state = getattr(combat, "_primary_player_state", None)
    player_state = getattr(state, "player_state", None)
    run_state = getattr(player_state, "run_state", None)
    if run_state is None:
        return []
    shared = []
    for attr in _SHARED_RUN_STATE_ATTRS:
        value = getattr(run_state, attr, None)
        if value is not None:
            shared.append(value)
    return shared


def clone_combat(combat: "CombatState") -> "CombatState":
    """An independent copy of `combat`.

    Raises `CloneError` rather than returning a copy that shares state with the
    original, because the sharing is invisible afterwards. The one deliberate
    exception is the act map -- see `_shared_with_the_original`, which explains
    why it is shared and why that cannot affect a fight.
    """
    blockers = clone_blockers(combat)
    if blockers:
        raise CloneError(
            "Cannot clone this combat: " + "; ".join(blockers) + "."
        )
    # Pre-seeding the memo makes deepcopy return these objects as-is.
    memo = {id(obj): obj for obj in _shared_with_the_original(combat)}

    clone = copy.deepcopy(combat, memo)

    _rebind_pending_turn_setup(clone)
    _apply_branch_policy(clone)
    return clone


def _apply_branch_policy(clone: "CombatState") -> None:
    """Make the CLONE plan against the worst random branch, if the policy says so.

    Set here and only here, so the authoritative combat keeps rolling its
    branches honestly. Offline that combat IS the game: biasing it would not be
    pessimistic planning, it would be changing the fight. Prediction 17.
    """
    try:
        from sts2_env.policy_config import active_policy
        if getattr(active_policy(), "random_branch", "sample") != "worst":
            return
    except Exception:  # noqa: BLE001 - config must never break a clone
        return
    for ai in (getattr(clone, "enemy_ais", None) or {}).values():
        if hasattr(ai, "assume_worst_branch"):
            ai.assume_worst_branch = True
