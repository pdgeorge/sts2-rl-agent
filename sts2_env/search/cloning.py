"""Copying a combat so it can be played out without touching the real one.

Search is nothing but this plus a loop: clone, play a line, look at the result,
throw the clone away. It works because `CombatState` deep-copies in well under a
millisecond -- 0.62 ms on a floor-1 deck, 0.90 ms on a floor-14 one with a run
state attached.

Deep-copying the run state matters and is not incidental. `shuffle_rng` and
`monster_ai_rng` resolve through it, so a clone that shared it would advance the
real run's RNG every time search looked at a line, and the fight the agent
actually plays would depend on how hard it thought. The clone gets its own
streams.

THE ONE THING THAT WOULD BREAK SILENTLY

`CombatState._pending_turn_setup` holds a lambda closing over `self`
(`core/combat.py` around 882, and its neighbours). `copy.deepcopy` treats
functions as atomic and returns the *same object*, so a clone taken while one is
pending would carry a callback pointing at the original combat: resuming the
clone's turn would mutate the fight it was copied from. Nothing would raise, and
the corruption would look like a mysteriously wrong simulation.

It is `None` at every point a decision is asked for, which is why this is an
assertion rather than a repair. If it ever fires, the honest fix is to find why
search is being asked to think mid-setup, not to null the callback and hope.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sts2_env.core.combat import CombatState


class CloneError(RuntimeError):
    """A combat that cannot be safely copied."""


def clone_blockers(combat: "CombatState") -> list[str]:
    """Reasons this state cannot be cloned safely. Empty means it can."""
    blockers = []
    if getattr(combat, "_pending_turn_setup", None) is not None:
        blockers.append(
            "a turn-setup callback is pending; deepcopy would share it with the "
            "original, so resuming the copy would mutate the real fight"
        )
    return blockers


def can_clone(combat: "CombatState") -> bool:
    return not clone_blockers(combat)


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
    return copy.deepcopy(combat)
