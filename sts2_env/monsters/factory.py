"""Build any monster from the id the bridge calls it.

    creature, ai = create_monster_by_id("EYE_WITH_TEETH", rng)

WHY THIS EXISTS
---------------
An encounter's opening roster is not the set of monsters the fight contains.
Several act 1-4 monsters summon reinforcements mid-fight -- Fogmog's Eye,
and every enemy carrying MinionPower that gets resummoned after it dies.

`CombatSituation.to_combat` rebuilds the *opening* roster, so when the live
bridge reports a summoned monster there is no slot for it. The reconciliation
in `situation.py` then had nowhere to put it, claimed a slot belonging to a
different monster, and dropped the real one. Measured on Fogmog: the sim ended
up holding a single creature labelled FOGMOG carrying the Eye's 6 HP, and every
attack the search aimed at "Fogmog" was sent at the immortal Eye instead.

BUILT BY CALLING, NOT BY NAME
-----------------------------
The map is derived by invoking each `create_*` factory once and reading the
`monster_id` off the creature it returns. Deriving the id from the function
name instead (`create_eye_with_teeth` -> `EYE_WITH_TEETH`) is right for most and
silently wrong for any that disagree, and a silently wrong monster id is exactly
the failure this module exists to end.

Factories that need arguments this cannot supply are skipped rather than
guessed at. They stay unavailable, the caller falls back to its old behaviour,
and nothing pretends to a monster it could not build.
"""

from __future__ import annotations

import functools
import inspect
import logging

from sts2_env.core.creature import Creature
from sts2_env.core.rng import Rng
from sts2_env.monsters.state_machine import MonsterAI

logger = logging.getLogger(__name__)

#: Modules holding `create_*` monster factories.
_MODULES = (
    "sts2_env.monsters.act1",
    "sts2_env.monsters.act1_weak",
    "sts2_env.monsters.act2",
    "sts2_env.monsters.act3",
    "sts2_env.monsters.act4",
)


def _callable_with_rng_alone(func) -> bool:
    """Can this factory be called with nothing but an Rng?

    Every parameter after the first must carry a default. `create_fogmog(rng,
    ascension_level=0)` qualifies; a factory needing a live callback such as
    `create_living_shield(rng, get_ally_count)` does not, and is left out rather
    than called with a placeholder that would misreport the monster.
    """
    try:
        params = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError):
        return False
    if not params:
        return False
    return all(
        p.default is not inspect.Parameter.empty
        or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in params[1:]
    )


@functools.lru_cache(maxsize=1)
def _factories() -> dict[str, object]:
    """{MONSTER_ID: factory}, built once by probing every create_* function."""
    import importlib

    found: dict[str, object] = {}
    for module_name in _MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - a missing act must not break the rest
            logger.debug("could not import %s", module_name, exc_info=True)
            continue
        for name, func in vars(module).items():
            if not name.startswith("create_") or not callable(func):
                continue
            if not _callable_with_rng_alone(func):
                continue
            try:
                creature, ai = func(Rng(0))
            except Exception:  # noqa: BLE001 - probing must never raise
                continue
            if not isinstance(creature, Creature) or not isinstance(ai, MonsterAI):
                continue
            monster_id = str(creature.monster_id or "").upper()
            # First definition wins, so an act 1 monster reappearing in a later
            # act's module does not quietly replace the one already registered.
            if monster_id and monster_id not in found:
                found[monster_id] = func
    return found


def known_monster_ids() -> frozenset[str]:
    return frozenset(_factories())


def create_monster_by_id(
    monster_id: str, rng: Rng | None = None
) -> tuple[Creature, MonsterAI] | None:
    """A fresh creature and AI for this id, or None if it cannot be built.

    None rather than a raise: a monster this simulator does not model is a
    parity gap to log and work around, not a reason to end a live run.
    """
    factory = _factories().get(str(monster_id or "").upper())
    if factory is None:
        return None
    try:
        return factory(rng if rng is not None else Rng(0))
    except Exception:  # noqa: BLE001
        logger.debug("factory for %s raised", monster_id, exc_info=True)
        return None
