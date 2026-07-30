"""Which relics and potions she has -- by identity, not by count.

Until now the observation carried `relic_count` and `potion_count`: how MANY,
never WHICH. So every fight was played without knowing whether she holds Snecko
Eye or Ice Cream or Burning Blood. A large share of strong Spire decisions are
relic-driven, so near-perfect play was structurally impossible -- not a training
shortfall, an information one.

Relics are a set: order does not matter, identity does, so they are multi-hot
over every RelicId. Potions are positional -- the action space picks a SLOT, so
knowing "I hold a Fire Potion" without knowing which slot it is in cannot be
acted on -- so they are one-hot per slot.

Shared by the simulator and the bridge on purpose. Both sides importing the same
encoder is what stops them drifting into agreeing about the observation while
disagreeing about what it means, which is the bug class that has cost this
project the most runs.

INDEX STABILITY MATTERS. Every index here is a column in a trained model's input
layer. If RelicId gains a member in the middle, or a potion is added, the columns
shift and every existing model silently misreads its own observation -- no error,
just worse play. test_relic_potion_encoding.py pins the sizes so that change
fails loudly instead.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

import sts2_env.potions.all  # noqa: F401  (registers the potion models)
from sts2_env.potions.base import _POTION_MODELS
from sts2_env.relics.base import RelicId

# --- relics -----------------------------------------------------------------

RELIC_IDS: tuple[str, ...] = tuple(r.name for r in RelicId)
RELIC_OBS_SIZE: int = len(RELIC_IDS)

# --- potions ----------------------------------------------------------------

# Five, not the default three: Potion Belt and friends raise the cap, and a slot
# that can exist must have a column or the observation truncates silently.
POTION_SLOTS: int = 5
POTION_IDS: tuple[str, ...] = tuple(sorted(_POTION_MODELS))
POTION_TYPE_COUNT: int = len(POTION_IDS)
POTION_OBS_SIZE: int = POTION_SLOTS * POTION_TYPE_COUNT

RELIC_POTION_OBS_SIZE: int = RELIC_OBS_SIZE + POTION_OBS_SIZE


def _canonical(name: object) -> str:
    """Fold naming conventions to one key.

    The simulator spells relics BURNING_BLOOD and potions BlockPotion, and the
    game sends ModelId.Entry strings whose convention has differed from the
    simulator's three times already (VULNERABLE_POWER/VULNERABLE,
    RestSite/REST_SITE, VICIOUS/VICIOUS_CARD). Comparing on letters and digits
    alone makes those differences stop mattering instead of silently encoding a
    relic as absent.
    """
    # Enum members stringify as "RelicId.BURNING_BLOOD", which folds to
    # RELICIDBURNINGBLOOD and matches nothing. Prefer .name.
    name = getattr(name, "name", name)
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


_RELIC_INDEX: dict[str, int] = {_canonical(r): i for i, r in enumerate(RELIC_IDS)}
_POTION_INDEX: dict[str, int] = {_canonical(p): i for i, p in enumerate(POTION_IDS)}


def relic_index(name: object) -> int | None:
    """Column for a relic, or None if it is not one we know."""
    return _RELIC_INDEX.get(_canonical(name))


def potion_index(name: object) -> int | None:
    return _POTION_INDEX.get(_canonical(name))


def encode_relics(relic_names: Iterable[Any]) -> np.ndarray:
    """Multi-hot over every RelicId."""
    out = np.zeros(RELIC_OBS_SIZE, dtype=np.float32)
    for name in relic_names or ():
        idx = relic_index(name)
        if idx is not None:
            out[idx] = 1.0
    return out


def encode_potions(slot_names: Sequence[Any]) -> np.ndarray:
    """One-hot per slot; an empty or unknown slot stays all zero.

    Positional because the action space is: the policy chooses "use the potion in
    slot 2", so the mapping from slot to identity is the part it needs.
    """
    out = np.zeros(POTION_OBS_SIZE, dtype=np.float32)
    for slot, name in enumerate(slot_names or ()):
        if slot >= POTION_SLOTS:
            break
        if name is None:
            continue
        idx = potion_index(name)
        if idx is not None:
            out[slot * POTION_TYPE_COUNT + idx] = 1.0
    return out


def encode_relics_and_potions(
    relic_names: Iterable[Any], slot_names: Sequence[Any],
) -> np.ndarray:
    return np.concatenate([encode_relics(relic_names), encode_potions(slot_names)])


# --- reading the two sides --------------------------------------------------


def relics_from_player_state(player: Any) -> list[str]:
    """Simulator: PlayerState.relics is a list of RelicId names."""
    return list(getattr(player, "relics", None) or [])


def potion_slots_from_player_state(player: Any) -> list[str | None]:
    """Simulator: PlayerState.potions is a slot list holding PotionInstance|None."""
    slots: list[str | None] = []
    for entry in (getattr(player, "potions", None) or [])[:POTION_SLOTS]:
        if entry is None:
            slots.append(None)
            continue
        model = getattr(entry, "model", None)
        slots.append(getattr(model, "potion_id", None) if model is not None else None)
    return slots


def relics_from_bridge_state(state: dict[str, Any]) -> list[str]:
    """Bridge: the mod sends a list of relic id strings under "relics".

    Accepts bare strings or objects with an "id", because the mod has sent both
    shapes for cards and being strict here would encode an empty relic set --
    which reads as "she owns nothing" rather than as an error.
    """
    out: list[str] = []
    for entry in state.get("relics") or ():
        if isinstance(entry, dict):
            rid = entry.get("id") or entry.get("relic_id") or entry.get("name")
        else:
            rid = entry
        if rid is not None:
            out.append(str(rid))
    return out


def potion_slots_from_bridge_state(state: dict[str, Any]) -> list[str | None]:
    """Bridge: "potion_slots" is positional, one entry per slot, null when empty.

    Deliberately NOT read from the combat "potions" list. That one is a list of
    usable potions and is not slot-indexed, so using it would put a potion in the
    wrong column whenever an earlier slot was empty.
    """
    slots: list[str | None] = []
    for entry in (state.get("potion_slots") or ())[:POTION_SLOTS]:
        if entry is None:
            slots.append(None)
        elif isinstance(entry, dict):
            slots.append(entry.get("id") or entry.get("potion_id") or entry.get("name"))
        else:
            slots.append(str(entry))
    return slots
