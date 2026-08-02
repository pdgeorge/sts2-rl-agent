"""Which relics and potions she has -- by identity, not by count.

Uses feature hashing so the observation space stays fixed regardless of how many
relics or potion types the game has. A model trained before a patch can load
after it and resume fine-tuning instead of starting from scratch.

Relics are hashed into 256 buckets (was 299-wide multi-hot). Potions are hashed
into 128 buckets preserving slot identity via the hash key (was 320 dims = 64
types x 5 slots).

Shared by the simulator and the bridge on purpose. Both sides importing the same
encoder is what stops them drifting into agreeing about the observation while
disagreeing about what it means, which is the bug class that has cost this
project the most runs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from sts2_env.gym_env.hashing import (
    RELIC_HASHER,
    POTION_HASHER,
    RELIC_POTION_OBS_SIZE,
)

# Five slots, not three: Potion Belt and friends raise the cap.
POTION_SLOTS: int = 5


def _canonical(name: object) -> str:
    """Fold naming conventions to one uppercased alphanum string."""
    name = getattr(name, "name", name)
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def _relic_key(name: object) -> str:
    return f"relic_{_canonical(name)}"


def _potion_key(slot: int, name: object) -> str:
    return f"potion_slot_{slot}_{_canonical(name)}"


def encode_relics(relic_names: Iterable[Any]) -> np.ndarray:
    """256 hashed buckets encoding which relics are owned."""
    names = [_relic_key(r) for r in relic_names or () if r is not None]
    return RELIC_HASHER.encode_set(names)


def encode_potions(slot_names: Sequence[Any]) -> np.ndarray:
    """128 hashed buckets encoding potion identity per slot.

    Slot is baked into the hash key so the policy can still learn
    "Fire Potion in slot 2" versus "Fire Potion in slot 0".
    """
    keys = []
    for slot, name in enumerate(slot_names or ()):
        if slot >= POTION_SLOTS:
            break
        if name is None:
            continue
        keys.append(_potion_key(slot, name))
    return POTION_HASHER.encode_set(keys)


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
    """Bridge: the mod sends a list of relic id strings under "relics"."""
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
    """Bridge: "potion_slots" is positional, one entry per slot, null when empty."""
    slots: list[str | None] = []
    for entry in (state.get("potion_slots") or ())[:POTION_SLOTS]:
        if entry is None:
            slots.append(None)
        elif isinstance(entry, dict):
            slots.append(entry.get("id") or entry.get("potion_id") or entry.get("name"))
        else:
            slots.append(str(entry))
    return slots
