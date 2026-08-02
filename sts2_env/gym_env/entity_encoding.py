"""What she is fighting, what she is holding, and what is on everyone.

Uses feature hashing so the observation space stays fixed regardless of how many
cards, powers, monsters, or relics the game has. A model trained before a patch
can load after it and resume fine-tuning instead of starting from scratch.

The original 131-dim combat block is left untouched; this module only replaces
the appended identity blocks that used to be multi-hot over enum-sized vectors.

Sizing (all fixed, all patch-stable):
  Player powers:   128 buckets (was 279)
  Hand card set:   256 buckets (was 599)
  Hand extra:       90 dims    (unchanged, slot-based)
  Deck card set:   256 buckets (was 599)
  Enemies:         5 x (64 identity + 128 power buckets) = 960 (was 1,575)
  Total entity:    1,690 dims  (was 3,142)
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from sts2_env.core.constants import MAX_HAND_SIZE as MAX_HAND_SLOTS
from sts2_env.gym_env.hashing import (
    DECK_CARD_HASHER,
    ENEMY_IDENTITY_HASHER,
    ENEMY_POWER_HASHER,
    HAND_CARD_HASHER,
    PLAYER_POWER_HASHER,
    ENTITY_OBS_SIZE,
    HAND_EXTRA_BLOCK,
    HAND_EXTRA_FEATURES,
    MAX_ENEMY_SLOTS,
    ENEMY_EXT_PER_SLOT,
)

# Exported for backward compat with tests and callers that import the size.
# The old multi-hot sizes (POWER_VEC_SIZE, CARD_SET_SIZE, etc.) are removed.


def _canonical(name: object) -> str:
    """Fold naming conventions to one uppercased alphanum string."""
    name = getattr(name, "name", name)
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def _power_key(name: object) -> str:
    return f"power_{_canonical(name)}"


def _card_key(name: object) -> str:
    return f"card_{_canonical(name)}"


def _monster_key(slot: int, name: object) -> str:
    return f"enemy_{slot}_id_{_canonical(name)}"


def _enemy_power_key(slot: int, name: object) -> str:
    return f"enemy_{slot}_power_{_canonical(name)}"


# --- encoders ----------------------------------------------------------------


def encode_powers(powers: Any) -> np.ndarray:
    """128 hashed buckets encoding which powers are present and their amounts."""
    if not powers:
        return np.zeros(PLAYER_POWER_HASHER.n_buckets, dtype=np.float32)

    items: Iterable[tuple[Any, Any]]
    if isinstance(powers, Mapping):
        items = powers.items()
    else:
        pairs = []
        for entry in powers:
            if isinstance(entry, Mapping):
                pairs.append((entry.get("id") or entry.get("name"),
                              entry.get("amount", entry.get("stacks", 0))))
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                pairs.append((entry[0], entry[1]))
            else:
                pairs.append((entry, 1))
        items = pairs

    def _get_key(item):
        key, _ = item
        return _power_key(key)

    def _get_value(item):
        _, value = item
        amount = getattr(value, "amount", value)
        try:
            return float(amount) / 20.0  # same scale as before
        except (TypeError, ValueError):
            return 1.0 / 20.0

    return PLAYER_POWER_HASHER.encode_set(items, get_key=_get_key, get_value=_get_value)


def encode_monster(slot: int, monster_id: object) -> np.ndarray:
    """64 hashed buckets encoding enemy identity for this slot."""
    return ENEMY_IDENTITY_HASHER.encode_binary(_monster_key(slot, monster_id))


def encode_card_set(card_names: Iterable[Any]) -> np.ndarray:
    """256 hashed buckets encoding which cards are present."""
    names = [_card_key(c) for c in card_names or () if c is not None]
    return HAND_CARD_HASHER.encode_set(names)


def encode_deck_set(card_names: Iterable[Any]) -> np.ndarray:
    """256 hashed buckets encoding deck composition."""
    names = [_card_key(c) for c in card_names or () if c is not None]
    return DECK_CARD_HASHER.encode_set(names)


def encode_hand_extra(cards: list[Any]) -> np.ndarray:
    """Per-slot features that the original 5 did not carry."""
    out = np.zeros(HAND_EXTRA_BLOCK, dtype=np.float32)
    card_types = ("ATTACK", "SKILL", "POWER", "CURSE", "STATUS")
    for slot, card in enumerate(cards or ()):
        if slot >= MAX_HAND_SLOTS:
            break
        base = slot * HAND_EXTRA_FEATURES
        ctype = _canonical(card.get("type"))
        for j, t in enumerate(card_types):
            if ctype == t:
                out[base + j] = 1.0
        n = len(card_types)
        out[base + n] = 1.0 if card.get("upgraded") else 0.0
        out[base + n + 1] = 1.0 if card.get("playable", True) else 0.0
        out[base + n + 2] = 1.0 if card.get("targets_enemy") else 0.0
        out[base + n + 3] = 1.0 if card.get("cost_x") else 0.0
    return out


def encode_entities(
    player_powers: Any,
    hand_card_names: Iterable[Any],
    hand_features: list[Any],
    deck_card_names: Iterable[Any],
    enemies: list[tuple[Any, Any]],
) -> np.ndarray:
    """The whole appended block, in the one order both sides must use.

    enemies is [(monster_id, powers), ...] in slot order.
    """
    parts = [
        encode_powers(player_powers),
        encode_card_set(hand_card_names),
        encode_hand_extra(hand_features),
        encode_deck_set(deck_card_names),
    ]
    enemy_block = np.zeros(MAX_ENEMY_SLOTS * ENEMY_EXT_PER_SLOT, dtype=np.float32)
    for slot, (monster_id, powers) in enumerate(enemies or ()):
        if slot >= MAX_ENEMY_SLOTS:
            break
        base = slot * ENEMY_EXT_PER_SLOT
        enemy_block[base:base + ENEMY_IDENTITY_HASHER.n_buckets] = encode_monster(slot, monster_id)
        power_start = base + ENEMY_IDENTITY_HASHER.n_buckets
        power_end = base + ENEMY_EXT_PER_SLOT
        if powers:
            def _get_key(item):
                key, _ = item
                return _enemy_power_key(slot, key)

            def _get_value(item):
                _, value = item
                amount = getattr(value, "amount", value)
                try:
                    return float(amount) / 20.0
                except (TypeError, ValueError):
                    return 1.0 / 20.0

            if isinstance(powers, Mapping):
                items = powers.items()
            else:
                items = []
                for entry in powers:
                    if isinstance(entry, Mapping):
                        items.append((entry.get("id") or entry.get("name"),
                                      entry.get("amount", entry.get("stacks", 0))))
                    elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                        items.append((entry[0], entry[1]))
                    else:
                        items.append((entry, 1))

            enemy_block[power_start:power_end] = ENEMY_POWER_HASHER.encode_set(
                items, get_key=_get_key, get_value=_get_value)
    parts.append(enemy_block)
    return np.concatenate(parts)


# --- reading the simulator --------------------------------------------------


def entities_from_combat(combat: Any, deck: Iterable[Any] = ()) -> dict[str, Any]:
    hand = list(getattr(combat, "hand", []) or [])
    hand_features = [{
        "type": getattr(getattr(c, "card_type", None), "name", getattr(c, "card_type", "")),
        "upgraded": getattr(c, "upgraded", False),
        "playable": True,
        "targets_enemy": "ENEMY" in _canonical(
            getattr(getattr(c, "target_type", None), "name", getattr(c, "target_type", ""))),
        "cost_x": getattr(c, "has_energy_cost_x", False),
    } for c in hand]

    enemies = []
    for enemy in (getattr(combat, "enemies", []) or [])[:MAX_ENEMY_SLOTS]:
        enemies.append((getattr(enemy, "monster_id", None), getattr(enemy, "powers", None)))

    return {
        "player_powers": getattr(getattr(combat, "player", None), "powers", None),
        "hand_card_names": [getattr(c, "card_id", None) for c in hand],
        "hand_features": hand_features,
        "deck_card_names": [getattr(c, "card_id", None) for c in (deck or ())],
        "enemies": enemies,
    }


# --- reading the bridge -----------------------------------------------------


def entities_from_bridge_state(state: Mapping[str, Any]) -> dict[str, Any]:
    hand = list(state.get("hand") or [])
    hand_features = [{
        "type": c.get("type"),
        "upgraded": c.get("upgraded", False),
        "playable": c.get("playable", True),
        "targets_enemy": "ENEMY" in _canonical(c.get("target", "")),
        "cost_x": c.get("cost_x", False),
    } for c in hand]

    enemies = []
    for enemy in (state.get("enemies") or [])[:MAX_ENEMY_SLOTS]:
        enemies.append((enemy.get("id"), enemy.get("powers")))

    return {
        "player_powers": (state.get("player") or {}).get("powers"),
        "hand_card_names": [c.get("id") for c in hand],
        "hand_features": hand_features,
        "deck_card_names": [
            c.get("id") if isinstance(c, Mapping) else c
            for c in (state.get("deck") or ())
        ],
        "enemies": enemies,
    }
