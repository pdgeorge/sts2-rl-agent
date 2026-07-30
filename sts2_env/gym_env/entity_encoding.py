"""What she is fighting, what she is holding, and what is on everyone.

The 131-dim combat observation showed 6 of 279 powers on the player, 3 of 279 on
each enemy, no enemy identity at all, and card identity as a single scalar
`(index + 1) / 600` -- an ordinal encoding of a categorical, so Bash was 0.043 and
its enum neighbour 0.045: numerically adjacent, semantically unrelated. In
practice that is noise, and she was playing "a 1-cost attack for 8" rather than a
card she recognised.

This module adds the identity blocks. They are APPENDED to the existing 131 dims
rather than replacing them: those dims have parity tests and known-good bridge
agreement, and rewriting them would put every one of those guarantees back in
play for no gain.

Sizing choices worth knowing:

- Powers are a 279-wide vector of normalised AMOUNTS, not a bitmask. Poison 12
  and Poison 1 are different situations, and one vector answers both "does she
  have it" and "how much".
- The hand is a 599-wide multi-hot of WHICH cards are held, plus a few features
  per slot. Full per-slot identity would be 10 x 599 = 5990 dims to disambiguate
  the rare case of two mechanically identical cards in different slots -- the set
  plus per-slot features costs 689 and loses almost nothing.
- The deck is a multi-hot too, because deckbuilding was being done blind: the
  observation carried deck_size and nothing about what was in it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from sts2_env.cards.base import CardId
from sts2_env.powers.base import PowerId

# --- powers -----------------------------------------------------------------

POWER_IDS: tuple[str, ...] = tuple(p.name for p in PowerId)
POWER_VEC_SIZE: int = len(POWER_IDS)

# Most stacks are small; Poison and Strength in a long fight are the tail. A
# scale of 20 keeps the common range readable without clipping the tail to 1.0
# immediately -- the observation is clipped at OBS_VALUE_HIGH anyway.
POWER_AMOUNT_SCALE = 20.0

# --- enemies ----------------------------------------------------------------

# Explicit and sorted so the column for a monster never moves. Derived from every
# monster_id the simulator constructs; a monster missing here encodes as all-zero
# (unknown) rather than raising, and test_entity_encoding pins the count so a new
# monster is a visible change instead of a silent column shift.
MONSTER_IDS: tuple[str, ...] = (
    "ARCHITECT", "AXEBOT", "BATTLE_FRIEND_V1", "BATTLE_FRIEND_V2",
    "BATTLE_FRIEND_V3", "BIG_DUMMY", "CUBEX_CONSTRUCT", "EYE_WITH_TEETH",
    "FAKE_MERCHANT_MONSTER", "FLYCONID", "FOGMOG", "FUZZY_WURM_CRAWLER",
    "INKLET", "LEAF_SLIME_M", "LEAF_SLIME_S", "LIVING_SHIELD", "MAWLER",
    "MULTI_ATTACK_MOVE_MONSTER", "MYSTERIOUS_KNIGHT", "NIBBIT",
    "ONE_HP_MONSTER", "OSTY", "ROYAL_GUARD", "SCROLL_OF_BITING",
    "SHRINKER_BEETLE", "SINGLE_ATTACK_MOVE_MONSTER", "SLITHERING_STRANGLER",
    "SNAPPING_JAXFRUIT", "TEN_HP_MONSTER", "THE_ADVERSARY_MK_ONE",
    "THE_ADVERSARY_MK_THREE", "THE_ADVERSARY_MK_TWO", "TWIG_SLIME_M",
    "TWIG_SLIME_S", "VINE_SHAMBLER", "WRIGGLER",
)
MONSTER_VEC_SIZE: int = len(MONSTER_IDS)

# --- cards ------------------------------------------------------------------

CARD_IDS: tuple[str, ...] = tuple(c.name for c in CardId)
CARD_SET_SIZE: int = len(CARD_IDS)

# Per hand slot, on top of the 5 features the original block already carries.
# Type matters because "a 1-cost attack for 8" and "a 1-cost skill for 8 block"
# were previously distinguished only by is_attack.
CARD_TYPES: tuple[str, ...] = ("ATTACK", "SKILL", "POWER", "CURSE", "STATUS")
HAND_EXTRA_FEATURES: int = len(CARD_TYPES) + 4  # + upgraded, playable, targets_enemy, cost_x
MAX_HAND_SLOTS = 10
MAX_ENEMY_SLOTS = 5

# --- block sizes ------------------------------------------------------------

PLAYER_POWER_BLOCK = POWER_VEC_SIZE
HAND_SET_BLOCK = CARD_SET_SIZE
HAND_EXTRA_BLOCK = MAX_HAND_SLOTS * HAND_EXTRA_FEATURES
DECK_SET_BLOCK = CARD_SET_SIZE
ENEMY_EXT_PER_SLOT = MONSTER_VEC_SIZE + POWER_VEC_SIZE
ENEMY_EXT_BLOCK = MAX_ENEMY_SLOTS * ENEMY_EXT_PER_SLOT

ENTITY_OBS_SIZE: int = (
    PLAYER_POWER_BLOCK + HAND_SET_BLOCK + HAND_EXTRA_BLOCK
    + DECK_SET_BLOCK + ENEMY_EXT_BLOCK
)


def _canonical(name: object) -> str:
    # Enum members stringify as "CardId.STRIKE_IRONCLAD", so folding str() alone
    # produced CARDIDSTRIKEIRONCLAD and matched nothing -- the whole block
    # encoded as zero while looking like it worked. Prefer .name.
    name = getattr(name, "name", name)
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


_POWER_INDEX = {_canonical(p): i for i, p in enumerate(POWER_IDS)}
_MONSTER_INDEX = {_canonical(m): i for i, m in enumerate(MONSTER_IDS)}
_CARD_INDEX = {_canonical(c): i for i, c in enumerate(CARD_IDS)}


def power_index(name: object) -> int | None:
    return _POWER_INDEX.get(_canonical(name))


def monster_index(name: object) -> int | None:
    return _MONSTER_INDEX.get(_canonical(name))


def card_index(name: object) -> int | None:
    idx = _CARD_INDEX.get(_canonical(name))
    if idx is not None:
        return idx
    # The game and the simulator have disagreed on suffixes three times
    # (VULNERABLE_POWER/VULNERABLE, VICIOUS/VICIOUS_CARD, RestSite/REST_SITE).
    for suffix in ("CARD", "STATUS", "POWER"):
        idx = _CARD_INDEX.get(_canonical(name) + suffix)
        if idx is not None:
            return idx
    return None


# --- encoders ---------------------------------------------------------------


def encode_powers(powers: Any) -> np.ndarray:
    """279 normalised amounts.

    Accepts what either side actually has: the simulator's
    {PowerId: PowerInstance} dict, or the bridge's list of {"id": ..., "amount":
    ...} objects.
    """
    out = np.zeros(POWER_VEC_SIZE, dtype=np.float32)
    if not powers:
        return out

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

    for key, value in items:
        name = getattr(key, "name", key)
        idx = power_index(name)
        if idx is None:
            continue
        amount = getattr(value, "amount", value)
        try:
            out[idx] = float(amount) / POWER_AMOUNT_SCALE
        except (TypeError, ValueError):
            out[idx] = 1.0 / POWER_AMOUNT_SCALE
    return out


def encode_monster(monster_id: object) -> np.ndarray:
    out = np.zeros(MONSTER_VEC_SIZE, dtype=np.float32)
    idx = monster_index(monster_id)
    if idx is not None:
        out[idx] = 1.0
    return out


def encode_card_set(card_names: Iterable[Any]) -> np.ndarray:
    """Multi-hot over every CardId. Counts saturate at 1 deliberately.

    How many copies of Strike is already carried by deck_size and the pile
    counts; which cards exist at all is what was missing.
    """
    out = np.zeros(CARD_SET_SIZE, dtype=np.float32)
    for name in card_names or ():
        idx = card_index(name)
        if idx is not None:
            out[idx] = 1.0
    return out


def encode_hand_extra(cards: list[Any]) -> np.ndarray:
    """Per-slot features that the original 5 did not carry."""
    out = np.zeros(HAND_EXTRA_BLOCK, dtype=np.float32)
    for slot, card in enumerate(cards or ()):
        if slot >= MAX_HAND_SLOTS:
            break
        base = slot * HAND_EXTRA_FEATURES
        ctype = _canonical(card.get("type"))
        for j, t in enumerate(CARD_TYPES):
            if ctype == t:
                out[base + j] = 1.0
        n = len(CARD_TYPES)
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
        encode_card_set(deck_card_names),
    ]
    enemy_block = np.zeros(ENEMY_EXT_BLOCK, dtype=np.float32)
    for slot, (monster_id, powers) in enumerate(enemies or ()):
        if slot >= MAX_ENEMY_SLOTS:
            break
        base = slot * ENEMY_EXT_PER_SLOT
        enemy_block[base:base + MONSTER_VEC_SIZE] = encode_monster(monster_id)
        enemy_block[base + MONSTER_VEC_SIZE:base + ENEMY_EXT_PER_SLOT] = encode_powers(powers)
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
        # Absent means playable, matching the mask's own fallback: only an
        # explicit False is authoritative for illegality.
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
        # The mod has to send this; without it deckbuilding stays blind and the
        # block reads as an empty deck, which is wrong rather than absent.
        "deck_card_names": [
            c.get("id") if isinstance(c, Mapping) else c
            for c in (state.get("deck") or ())
        ],
        "enemies": enemies,
    }
