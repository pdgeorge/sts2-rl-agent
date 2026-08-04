"""Engineered deck quality features for the meta-policy.

The meta-policy cannot learn deckbuilding from hash patterns alone. These features
are pre-computed synergy signals that tell the policy "this deck generates a lot
of block and has a block-to-damage converter" without it having to discover that
pattern from scratch.

Every feature is derived from card PROPERTIES (cost, type, damage, block, keywords)
not from card IDENTITY. This means a new card automatically contributes to the
right features as long as its properties are in the static metadata. A card that
deals 8 damage and costs 2 contributes to damage_sources and damage_per_energy
whether it is Bash, Strike, or a new card from next patch.

Features (32 dims):
  Defensive (5)
    block_sources        count of cards with base_block > 0
    block_per_energy     total block / total energy cost of block cards
    healing_sources      cards with HEAL keyword
    defensive_ratio      (block + heal cards) / deck size
    survival_score       block_per_energy * block_sources / deck_size

  Offensive (5)
    damage_sources       count of cards with base_damage > 0
    damage_per_energy    total damage / total energy cost of damage cards
    scaling_sources      cards with STRENGTH, DEXTERITY, VULNERABLE keywords
    burst_potential      max base_damage among damage cards
    offensive_ratio      damage cards / deck size

  Synergy (8)
    has_block_consumer    1.0 if any card converts block to damage/effect
    has_exhaust_synergy     1.0 if exhaust generators AND beneficiaries
    has_strength_scaling    1.0 if Strength source + multi-hit attack
    has_vulnerable         1.0 if any card applies VULNERABLE
    draw_density           draw cards / deck size
    energy_generation      cards that give ENERGY
    retain_count           cards with RETAIN
    combo_potential        (draw + energy + retain) / deck size

  Economy (6)
    x_cost_count         cards with has_energy_cost_x
    high_cost_count      cards with cost >= 3
    low_cost_count       cards with cost <= 1
    avg_cost             mean energy cost
    free_count           cards with cost == 0
    cost_variance        variance of costs

  Composition (5)
    attack_ratio         attacks / deck size
    skill_ratio          skills / deck size
    power_ratio          powers / deck size
    upgraded_ratio       upgraded cards / deck size
    curse_ratio          curses / deck size

  Archetype signals (3)
    barricade_signal     block sources + block consumer + high skill ratio
    exhaust_signal       exhaust generators + exhaust beneficiaries
    strength_signal      strength sources + multi-hit attacks + vulnerable
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from sts2_env.cards.reference_static_metadata import (
    ReferenceCardStaticMetadata,
    reference_metadata_by_card_id,
)
from sts2_env.core.enums import CardId, CardType, CardRarity


# Cache the metadata lookup
_METADATA_CACHE: dict[CardId, ReferenceCardStaticMetadata] = {}


def _get_metadata(card_id: Any) -> ReferenceCardStaticMetadata | None:
    """Lookup card metadata by CardId enum or string name."""
    if card_id is None:
        return None

    # Normalize to CardId enum
    if isinstance(card_id, str):
        # Try direct lookup
        name = card_id.upper()
        if name in CardId.__members__:
            card_id = CardId[name]
        else:
            # Try common suffixes
            for suffix in ("", "_CARD", "_STATUS"):
                if name + suffix in CardId.__members__:
                    card_id = CardId[name + suffix]
                    break
            else:
                return None
    elif isinstance(card_id, Mapping):
        # Bridge sends deck cards as dicts with "id" / "name" keys
        card_id = card_id.get("id") or card_id.get("name")
        if card_id is None:
            return None
        return _get_metadata(card_id)
    elif not isinstance(card_id, CardId):
        # Might be an enum from another module or a card instance
        if hasattr(card_id, "card_id"):
            card_id = card_id.card_id
        elif hasattr(card_id, "name"):
            return _get_metadata(card_id.name)
        else:
            return None

    if card_id in _METADATA_CACHE:
        return _METADATA_CACHE[card_id]

    # Bulk load on first miss
    all_meta = reference_metadata_by_card_id()
    for cid, meta in all_meta.items():
        _METADATA_CACHE[cid] = meta

    return _METADATA_CACHE.get(card_id)


# ─── feature constants ───────────────────────────────────────────────────

DECK_FEATURE_SIZE: int = 32

# Keyword sets we care about
BLOCK_CONSUMER_KEYWORDS: frozenset[str] = frozenset({
    "body_slam", "barricade", "juggernaut", "entrench",
})
EXHAUST_GENERATOR_KEYWORDS: frozenset[str] = frozenset({
    "corruption", "feel_no_pain", "dark_embrace", "second_wind",
    "burning_pact", "true_grit",
})
EXHAUST_BENEFICIARY_KEYWORDS: frozenset[str] = frozenset({
    "feel_no_pain", "dark_embrace", "charons_ashes", "death_nihil",
})
STRENGTH_KEYWORDS: frozenset[str] = frozenset({
    "strength", "inflame", "demon_form", "limit_break", "brutality",
})
VULNERABLE_KEYWORDS: frozenset[str] = frozenset({
    "vulnerable", "thunderclap", "bash", "shockwave",
})
HEAL_KEYWORDS: frozenset[str] = frozenset({
    "heal", "reaper", "feed", "burning_blood",
})
ENERGY_KEYWORDS: frozenset[str] = frozenset({
    "energy", "adrenaline", "offering", "bloodletting", "pommel_strike",
})
DRAW_KEYWORDS: frozenset[str] = frozenset({
    "draw", "battle_trance", "pommel_strike", "shrug_it_off", "escape_plan",
})
RETAIN_KEYWORDS: frozenset[str] = frozenset({
    "retain", "equilibrium", "well_laid_plans", "ghostly",
})
MULTI_HIT_KEYWORDS: frozenset[str] = frozenset({
    "twin_strike", "sword_boomerang", "pummel", "whirlwind",
})


def _card_name_lower(card_id: Any) -> str:
    """Get a lowercased name for keyword matching."""
    if card_id is None:
        return ""
    if isinstance(card_id, CardId):
        return card_id.name.lower()
    if hasattr(card_id, "name"):
        return str(card_id.name).lower()
    return str(card_id).lower()


def _has_keyword(card_id: Any, keywords: frozenset[str]) -> bool:
    """Check if a card's name contains any of the given keywords."""
    name = _card_name_lower(card_id)
    return any(kw in name for kw in keywords)


def _is_upgraded(card: Any) -> bool:
    """Check if a card instance is upgraded.

    Handles the bridge's dict form as well as CardInstance. Without the Mapping
    branch the upgrade features read zero for every live deck, because a dict has
    no `upgraded` attribute -- only a key of that name.
    """
    if isinstance(card, Mapping):
        return bool(card.get("upgraded") or card.get("is_upgraded"))
    if isinstance(card, str):
        return card.endswith("+")
    return bool(getattr(card, "upgraded", False))


def encode_deck_features(cards: Iterable[Any]) -> np.ndarray:
    """Compute 32 engineered deck quality features.

    cards may be CardInstance objects, CardId enums, or strings.
    """
    features = np.zeros(DECK_FEATURE_SIZE, dtype=np.float32)
    if not cards:
        return features

    card_list = list(cards)
    n = len(card_list)
    if n == 0:
        return features

    # Collect properties
    costs: list[int] = []
    damages: list[int] = []
    blocks: list[int] = []
    types: list[CardType] = []
    is_upgraded_list: list[bool] = []

    for card in card_list:
        meta = _get_metadata(card)
        if meta is None:
            continue

        costs.append(meta.cost)
        types.append(meta.card_type)
        is_upgraded_list.append(_is_upgraded(card))

        # Damage and block from static metadata... wait, it's not there.
        # We need to construct a temporary card or read from effect_vars.
        # For now, use a heuristic: look up the card instance to get base_damage/block.
        dmg = 0
        blk = 0
        if hasattr(card, "base_damage") and card.base_damage is not None:
            dmg = card.base_damage
        if hasattr(card, "base_block") and card.base_block is not None:
            blk = card.base_block
        damages.append(dmg)
        blocks.append(blk)

    if not costs:
        return features

    costs_arr = np.array(costs, dtype=np.float32)
    damages_arr = np.array(damages, dtype=np.float32)
    blocks_arr = np.array(blocks, dtype=np.float32)

    # Defensive features (0-4)
    block_mask = blocks_arr > 0
    features[0] = np.count_nonzero(block_mask) / n  # block_sources ratio
    if np.any(block_mask):
        block_costs = costs_arr[block_mask]
        features[1] = np.sum(blocks_arr[block_mask]) / np.maximum(np.sum(block_costs), 1.0)  # block_per_energy
    features[2] = sum(1 for c in card_list if _has_keyword(c, HEAL_KEYWORDS)) / n
    features[3] = features[0] + features[2]  # defensive_ratio
    features[4] = features[1] * features[0]  # survival_score

    # Offensive features (5-9)
    damage_mask = damages_arr > 0
    features[5] = np.count_nonzero(damage_mask) / n
    if np.any(damage_mask):
        damage_costs = costs_arr[damage_mask]
        features[6] = np.sum(damages_arr[damage_mask]) / np.maximum(np.sum(damage_costs), 1.0)
    features[7] = sum(1 for c in card_list if _has_keyword(c, STRENGTH_KEYWORDS)) / n
    features[8] = np.max(damages_arr) / 50.0 if len(damages_arr) > 0 else 0.0  # burst_potential, scale by 50
    features[9] = features[5]  # offensive_ratio (same as damage_sources)

    # Synergy features (10-17)
    features[10] = 1.0 if any(_has_keyword(c, BLOCK_CONSUMER_KEYWORDS) for c in card_list) else 0.0
    has_exhaust_gen = any(_has_keyword(c, EXHAUST_GENERATOR_KEYWORDS) for c in card_list)
    has_exhaust_ben = any(_has_keyword(c, EXHAUST_BENEFICIARY_KEYWORDS) for c in card_list)
    features[11] = 1.0 if has_exhaust_gen and has_exhaust_ben else 0.0
    has_strength = any(_has_keyword(c, STRENGTH_KEYWORDS) for c in card_list)
    has_multi_hit = any(_has_keyword(c, MULTI_HIT_KEYWORDS) for c in card_list)
    features[12] = 1.0 if has_strength and has_multi_hit else 0.0
    features[13] = 1.0 if any(_has_keyword(c, VULNERABLE_KEYWORDS) for c in card_list) else 0.0
    features[14] = sum(1 for c in card_list if _has_keyword(c, DRAW_KEYWORDS)) / n
    features[15] = sum(1 for c in card_list if _has_keyword(c, ENERGY_KEYWORDS)) / n
    features[16] = sum(1 for c in card_list if _has_keyword(c, RETAIN_KEYWORDS)) / n
    features[17] = features[14] + features[15] + features[16]  # combo_potential

    # Economy features (18-23)
    x_cost = sum(1 for c in card_list if hasattr(c, "has_energy_cost_x") and c.has_energy_cost_x)
    # Also check metadata
    x_cost += sum(1 for c in card_list
                   if _get_metadata(c) is not None and _get_metadata(c).has_energy_cost_x)
    features[18] = x_cost / n
    features[19] = np.count_nonzero(costs_arr >= 3) / n
    features[20] = np.count_nonzero(costs_arr <= 1) / n
    features[21] = np.mean(costs_arr) / 5.0  # avg_cost scaled
    features[22] = np.count_nonzero(costs_arr == 0) / n
    features[23] = np.var(costs_arr) / 25.0 if len(costs_arr) > 1 else 0.0  # cost_variance scaled

    # Composition features (24-28)
    type_counts = {t: 0 for t in CardType}
    for t in types:
        type_counts[t] = type_counts.get(t, 0) + 1
    features[24] = type_counts.get(CardType.ATTACK, 0) / n
    features[25] = type_counts.get(CardType.SKILL, 0) / n
    features[26] = type_counts.get(CardType.POWER, 0) / n
    features[27] = sum(is_upgraded_list) / n
    features[28] = sum(1 for c in card_list
                       if _get_metadata(c) is not None
                       and _get_metadata(c).rarity == CardRarity.CURSE) / n

    # Archetype signals (29-31)
    features[29] = features[0] + features[10] + features[25]  # barricade = block + consumer + skills
    features[30] = (sum(1 for c in card_list if _has_keyword(c, EXHAUST_GENERATOR_KEYWORDS)) / n +
                    sum(1 for c in card_list if _has_keyword(c, EXHAUST_BENEFICIARY_KEYWORDS)) / n)
    features[31] = features[7] + features[12] + features[13]  # strength archetype

    # Clip to reasonable bounds
    np.clip(features, -1.0, 10.0, out=features)
    return features
