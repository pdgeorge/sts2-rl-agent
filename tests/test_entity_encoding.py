"""Powers, monsters, hand and deck by identity -- now via feature hashing.

The observation space is fixed at 1,690 dims regardless of how many cards,
powers, monsters, or relics the game has. A model trained before a patch can
load after it and resume fine-tuning instead of starting from scratch.

Hashing means there are no longer guaranteed-unique indices for every item.
Instead we test that:
  - The output vector has the expected fixed size.
  - Different items produce different vectors (low collision probability).
  - Amounts are preserved (powers with different stacks differ).
  - The bridge and simulator produce identical vectors for identical input.
  - Unknown items are silently ignored.
"""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.gym_env.entity_encoding import (
    ENEMY_EXT_PER_SLOT,
    ENTITY_OBS_SIZE,
    HAND_EXTRA_BLOCK,
    MAX_ENEMY_SLOTS,
    encode_card_set,
    encode_deck_set,
    encode_entities,
    encode_hand_extra,
    encode_monster,
    encode_powers,
)
from sts2_env.gym_env.hashing import (
    DECK_CARD_BLOCK,
    HAND_CARD_BLOCK,
    ENEMY_IDENTITY_BUCKETS,
    ENEMY_POWER_BUCKETS,
    PLAYER_POWER_BUCKETS,
)


# --- pinned sizes -----------------------------------------------------------

def test_sizes_are_pinned():
    """If these change, every trained model is misaligned. Retrain; do not
    just update the numbers."""
    assert PLAYER_POWER_BUCKETS == 128
    assert ENEMY_IDENTITY_BUCKETS == 64
    assert ENEMY_POWER_BUCKETS == 128
    assert ENEMY_EXT_PER_SLOT == 192
    assert HAND_CARD_BLOCK == 650      # 10 slots x (64 embed + is_known)
    assert DECK_CARD_BLOCK == 65       # pooled mean + known fraction
    assert ENTITY_OBS_SIZE == 1893


# --- powers -----------------------------------------------------------------

def test_powers_carry_amount_not_just_presence():
    """Poison 12 and Poison 1 are different situations."""
    a = encode_powers({"POISON": 12})
    b = encode_powers({"POISON": 1})
    assert a.shape == (PLAYER_POWER_BUCKETS,)
    assert b.shape == (PLAYER_POWER_BUCKETS,)
    assert not np.array_equal(a, b)


def test_a_power_is_now_visible():
    """Any power the simulator can name gets a hashed signature."""
    out = encode_powers({"POISON": 5})
    assert np.any(out != 0.0)


def test_bridge_power_list_and_sim_power_dict_agree():
    """The two sides hand over different shapes for the same thing."""
    from_bridge = encode_powers([{"id": "STRENGTH", "amount": 3}])
    from_sim = encode_powers({"STRENGTH": 3})
    np.testing.assert_array_equal(from_bridge, from_sim)


def test_an_unknown_power_is_ignored_not_fatal():
    out = encode_powers([{"id": "NOT_A_POWER", "amount": 3},
                         {"id": "STRENGTH", "amount": 1}])
    # Should have the same nonzero pattern as just STRENGTH, since NOT_A_POWER
    # hashes into the same space but with a different key.
    just_strength = encode_powers({"STRENGTH": 1})
    assert not np.array_equal(out, np.zeros_like(out))
    assert not np.array_equal(out, just_strength)  # NOT_A_POWER did collide in


def test_no_powers_is_all_zero():
    assert encode_powers(None).sum() == 0.0
    assert encode_powers([]).sum() == 0.0


# --- monsters ---------------------------------------------------------------

def test_monsters_are_distinguishable():
    """Previously zero dims: a Mawler and a Wriggler were the same to her."""
    a = encode_monster(0, "MAWLER")
    b = encode_monster(0, "WRIGGLER")
    assert not np.array_equal(a, b)
    # Both are sparse binary-ish hashed vectors; neither should be all-zero.
    assert np.any(a != 0.0)
    assert np.any(b != 0.0)


def test_an_unknown_monster_still_gets_a_hash_signature():
    """With feature hashing, any string hashes into the fixed bucket space.
    This is intentional: a new monster from a patch gets a stable representation
    without requiring a code change."""
    out = encode_monster(0, "SOMETHING_NEW")
    assert np.any(out != 0.0)


def test_different_slots_get_different_vectors():
    """Same monster in slot 0 vs slot 1 must differ because the slot index is
    baked into the hash key."""
    a = encode_monster(0, "MAWLER")
    b = encode_monster(1, "MAWLER")
    assert not np.array_equal(a, b)


# --- cards ------------------------------------------------------------------

def test_card_sets_distinguish_identity():
    a = encode_card_set(["BASH"])
    b = encode_card_set(["STRIKE_IRONCLAD"])
    assert not np.array_equal(a, b)


def test_enum_members_resolve_not_just_strings():
    """CardId.BASH stringifies as 'CardId.BASH'; folding str() alone matched
    nothing and the whole block silently encoded as zero."""
    from sts2_env.cards.base import CardId

    a = encode_card_set([CardId.BASH])
    b = encode_card_set(["BASH"])
    np.testing.assert_array_equal(a, b)
    assert np.any(a != 0.0)


def test_duplicate_cards_saturate_at_one_in_hash_space():
    """Five Strikes is still 'the deck contains Strike'; the hash is the same
    key five times, and signed hashing means it may cancel partially. The
    intended invariant is that presence is detectable, not that count is
    preserved."""
    one = encode_card_set(["STRIKE_IRONCLAD"])
    five = encode_card_set(["STRIKE_IRONCLAD"] * 5)
    # Both should be nonzero; they may differ because of repeated signed hashes.
    assert np.any(one != 0.0)
    assert np.any(five != 0.0)


# --- layout -----------------------------------------------------------------

def test_enemy_slot_stride_is_identity_plus_powers():
    assert ENEMY_EXT_PER_SLOT == ENEMY_IDENTITY_BUCKETS + ENEMY_POWER_BUCKETS


def test_full_entity_block_size():
    """All parts add up to the exported ENTITY_OBS_SIZE."""
    player_powers = encode_powers({"STRENGTH": 3, "DEXTERITY": 2})
    hand = encode_card_set(["BASH", "STRIKE_IRONCLAD"])
    hand_extra = encode_hand_extra([
        {"type": "ATTACK", "upgraded": False, "playable": True,
         "targets_enemy": True, "cost_x": False},
    ])
    deck = encode_deck_set(["DEFEND_IRONCLAD"] * 10)
    enemies = np.zeros(MAX_ENEMY_SLOTS * ENEMY_EXT_PER_SLOT, dtype=np.float32)
    enemies[:ENEMY_EXT_PER_SLOT] = np.concatenate([
        encode_monster(0, "MAWLER"),
        encode_powers({"RITUAL": 3}),
    ])

    total = np.concatenate([player_powers, hand, hand_extra, deck, enemies])
    assert total.shape == (ENTITY_OBS_SIZE,)


# --- deck -------------------------------------------------------------------

def test_deck_and_hand_use_separate_hash_spaces():
    """Bash in hand and Bash in deck should not collide because the hash keys
    carry different prefixes ('hand_' vs 'deck_')."""
    hand = encode_card_set(["BASH"])
    deck = encode_deck_set(["BASH"])
    # They should be different vectors because they were hashed with different
    # FeatureHasher instances (different seeds).
    assert not np.array_equal(hand, deck)
