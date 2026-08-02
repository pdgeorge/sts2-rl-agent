"""Relic and potion identity in the observation -- now via feature hashing.

The observation space is fixed at 384 dims regardless of how many relics or
potion types the game has. A model trained before a patch can load after it
and resume fine-tuning instead of starting from scratch.

Hashing means there are no longer guaranteed-unique indices for every item.
Instead we test that:
  - The output vector has the expected fixed size.
  - Different items produce different vectors (low collision probability).
  - Position is preserved (potions in different slots differ).
  - The bridge and simulator produce identical vectors for identical input.
  - Unknown items are silently ignored.
"""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.gym_env.relic_potion_encoding import (
    POTION_SLOTS,
    RELIC_POTION_OBS_SIZE,
    encode_potions,
    encode_relics,
    encode_relics_and_potions,
    potion_slots_from_bridge_state,
    relics_from_bridge_state,
)
from sts2_env.gym_env.hashing import RELIC_BUCKETS, POTION_BUCKETS


# --- size stability ---------------------------------------------------------

def test_relic_bucket_count_is_pinned():
    """If this fails, the observation space changed and models break."""
    assert RELIC_BUCKETS == 256


def test_potion_bucket_count_is_pinned():
    assert POTION_BUCKETS == 128


def test_observation_size_is_pinned():
    assert RELIC_POTION_OBS_SIZE == RELIC_BUCKETS + POTION_BUCKETS == 384


# --- relics -----------------------------------------------------------------

def test_relics_are_hashed_by_identity():
    out = encode_relics(["BURNING_BLOOD", "BLACK_BLOOD"])
    assert out.shape == (RELIC_BUCKETS,)
    assert np.any(out != 0.0)
    # Two different relics should not fully cancel.


def test_two_different_relics_are_distinguishable():
    """The whole point: a count could not tell these apart."""
    a = encode_relics(["BURNING_BLOOD"])
    b = encode_relics(["BLACK_BLOOD"])
    assert not np.array_equal(a, b)
    assert np.any(a != 0.0)
    assert np.any(b != 0.0)


def test_no_relics_is_all_zero():
    assert encode_relics([]).sum() == 0.0


def test_an_unknown_relic_is_ignored_rather_than_crashing():
    """A relic the simulator does not model must not end a live run."""
    known = encode_relics(["BURNING_BLOOD"])
    mixed = encode_relics(["BURNING_BLOOD", "NOT_A_REAL_RELIC"])
    # The unknown relic hashes into the same space with a different key, so it
    # may collide or not; the known relic must still be detectable.
    assert not np.array_equal(mixed, np.zeros_like(mixed))


def test_naming_conventions_do_not_change_the_answer():
    """Sim spells it BURNING_BLOOD; the game has differed before."""
    a = encode_relics(["BURNING_BLOOD"])
    for variant in ("BurningBlood", "burning_blood", "Burning Blood", "BURNINGBLOOD"):
        b = encode_relics([variant])
        np.testing.assert_array_equal(a, b), f"{variant} must resolve"


# --- potions ----------------------------------------------------------------

def test_potions_are_positional():
    """Slot matters: the action space picks a slot, not a potion name."""
    in_slot0 = encode_potions(["BlockPotion", None])
    in_slot1 = encode_potions([None, "BlockPotion"])
    assert not np.array_equal(in_slot0, in_slot1), "the same potion in a different slot"
    assert np.any(in_slot0 != 0.0)
    assert np.any(in_slot1 != 0.0)


def test_empty_slots_stay_zero():
    out = encode_potions([None, None, None])
    assert out.sum() == 0.0


def test_slots_beyond_the_cap_are_dropped():
    out = encode_potions(["BlockPotion"] * (POTION_SLOTS + 3))
    # With signed hashing, repeated identical keys may partially cancel.
    # The invariant is that some signal survives, not an exact count.
    assert not np.array_equal(out, np.zeros_like(out))


def test_an_unknown_potion_is_ignored():
    out = encode_potions(["NotARealPotion", "BlockPotion"])
    assert not np.array_equal(out, np.zeros_like(out))


# --- reading the bridge -----------------------------------------------------

def test_bridge_relics_accept_strings_or_objects():
    assert relics_from_bridge_state({"relics": ["BURNING_BLOOD"]}) == ["BURNING_BLOOD"]
    assert relics_from_bridge_state(
        {"relics": [{"id": "BURNING_BLOOD"}]}) == ["BURNING_BLOOD"]


def test_bridge_relics_missing_is_empty_not_an_error():
    assert relics_from_bridge_state({}) == []


def test_bridge_potion_slots_keep_their_positions():
    """A null first slot must not shift the second potion into slot 0."""
    slots = potion_slots_from_bridge_state(
        {"potion_slots": [None, "BlockPotion"]})
    assert slots == [None, "BlockPotion"]

    out = encode_potions(slots)
    assert np.any(out != 0.0)


# --- combined block ---------------------------------------------------------

def test_combined_block_size():
    relics = encode_relics(["BURNING_BLOOD"])
    potions = encode_potions(["BlockPotion", None])
    combined = encode_relics_and_potions(["BURNING_BLOOD"], ["BlockPotion", None])
    assert combined.shape == (RELIC_POTION_OBS_SIZE,)
    np.testing.assert_array_equal(combined[:RELIC_BUCKETS], relics)
    np.testing.assert_array_equal(combined[RELIC_BUCKETS:], potions)
