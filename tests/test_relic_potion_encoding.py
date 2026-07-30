"""Relic and potion identity in the observation, and the indices staying put.

The observation used to carry relic_count and potion_count -- how many, never
which -- so every decision was made without knowing whether she holds Snecko Eye
or Ice Cream. These tests cover the encoding that fixes that, and one thing that
matters more than the encoding: index stability.

Every index here is a column in a trained model's input layer. If RelicId gains a
member in the middle, or a potion is added to the registry, the columns shift and
every existing model silently misreads its own observation. No exception, no
warning, just worse play that looks like a bad training run. The size tests exist
to make that change fail loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.gym_env.relic_potion_encoding import (
    POTION_IDS,
    POTION_OBS_SIZE,
    POTION_SLOTS,
    POTION_TYPE_COUNT,
    RELIC_IDS,
    RELIC_OBS_SIZE,
    RELIC_POTION_OBS_SIZE,
    encode_potions,
    encode_relics,
    potion_index,
    potion_slots_from_bridge_state,
    relic_index,
    relics_from_bridge_state,
)


# --- index stability --------------------------------------------------------

def test_relic_count_is_pinned():
    """If this fails, RelicId changed and every trained model is now misaligned.

    Retrain, or map old columns to new ones. Do not just update the number.
    """
    assert RELIC_OBS_SIZE == 299


def test_potion_count_is_pinned():
    """Same: a new potion shifts every column after it."""
    assert POTION_TYPE_COUNT == 64


def test_observation_size_is_pinned():
    assert POTION_OBS_SIZE == POTION_SLOTS * POTION_TYPE_COUNT
    assert RELIC_POTION_OBS_SIZE == RELIC_OBS_SIZE + POTION_OBS_SIZE == 299 + 5 * 64


def test_the_starting_relic_has_a_stable_index():
    """Burning Blood is RelicId member 1, so it anchors the ordering."""
    assert relic_index("BURNING_BLOOD") == 0
    assert RELIC_IDS[0] == "BURNING_BLOOD"


# --- relics -----------------------------------------------------------------

def test_relics_are_multi_hot_by_identity():
    out = encode_relics(["BURNING_BLOOD", "BLACK_BLOOD"])
    assert out[relic_index("BURNING_BLOOD")] == 1.0
    assert out[relic_index("BLACK_BLOOD")] == 1.0
    assert out.sum() == 2.0, "exactly the relics held, nothing else"


def test_two_different_relics_are_distinguishable():
    """The whole point: a count could not tell these apart."""
    a = encode_relics(["BURNING_BLOOD"])
    b = encode_relics(["BLACK_BLOOD"])
    assert not np.array_equal(a, b)
    assert a.sum() == b.sum() == 1.0, "same count, different observation"


def test_no_relics_is_all_zero():
    assert encode_relics([]).sum() == 0.0


def test_an_unknown_relic_is_ignored_rather_than_crashing():
    """A relic the simulator does not model must not end a live run."""
    out = encode_relics(["BURNING_BLOOD", "NOT_A_REAL_RELIC"])
    assert out.sum() == 1.0


def test_naming_conventions_do_not_change_the_answer():
    """Sim spells it BURNING_BLOOD; the game has differed before."""
    expected = relic_index("BURNING_BLOOD")
    for variant in ("BurningBlood", "burning_blood", "Burning Blood", "BURNINGBLOOD"):
        assert relic_index(variant) == expected, f"{variant} must resolve"


# --- potions ----------------------------------------------------------------

def test_potions_are_positional():
    """Slot matters: the action space picks a slot, not a potion name."""
    first = POTION_IDS[0]
    in_slot0 = encode_potions([first, None])
    in_slot1 = encode_potions([None, first])
    assert not np.array_equal(in_slot0, in_slot1), "the same potion in a different slot"
    assert in_slot0[potion_index(first)] == 1.0
    assert in_slot1[POTION_TYPE_COUNT + potion_index(first)] == 1.0


def test_empty_slots_stay_zero():
    out = encode_potions([None, None, None])
    assert out.sum() == 0.0


def test_slots_beyond_the_cap_are_dropped_not_wrapped():
    """Writing past the cap would land on another slot's columns."""
    out = encode_potions([POTION_IDS[0]] * (POTION_SLOTS + 3))
    assert out.sum() == POTION_SLOTS


def test_an_unknown_potion_is_ignored(caplog):
    out = encode_potions(["NotARealPotion", POTION_IDS[0]])
    assert out.sum() == 1.0


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
    assert out[potion_index("BlockPotion")] == 0.0, "slot 0 must stay empty"
    assert out[POTION_TYPE_COUNT + potion_index("BlockPotion")] == 1.0
