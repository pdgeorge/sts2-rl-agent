"""Powers, monsters, hand and deck by identity -- and the columns staying put.

The 131-dim combat observation showed 6 of 279 powers on the player, 3 of 279 on
each enemy, no monster identity at all, and card identity as a single scalar
`(index + 1) / 600`. Poison, Thorns, Curl Up, Ritual, Metallicize, Barricade and
270 others were simply invisible, and the deck -- the thing every card reward is
deciding about -- was a single number.

As with the relic block, the sizes are pinned. Every index is a column in a
trained model's input layer, so a PowerId inserted in the middle silently
misaligns every model: no error, just worse play that looks like a bad run.
"""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.gym_env.entity_encoding import (
    CARD_SET_SIZE,
    ENEMY_EXT_PER_SLOT,
    ENTITY_OBS_SIZE,
    MONSTER_VEC_SIZE,
    POWER_AMOUNT_SCALE,
    POWER_VEC_SIZE,
    card_index,
    encode_card_set,
    encode_monster,
    encode_powers,
    monster_index,
    power_index,
)


# --- pinned sizes -----------------------------------------------------------

def test_sizes_are_pinned():
    """If these change, every trained model is misaligned. Retrain; do not
    just update the numbers."""
    assert POWER_VEC_SIZE == 279
    assert MONSTER_VEC_SIZE == 36
    assert CARD_SET_SIZE == 599
    assert ENTITY_OBS_SIZE == 3142


# --- powers -----------------------------------------------------------------

def test_powers_carry_amount_not_just_presence():
    """Poison 12 and Poison 1 are different situations."""
    a = encode_powers({"POISON": 12})
    b = encode_powers({"POISON": 1})
    assert a[power_index("POISON")] == pytest.approx(12 / POWER_AMOUNT_SCALE)
    assert b[power_index("POISON")] == pytest.approx(1 / POWER_AMOUNT_SCALE)
    assert not np.array_equal(a, b)


def test_a_power_outside_the_old_nine_is_now_visible():
    """Poison was one of the 270 the observation could not represent."""
    out = encode_powers({"POISON": 5})
    assert out.sum() > 0


def test_bridge_power_list_and_sim_power_dict_agree():
    """The two sides hand over different shapes for the same thing."""
    from_bridge = encode_powers([{"id": "STRENGTH", "amount": 3}])
    from_sim = encode_powers({"STRENGTH": 3})
    np.testing.assert_array_equal(from_bridge, from_sim)


def test_an_unknown_power_is_ignored_not_fatal():
    out = encode_powers([{"id": "NOT_A_POWER", "amount": 3},
                         {"id": "STRENGTH", "amount": 1}])
    assert out[power_index("STRENGTH")] == pytest.approx(1 / POWER_AMOUNT_SCALE)


def test_no_powers_is_all_zero():
    assert encode_powers(None).sum() == 0.0
    assert encode_powers([]).sum() == 0.0


# --- monsters ---------------------------------------------------------------

def test_monsters_are_distinguishable():
    """Previously zero dims: a Mawler and a Wriggler were the same to her."""
    a, b = encode_monster("MAWLER"), encode_monster("WRIGGLER")
    assert not np.array_equal(a, b)
    assert a.sum() == b.sum() == 1.0


def test_an_unknown_monster_encodes_as_all_zero():
    assert encode_monster("SOMETHING_NEW").sum() == 0.0


# --- cards ------------------------------------------------------------------

def test_card_sets_distinguish_identity():
    a = encode_card_set(["BASH"])
    b = encode_card_set(["STRIKE_IRONCLAD"])
    assert not np.array_equal(a, b)


def test_enum_members_resolve_not_just_strings():
    """CardId.BASH stringifies as "CardId.BASH"; folding str() alone matched
    nothing and the whole block silently encoded as zero."""
    from sts2_env.cards.base import CardId

    assert card_index(CardId.BASH) == card_index("BASH")
    assert encode_card_set([CardId.STRIKE_IRONCLAD]).sum() == 1.0


def test_relic_enum_members_resolve_too():
    from sts2_env.gym_env.relic_potion_encoding import encode_relics, relic_index
    from sts2_env.relics.base import RelicId

    assert relic_index(RelicId.BURNING_BLOOD) == relic_index("BURNING_BLOOD")
    assert encode_relics([RelicId.BURNING_BLOOD]).sum() == 1.0


def test_duplicate_cards_saturate_at_one():
    """Five Strikes is still "the deck contains Strike"; counts live elsewhere."""
    assert encode_card_set(["STRIKE_IRONCLAD"] * 5).sum() == 1.0


def test_naming_variants_resolve():
    expected = card_index("BASH")
    for variant in ("Bash", "bash", "BASH"):
        assert card_index(variant) == expected


# --- layout -----------------------------------------------------------------

def test_enemy_slot_stride_is_identity_plus_powers():
    assert ENEMY_EXT_PER_SLOT == MONSTER_VEC_SIZE + POWER_VEC_SIZE
