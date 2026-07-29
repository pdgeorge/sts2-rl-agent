"""Ironclad factory upgrade parity tests backed by reference card models."""

import pytest

import sts2_env.powers  # noqa: F401

from sts2_env.cards.factory import create_card, create_reference_card
from sts2_env.core.enums import CardId


IRONCLAD_FACTORY_UPGRADE_CARD_IDS = (
    CardId.AGGRESSION_CARD,
    CardId.ANGER,
    CardId.ASHEN_STRIKE,
    CardId.BASH,
    CardId.BATTLE_TRANCE,
    CardId.BLOODLETTING,
    CardId.BLOOD_WALL,
    CardId.BODY_SLAM,
    CardId.BRAND,
    CardId.BREAK,
    CardId.BREAKTHROUGH,
    CardId.BULLY,
    CardId.CASCADE,
    CardId.CINDER,
    CardId.COLOSSUS_CARD,
    CardId.CONFLAGRATION,
    CardId.CRIMSON_MANTLE,
    CardId.CRUELTY_CARD,
    CardId.DARK_EMBRACE_CARD,
    CardId.DEFEND_IRONCLAD,
    CardId.DISMANTLE,
    CardId.DOMINATE,
    CardId.DRUM_OF_BATTLE_CARD,
    CardId.EVIL_EYE,
    CardId.EXPECT_A_FIGHT,
    CardId.FEED,
    CardId.FEEL_NO_PAIN_CARD,
    CardId.FIEND_FIRE,
    CardId.FIGHT_ME,
    CardId.FORGOTTEN_RITUAL,
    CardId.HAVOC,
    CardId.HELLRAISER_CARD,
    CardId.HEMOKINESIS,
    CardId.HOWL_FROM_BEYOND,
    CardId.IMPERVIOUS,
    CardId.INFERNAL_BLADE,
    CardId.INFERNO_CARD,
    CardId.INFLAME,
    CardId.IRON_WAVE,
    CardId.JUGGERNAUT_CARD,
    CardId.JUGGLING_CARD,
    CardId.MANGLE,
    CardId.MOLTEN_FIST,
    CardId.OFFERING,
    CardId.ONE_TWO_PUNCH_CARD,
    CardId.PACTS_END,
    CardId.PERFECTED_STRIKE,
    CardId.PILLAGE,
    CardId.POMMEL_STRIKE,
    CardId.PRIMAL_FORCE,
    CardId.PYRE,
    CardId.RAGE_CARD,
    CardId.RAMPAGE,
    CardId.RUPTURE_CARD,
    CardId.SECOND_WIND,
    CardId.SETUP_STRIKE_CARD,
    CardId.SHRUG_IT_OFF,
    CardId.SPITE,
    CardId.STAMPEDE_CARD,
    CardId.STOKE,
    CardId.STOMP,
    CardId.STONE_ARMOR,
    CardId.STRIKE_IRONCLAD,
    CardId.SWORD_BOOMERANG,
    CardId.TANK_CARD,
    CardId.TAUNT,
    CardId.TEAR_ASUNDER,
    CardId.THRASH,
    CardId.THUNDERCLAP,
    CardId.TREMBLE,
    CardId.TWIN_STRIKE,
    CardId.UNMOVABLE,
    CardId.UNRELENTING,
    CardId.UPPERCUT,
    CardId.VICIOUS_CARD,
)


def _ironclad_card_id(card_id: CardId) -> str:
    return card_id.name


@pytest.mark.parametrize("card_id", IRONCLAD_FACTORY_UPGRADE_CARD_IDS, ids=_ironclad_card_id)
def test_ironclad_factory_upgrade_core_metadata_matches_reference(card_id: CardId):
    actual = create_card(card_id, upgraded=True)
    expected = create_reference_card(card_id, upgraded=True, allow_generation=True)

    assert actual.upgraded is expected.upgraded
    assert actual.cost == expected.cost
    assert actual.has_energy_cost_x == expected.has_energy_cost_x
    assert actual.card_type == expected.card_type
    assert actual.target_type == expected.target_type
    assert actual.rarity == expected.rarity
    assert actual.star_cost == expected.star_cost
    assert actual.keywords == expected.keywords
    assert actual.tags == expected.tags


@pytest.mark.parametrize("card_id", IRONCLAD_FACTORY_UPGRADE_CARD_IDS, ids=_ironclad_card_id)
def test_ironclad_factory_upgrade_dynamic_values_match_reference(card_id: CardId):
    actual = create_card(card_id, upgraded=True)
    expected = create_reference_card(card_id, upgraded=True, allow_generation=True)

    for key, expected_value in expected.effect_vars.items():
        actual_value = actual.effect_vars.get(key)
        if key in {"damage", "calc_base"} and actual_value is None:
            assert actual.base_damage == expected_value
            continue
        if key == "block" and actual_value is None:
            assert actual.base_block == expected_value
            continue
        assert actual_value == expected_value
