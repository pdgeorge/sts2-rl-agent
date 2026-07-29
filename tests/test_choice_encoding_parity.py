"""The simulator and the live game must encode the same decision identically.

This is the anti-drift mechanism for the run observation. The two sides describe
the same options with different field names -- card_id against id, point_type
against type, option_id against id -- and their vocabularies differ in spelling,
REST_SITE against RestSite. Every one of those has already caused a bug that
looked like something else: missing content, a broken mod, simulator drift.

If these tests pass, a policy trained in the simulator reads the live game the
same way it read training. If they fail, it does not, and the failure would
otherwise show up only as "she plays worse on stream than she did in testing".
"""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.gym_env.choice_encoding import (
    CARD_BLOCK,
    CHOICE_OBS_SIZE,
    CHOICE_SLOTS,
    MAP_POINT_TYPES,
    NODE_BLOCK,
    OPTION_BLOCK,
    choices_from_bridge_state,
    choices_from_sim_actions,
    encode_choices,
    normalize_enum_name,
    resolve_card_id,
)


def _encode(extracted: dict) -> np.ndarray:
    return encode_choices(**extracted)


class TestNameNormalisation:
    @pytest.mark.parametrize("game, simulator", [
        ("RestSite", "REST_SITE"),
        ("Monster", "MONSTER"),
        ("Elite", "ELITE"),
        ("Boss", "BOSS"),
        ("Unassigned", "UNASSIGNED"),
        # Already-normalised names must survive unchanged, since the simulator
        # side sends them that way.
        ("REST_SITE", "REST_SITE"),
        ("MONSTER", "MONSTER"),
    ])
    def test_both_spellings_normalise_together(self, game, simulator):
        assert normalize_enum_name(game) == normalize_enum_name(simulator)

    def test_every_map_point_type_is_reachable_from_the_game_spelling(self):
        """A node type the game sends must land in the one-hot, not fall through.

        The game's enum is Unassigned/Unknown/Shop/Treasure/RestSite/Monster/
        Elite/Boss/Ancient; the simulator's is the same nine in UPPER_SNAKE. If a
        member stopped matching, that node would encode as all-zero and the agent
        would be blind to one room type without anything failing.
        """
        game_spellings = ["Unassigned", "Unknown", "Shop", "Treasure",
                          "RestSite", "Monster", "Elite", "Boss", "Ancient"]
        assert len(game_spellings) == len(MAP_POINT_TYPES)
        for spelling in game_spellings:
            assert normalize_enum_name(spelling) in MAP_POINT_TYPES


class TestCardIdResolution:
    def test_plain_name_resolves(self):
        assert resolve_card_id("STRIKE_IRONCLAD") is not None

    def test_suffixed_card_resolves_from_the_bare_name(self):
        """Both payloads say VICIOUS; the simulator calls it VICIOUS_CARD.

        Looking the bare name up directly is what made 24 steps of a real trace
        uncomparable and had me report the card as missing from the simulator.
        """
        assert resolve_card_id("VICIOUS") is not None
        assert resolve_card_id("BARRICADE") is not None

    def test_unknown_card_is_none_rather_than_raising(self):
        assert resolve_card_id("NOT_A_REAL_CARD") is None


class TestSimAndBridgeAgree:
    def test_card_reward_encodes_identically(self):
        sim_actions = [
            {"action": "skip"},
            {"action": "pick_card", "index": 0, "card_id": "TAUNT",
             "rarity": "COMMON", "upgraded": False},
            {"action": "pick_card", "index": 1, "card_id": "TRUE_GRIT",
             "rarity": "COMMON", "upgraded": False},
            {"action": "pick_card", "index": 2, "card_id": "HEMOKINESIS",
             "rarity": "UNCOMMON", "upgraded": False},
        ]
        # Exactly the shape the mod sends: different keys, and cost/type present
        # where the simulator had rarity instead.
        bridge_state = {
            "type": "card_reward",
            "can_skip": True,
            "cards": [
                {"index": 0, "id": "TAUNT", "type": "Skill", "cost": 1},
                {"index": 1, "id": "TRUE_GRIT", "type": "Skill", "cost": 1},
                {"index": 2, "id": "HEMOKINESIS", "type": "Attack", "cost": 1},
            ],
        }
        np.testing.assert_array_equal(
            _encode(choices_from_sim_actions(sim_actions)),
            _encode(choices_from_bridge_state(bridge_state)),
        )

    def test_map_choice_encodes_identically(self):
        sim_actions = [
            {"action": "move", "coord": [1, 0], "point_type": "MONSTER"},
            {"action": "move", "coord": [1, 2], "point_type": "SHOP"},
            {"action": "move", "coord": [1, 5], "point_type": "REST_SITE"},
        ]
        bridge_state = {
            "type": "map_select",
            "nodes": [
                {"index": 0, "row": 1, "col": 0, "type": "Monster"},
                {"index": 1, "row": 1, "col": 2, "type": "Shop"},
                {"index": 2, "row": 1, "col": 5, "type": "RestSite"},
            ],
        }
        np.testing.assert_array_equal(
            _encode(choices_from_sim_actions(sim_actions)),
            _encode(choices_from_bridge_state(bridge_state)),
        )

    def test_rest_site_encodes_identically(self):
        sim_actions = [
            {"action": "rest_option", "option_id": "HEAL", "label": "Rest",
             "enabled": True, "description": "Heal 30% of max HP"},
            {"action": "rest_option", "option_id": "SMITH", "label": "Smith",
             "enabled": True, "description": "Upgrade a card"},
        ]
        bridge_state = {
            "type": "rest_site",
            "options": [
                {"index": 0, "action": "rest_option", "id": "HEAL",
                 "label": "Rest", "enabled": True},
                {"index": 1, "action": "rest_option", "id": "SMITH",
                 "label": "Smith", "enabled": True},
            ],
        }
        np.testing.assert_array_equal(
            _encode(choices_from_sim_actions(sim_actions)),
            _encode(choices_from_bridge_state(bridge_state)),
        )


class TestEncodingProperties:
    def test_block_is_the_declared_size(self):
        assert encode_choices().shape == (CHOICE_OBS_SIZE,)
        assert CHOICE_OBS_SIZE == CARD_BLOCK + NODE_BLOCK + OPTION_BLOCK

    def test_nothing_offered_is_all_zero(self):
        """An idle phase must look empty, exactly as the combat slice does."""
        assert not encode_choices().any()

    def test_different_cards_produce_different_vectors(self):
        """The property whose absence caused all of this.

        Before, a card reward's observation was identical whatever was offered.
        If this ever passes trivially again, the agent is blind and nothing else
        in the run observation will say so.
        """
        a = encode_choices(card_names=["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"])
        b = encode_choices(card_names=["BASH", "HEMOKINESIS"])
        assert not np.array_equal(a, b)

    def test_slot_order_is_preserved(self):
        """Swapping two offers must change the vector.

        Slots line up with action indices, so order carrying no information would
        mean the agent could not tell which slot it was selecting.
        """
        a = encode_choices(card_names=["STRIKE_IRONCLAD", "BASH"])
        b = encode_choices(card_names=["BASH", "STRIKE_IRONCLAD"])
        assert not np.array_equal(a, b)

    def test_unknown_card_holds_its_slot(self):
        """An unimplemented card must not shift the cards after it.

        Slots map to action indices. Compacting the list would make the agent
        select a different card than the one it evaluated.
        """
        known_second = encode_choices(card_names=["NOT_A_REAL_CARD", "BASH"])
        bash_alone = encode_choices(card_names=["BASH"])
        assert not np.array_equal(known_second, bash_alone)

    def test_more_options_than_slots_does_not_overflow(self):
        many = [f"CARD_{i}" for i in range(CHOICE_SLOTS * 3)]
        assert encode_choices(card_names=many).shape == (CHOICE_OBS_SIZE,)
