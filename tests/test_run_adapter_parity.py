"""The live game and the simulator must produce the same run observation.

test_choice_encoding_parity covers the 126 choice dims. This covers the whole
277: combat block, run-level block, choice block, from a bridge message against
the same situation in the simulator.

It exists because the failure it prevents is invisible. A policy trained on the
simulator and deployed on a bridge that scales gold differently, orders the phase
one-hot differently, or reads a 1-based act as 0-based, does not crash. It plays
slightly wrong, forever, and the only symptom is "she is worse on stream than in
testing" -- which is indistinguishable from the model just being mediocre.

Every mismatch found by hand today had that shape: VULNERABLE_POWER against
VULNERABLE, RestSite against REST_SITE, VICIOUS against VICIOUS_CARD, and a
recorder that dropped fields the mod was sending correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.bridge.run_adapter import RunStateAdapter
from sts2_env.gym_env.observation import OBS_SIZE as COMBAT_OBS_SIZE
from sts2_env.gym_env.run_env import RUN_OBS_SIZE, TOTAL_ACTIONS
from sts2_env.gym_env.run_level_encoding import (
    RUN_LEVEL_SIZE,
    encode_run_level,
    run_level_from_bridge_state,
)


RUN_SLICE = slice(COMBAT_OBS_SIZE, COMBAT_OBS_SIZE + RUN_LEVEL_SIZE)
CHOICE_SLICE = slice(COMBAT_OBS_SIZE + RUN_LEVEL_SIZE, RUN_OBS_SIZE)


@pytest.fixture
def adapter():
    return RunStateAdapter()


def _bridge_map_state():
    """A map screen exactly as the mod sends one, run fields included."""
    return {
        "type": "map_select",
        "act": 2,               # 1-based on the wire
        "floor": 21,
        "act_floor": 4,
        "run_hp": 42,
        "run_max_hp": 80,
        "gold": 315,
        "deck_size": 18,
        "relic_count": 5,
        "potion_count": 2,
        "max_potion_slots": 3,
        "ascension": 0,
        "room_type": "Elite",
        "nodes": [
            {"index": 0, "row": 5, "col": 1, "type": "Monster"},
            {"index": 1, "row": 5, "col": 3, "type": "RestSite"},
            {"index": 2, "row": 5, "col": 4, "type": "Elite"},
        ],
    }


class TestShape:
    def test_observation_is_the_declared_size(self, adapter):
        obs = adapter.encode_observation(_bridge_map_state())
        assert obs.shape == (RUN_OBS_SIZE,)
        assert obs.dtype == np.float32

    def test_mask_is_the_declared_size(self, adapter):
        mask = adapter.compute_action_mask(_bridge_map_state())
        assert mask.shape == (TOTAL_ACTIONS,)

    def test_mask_is_never_empty(self, adapter):
        """MaskablePPO raises on an all-zero mask, which would end a live run.

        Any state must be answerable, including ones this adapter does not know.
        """
        for state in (
            _bridge_map_state(),
            {"type": "card_reward", "cards": [], "can_skip": True},
            {"type": "something_new_the_mod_added"},
            {},
        ):
            assert adapter.compute_action_mask(state).sum() >= 1


class TestRunLevelParity:
    def test_act_is_converted_from_one_based_to_zero_based(self):
        """The wire is 1-based for readability; the observation is 0-based.

        Getting this wrong shifts every act reading by a third of its scale and
        nothing raises.
        """
        kwargs = run_level_from_bridge_state(_bridge_map_state())
        assert kwargs["act_index"] == 1, "act 2 on the wire is index 1"

    def test_matches_a_hand_built_simulator_vector(self, adapter):
        """The adapter's run block equals what the simulator would write."""
        expected = encode_run_level(
            act_index=1, total_floor=21, act_floor=4,
            hp=42, max_hp=80, gold=315,
            deck_size=18, relic_count=5,
            potion_count=2, max_potion_slots=3,
            phase="MAP_CHOICE", ascension=0,
            is_elite=True, is_boss=False,
        )
        actual = adapter.encode_observation(_bridge_map_state())[RUN_SLICE]
        np.testing.assert_array_equal(actual, expected)

    def test_encoded_values_are_pinned_to_known_numbers(self, adapter):
        """Golden values, because comparing the encoder to itself proves nothing.

        test_matches_a_hand_built_simulator_vector builds its expectation with the
        same encoder it checks, so the two move together: changing GOLD_SCALE from
        1000 to 100 passed every other test in this file. A model trained before
        such a change would read every gold value ten times too large and nothing
        would fail.

        These are the numbers a model in training right now is reading. Changing
        one means retraining, so it should require deliberately editing this test.
        """
        run = adapter.encode_observation(_bridge_map_state())[RUN_SLICE]

        assert run[0] == pytest.approx(1 / 3.0), "act index 1 over ACT_SCALE 3"
        assert run[1] == pytest.approx(21 / 50.0), "floor 21 over TOTAL_FLOOR_SCALE 50"
        assert run[2] == pytest.approx(4 / 20.0), "act_floor 4 over ACT_FLOOR_SCALE 20"
        assert run[3] == pytest.approx(42 / 80.0), "hp ratio"
        assert run[4] == pytest.approx(315 / 1000.0), "gold 315 over GOLD_SCALE 1000"
        assert run[5] == pytest.approx(18 / 40.0), "deck 18 over DECK_SIZE_SCALE 40"
        assert run[6] == pytest.approx(5 / 30.0), "relics 5 over RELIC_COUNT_SCALE 30"
        assert run[7] == pytest.approx(2 / 3.0), "potions held over slots"
        assert run[8] == pytest.approx(3 / 5.0), "slots 3 over MAX_POTION_SLOTS_SCALE 5"
        assert run[9] == 1.0, "MAP_CHOICE is the first phase in PHASE_ORDER"
        assert run[10:17].sum() == 0.0, "exactly one phase may be set"
        assert run[17] == 0.0, "ascension 0"

    def test_room_type_reaches_the_elite_flag(self, adapter):
        """Elite/Boss come from MapPointType on the wire and RoomType in the sim.

        Different enums, same meaning, and the normaliser is what joins them.
        """
        obs = adapter.encode_observation(_bridge_map_state())
        assert obs[RUN_SLICE][18] == 1.0, "Elite room should set is_elite"
        assert obs[RUN_SLICE][19] == 0.0

    def test_missing_fields_do_not_raise(self, adapter):
        """An older mod, or a state a handler has not been taught to enrich.

        Zero is the honest answer; the fix is to teach the mod, not to guess here.
        """
        obs = adapter.encode_observation({"type": "map_select", "nodes": []})
        assert obs.shape == (RUN_OBS_SIZE,)


class TestChoiceParity:
    def test_map_nodes_reach_the_choice_block(self, adapter):
        obs = adapter.encode_observation(_bridge_map_state())
        assert obs[CHOICE_SLICE].any(), "three offered nodes must not encode as nothing"

    def test_different_maps_encode_differently(self, adapter):
        """The property whose absence started all of this."""
        a = _bridge_map_state()
        b = _bridge_map_state()
        b["nodes"] = [{"index": 0, "row": 5, "col": 1, "type": "Shop"}]
        obs_a = adapter.encode_observation(a)[CHOICE_SLICE]
        obs_b = adapter.encode_observation(b)[CHOICE_SLICE]
        assert not np.array_equal(obs_a, obs_b)


class TestActionRoundTrip:
    def test_map_choice_decodes_to_the_offered_index(self, adapter):
        state = _bridge_map_state()
        mask = adapter.compute_action_mask(state)
        chosen = int(np.flatnonzero(mask)[1])          # second offered node
        assert adapter.decode_action(chosen, state) == {"action": "choose", "index": 1}

    def test_only_offered_options_are_unmasked(self, adapter):
        """Three nodes offered means three legal actions, not five.

        An unmasked slot for a node that is not there is a move the game will
        reject, and the mod would then sit waiting for a valid one.
        """
        state = _bridge_map_state()
        assert adapter.compute_action_mask(state).sum() == 3

    def test_card_reward_skip_is_reachable_when_allowed(self, adapter):
        state = {
            "type": "card_reward",
            "can_skip": True,
            "cards": [
                {"index": 0, "id": "TAUNT", "type": "Skill", "cost": 1},
                {"index": 1, "id": "TRUE_GRIT", "type": "Skill", "cost": 1},
            ],
        }
        mask = adapter.compute_action_mask(state)
        skip_action = int(np.flatnonzero(mask)[-1])
        assert adapter.decode_action(skip_action, state) == {"action": "skip"}

    def test_combat_still_uses_the_combat_layout(self, adapter):
        """Combat must be untouched: the combat model shares this action space."""
        state = {
            "type": "combat_action",
            "player": {"hp": 60, "max_hp": 80, "block": 0, "energy": 3,
                       "max_energy": 3, "powers": []},
            "hand": [{"id": "STRIKE_IRONCLAD", "cost": 1, "type": "Attack",
                      "target": "AnyEnemy", "playable": True}],
            "enemies": [{"id": "MAWLER", "hp": 40, "max_hp": 40, "block": 0,
                         "is_alive": True, "powers": []}],
            "draw_pile_count": 5, "discard_pile_count": 0,
            "exhaust_pile_count": 0, "round": 1,
        }
        mask = adapter.compute_action_mask(state)
        assert mask[:115].any(), "combat actions must be in the combat slice"
        assert not mask[115:].any(), "combat must not unmask run-level actions"
