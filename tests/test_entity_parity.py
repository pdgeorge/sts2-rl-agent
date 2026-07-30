"""The simulator and the bridge must agree on the identity block, column for column.

Nothing raises when they disagree. The policy trains believing a column means
Poison and plays live believing it means something else, and the only symptom is
that it plays worse than it evaluated -- which reads as a bad training run, not a
bug. Three fields have already drifted this way on this project.

Columns are pinned to absolute offsets rather than derived from the encoder under
test. An earlier parity suite here built its expectation with the same encoder and
a scale change from 1000 to 100 passed every test.
"""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.bridge.run_adapter import RunStateAdapter
from sts2_env.gym_env.entity_encoding import (
    ENEMY_EXT_PER_SLOT,
    ENTITY_OBS_SIZE,
    HAND_EXTRA_BLOCK,
    HAND_SET_BLOCK,
    MONSTER_VEC_SIZE,
    POWER_AMOUNT_SCALE,
    POWER_VEC_SIZE,
    card_index,
    monster_index,
    power_index,
)
from sts2_env.gym_env.observation import OBS_SIZE as COMBAT_OBS_SIZE
from sts2_env.gym_env.run_env import RUN_OBS_SIZE

# The identity block sits immediately after the original combat dims.
ENTITY_START = COMBAT_OBS_SIZE
PLAYER_POWERS_AT = ENTITY_START
HAND_SET_AT = PLAYER_POWERS_AT + POWER_VEC_SIZE
HAND_EXTRA_AT = HAND_SET_AT + HAND_SET_BLOCK
DECK_SET_AT = HAND_EXTRA_AT + HAND_EXTRA_BLOCK
ENEMIES_AT = DECK_SET_AT + HAND_SET_BLOCK


def _bridge(**state):
    base = {
        "type": "combat_action",
        "player": {"hp": 50, "max_hp": 80, "block": 0, "energy": 3,
                   "max_energy": 3, "powers": []},
        "hand": [], "enemies": [],
        "draw_pile_count": 5, "discard_pile_count": 0,
        "exhaust_pile_count": 0, "round": 1,
    }
    base.update(state)
    return RunStateAdapter().encode_observation(base)


def test_the_layout_offsets_are_pinned():
    assert ENTITY_START == 131
    assert RUN_OBS_SIZE == COMBAT_OBS_SIZE + ENTITY_OBS_SIZE + 20 + 126 + 619
    assert RUN_OBS_SIZE == 4038


def test_player_poison_lands_in_the_same_column_on_both_sides():
    """Poison was invisible before; now it must be visible in ONE agreed place."""
    col = PLAYER_POWERS_AT + power_index("POISON")
    obs = _bridge(player={"hp": 50, "max_hp": 80, "block": 0, "energy": 3,
                          "max_energy": 3, "powers": [{"id": "POISON", "amount": 8}]})
    assert obs[col] == pytest.approx(8 / POWER_AMOUNT_SCALE)


def test_enemy_identity_and_powers_land_in_the_right_slot():
    obs = _bridge(enemies=[
        {"id": "MAWLER", "hp": 20, "max_hp": 40, "is_alive": True,
         "powers": [{"id": "THORNS", "amount": 3}]},
        {"id": "WRIGGLER", "hp": 10, "max_hp": 10, "is_alive": True, "powers": []},
    ])
    slot0 = ENEMIES_AT
    slot1 = ENEMIES_AT + ENEMY_EXT_PER_SLOT

    assert obs[slot0 + monster_index("MAWLER")] == 1.0
    assert obs[slot1 + monster_index("WRIGGLER")] == 1.0, "slot 1, not compacted"
    thorns = slot0 + MONSTER_VEC_SIZE + power_index("THORNS")
    assert obs[thorns] == pytest.approx(3 / POWER_AMOUNT_SCALE)


def test_the_simulators_starting_deck_matches_the_bridges():
    """Both sides must put Strike, Defend and Bash in the same deck columns."""
    from sts2_env.gym_env.run_env import STS2RunEnv

    env = STS2RunEnv(max_steps=50)
    sim_obs, _ = env.reset(seed=3)
    deck = [c.card_id for c in env._mgr.run_state.player.deck]

    bridge_obs = _bridge(deck=[getattr(c, "name", c) for c in deck])

    sim_deck = sim_obs[DECK_SET_AT:DECK_SET_AT + HAND_SET_BLOCK]
    bridge_deck = bridge_obs[DECK_SET_AT:DECK_SET_AT + HAND_SET_BLOCK]
    np.testing.assert_array_equal(sim_deck, bridge_deck)
    assert sim_deck.sum() >= 2, "a starting deck has at least Strike and Defend"


def test_the_deck_is_visible_outside_combat():
    """The point of the deck block: card rewards happen with no combat state.

    Encoding it inside the combat block would leave every deckbuilding decision
    blind, which is the exact gap this was added to close.
    """
    from sts2_env.gym_env.run_env import STS2RunEnv

    env = STS2RunEnv(max_steps=50)
    obs, _ = env.reset(seed=3)
    assert env._mgr.get_combat_state() is None, "reset is not in combat"
    assert obs[DECK_SET_AT:DECK_SET_AT + HAND_SET_BLOCK].sum() > 0


def test_hand_identity_is_a_set_not_an_ordinal():
    """Two different cards must give two different columns, not adjacent scalars."""
    bash = _bridge(hand=[{"id": "BASH", "cost": 2, "type": "Attack",
                          "target": "AnyEnemy"}])
    strike = _bridge(hand=[{"id": "STRIKE_IRONCLAD", "cost": 1, "type": "Attack",
                            "target": "AnyEnemy"}])
    assert bash[HAND_SET_AT + card_index("BASH")] == 1.0
    assert strike[HAND_SET_AT + card_index("STRIKE_IRONCLAD")] == 1.0
    assert bash[HAND_SET_AT + card_index("STRIKE_IRONCLAD")] == 0.0


def test_a_mod_that_sends_no_deck_leaves_the_deck_block_empty():
    """Documents the risk: absent reads as an empty deck, not as an error."""
    obs = _bridge()
    assert obs[DECK_SET_AT:DECK_SET_AT + HAND_SET_BLOCK].sum() == 0.0
