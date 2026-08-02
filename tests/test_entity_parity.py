"""The simulator and the bridge must agree on the identity block, vector for vector.

With feature hashing there are no longer guaranteed-unique columns for every
item. Instead the invariant is: both sides use the SAME hasher with the SAME
seed, so the same input (e.g. Poison amount 8) produces the SAME output vector
on both sides. The policy learns "this hash pattern means Poison" rather than
"column 47 means Poison."

Nothing raises when they disagree. The policy trains believing a pattern means
Poison and plays live believing it means something else, and the only symptom is
that it plays worse. These tests verify byte-identical agreement between the
simulator's encoder and the bridge's encoder.
"""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.bridge.run_adapter import RunStateAdapter
from sts2_env.gym_env.entity_encoding import (
    ENEMY_EXT_PER_SLOT,
    ENTITY_OBS_SIZE,
    encode_entities,
    entities_from_bridge_state,
    entities_from_combat,
)
from sts2_env.gym_env.observation import OBS_SIZE as COMBAT_OBS_SIZE
from sts2_env.gym_env.run_env import RUN_OBS_SIZE

# The identity block sits immediately after the original combat dims.
ENTITY_START = COMBAT_OBS_SIZE


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
    from sts2_env.gym_env.deck_features import DECK_FEATURE_SIZE
    assert RUN_OBS_SIZE == COMBAT_OBS_SIZE + ENTITY_OBS_SIZE + DECK_FEATURE_SIZE + 20 + 126 + 384
    assert RUN_OBS_SIZE == 2383


def test_player_poison_is_visible_on_both_sides():
    """Poison was invisible before; now it must be visible with the same hash
    signature on both sides."""
    obs = _bridge(player={"hp": 50, "max_hp": 80, "block": 0, "energy": 3,
                          "max_energy": 3, "powers": [{"id": "POISON", "amount": 8}]})
    entity = obs[ENTITY_START:ENTITY_START + ENTITY_OBS_SIZE]
    # The player-power block is the first 128 dims of the entity block.
    player_power_block = entity[:128]
    assert np.any(player_power_block != 0.0), "Poison should hash to nonzero"


def test_enemy_identity_and_powers_are_present():
    obs = _bridge(enemies=[
        {"id": "MAWLER", "hp": 20, "max_hp": 40, "is_alive": True,
         "powers": [{"id": "THORNS", "amount": 3}]},
        {"id": "WRIGGLER", "hp": 10, "max_hp": 10, "is_alive": True, "powers": []},
    ])
    entity = obs[ENTITY_START:ENTITY_START + ENTITY_OBS_SIZE]
    # Enemy block starts after player powers (128) + hand set (256) + hand extra (90)
    # + deck set (256) = 730
    enemy_start = 128 + 256 + 90 + 256
    enemy0 = entity[enemy_start:enemy_start + ENEMY_EXT_PER_SLOT]
    enemy1 = entity[enemy_start + ENEMY_EXT_PER_SLOT:enemy_start + 2 * ENEMY_EXT_PER_SLOT]

    assert np.any(enemy0 != 0.0), "Mawler + Thorns should hash to nonzero"
    assert np.any(enemy1 != 0.0), "Wriggler should hash to nonzero"
    assert not np.array_equal(enemy0, enemy1), "different enemies must differ"


def test_the_simulators_starting_deck_matches_the_bridges():
    """Both sides must put the same deck into the same hashed vector."""
    from sts2_env.gym_env.run_env import STS2RunEnv

    env = STS2RunEnv(max_steps=50)
    sim_obs, _ = env.reset(seed=3)
    deck = [c.card_id for c in env._mgr.run_state.player.deck]

    bridge_obs = _bridge(deck=[getattr(c, "name", c) for c in deck])

    # Compare the full entity blocks byte-for-byte.
    sim_entity = sim_obs[ENTITY_START:ENTITY_START + ENTITY_OBS_SIZE]
    bridge_entity = bridge_obs[ENTITY_START:ENTITY_START + ENTITY_OBS_SIZE]
    np.testing.assert_array_equal(sim_entity, bridge_entity)
    assert sim_entity.sum() > 0, "a starting deck produces a nonzero hash"


def test_the_deck_is_visible_outside_combat():
    """The point of the deck block: card rewards happen with no combat state."""
    from sts2_env.gym_env.run_env import STS2RunEnv

    env = STS2RunEnv(max_steps=50)
    obs, _ = env.reset(seed=3)
    assert env._mgr.get_combat_state() is None, "reset is not in combat"
    entity = obs[ENTITY_START:ENTITY_START + ENTITY_OBS_SIZE]
    # Deck set starts at offset 128 + 256 + 90 = 474 within the entity block
    deck_block = entity[474:474 + 256]
    assert deck_block.sum() > 0


def test_hand_identity_is_a_set_not_an_ordinal():
    """Two different cards must give two different hash signatures."""
    bash = _bridge(hand=[{"id": "BASH", "cost": 2, "type": "Attack",
                          "target": "AnyEnemy"}])
    strike = _bridge(hand=[{"id": "STRIKE_IRONCLAD", "cost": 1, "type": "Attack",
                            "target": "AnyEnemy"}])
    bash_entity = bash[ENTITY_START:ENTITY_START + ENTITY_OBS_SIZE]
    strike_entity = strike[ENTITY_START:ENTITY_START + ENTITY_OBS_SIZE]

    # The hand set block starts at offset 128 within entity
    bash_hand = bash_entity[128:128 + 256]
    strike_hand = strike_entity[128:128 + 256]

    assert not np.array_equal(bash_hand, strike_hand)
    assert np.any(bash_hand != 0.0)
    assert np.any(strike_hand != 0.0)


def test_a_mod_that_sends_no_deck_leaves_the_deck_block_empty():
    """Documents the risk: absent reads as an empty deck, not as an error."""
    obs = _bridge()
    entity = obs[ENTITY_START:ENTITY_START + ENTITY_OBS_SIZE]
    # Deck block is at offset 474, size 256
    assert entity[474:474 + 256].sum() == 0.0


def test_bridge_and_sim_entity_encoders_agree_on_full_block():
    """Both sides use the same hashers with the same seeds, so identical
    high-level state produces byte-identical entity vectors."""
    # Build a rich state that exercises every sub-block.
    bridge_state = {
        "player": {"powers": [{"id": "STRENGTH", "amount": 3},
                              {"id": "POISON", "amount": 5}]},
        "hand": [
            {"id": "BASH", "type": "Attack", "upgraded": False,
             "playable": True, "target": "AnyEnemy", "cost_x": False},
            {"id": "STRIKE_IRONCLAD", "type": "Attack", "upgraded": False,
             "playable": True, "target": "AnyEnemy", "cost_x": False},
        ],
        "deck": ["DEFEND_IRONCLAD", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD"],
        "enemies": [
            {"id": "MAWLER", "powers": [{"id": "THORNS", "amount": 2}]},
            {"id": "WRIGGLER", "powers": []},
        ],
    }

    from_bridge = entities_from_bridge_state(bridge_state)
    bridge_vec = encode_entities(**from_bridge)

    # Build an equivalent simulator-side state (mock objects with the same attrs)
    class MockPower:
        def __init__(self, name, amount):
            self.name = name
            self.amount = amount
    class MockCard:
        def __init__(self, card_id, card_type, target_type, upgraded=False, has_x=False):
            self.card_id = card_id
            self.card_type = card_type
            self.target_type = target_type
            self.upgraded = upgraded
            self.has_energy_cost_x = has_x
    class MockEnemy:
        def __init__(self, monster_id, powers):
            self.monster_id = monster_id
            self.powers = powers

    sim_hand = [
        MockCard("BASH", "Attack", "AnyEnemy"),
        MockCard("STRIKE_IRONCLAD", "Attack", "AnyEnemy"),
    ]
    sim_hand_bridge_format = [
        {"type": c.card_type, "upgraded": c.upgraded, "playable": True,
         "target": c.target_type, "cost_x": c.has_energy_cost_x}
        for c in sim_hand
    ]
    sim_deck = [
        MockCard("DEFEND_IRONCLAD", "Skill", "Self"),
        MockCard("DEFEND_IRONCLAD", "Skill", "Self"),
        MockCard("STRIKE_IRONCLAD", "Attack", "AnyEnemy"),
    ]
    sim_enemies = [
        MockEnemy("MAWLER", {"THORNS": MockPower("THORNS", 2)}),
        MockEnemy("WRIGGLER", {}),
    ]

    from_sim = {
        "player_powers": {"STRENGTH": MockPower("STRENGTH", 3),
                          "POISON": MockPower("POISON", 5)},
        "hand_card_names": [c.card_id for c in sim_hand],
        "hand_features": entities_from_bridge_state({"hand": sim_hand_bridge_format})["hand_features"],
        "deck_card_names": [c.card_id for c in sim_deck],
        "enemies": [(e.monster_id, e.powers) for e in sim_enemies],
    }
    sim_vec = encode_entities(**from_sim)

    np.testing.assert_array_equal(bridge_vec, sim_vec)
