"""Flying live combat with the evaluation pilot instead of the trained model.

The model measures level with a greedy heuristic -- 33% against 36% on act 1
elites after 40M steps. The D0 pilot is not level with greedy: over 180 act-1
boss fights on the same scaling deck, greedy wins 3% and value_pilot wins 96%,
because greedy cannot play a Power at all.

The subtle part is not the decision, it is the translation. The simulator and
the bridge use DIFFERENT action encodings, and passing the index straight
through is silently wrong rather than loud: the simulator encodes (hand 4,
target 0) as 31 while the bridge encodes the same play as 5.
"""

from __future__ import annotations

import numpy as np

from sts2_env.bridge.agent_runner import _pilot_combat_action
from sts2_env.bridge.state_adapter import StateAdapter

HAND = [
    {"id": "STRIKE_IRONCLAD", "cost": 1, "target": "ANY_ENEMY"},
    {"id": "DEFEND_IRONCLAD", "cost": 1, "target": "SELF"},
    {"id": "INFLAME", "cost": 1, "target": "SELF"},
    {"id": "IRON_WAVE", "cost": 1, "target": "ANY_ENEMY"},
    {"id": "BASH", "cost": 2, "target": "ANY_ENEMY"},
]


def _enemy(hp, monster="CUBEX_CONSTRUCT"):
    return {"id": monster, "hp": hp, "max_hp": 65, "block": 0,
            "powers": [], "is_alive": True}


def _state(enemies, energy=3):
    return {
        "type": "combat_action",
        "player": {"hp": 62, "max_hp": 80, "block": 0, "energy": energy, "powers": []},
        "enemies": enemies,
        "hand": HAND,
    }


def _decide(state):
    adapter = StateAdapter()
    mask = adapter.compute_action_mask(state)
    action = _pilot_combat_action(state, mask)
    if action is None:
        return None
    assert mask[action], "the pilot returned an action the game would reject"
    return adapter.decode_action(action, state)


def test_the_chosen_action_is_always_legal_live():
    """The whole risk of this path. An illegal action is not rejected loudly --
    the game ignores it, the state comes back unchanged, and a deterministic
    agent sends it again. That loop is what STUCK_REPEAT_LIMIT exists for."""
    for enemies in ([_enemy(45)], [_enemy(45), _enemy(6)], [_enemy(6), _enemy(45)]):
        assert _decide(_state(enemies)) is not None


def test_it_targets_the_enemy_it_meant_to():
    """Targeting has to survive the translation, and it is the part most easily
    lost: with ONE enemy the bridge omits targeted actions entirely because the
    game auto-targets, so a faithful (card, target) encoding is never legal and
    the untargeted form has to be tried second."""
    assert _decide(_state([_enemy(45), _enemy(6)]))["target_index"] == 1
    assert _decide(_state([_enemy(6), _enemy(45)]))["target_index"] == 0


def test_no_energy_ends_the_turn():
    decision = _decide(_state([_enemy(45)], energy=0))
    assert decision == {"type": "END_TURN"}


def test_an_unmodelled_monster_falls_back_to_the_model():
    """`rebuild_combat` returns None for anything the simulator does not have.
    Degrading to the trained model is correct; crashing mid-run is not."""
    adapter = StateAdapter()
    state = _state([{"id": "NOT_A_REAL_MONSTER", "hp": 10, "max_hp": 10,
                     "block": 0, "powers": [], "is_alive": True}])
    assert _pilot_combat_action(state, adapter.compute_action_mask(state)) is None


def test_a_malformed_state_falls_back_rather_than_raising():
    """Only the pilot path is under test here. A mask is supplied directly
    because `compute_action_mask` itself raises on `player: None`, which is a
    separate fragility and not one this path should be asserting about.
    """
    mask = np.ones(200, dtype=np.int8)
    for broken in ({}, {"type": "combat_action"}, {"player": None, "enemies": None},
                   {"player": {}, "enemies": [], "hand": [{"id": "NOT_A_CARD"}]}):
        assert _pilot_combat_action(broken, mask) in (None, 0)
