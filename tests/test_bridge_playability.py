"""The bridge mask must not offer a card the game will reject.

Observed live: alpha tried an unplayable card repeatedly within one turn. The mask
inferred playability from cost alone, so a 0-cost curse passed the energy check and
was offered; the game rejected the play, nothing about the state changed, and a
deterministic policy selected the same card again.

Worse than the wasted turn is why it was never caught in training. The simulator
masks with combat.can_play_card, which is a full check, so an unplayable card could
never be selected there. The policy had no opportunity to learn that such a card is
a dead end, because in its whole training it was never allowed to try one.

That asymmetry is the bug class these tests exist for: the simulator and the bridge
agreeing about the observation while disagreeing about what is legal.
"""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.bridge.state_adapter import StateAdapter


def _combat_state(hand):
    return {
        "type": "combat_action",
        "player": {"hp": 60, "max_hp": 80, "block": 0, "energy": 3,
                   "max_energy": 3, "powers": []},
        "hand": hand,
        "enemies": [{"id": "MAWLER", "hp": 40, "max_hp": 40, "block": 0,
                     "is_alive": True, "powers": []}],
        "draw_pile_count": 5, "discard_pile_count": 0,
        "exhaust_pile_count": 0, "round": 1,
    }


@pytest.fixture
def adapter():
    return StateAdapter()


def test_unplayable_card_is_masked_out(adapter):
    """A curse costs 0 and is illegal. Cost alone cannot tell.

    This is the live failure: ASCENDERS_BANE at cost 0 passed the energy check,
    got offered, and was chosen over and over.
    """
    state = _combat_state([
        {"id": "ASCENDERS_BANE", "cost": 0, "type": "Curse",
         "target": "Self", "playable": False},
    ])
    mask = adapter.compute_action_mask(state)
    # Index 1 is "play hand card 0". Only END_TURN should remain.
    assert mask[1] == 0, "an unplayable card must not be offered"
    assert mask[0] == 1, "end turn must always be available"
    assert mask.sum() == 1


def test_playable_card_is_still_offered(adapter):
    state = _combat_state([
        {"id": "STRIKE_IRONCLAD", "cost": 1, "type": "Attack",
         "target": "AnyEnemy", "playable": True},
    ])
    mask = adapter.compute_action_mask(state)
    assert mask.sum() > 1, "a playable card must be offered"


def test_unplayable_card_does_not_shift_the_others(adapter):
    """Masking one card must not renumber the rest.

    Action indices are positional. If skipping a card compacted the hand, the
    policy would evaluate one card and play a different one.
    """
    state = _combat_state([
        {"id": "ASCENDERS_BANE", "cost": 0, "type": "Curse",
         "target": "Self", "playable": False},
        {"id": "DEFEND_IRONCLAD", "cost": 1, "type": "Skill",
         "target": "Self", "playable": True},
    ])
    mask = adapter.compute_action_mask(state)
    assert mask[1] == 0, "slot 0 is the curse and stays masked"
    assert mask[2] == 1, "slot 1 is Defend and stays at slot 1"


def test_absent_flag_falls_back_to_the_cost_check(adapter):
    """An older mod sends no flag; behaviour there must not change."""
    state = _combat_state([
        {"id": "STRIKE_IRONCLAD", "cost": 1, "type": "Attack", "target": "AnyEnemy"},
        {"id": "BASH", "cost": 99, "type": "Attack", "target": "AnyEnemy"},
    ])
    mask = adapter.compute_action_mask(state)
    assert mask[1 + 10 + 0] == 1 or mask[1] == 1, "affordable card offered"
    targeted_bash = mask[1 + 10 + 1 * 5: 1 + 10 + 2 * 5]
    assert not targeted_bash.any(), "a card costing 99 with 3 energy is not offered"


def test_playable_true_is_not_enough_on_its_own(adapter):
    """playable=True must not override an obvious energy shortfall.

    The flag is authoritative for illegality, not for affordability -- a state
    captured a moment before energy was spent could still say True.
    """
    state = _combat_state([
        {"id": "BASH", "cost": 99, "type": "Attack",
         "target": "AnyEnemy", "playable": True},
    ])
    mask = adapter.compute_action_mask(state)
    assert mask.sum() == 1, "only end turn; 99 cost with 3 energy is unaffordable"


def test_mask_is_never_empty_even_if_everything_is_unplayable(adapter):
    """A hand of curses still has to be answerable.

    An all-zero mask makes MaskablePPO raise, which would end a live run.
    """
    state = _combat_state([
        {"id": "ASCENDERS_BANE", "cost": 0, "type": "Curse",
         "target": "Self", "playable": False},
        {"id": "REGRET", "cost": 0, "type": "Curse",
         "target": "Self", "playable": False},
    ])
    mask = adapter.compute_action_mask(state)
    assert mask.sum() >= 1
    assert mask[0] == 1, "end turn is the only move and must be offered"
