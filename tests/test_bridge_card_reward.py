"""The bridge must not offer skip, and must not confuse skip with a fourth card.

Both tests here are regressions from one live failure. A run 2.8 hours into training
reached a card reward, the model chose skip, the mod logged "Skip action could not be
executed", the screen re-presented, and a deterministic policy chose skip again --
three times, until the run ended. Nothing crashed; the run just stopped progressing.

The mod claimed can_skip=true for an action it had no way to perform.
NCardRewardSelectionScreen has no skip control and the game's own
CardRewardScreenHandler always picks a card, so the claim was never true.

The second test covers what that fix exposed. run_env treats the last card_reward
slot as skip unconditionally and puts a fourth card in a separate region, so the
bridge unmasking that slot as "card 4" would mistranslate an intended skip into a
pick -- the simulator and the bridge agreeing on the observation while disagreeing
about what an action means.
"""

from __future__ import annotations

import pytest

from sts2_env.bridge.run_adapter import RunStateAdapter, _CARD_RWD_START

_CARD_RWD_WIDTH = 4
_SKIP_SLOT = _CARD_RWD_START + _CARD_RWD_WIDTH - 1


def _card_reward(card_ids, can_skip):
    return {
        "type": "card_reward",
        "cards": [{"id": c, "cost": 1, "type": "Attack"} for c in card_ids],
        "can_skip": can_skip,
        "act": 1, "floor": 2, "act_floor": 2, "gold": 99,
        "run_hp": 60, "run_max_hp": 80, "deck_size": 10,
    }


@pytest.fixture
def adapter():
    return RunStateAdapter()


def test_skip_is_not_offered_when_the_game_cannot_skip(adapter):
    """can_skip=false must close the skip slot. This is the live hang."""
    mask = adapter.compute_action_mask(_card_reward(["BASH", "CLEAVE", "ANGER"], False))
    assert mask[_SKIP_SLOT] == 0, "skip must not be offered when the mod cannot skip it"
    assert mask[_CARD_RWD_START] == 1, "the cards are still choosable"


def test_skip_is_offered_when_the_game_can_skip(adapter):
    mask = adapter.compute_action_mask(_card_reward(["BASH", "CLEAVE", "ANGER"], True))
    assert mask[_SKIP_SLOT] == 1


def test_a_fourth_card_never_lands_on_the_skip_slot(adapter):
    """Four cards must not spill into slot 3; run_env reserves it for skip."""
    mask = adapter.compute_action_mask(
        _card_reward(["BASH", "CLEAVE", "ANGER", "IMPERVIOUS"], False))
    assert mask[_SKIP_SLOT] == 0, "slot 3 is skip, not the fourth card"
    for offset in range(3):
        assert mask[_CARD_RWD_START + offset] == 1, "the first three stay choosable"


def test_the_skip_slot_always_decodes_to_skip(adapter):
    """Even with can_skip false, the slot means skip -- it is merely masked out.

    Decoding it as choose-index-3 would send a card pick when the policy asked to
    skip, which is worse than refusing: the run continues and the log looks clean.
    """
    state = _card_reward(["BASH", "CLEAVE", "ANGER", "IMPERVIOUS"], False)
    assert adapter.decode_action(_SKIP_SLOT, state) == {"action": "skip"}


def test_cards_decode_to_their_own_index(adapter):
    state = _card_reward(["BASH", "CLEAVE", "ANGER"], True)
    for offset in range(3):
        decoded = adapter.decode_action(_CARD_RWD_START + offset, state)
        assert decoded == {"action": "choose", "index": offset}


def test_a_card_reward_is_always_answerable(adapter):
    """No options and no skip still needs one legal action, or MaskablePPO raises."""
    mask = adapter.compute_action_mask(_card_reward([], False))
    assert mask[_CARD_RWD_START:_CARD_RWD_START + _CARD_RWD_WIDTH].sum() >= 1
