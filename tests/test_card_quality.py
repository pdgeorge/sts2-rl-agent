"""Which card to take, judged by what the card actually is.

Grounded in the 30-run live session of 2026-08-05: 183 card rewards, none
refused, 61% of every card played a basic Strike or Defend, and this exact offer
answered with BLIGHT_STRIKE because it was listed first.
"""

from __future__ import annotations

import pytest

from sts2_env.bridge.agent_runner import _pick_card_reward_index
from sts2_env.bridge.card_quality import (
    CARD_RATINGS,
    deck_shape,
    rank_cards,
    score_card,
)

STARTER = ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"]


def _card(name, index=0, **extra):
    return {"id": name, "index": index, **extra}


def _reward(names, deck=None, can_skip=False, **state):
    return {
        "type": "card_reward",
        "can_skip": can_skip,
        "deck": deck if deck is not None else STARTER,
        "deck_size": len(deck if deck is not None else STARTER),
        "cards": [_card(n, i) for i, n in enumerate(names)],
        **state,
    }


# -- the offer that started this ---------------------------------------------

def test_the_real_offer_from_the_live_log() -> None:
    """BLIGHT_STRIKE is 8 damage for 1 energy; SUNDER is 26 for 3. The old rule
    took Blight Strike because it came first in the list."""
    state = _reward(["BLIGHT_STRIKE", "SUNDER", "FLICK_FLACK"])
    assert _pick_card_reward_index(state) == 1


def test_more_output_per_energy_scores_higher() -> None:
    assert score_card(_card("SUNDER"), STARTER) > score_card(_card("BLIGHT_STRIKE"), STARTER)


# -- things never worth taking -----------------------------------------------

@pytest.mark.parametrize("name", ["CLUMSY", "BURN"])
def test_curses_and_statuses_are_refused(name) -> None:
    assert score_card(_card(name), STARTER) < 0


def test_a_curse_is_skipped_when_the_game_allows_it() -> None:
    state = _reward(["CLUMSY"], can_skip=True)
    assert _pick_card_reward_index(state) is None


def test_a_curse_is_still_answered_when_the_screen_cannot_skip() -> None:
    """The mod's screen path reports can_skip false and its handler always takes
    a card. Returning None there claims a decision the game cannot perform, and
    an unanswered screen is what loops a run forever."""
    state = _reward(["CLUMSY"], can_skip=False)
    assert _pick_card_reward_index(state) == 0


# -- rarity and scaling ------------------------------------------------------

def test_rarer_cards_beat_common_ones_all_else_equal() -> None:
    ranked = rank_cards([_card("SUNDER", 0), _card("BLIGHT_STRIKE", 1)], STARTER)
    assert ranked[0][2]["id"] == "SUNDER"


def test_a_power_is_worth_more_than_its_numbers() -> None:
    """Inflame has no damage and no block, and it is how a deck beats a boss.
    Every one of the six live boss attempts died with no scaling in the deck."""
    assert score_card(_card("INFLAME"), STARTER) > score_card(_card("BLIGHT_STRIKE"), STARTER)


def test_the_second_power_is_worth_less_than_the_first() -> None:
    with_one = STARTER + ["INFLAME"]
    assert score_card(_card("INFLAME"), with_one) < score_card(_card("INFLAME"), STARTER)


# -- judged against what the deck needs --------------------------------------

def test_a_deck_with_no_block_values_a_block_card() -> None:
    no_block = ["STRIKE_IRONCLAD"] * 10
    assert score_card(_card("IRON_WAVE"), no_block) > score_card(_card("IRON_WAVE"), STARTER)


def test_deck_shape_reads_the_deck() -> None:
    shape = deck_shape(STARTER)
    assert shape["size"] == 10
    assert 0.0 < shape["block_density"] < 1.0
    assert shape["powers"] == 0


def test_an_empty_deck_does_not_divide_by_zero() -> None:
    assert deck_shape([])["size"] == 0
    assert score_card(_card("SUNDER"), []) > 0


# -- a bloated deck ----------------------------------------------------------

def test_a_very_large_deck_skips_when_it_can() -> None:
    huge = STARTER * 4
    state = _reward(["SUNDER"], deck=huge, can_skip=True, deck_size=len(huge))
    assert _pick_card_reward_index(state) is None


# -- surviving a game update -------------------------------------------------

def test_a_card_this_build_has_never_heard_of_is_neutral_not_refused() -> None:
    """After an update a new card should be considered, not treated as a curse."""
    assert score_card(_card("A_CARD_ADDED_NEXT_PATCH"), STARTER) == 0.0


def test_an_unknown_card_is_still_pickable() -> None:
    state = _reward(["A_CARD_ADDED_NEXT_PATCH"])
    assert _pick_card_reward_index(state) == 0


# -- the override the ratings go in ------------------------------------------

def test_supplied_ratings_win_over_everything_computed() -> None:
    CARD_RATINGS["BLIGHT_STRIKE"] = 99.0
    try:
        state = _reward(["BLIGHT_STRIKE", "SUNDER"])
        assert _pick_card_reward_index(state) == 0
    finally:
        CARD_RATINGS.pop("BLIGHT_STRIKE", None)


def test_no_cards_on_offer_does_not_raise() -> None:
    assert _pick_card_reward_index(_reward([])) == 0
    assert _pick_card_reward_index(_reward([], can_skip=True)) is None
