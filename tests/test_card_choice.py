"""Tests for measured card-reward choice.

The behavioural tests use decks whose ordering is not a matter of opinion -- a
0-cost 6-damage attack against a starter deck, an unplayable curse against
nothing. If the evaluator cannot get those right it cannot be trusted on the
close calls it will actually face.
"""

from __future__ import annotations

import pytest

from sts2_env.cards.factory import create_card
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.core.enums import CardId
from sts2_env.evaluation.battery import Tier
from sts2_env.evaluation.card_choice import (
    HP_WEIGHT,
    best_index,
    margin,
    rank_candidates,
    score_candidate,
    tiers_for_floor,
)
from sts2_env.evaluation.pilots import greedy_pilot


def _starter():
    return create_ironclad_starter_deck()


# --- tier selection ---------------------------------------------------------


def test_tiers_track_the_act_the_decision_is_made_in():
    assert tiers_for_floor(1) == (Tier(1, "normal"), Tier(1, "elite"))
    assert tiers_for_floor(17) == (Tier(1, "normal"), Tier(1, "elite"))
    assert tiers_for_floor(20) == (Tier(2, "normal"), Tier(2, "elite"))
    assert tiers_for_floor(40) == (Tier(3, "normal"), Tier(3, "elite"))


def test_bosses_are_excluded_from_draft_comparison():
    """With a mid-act deck every candidate scores zero against a boss, and a
    column of zeros ranks nothing."""
    for floor in (1, 20, 40):
        assert all(t.kind != "boss" for t in tiers_for_floor(floor))


# --- scoring ----------------------------------------------------------------


def test_score_combines_winning_with_hp_cost():
    """Two decks that win equally often are separated by what winning cost.

    This is the whole reason the battery reports two numbers: on the reference
    decks every candidate won 91.7% of act 1 normal fights while HP cost ranged
    33.8 to 15.6, so win rate alone called them identical.
    """
    from sts2_env.evaluation.card_choice import _combined

    cheap = _combined(win_rate=0.9, hp_lost=10.0, max_hp=80)
    costly = _combined(win_rate=0.9, hp_lost=40.0, max_hp=80)
    assert cheap > costly
    assert costly == pytest.approx(0.9 - HP_WEIGHT * (40.0 / 80))


def test_winning_dominates_hp():
    """A candidate that wins far more often should not lose on HP alone."""
    from sts2_env.evaluation.card_choice import _combined

    wins_more = _combined(win_rate=0.9, hp_lost=60.0, max_hp=80)
    wins_less = _combined(win_rate=0.5, hp_lost=0.0, max_hp=80)
    assert wins_more > wins_less


def test_never_winning_scores_on_win_rate_alone():
    """HP lost is NaN when nothing was won; that must not poison the score."""
    from sts2_env.evaluation.card_choice import _combined

    score = _combined(win_rate=0.0, hp_lost=float("nan"), max_hp=80)
    assert score == 0.0


# --- ranking ----------------------------------------------------------------


@pytest.mark.slow
def test_a_strong_card_outranks_a_curse():
    """Ordering that is not a matter of opinion."""
    deck = _starter()
    anger = create_card(CardId.ANGER)
    clumsy = create_card(CardId.CLUMSY)

    ranked = rank_candidates(
        deck, [clumsy, anger], greedy_pilot, floor=6, include_skip=False
    )
    assert ranked[0].card is anger, [r.label for r in ranked]


@pytest.mark.slow
def test_skip_is_a_real_option():
    deck = _starter()
    ranked = rank_candidates(
        deck, [create_card(CardId.CLUMSY)], greedy_pilot, floor=6
    )
    assert any(r.card is None for r in ranked), "skip should be scored"


def test_best_index_maps_back_to_the_offered_list():
    deck = _starter()
    cards = [create_card(CardId.ANGER), create_card(CardId.IRON_WAVE)]
    ranked = rank_candidates(
        deck, cards, greedy_pilot, floor=6, seeds=(0,), include_skip=False
    )
    index = best_index(ranked, cards)
    assert index in (0, 1)
    assert cards[index] is ranked[0].card


def test_best_index_returns_none_when_skipping_wins():
    from sts2_env.evaluation.card_choice import CandidateScore

    card = object()
    ranked = [
        CandidateScore(card=None, score=1.0, win_rate=1.0, hp_lost=0.0, fights=1),
        CandidateScore(card=card, score=0.5, win_rate=0.5, hp_lost=0.0, fights=1),
    ]
    assert best_index(ranked, [card]) is None


def test_margin_reports_how_close_the_call_was():
    from sts2_env.evaluation.card_choice import CandidateScore

    def make(score):
        return CandidateScore(
            card=object(), score=score, win_rate=0.0, hp_lost=0.0, fights=1
        )

    assert margin([make(1.0), make(0.4)]) == pytest.approx(0.6)
    assert margin([make(1.0)]) == float("inf")


def test_margin_matters_because_small_samples_are_confidently_wrong():
    """Documented, measured: at 45 fights an unplayable curse ranks ABOVE skip,
    and only at ~240 fights does the ordering correct itself. A caller that
    ignores margin will act on that."""
    from sts2_env.evaluation.card_choice import DEFAULT_SEEDS

    assert len(DEFAULT_SEEDS) >= 6, (
        "three seeds was measured ranking a curse better than taking nothing"
    )


# --- the deck is not mutated ------------------------------------------------


def test_scoring_does_not_modify_the_deck_it_was_given():
    """Cards carry mutable per-combat state; a leak here would make every later
    evaluation measure a deck nobody has."""
    deck = _starter()
    before = [(c.card_id, c.cost, c.upgraded) for c in deck]

    score_candidate(
        deck, create_card(CardId.ANGER), greedy_pilot,
        tiers=(Tier(1, "weak"),), seeds=(0,),
    )

    after = [(c.card_id, c.cost, c.upgraded) for c in deck]
    assert before == after
    assert len(deck) == len(before)
