"""Which card to upgrade, and which to transform away."""

from __future__ import annotations

import sts2_env.cards  # noqa: F401  (resolves package import order)
from sts2_env.bridge.card_quality import score_card
from sts2_env.bridge.upgrade_targets import (
    BEHAVIOURAL_UPGRADES,
    pick_transform_target,
    pick_upgrade_target,
    upgrade_gain,
)


def test_score_card_reads_the_upgrade_flag():
    """It did not, and every numeric upgrade delta was therefore exactly 0.0."""
    assert score_card({"id": "BLUDGEON", "upgraded": True}) > score_card({"id": "BLUDGEON"})


def test_a_behavioural_upgrade_beats_a_bigger_number():
    """Armaments upgraded has identical cost, damage, block and flags -- the
    change is an `if (IsUpgraded)` branch in OnPlay, so its numeric delta is
    exactly zero and it would otherwise rank last."""
    deck = [{"id": "BLUDGEON"}, {"id": "ARMAMENTS"}]
    assert upgrade_gain(deck[1], deck, 1) > upgrade_gain(deck[0], deck, 0)
    assert pick_upgrade_target(deck) == 1


def test_the_behavioural_list_came_from_the_decompile():
    """Derived by grepping IsUpgraded, not hand-listed, so a patch updates it."""
    assert "ARMAMENTS" in BEHAVIOURAL_UPGRADES
    assert "BLUDGEON" not in BEHAVIOURAL_UPGRADES
    assert 20 < len(BEHAVIOURAL_UPGRADES) < 40


def test_upgrading_scores_the_gain_not_the_absolute():
    """An already-strong card has little headroom, so ranking by the upgraded
    card's absolute value would just pick whatever was already best."""
    deck = [{"id": "BLUDGEON"}, {"id": "SHRUG_IT_OFF"}]
    # Bludgeon is by far the stronger card...
    assert score_card(deck[0]) > score_card(deck[1]) * 2
    # ...but the gains are comparable, which absolute scoring would never show.
    assert upgrade_gain(deck[1], deck, 1) > 0.5 * upgrade_gain(deck[0], deck, 0)


def test_a_card_is_scored_against_the_deck_without_itself():
    """Leaving it in makes it compete with its own contribution -- a deck's only
    block card looks less needed because the deck has a block card, itself."""
    from sts2_env.bridge.upgrade_targets import _without

    deck = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
    assert _without(deck, 1) == [{"id": "A"}, {"id": "C"}]


def test_an_empty_deck_has_no_target():
    assert pick_upgrade_target([]) is None
    assert pick_transform_target([]) is None


# --- transform -------------------------------------------------------------

def test_transform_never_targets_a_curse():
    """Transforming a curse yields another curse, sometimes a worse one."""
    deck = [{"id": "REGRET", "type": "Curse"}, {"id": "STRIKE_IRONCLAD"}]
    assert pick_transform_target(deck) == 1


def test_transform_takes_the_least_valuable_card():
    deck = [{"id": "BLUDGEON"}, {"id": "STRIKE_IRONCLAD"}]
    chosen = pick_transform_target(deck)
    assert deck[chosen]["id"] == "STRIKE_IRONCLAD"


def test_a_deck_of_only_curses_has_nothing_to_transform():
    deck = [{"id": "REGRET", "type": "Curse"}]
    assert pick_transform_target(deck) is None
