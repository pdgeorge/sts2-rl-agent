"""Which deck is this, and does this card belong in it.

The property that matters most here is that a card may belong to more than one
archetype. Rupture grants Strength whenever you lose HP -- it is a bloodletting
card *and* a strength card, and a design that forces it to pick one is wrong
about the game.
"""

from __future__ import annotations

import sts2_env.cards  # noqa: F401  (resolves package import order)
from sts2_env.search.archetypes import (
    ARCHETYPE_SEEDS,
    DeckDirection,
    archetype_names,
    card_affinities,
    peakedness,
)


def test_every_archetype_has_a_vector():
    assert set(archetype_names()) == set(ARCHETYPE_SEEDS)


def test_a_single_archetype_card_spikes_on_one_and_ignores_the_rest():
    affinities = card_affinities("BARRICADE_CARD")
    ranked = sorted(affinities.values(), reverse=True)
    assert ranked[0] > 0.5
    assert ranked[1] < 0.2, f"Barricade should not read as a second archetype: {affinities}"


def test_a_dual_archetype_card_scores_on_both():
    """Rupture is bloodletting AND strength. Forcing one would be wrong."""
    affinities = card_affinities("RUPTURE_CARD")
    assert affinities["bloodletting"] > 0.3
    assert affinities["strength"] > 0.1
    assert affinities["bloodletting"] > affinities["strength"]


def test_peakedness_separates_deck_defining_from_generically_good():
    assert peakedness("BARRICADE_CARD") > 2 * peakedness("IRON_WAVE")
    assert peakedness("PERFECTED_STRIKE") > 2 * peakedness("SHRUG_IT_OFF")


def test_a_card_serving_two_archetypes_commits_you_less_than_one_serving_one():
    """A property worth having, not an accident: Rupture fits two decks, so it
    decides less about which deck you are building than Barricade does."""
    assert peakedness("RUPTURE_CARD") < peakedness("BARRICADE_CARD")


def test_an_unknown_card_has_no_opinion():
    assert card_affinities("NOT_A_REAL_CARD") == {}
    assert peakedness("NOT_A_REAL_CARD") == 0.0


# --- accumulating a direction ---------------------------------------------

def test_starter_cards_carry_no_direction():
    """5 Strike, 4 Defend and a Bash would otherwise decide every run."""
    direction = DeckDirection()
    direction.observe_deck(["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"])
    assert direction.counted == 0
    assert direction.leader == (None, 0.0)
    assert direction.committed is None


def test_no_direction_until_there_are_enough_cards():
    direction = DeckDirection()
    direction.observe_deck(["BARRICADE_CARD", "ENTRENCH"])
    assert direction.committed is None, "two cards is not a plan"


def test_a_block_draft_commits_to_block_scaling():
    direction = DeckDirection()
    direction.observe_deck(["BODY_SLAM", "BARRICADE_CARD", "ENTRENCH"])
    assert direction.committed == "block-scaling"


def test_a_strike_draft_commits_to_strike_synergy():
    direction = DeckDirection()
    direction.observe_deck(["PERFECTED_STRIKE", "TWIN_STRIKE", "POMMEL_STRIKE"])
    assert direction.committed == "strike-synergy"


def test_commitment_is_sticky():
    """A deck that changes its mind on floor 12 has two half-decks."""
    direction = DeckDirection()
    direction.observe_deck(["BODY_SLAM", "BARRICADE_CARD", "ENTRENCH"])
    assert direction.committed == "block-scaling"
    direction.observe_deck(["PERFECTED_STRIKE"] * 6)
    assert direction.committed == "block-scaling"


def test_fit_is_peakedness_before_commitment_and_affinity_after():
    """With no plan the useful card is the one that supplies a plan."""
    fresh = DeckDirection()
    assert fresh.fit("BARRICADE_CARD") == peakedness("BARRICADE_CARD")

    committed = DeckDirection()
    committed.observe_deck(["BODY_SLAM", "BARRICADE_CARD", "ENTRENCH"])
    assert committed.committed == "block-scaling"
    assert committed.fit("SHRUG_IT_OFF") > committed.fit("PERFECTED_STRIKE")
