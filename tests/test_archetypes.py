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


# --- the unified scorer: quality x fit -------------------------------------

def test_a_curse_stays_refused_however_well_it_fits():
    """Letting fit soften a negative would make a well-themed curse takeable."""
    from sts2_env.bridge.card_quality import score_card, score_card_for_deck

    direction = DeckDirection()
    direction.observe_deck(["BODY_SLAM", "BARRICADE_CARD", "ENTRENCH"])
    curse = {"id": "REGRET", "type": "Curse"}

    assert score_card(curse) < 0
    assert score_card_for_deck(curse, [], direction) == score_card(curse, [])


def test_fit_decides_when_card_quality_abstains():
    """Entrench is "double your Block" -- no base damage, no base block, no
    scored effect vars, so whatever number the quality scorer returns comes from
    rarity and cost and says nothing about whether this deck wants it. A block
    deck would otherwise never take its own payoff.
    """
    from sts2_env.bridge.card_quality import (
        quality_is_uninformative,
        score_card_for_deck,
    )

    deck = [{"id": "BODY_SLAM"}, {"id": "BARRICADE_CARD"}, {"id": "ENTRENCH"}]
    direction = DeckDirection()
    direction.observe_deck([c["id"] for c in deck])

    assert quality_is_uninformative({"id": "ENTRENCH"})
    assert not quality_is_uninformative({"id": "BLUDGEON"})
    assert score_card_for_deck({"id": "ENTRENCH"}, deck, direction) > 0.4


def test_the_plan_reorders_cards_of_similar_quality():
    from sts2_env.bridge.card_quality import rank_cards

    deck = [{"id": "BODY_SLAM"}, {"id": "BARRICADE_CARD"}, {"id": "ENTRENCH"}]
    direction = DeckDirection()
    direction.observe_deck([c["id"] for c in deck])
    offered = [{"id": "PERFECTED_STRIKE"}, {"id": "SHRUG_IT_OFF"}]

    with_plan = [c["id"] for _, _, c in rank_cards(offered, deck, direction)]
    assert with_plan[0] == "SHRUG_IT_OFF"


def test_no_direction_behaves_exactly_as_before():
    """The archetype data is an enhancement; its absence must change nothing."""
    from sts2_env.bridge.card_quality import rank_cards

    offered = [{"id": "BLUDGEON"}, {"id": "STRIKE_IRONCLAD"}]
    assert rank_cards(offered, []) == rank_cards(offered, [], None)


def test_the_runner_reads_a_direction_off_the_bridge_deck():
    from sts2_env.bridge.agent_runner import _deck_direction

    state = {"deck": [{"id": "BODY_SLAM"}, {"id": "BARRICADE_CARD"}, {"id": "ENTRENCH"}]}
    direction = _deck_direction(state)
    assert direction is not None
    assert direction.committed == "block-scaling"


def test_the_runner_survives_a_deck_it_cannot_read():
    from sts2_env.bridge.agent_runner import _deck_direction

    assert _deck_direction({}) is not None
    assert _deck_direction({"deck": [{"id": "NOT_A_CARD"}]}).committed is None


# --- the milestone: the one moment she states a plan ------------------------

def _watcher():
    from sts2_env.bridge.milestones import MilestoneWatcher

    watcher = MilestoneWatcher()
    watcher.reset()
    return watcher


def test_she_names_the_deck_in_her_own_register():
    """Terse, lowercase, no subject pronoun -- matching "heading for the
    Ancient" and "took down an elite". And never the internal slug."""
    direction = DeckDirection()
    direction.observe_deck(["BODY_SLAM", "BARRICADE_CARD", "ENTRENCH"])

    text = _watcher().archetype_chosen(direction.committed, direction.confidence)["text"]
    assert text.startswith("building a block deck.")
    assert "block-scaling" not in text, "internal slug leaked into what she says"
    assert not text.startswith("I")


def test_confidence_is_per_card_so_a_big_deck_does_not_sound_certain():
    """The raw margin is a sum and grows with deck size, so a fifteen-card deck
    would clear gut_phrase's "obvious" threshold on arithmetic alone."""
    small = DeckDirection()
    small.observe_deck(["BODY_SLAM", "BARRICADE_CARD", "ENTRENCH"])
    big = DeckDirection()
    big.observe_deck(["BODY_SLAM", "BARRICADE_CARD", "ENTRENCH"] * 5)

    assert big.leader[1] > small.leader[1] * 3, "raw margin should grow with size"
    assert abs(big.confidence - small.confidence) < 0.05, "confidence should not"


def test_a_muddled_draft_sounds_uncertain_and_a_clear_one_does_not():
    muddled = DeckDirection()
    muddled.observe_deck(["BLUDGEON", "SHRUG_IT_OFF", "IRON_WAVE", "INFLAME"])
    clear = DeckDirection()
    clear.observe_deck(["PERFECTED_STRIKE", "TWIN_STRIKE", "POMMEL_STRIKE"])

    assert "either way" in _watcher().archetype_chosen(
        muddled.committed, muddled.confidence)["text"]
    assert "obvious" in _watcher().archetype_chosen(
        clear.committed, clear.confidence)["text"]


def test_it_fires_once_a_run():
    direction = DeckDirection()
    direction.observe_deck(["BODY_SLAM", "BARRICADE_CARD", "ENTRENCH"])
    watcher = _watcher()

    assert watcher.archetype_chosen(direction.committed, direction.confidence)
    assert watcher.archetype_chosen(direction.committed, direction.confidence) is None
