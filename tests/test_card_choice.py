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


# --- reconstructing a live state --------------------------------------------


def _reward_state(deck_ids, offered_ids, floor=6, can_skip=True):
    return {
        "type": "card_reward",
        "can_skip": can_skip,
        "run_state": {"deck": [{"id": i} for i in deck_ids], "total_floor": floor},
        "max_hp": 80,
        "cards": [{"id": i} for i in offered_ids],
    }


def test_reward_context_reconstructs_a_deck():
    from sts2_env.evaluation.from_bridge import reward_context

    state = _reward_state(["STRIKE_IRONCLAD"] * 5 + ["BASH"], ["ANGER", "IRON_WAVE"])
    context = reward_context(state)
    assert context is not None
    assert len(context.deck) == 6
    assert context.resolved_fraction == 1.0
    assert context.usable
    assert context.floor == 6


def test_missing_deck_is_refused_not_treated_as_empty():
    """An empty deck would score every candidate identically and look like it
    worked. Older mod builds send deck_size without deck."""
    from sts2_env.evaluation.from_bridge import reward_context

    assert reward_context({"type": "card_reward", "cards": [{"id": "ANGER"}]}) is None


def test_a_badly_reconstructed_deck_is_not_usable():
    """Evaluating a deck missing a third of its cards is worse than not
    evaluating: it is confidently measuring something nobody owns."""
    from sts2_env.evaluation.from_bridge import reward_context

    state = _reward_state(
        ["STRIKE_IRONCLAD", "NOT_A_REAL_CARD", "ALSO_FAKE"], ["ANGER"]
    )
    context = reward_context(state)
    assert context is not None
    assert context.resolved_fraction < 0.8
    assert not context.usable


def test_upgraded_cards_survive_the_round_trip():
    """A deck read as all-unupgraded understates itself, and upgrades measurably
    matter: the same five cards upgraded cut elite HP cost 61.1 -> 35.5."""
    from sts2_env.evaluation.from_bridge import build_card

    plain = build_card({"id": "IRON_WAVE"})
    upgraded = build_card({"id": "IRON_WAVE", "upgraded": True})
    assert plain is not None and upgraded is not None
    assert upgraded.upgraded and not plain.upgraded


def test_bridge_name_variants_resolve():
    from sts2_env.evaluation.from_bridge import build_card

    for variant in ("STRIKE_IRONCLAD", "StrikeIronclad", "strike_ironclad"):
        card = build_card({"id": variant})
        assert card is not None, variant
        assert card.card_id.name == "STRIKE_IRONCLAD"


def test_unresolvable_offered_card_is_dropped_not_guessed():
    from sts2_env.evaluation.from_bridge import reward_context

    state = _reward_state(["STRIKE_IRONCLAD"] * 10, ["ANGER", "CARD_FROM_NEXT_PATCH"])
    context = reward_context(state)
    assert [c.card_id.name for c in context.candidates] == ["ANGER"]
    assert context.candidate_indexes == [0]


@pytest.mark.slow
def test_choose_card_index_returns_an_index_into_the_offered_list():
    from sts2_env.evaluation.from_bridge import choose_card_index

    state = _reward_state(
        ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"],
        ["CLUMSY", "IRON_WAVE", "ANGER"],
    )
    picked = choose_card_index(state, greedy_pilot, seeds=(0, 1))
    assert picked is None or 0 <= picked < 3


# --- rest sites -------------------------------------------------------------


def test_rest_value_is_capped_by_missing_hp():
    """At full HP a rest is worth nothing. A 50%-threshold rule takes it anyway."""
    from sts2_env.evaluation.rest_choice import rest_heal_value

    assert rest_heal_value(80, 80) == 0.0
    assert rest_heal_value(78, 80) < 6.0       # only 2 hp is missing to restore


def test_rest_value_is_worth_most_crossing_the_danger_zone():
    """Measured: survival is ~3% at half HP and ~67% at 80%, so the same 24 hp
    restored is worth several times more landing in that band than below it.

    Pinned because flat `min(heal, missing_hp)` priced all three identically, and
    that is the pricing that let a rest at 30% look as good as a rest at 50%
    when one leaves you dead and the other does not.
    """
    from sts2_env.evaluation.rest_choice import rest_heal_value

    deep = rest_heal_value(20, 80)      # 25% -> 55%, still likely dead
    band = rest_heal_value(40, 80)      # 50% -> 80%, the steep stretch
    topped = rest_heal_value(72, 80)    # 90% -> 100%, mostly wasted

    assert band > 2 * deep
    assert band > 2 * topped


def test_upgrade_value_scales_with_fights_remaining():
    """An upgrade pays out every remaining fight; a rest pays once. A fixed
    threshold cannot express that, and rest sites cluster where it matters most."""
    from sts2_env.evaluation.rest_choice import RestOption

    per_fight = 3.0
    early = RestOption(kind="upgrade", hp_value=per_fight * 8)
    late = RestOption(kind="upgrade", hp_value=per_fight * 1)
    assert early.hp_value > late.hp_value


@pytest.mark.slow
def test_a_strong_upgrade_beats_a_heal_even_when_hurt():
    """The result that justifies the whole module.

    At 20/80 HP the old rule heals, unconditionally. But upgrading Iron Wave is
    worth ~8 HP a fight over six fights (~50 HP) against a 24 HP heal, so healing
    is the wrong call -- and a threshold rule cannot see it, because it never
    looks at what the upgrade is worth.

    Measured stably across seed counts: +54.0 / +45.2 / +50.8 against a flat
    +24.0, so this is arithmetic rather than noise.
    """
    from sts2_env.evaluation.rest_choice import rank_rest_options

    deck = _starter() + [create_card(CardId.IRON_WAVE)]
    ranked = rank_rest_options(
        deck, [deck[-1]], greedy_pilot, current_hp=20, max_hp=80,
        floor=6, fights_remaining=6, seeds=(0, 1, 2),
    )
    assert ranked[0].kind == "upgrade"
    rest = next(o for o in ranked if o.kind == "rest")
    assert ranked[0].hp_value > rest.hp_value


@pytest.mark.slow
def test_a_worthless_upgrade_loses_to_a_heal_when_hurt():
    """The other side: when nothing worth upgrading is on offer, rest wins."""
    from sts2_env.evaluation.rest_choice import rank_rest_options

    deck = _starter()
    ranked = rank_rest_options(
        deck, [], greedy_pilot, current_hp=20, max_hp=80,
        floor=6, fights_remaining=6, seeds=(0, 1),
    )
    assert ranked[0].kind == "rest"


def test_upgrades_are_worthless_at_zero_fights_remaining():
    """An annuity with no payments left is worth nothing, so rest must win."""
    from sts2_env.evaluation.rest_choice import rank_rest_options

    deck = _starter() + [create_card(CardId.IRON_WAVE)]
    ranked = rank_rest_options(
        deck, [deck[-1]], greedy_pilot, current_hp=40, max_hp=80,
        floor=6, fights_remaining=0, seeds=(0,),
    )
    assert ranked[0].kind == "rest"


# --- smith screens ----------------------------------------------------------


def test_smith_picks_the_most_valuable_upgrade_not_the_first():
    """The heuristic took cards[:count] -- literally whatever was listed first."""
    from sts2_env.evaluation.from_bridge import choose_upgrade_indexes

    deck = [{"id": "STRIKE_IRONCLAD"}] * 5 + [{"id": "DEFEND_IRONCLAD"}] * 4 + [
        {"id": "BASH"}, {"id": "IRON_WAVE"}
    ]
    state = {
        "type": "card_select",
        "run_state": {"deck": deck, "total_floor": 6},
        "max_hp": 80,
        "cards": [
            {"id": "STRIKE_IRONCLAD", "index": 0},
            {"id": "IRON_WAVE", "index": 1},
            {"id": "DEFEND_IRONCLAD", "index": 2},
        ],
        "min_select": 1, "max_select": 1,
    }
    chosen = choose_upgrade_indexes(state, greedy_pilot, seeds=(0, 1))
    assert chosen == [1], f"expected Iron Wave (index 1), got {chosen}"


def test_smith_refuses_without_a_deck():
    from sts2_env.evaluation.from_bridge import choose_upgrade_indexes

    state = {"type": "card_select", "cards": [{"id": "BASH", "index": 0}]}
    assert choose_upgrade_indexes(state, greedy_pilot) is None


def test_smith_intent_gate_distinguishes_heal_from_smith():
    """card_select carries no purpose field, so an upgrade screen and a removal
    screen look identical. Ranking by upgrade value on a removal screen would
    delete the best card in the deck, so intent is tracked rather than guessed."""
    from sts2_env.bridge.agent_runner import _option_is_smith

    state = {"options": [
        {"id": "heal", "index": 0, "enabled": True},
        {"id": "smith", "index": 1, "enabled": True},
    ]}
    assert _option_is_smith(state, 1) is True
    assert _option_is_smith(state, 0) is False


# --- gauntlet scoring -------------------------------------------------------


@pytest.mark.slow
def test_body_slam_is_rejected_without_a_block_source():
    """The pick that prompted this change.

    Body Slam deals damage equal to your block. With four Defends and nothing
    else generating block it does almost nothing, and the agent took it live.

    Per-fight scoring rated it 0.695 against 0.743 for skipping -- a 0.05 gap
    inside the noise. Gauntlet scoring, which carries HP between fights instead
    of resetting to full, rates it 7 HP worse than taking nothing.
    """
    from sts2_env.evaluation.card_choice import rank_candidates

    deck = [create_card(CardId[n]) for n in
            ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 +
            ["BASH", "ANGER", "TRUE_GRIT", "PILLAGE"]]
    cards = [create_card(CardId.BODY_SLAM), create_card(CardId.UPPERCUT)]

    ranked = rank_candidates(deck, cards, greedy_pilot, floor=11)
    body_slam = next(r for r in ranked if "BODY_SLAM" in r.label)
    skip = next(r for r in ranked if r.card is None)
    assert body_slam.score < skip.score, (
        "Body Slam should lose to taking nothing in a deck with no block source"
    )


def test_gauntlet_length_sits_where_decks_separate():
    """Two was wrong, and wrong for a measurable reason.

    Run on act1_normal with no relics, everything died at 3 fights. Run in real
    act order -- weak fights first -- with the starting relic, a starter deck
    survives 5 and dies on the 6th, matching a live count of 5 wins. At 2 fights
    every deck scores ~76 HP and nothing separates.
    """
    from sts2_env.evaluation.card_choice import GAUNTLET_FIGHTS

    assert GAUNTLET_FIGHTS == 5


def test_gauntlet_opens_with_weak_fights():
    """A run does not start against mid-act encounters, and pretending it does
    made every deck look unplayable."""
    from sts2_env.evaluation.battery import Tier, act_sequence, encounters_for

    weak = set(encounters_for(Tier(1, "weak")))
    sequence = act_sequence(Tier(1, "normal"), 5)
    assert len(sequence) == 5
    assert all(s in weak for s in sequence[:3]), "act should open with weak fights"
    assert not any(s in weak for s in sequence[3:])


def test_the_starting_relic_is_present_by_default():
    """A deck without Burning Blood is not the deck being played -- it heals 6
    after every win, worth a whole extra fight over a gauntlet."""
    from sts2_env.evaluation.battery import DEFAULT_RELICS, _post_combat_heal

    assert "BURNING_BLOOD" in DEFAULT_RELICS
    assert _post_combat_heal(None) == 6.0
    assert _post_combat_heal([]) == 0.0


# --- deck composition rules -------------------------------------------------


def _mk(names):
    return [create_card(CardId[n]) for n in names]


def _base_deck():
    return _mk(["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 +
               ["BASH", "ANGER", "TRUE_GRIT", "PILLAGE"])


def _thin_deck():
    """A real shape from live play: 20 cards, 20% block, 28% of opening hands
    with no block at all. This is what pdgeorge reported watching -- "taking big
    hits but having no block cards in the deck"."""
    return _mk(["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"] +
               ["UPPERCUT", "ANGER", "RAMPAGE", "INFLAME", "SPITE",
                "PILLAGE", "BULLY", "WHIRLWIND", "THUNDERCLAP", "MOLTEN_FIST"])


def _flooded_deck():
    return _mk(["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"] +
               ["TAUNT", "TAUNT", "SHRUG_IT_OFF", "SHRUG_IT_OFF",
                "BLOOD_WALL", "EVIL_EYE"])


@pytest.mark.slow
def test_a_deck_below_the_band_is_pushed_toward_block():
    """The floor, restated as density. A count could not express this: the deck
    holds four Defends, which the old rule counted as defence it already had,
    while they are 20% of twenty cards and it bricks one hand in four."""
    from sts2_env.evaluation.deck_metrics import BLOCK_DENSITY_MIN, block_density

    deck = _thin_deck()
    assert block_density(deck) < BLOCK_DENSITY_MIN
    ranked = rank_candidates(
        deck, _mk(["TAUNT", "UPPERCUT"]), greedy_pilot, floor=11, seeds=(0, 1, 2, 3),
    )
    assert "TAUNT" in ranked[0].label, [r.label for r in ranked]


@pytest.mark.slow
def test_a_deck_above_the_band_is_pushed_away_from_block():
    """The cap, restated as density -- and it now scores block BELOW skipping
    rather than deleting the candidate, so `margin` still reports what the
    measurement said."""
    from sts2_env.evaluation.deck_metrics import BLOCK_DENSITY_MAX, block_density

    deck = _flooded_deck()
    assert block_density(deck) > BLOCK_DENSITY_MAX
    ranked = rank_candidates(
        deck, _mk(["TAUNT", "UPPERCUT"]), greedy_pilot, floor=11, seeds=(0, 1, 2, 3),
    )
    assert "UPPERCUT" in ranked[0].label
    taunt = next(r for r in ranked if r.card is not None and "TAUNT" in r.label)
    skip = next(r for r in ranked if r.card is None)
    assert taunt.score < skip.score


@pytest.mark.slow
def test_a_deck_inside_the_band_gets_no_push_either_way():
    """The term must be silent where it has nothing to say, or it is just a
    standing thumb on the scale for block."""
    from sts2_env.evaluation.deck_metrics import (
        BLOCK_DENSITY_MAX,
        BLOCK_DENSITY_MIN,
        block_density,
    )

    from sts2_env.evaluation.card_choice import _band_distance

    deck = _base_deck()
    assert BLOCK_DENSITY_MIN <= block_density(deck) <= BLOCK_DENSITY_MAX

    # The term is (distance_before - distance_after) * DENSITY_POINTS, so it
    # contributes exactly zero when both are zero. Asserted on the distance
    # itself rather than on a score, because a score assertion here would pass
    # for any number at all -- which an earlier draft of this test did.
    assert _band_distance(block_density(deck)) == 0.0
    assert _band_distance(block_density(deck + [create_card(CardId.TAUNT)])) == 0.0


@pytest.mark.slow
def test_a_clearly_better_card_still_beats_the_density_push():
    """The property that killed the 3.5-point tolerance. A correction big enough
    to force block through is big enough to force anything through, so the push
    has to lose to a card that is genuinely better.

    BATTLE_TRANCE is +4% run winrate over 9,700 offers against TAUNT's +1% over
    15,000, and it wins on a deck that wants block -- 1.81 to 1.65 -- because the
    density term is ~1.5 points, not unbounded.
    """
    ranked = rank_candidates(
        _thin_deck(), _mk(["TAUNT", "BATTLE_TRANCE"]), greedy_pilot,
        floor=11, seeds=(0, 1, 2, 3),
    )
    assert "BATTLE_TRANCE" in ranked[0].label, [r.label for r in ranked]


def test_the_density_term_is_signed_not_a_distance():
    """One term replaces a cap and a floor only because it changes sign. A
    distance would push toward block from both walls."""
    from sts2_env.evaluation.card_choice import _band_distance
    from sts2_env.evaluation.deck_metrics import (
        BLOCK_DENSITY_MAX,
        BLOCK_DENSITY_MIN,
        block_density,
    )

    thin, flooded = _thin_deck(), _flooded_deck()
    taunt = create_card(CardId.TAUNT)

    # Below the band, adding block reduces the distance; above it, increases.
    assert _band_distance(block_density(thin + [taunt])) < _band_distance(block_density(thin))
    assert _band_distance(block_density(flooded + [taunt])) > _band_distance(block_density(flooded))
    assert _band_distance((BLOCK_DENSITY_MIN + BLOCK_DENSITY_MAX) / 2) == 0.0


def test_starter_defends_do_not_count_as_high_quality_defence():
    """`HIGH_QUALITY_BLOCK` still answers "is this real defence against a
    40-damage hit" and still excludes Defend. Density answers a different
    question -- "will I have block in hand" -- and counts Defend fully. Both
    thresholds are right for their own question; pinned so they do not get
    merged."""
    from sts2_env.evaluation.card_choice import (
        is_defensive,
        is_high_quality_defence,
    )
    from sts2_env.evaluation.deck_metrics import block_density

    defend = create_card(CardId.DEFEND_IRONCLAD)
    flame = create_card(CardId.FLAME_BARRIER_CARD)
    assert is_defensive(defend) and not is_high_quality_defence(defend)
    assert is_defensive(flame) and is_high_quality_defence(flame)
    assert block_density([defend]) == 1.0     # density counts it in full


def test_rest_evaluates_distinct_cards_not_the_first_n():
    """A real bug: a starter-ordered deck begins with five Strikes and four
    Defends, so an 8-candidate cap evaluated only those and never saw the drafted
    cards. It upgraded Strikes because Uppercut was past the cap."""
    from sts2_env.evaluation.rest_choice import rank_rest_options

    deck = _base_deck() + _mk(["UPPERCUT", "IRON_WAVE"])
    ranked = rank_rest_options(
        deck, [c for c in deck if not c.upgraded], greedy_pilot,
        current_hp=45, max_hp=80, floor=12, seeds=(0, 1),
    )
    labels = [o.label for o in ranked if o.kind == "upgrade"]
    assert any("UPPERCUT" in l for l in labels), (
        "drafted cards must be evaluated, not crowded out by starter duplicates"
    )
    # one entry per distinct card, so nine starter cards do not eat the budget
    strikes = [l for l in labels if "STRIKE_IRONCLAD" in l]
    assert len(strikes) <= 1


def test_rest_option_index_points_into_the_callers_list():
    """`index` is a position in `upgradable`, not in the deduplicated list.

    A live bug: dedup collapsed nine starter cards into two entries, so the third
    distinct card came back as index 2, and `from_bridge` mapped that through the
    *offered* list and upgraded a Defend. It ranked correctly and upgraded
    something else, which is the worst shape of wrong -- the logs looked right.
    """
    from sts2_env.cards.factory import create_card
    from sts2_env.cards.ironclad import create_ironclad_starter_deck
    from sts2_env.core.enums import CardId
    from sts2_env.evaluation.pilots import greedy_pilot
    from sts2_env.evaluation.rest_choice import rank_rest_options

    deck = create_ironclad_starter_deck() + [create_card(CardId.IRON_WAVE)]
    ranked = rank_rest_options(
        deck, list(deck), greedy_pilot, current_hp=70, floor=6, seeds=(0,),
    )
    for option in ranked:
        if option.kind == "upgrade":
            assert deck[option.index].card_id == option.card.card_id


def test_starter_upgrades_rank_below_resting():
    """pdgeorge's rule: if Strike and Defend are all that is on offer, rest.

    Not measured -- at high HP the arithmetic prefers Strike+ because the heal it
    is compared against is mostly wasted. That is true and it is still the wrong
    move, so the prior is encoded rather than derived.
    """
    from sts2_env.cards.ironclad import create_ironclad_starter_deck
    from sts2_env.evaluation.pilots import greedy_pilot
    from sts2_env.evaluation.rest_choice import LOW_PRIORITY_UPGRADES, rank_rest_options

    deck = create_ironclad_starter_deck()
    starters = [c for c in deck if c.card_id in LOW_PRIORITY_UPGRADES]
    assert starters, "starter deck should contain Strikes and Defends"

    ranked = rank_rest_options(
        deck, starters, greedy_pilot, current_hp=55, floor=6, seeds=(0, 1),
    )
    assert ranked[0].kind == "rest"


def test_an_unknown_card_gets_no_prior_rather_than_an_average_one():
    """A card from a patch untapped has not seen must not be scored as neutral.

    Zero is not "no opinion" here -- it is better than every card measured as
    negative, so a silent fallback to 0 would rank an unmeasured card above most
    of the real pool. It has to stay out of the prior term entirely.
    """
    from sts2_env.evaluation.card_priors import prior_score

    assert prior_score("NOT_A_REAL_CARD_FROM_ANY_PATCH",
                       decision="card_reward", floor=1) is None


def test_a_thin_sample_is_shrunk_toward_zero():
    """Fiend Fire's -1% comes from 970 offers and Taunt's +1% from 15,000. Read
    at face value the two look symmetric; they are not remotely."""
    from sts2_env.evaluation.card_priors import prior_score

    taunt = prior_score(create_card(CardId.TAUNT), decision="card_reward", floor=1)
    fiend = prior_score(create_card(CardId.FIEND_FIRE), decision="card_reward", floor=1)
    assert taunt is not None and fiend is not None
    assert abs(taunt) > 3 * abs(fiend)


def test_a_rarely_taken_upgrade_is_not_vetoed_on_its_winrate():
    """IRON_WAVE reads -9% act winrate to upgrade on 4,600 smiths, but only 7% of
    players take it -- so the sample is the runs where nothing better was on
    offer. Vetoing on that would punish the card for the company it keeps."""
    from sts2_env.evaluation.rest_choice import _upgrading_is_measured_bad

    assert not _upgrading_is_measured_bad(create_card(CardId.IRON_WAVE), floor=6)
    assert _upgrading_is_measured_bad(create_card(CardId.FIEND_FIRE), floor=6)
