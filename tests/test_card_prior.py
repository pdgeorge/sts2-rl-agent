"""The community card prior: capped, gated, and inert unless a policy asks."""
from __future__ import annotations

import sts2_env.cards  # noqa: F401
from sts2_env.bridge.card_quality import (
    PRIOR_CAP,
    PRIOR_MIN_N,
    card_prior_bonus,
    score_card,
)
from sts2_env.policy_config import PolicyConfig, set_active_policy


def test_v001_is_unchanged_by_the_prior_existing():
    """The shipped policy must score exactly as it did before this term.

    Every measured number on the scoreboard was produced under `v001`. If
    adding the prior moved it, the baseline arm of prediction 14's A/B would be
    a different agent from the one those numbers describe.
    """
    set_active_policy(PolicyConfig.load("v001"))
    # The tie blocks the formula is known to produce, from the scoreboard row.
    assert score_card({"id": "OFFERING"}) == score_card({"id": "INFLAME"})
    assert score_card({"id": "INFLAME"}) == score_card({"id": "DEMON_FORM"})
    assert score_card({"id": "WHIRLWIND"}) == score_card({"id": "POMMEL_STRIKE"})


def test_the_prior_breaks_those_ties_when_switched_on():
    set_active_policy(PolicyConfig.load("v004_card_prior"))
    offering = score_card({"id": "OFFERING"})
    inflame = score_card({"id": "INFLAME"})
    demon = score_card({"id": "DEMON_FORM"})
    assert len({offering, inflame, demon}) == 3, "the tie block survived"
    # Demon Form's delta is the largest of the three, so it must now lead.
    assert demon > offering > inflame
    set_active_policy(PolicyConfig.load("v001"))


def test_zero_weight_is_exactly_zero():
    assert card_prior_bonus("DEMON_FORM", 0.0) == 0.0


def test_the_bonus_is_capped_both_ways():
    """Uncapped, a -18 delta would swamp a formula that lives in 0..5.

    `EvalWeights.powers_cap` exists because an uncapped term once made the
    searcher refuse to attack a sleeping elite. Same failure, same guard.
    """
    for weight in (1.0, 5.0, 50.0):
        for name in ("DEMON_FORM", "BODY_SLAM", "DRUM_OF_BATTLE"):
            assert abs(card_prior_bonus(name, weight)) <= PRIOR_CAP + 1e-9


def test_an_unknown_card_scores_no_bonus_rather_than_raising():
    """A card the prior has never seen is neutral, not refused. The prior goes
    stale on a patch; a new card must not be treated as a curse for it."""
    assert card_prior_bonus("NOT_A_REAL_CARD_XYZZY", 1.0) == 0.0


def test_thin_samples_are_ignored():
    assert PRIOR_MIN_N > 0
    from sts2_env.bridge.card_quality import _prior_table
    table = _prior_table()
    assert table, "the prior table did not load"
    thin = [n for n, e in table.items() if e.get("n", 0) < PRIOR_MIN_N]
    for name in thin:
        assert card_prior_bonus(name, 1.0) == 0.0
