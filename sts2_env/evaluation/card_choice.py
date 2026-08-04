"""Which of these cards should we add? Answered by playing, not by guessing.

This is the battery scoped down to the decision actually in front of the agent:
three cards on a reward screen and a deck to add one to. For each candidate,
build `deck + card`, play it against a slice of the grid, and take the best.

WHY THIS EXISTS RATHER THAN A TRAINED CARD-REWARD HEAD

A trained head needs the deck-quality signal this module produces, so it comes
second. Meanwhile the current policy picks by position -- 19 of 28 rewards went to
slot 2 and none to slot 0, with the mask, the choice encoding and the card ids all
verified working. It is not blind, it is untrained, and drafting is the decision
it has the least chance of learning from a sparse terminal reward.

Distilling this into a network later is Phase 5. Until then it can simply be
called, because it is fast enough to run live.

WHAT "BEST" MEANS HERE

`score` combines the two numbers the battery reports:

    win_rate - HP_WEIGHT * (hp_lost / max_hp)

Winning dominates, and among decks that win, losing less HP wins. That second
term is not a tiebreak detail -- on the reference decks every candidate won 91.7%
of act 1 normal fights while HP cost ranged 33.8 to 15.6. Ranking on win rate
alone would have called them all equal.

WHICH TIERS TO TEST AGAINST

Defaults to the tiers a run is about to face, not the whole grid. Testing a floor
6 draft against act 3 bosses measures nothing useful: every candidate scores zero
and the comparison is noise. `tiers_for_floor` picks the cells that discriminate
at the point the decision is being made.

HOW MANY FIGHTS IT TAKES, MEASURED

Ranking one curse (Clumsy) against skipping, on a 10-card starter deck:

    seeds   fights   curse    skip    correct?
      3       45     +0.578  +0.543   no  -- curse ranked BETTER than nothing
      8      120     +0.552  +0.549   no
     16      240     +0.545  +0.550   yes

Two things worth knowing before trusting a ranking.

It converges from the wrong side. At 45 fights an unplayable curse looks actively
good, so a small sample is not merely imprecise, it is confidently wrong.

And single-card deltas are genuinely tiny. Even at 240 fights the gap is 0.005
and the win rates are identical -- the whole difference is in the HP term. One
curse in a 10-card deck is often not even drawn in a given fight. That is the
real size of the effect, not a defect in the measurement, and it is why drafting
is close to unlearnable from run outcomes alone.

So DEFAULT_SEEDS is a compromise, not a recommendation: enough to separate an
obviously good card from an obviously bad one, not enough to settle near-ties.
Check `margin` before treating a winner as the answer. For offline work that can
afford it, pass more seeds -- cost is linear, about 0.28s per seed per candidate.

HONEST LIMITS

* The pilot decides what good means. Greedy-damage will undervalue block and
  scaling cards because it cannot convert them. This ranking inherits that
  ceiling entirely.
* It separates good from bad, not good from slightly better. See the table above.
* It is greedy over one card. It cannot see that two cards are only good
  together, which is exactly what a synergy measurement would add later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from sts2_env.core.constants import IRONCLAD_STARTING_HP
from sts2_env.evaluation.battery import Pilot, Tier, mean_gauntlet_hp

logger = logging.getLogger(__name__)

GAUNTLET_FIGHTS = 5
"""Consecutive fights per gauntlet, healing only from relics between them.

Five, because that is where a starter deck actually separates. The sequence
follows real act structure -- weak fights first, then the normal pool -- and
carries the character's starting relic, so:

    2 fights  76.1 hp     all weak, every deck looks identical
    3 fights  73.5 hp
    4 fights  46.6 hp
    5 fights  31.2 hp     discriminating
    6 fights   0.0 hp     everything dies, no resolution again

This was 2 for most of a day, chosen because "every deck dies at 3". That was an
artifact of running the gauntlet entirely on act1_normal with no relics, which is
a situation a run never encounters. Corrected, the same deck survives 5 fights --
matching a live count of 5 wins before dying to an elite.
"""

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
"""Six seeds, ~90 fights per candidate, ~1.7s. See the table in the module
docstring: this separates clear differences and will not settle close ones."""

# --- deck composition -------------------------------------------------------
#
# Two rules, kept apart on purpose: one is a measurement, the other is play
# knowledge the battery cannot currently produce. Blurring them would let a prior
# win an argument it has not earned.

HIGH_QUALITY_BLOCK = 8
"""Block a card must give to count as real defence.

Keyed off base_block rather than a list of card names, because the hardcoded
keyword lists in deck_features.py are exactly what rots on every patch. Flame
Barrier is 12 and Blood Wall 16; a 5-block Defend does not qualify, and against
the 40-damage hits of act 2 it genuinely is not defence.
"""

DEFENCE_CAP_BY_ACT = {1: 2, 2: 3, 3: 3}
"""MEASURED, for act 1. Clear rate over 75 runs of 3 weak + 1 hallway + elite,
across all three act 1 elites:

    0 defensive cards   77.3% +/- 4.8%
    1 defensive card    74.7% +/- 5.0%
    2 defensive cards   74.7% +/- 5.0%
    3 defensive cards   38.7% +/- 5.6%     <- collapse

One or two cost nothing detectable; the third is a 5+ sem drop. Acts 2 and 3 are
EXTRAPOLATED -- their enemies hit harder, so the ceiling should be higher, but
nobody has measured it. Treat those two numbers as guesses.
"""

DEFENCE_FLOOR_BY_ACT = {1: 1, 2: 2, 3: 2}
"""NOT MEASURED. This is pdgeorge's play knowledge, recorded as a prior.

A healthy Ironclad run picks up a Flame Barrier or Blood Wall early even over a
good attack, and wants a second by act 2 to have any answer to a 40-damage hit.
The battery does not reproduce that: it says one defensive card is *free*
(74.7% against 77.3%, 0.4 sem), not that it is good.

Including it anyway is defensible precisely because it is free -- it buys
something the measurement cannot see at no measured cost. There is a standing
reason the measurement may understate it: the greedy pilot attacks first and
blocks with leftover energy, so a defensive card does less in its hands than in a
player's.

If a better pilot ever shows defence actively helping, this graduates from prior
to result and the comment should say so.
"""

DEFENCE_TOLERANCE = 1.0
"""How far behind the winner a defensive card may sit and still be taken.

One run-winrate point, which is 0.08 in the gauntlet-HP units this used to be
scored in -- the same tolerance restated, not a loosened one. It fires on a
near-tie and not otherwise.

THE ACT 2 FLOOR NO LONGER FIRES AGAINST A STRONG ALTERNATIVE, and that is a real
behaviour change rather than an oversight:

    UPPERCUT     act 2 draft  run +1%   3,500 offers   picked ...  ->  +0.41
    BLOOD_WALL   act 2 draft  run -4%  10,000 offers   picked 22%  ->  -2.67
                                                            gap = 3.08

3.08 points apart, so at a 1.0 tolerance Blood Wall is not promoted. Raising the
tolerance to 3.5 to force it was tried and reverted: it also promoted BULWARK
over UPPERCUT on a floor-11 deck, which is precisely the hijacking the tolerance
exists to prevent. A prior cannot be given enough room to beat real data without
also giving it room to beat everything else.

The measurement disagrees with the act 2 floor consistently:

    act 1 defence    FLAME_BARRIER +0% (61% pick)   SHRUG_IT_OFF +1%   IMPERVIOUS +6%
    act 2 defence    FLAME_BARRIER -4% (46% pick)   SHRUG_IT_OFF -1%   BLOOD_WALL -4%

The act 1 floor is supported and now graduates from prior to result: near-neutral
to positive cards at high take rates, which is what "free, and buys something the
sim cannot see" looks like once someone checks. It still fires, because act 1
defensive cards are not 3 points behind anything.

The act 2 floor of two is NOT supported by anything here. It remains in
DEFENCE_FLOOR_BY_ACT because it was asked for twice and the failure it targets --
dying to a 40-damage hit with no block -- is a failure of THIS agent, which
untapped's human population does not share. But it will now only promote a
defensive card that is genuinely close, which on current data it rarely is.

Real reasons the human number might not transfer: a player converts block with
sequencing and potions the pilot cannot, and by act 2 a human is executing a deck
plan that a reactive block card dilutes -- while this agent has no plan to
dilute. None of that is measured. If the act 2 floor is ever dropped, drop this
back to 1.0 with it; it exists only to let the floor fire.
"""

BATTERY_POINTS_PER_UNIT = 5.0
"""Run-winrate points per unit of battery score. The rate at which simulation is
allowed to argue with tens of thousands of real runs.

The battery's spread across one reward screen is ~0.2-0.3 units, so at 5 it moves
a candidate by about a point. Priors between candidates on a screen typically
differ by 1-3 points. So the prior sets the ordering, and the battery adjusts
within it -- which is what "prior leads" has to mean numerically.

It was 12 first, giving the battery ~3 points of swing. That is more than most
prior gaps, so the sim still led and the prior was decoration. Worth stating
because "we added a prior" and "the prior decides anything" are different claims
and only the second one is worth having.

Raise it and simulation overrules real data; drop it to 0 and the agent drafts
off a tier list with no idea what is already in its deck.
"""

HP_WEIGHT = 0.5
"""How much a full bar of HP is worth against a win, in the combined score.

At 0.5 a candidate that wins 10% less often has to save more than a fifth of the
player's max HP to make up for it. Winning should dominate; HP should decide
between candidates that all win.
"""


@dataclass(frozen=True)
class CandidateScore:
    card: object
    score: float
    win_rate: float
    hp_lost: float
    fights: int
    prior: float = 0.0
    """Untapped's contribution to `score`, already included in it. Kept separate
    so a log line can say whether the sim or the real-run data made the call."""

    @property
    def label(self) -> str:
        return getattr(self.card, "name", str(self.card))


def tiers_for_floor(floor: int) -> tuple[Tier, ...]:
    """The cells that discriminate for a decision made on this floor.

    A draft is answering "what do I face next", so the useful comparison is the
    normal and elite fights of the current act. Bosses are excluded: with a
    mid-act deck most candidates score zero against them, and a column of zeros
    ranks nothing.
    """
    act = 1 if floor <= 17 else (2 if floor <= 34 else 3)
    return (Tier(act, "normal"), Tier(act, "elite"))


def _act_for_floor(floor: int) -> int:
    return 1 if floor <= 17 else (2 if floor <= 34 else 3)


def is_defensive(card) -> bool:
    return bool(getattr(card, "base_block", 0) or 0)


def is_high_quality_defence(card) -> bool:
    return (getattr(card, "base_block", 0) or 0) >= HIGH_QUALITY_BLOCK


def apply_composition_rules(
    ranked: list[CandidateScore], deck: Sequence, floor: int
) -> list[CandidateScore]:
    """Cap defensive cards, and floor them, in that order.

    The cap drops candidates outright; the floor only promotes one that is
    already close. Neither invents a score -- the ranking below the top is
    untouched, so `margin` still reports what the measurement actually said.
    """
    act = _act_for_floor(floor)
    # Both rules count HIGH-QUALITY defence only. Counting anything with block
    # would count the four starter Defends, so the cap fired on every deck from
    # turn one and the floor could never run. It would also not match what was
    # measured: the variants added Flame Barrier / Blood Wall / Bulwark on top of
    # those Defends, which were constant across every arm.
    held_good = sum(1 for c in deck if is_high_quality_defence(c))
    held = held_good

    cap = DEFENCE_CAP_BY_ACT.get(act, 3)
    if held >= cap:
        trimmed = [
            r for r in ranked
            if r.card is None or not is_high_quality_defence(r.card)
        ]
        if trimmed:
            logger.info(
                "composition: deck already holds %d high-quality defensive cards "
                "(act %d cap %d); further defence dropped", held, act, cap,
            )
            ranked = trimmed

    needed = DEFENCE_FLOOR_BY_ACT.get(act, 1)
    if held_good < needed and ranked:
        best = ranked[0].score
        for index, candidate in enumerate(ranked):
            if index == 0 or candidate.card is None:
                continue
            if not is_high_quality_defence(candidate.card):
                continue
            if best - candidate.score <= DEFENCE_TOLERANCE:
                logger.info(
                    "composition: deck has %d/%d high-quality defensive cards for "
                    "act %d; promoting %s (%.3f behind the winner, tolerance %.2f)",
                    held_good, needed, act, candidate.label,
                    best - candidate.score, DEFENCE_TOLERANCE,
                )
                return [candidate] + [r for r in ranked if r is not candidate]
            break

    return ranked


def _combined(win_rate: float, hp_lost: float, max_hp: int) -> float:
    if hp_lost != hp_lost:  # NaN: never won, so there is no HP cost to weigh
        return win_rate
    return win_rate - HP_WEIGHT * (hp_lost / max_hp)


def score_candidate(
    deck: Sequence,
    card,
    pilot: Pilot,
    *,
    tiers: Sequence[Tier],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_hp: int = IRONCLAD_STARTING_HP,
    fights: int = GAUNTLET_FIGHTS,
) -> CandidateScore:
    """Play `deck + card` through a gauntlet and report HP surviving.

    Scored on HP left after consecutive fights WITHOUT healing, because the
    per-fight version of this reset to full HP every fight and so could not value
    HP at all. On the deck that actually reached floor 11 live, ranking Body Slam
    against skipping:

        per-fight scoring   BODY_SLAM 0.695 vs SKIP 0.743   (0.05, inside noise)
        gauntlet scoring    BODY_SLAM 25.7hp vs SKIP 32.9hp (7.2 HP, clear)

    The agent took Body Slam. With four Defends and no other block source it
    deals almost nothing, and the old scoring could not see the difference.

    Cheaper as well as sharper: 24 fights per candidate against 90.
    """
    candidate_deck = list(deck) + ([card] if card is not None else [])

    tier = tiers[0] if tiers else Tier(1, "normal")
    hp_left = mean_gauntlet_hp(
        candidate_deck, tier, pilot, seeds=seeds, max_hp=max_hp, fights=fights
    )

    return CandidateScore(
        card=card,
        score=hp_left / max_hp,        # normalised so thresholds stay comparable
        win_rate=float("nan"),
        hp_lost=max_hp - hp_left,
        fights=len(list(seeds)) * fights,
    )


def rank_candidates(
    deck: Sequence,
    candidates: Sequence,
    pilot: Pilot,
    *,
    floor: int = 1,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_hp: int = IRONCLAD_STARTING_HP,
    include_skip: bool = True,
    use_priors: bool = True,
) -> list[CandidateScore]:
    """Rank the cards on offer, best first.

    `include_skip` adds the current deck unchanged as a candidate, so declining a
    card is a real option rather than something the caller has to special-case.
    Skipping genuinely wins sometimes -- a bad card is worse than no card, and
    that is a thing this can measure rather than assume.

    `use_priors` folds in untapped's real-run winrate for each card. The prior
    leads on card quality and the battery adjusts for deck fit, because the two
    are good at different questions: 16,000 real offers know Taunt beats Fiend
    Fire, and only the simulation knows the deck already holds three Taunts.
    """
    tiers = tiers_for_floor(floor)
    options = list(candidates) + ([None] if include_skip else [])

    scored = [
        score_candidate(
            deck, card, pilot, tiers=tiers, seeds=seeds, max_hp=max_hp
        )
        for card in options
    ]
    scored = _to_winrate_points(scored, floor, use_priors)
    scored.sort(key=lambda s: s.score, reverse=True)
    return apply_composition_rules(scored, deck, floor)


def _to_winrate_points(
    scored: list[CandidateScore], floor: int, use_priors: bool
) -> list[CandidateScore]:
    """Rescore everything in percentage points of run win rate.

    The prior sets the units because it is the quantity with 27,000 runs behind
    it, and the battery is converted onto that scale as an adjustment. Doing it
    the other way -- adding a prior into battery units -- was the first attempt
    and it did nothing at all:

        battery only       FIEND_FIRE +0.636   TAUNT +0.395
        prior as an add-on FIEND_FIRE +0.629   TAUNT +0.425   (still loses)

    A 0.24 battery gap cannot be overturned by a 0.03 nudge. If the prior is
    meant to lead, it has to be the base quantity, not a tiebreak.

    ZERO IS SKIPPING

    Untapped's deltas are already relative to not taking the card, so skip is the
    natural origin: its battery score becomes the reference and it lands at 0.0.
    A candidate's score then reads directly as "run win rate points against
    declining", which is what the decision actually is. Without a skip option the
    mean stands in, and only the ordering is meaningful.
    """
    import dataclasses

    from sts2_env.evaluation.card_priors import prior_score

    if not scored:
        return scored

    skips = [s.score for s in scored if s.card is None]
    reference = skips[0] if skips else sum(s.score for s in scored) / len(scored)

    rescored = []
    for candidate in scored:
        prior = 0.0
        if use_priors and candidate.card is not None:
            # None means untapped has never seen this card -- a patch's new card,
            # or one our enum names differently. It keeps a zero prior and is
            # ranked on the battery alone, rather than being scored as average.
            prior = prior_score(
                candidate.card, decision="card_reward", floor=floor
            ) or 0.0
        adjustment = (candidate.score - reference) * BATTERY_POINTS_PER_UNIT
        rescored.append(
            dataclasses.replace(candidate, score=prior + adjustment, prior=prior)
        )
    return rescored


def best_index(ranked: Sequence[CandidateScore], candidates: Sequence) -> int | None:
    """Index of the winning card in the original list, or None to skip."""
    if not ranked:
        return None
    winner = ranked[0].card
    if winner is None:
        return None
    for index, card in enumerate(candidates):
        if card is winner:
            return index
    return None


def margin(ranked: Sequence[CandidateScore]) -> float:
    """Gap between the top two. Small means the measurement did not separate them,
    which is worth knowing before treating the winner as the right answer."""
    if len(ranked) < 2:
        return float("inf")
    return ranked[0].score - ranked[1].score
