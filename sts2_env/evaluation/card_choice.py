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

DEFENCE_TOLERANCE = 0.08
"""How far behind the winner a defensive card may sit and still be taken.

In score units, which are gauntlet HP over max HP, so roughly 6 HP on an 80 HP
bar. For scale: Uppercut beat skipping by 0.135 on the real floor-11 deck, so at
this tolerance the floor would NOT have overridden that pick, and only fires on a
genuine near-tie.
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
) -> list[CandidateScore]:
    """Rank the cards on offer, best first.

    `include_skip` adds the current deck unchanged as a candidate, so declining a
    card is a real option rather than something the caller has to special-case.
    Skipping genuinely wins sometimes -- a bad card is worse than no card, and
    that is a thing this can measure rather than assume.
    """
    tiers = tiers_for_floor(floor)
    options = list(candidates) + ([None] if include_skip else [])

    scored = [
        score_candidate(
            deck, card, pilot, tiers=tiers, seeds=seeds, max_hp=max_hp
        )
        for card in options
    ]
    scored.sort(key=lambda s: s.score, reverse=True)
    return apply_composition_rules(scored, deck, floor)


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
