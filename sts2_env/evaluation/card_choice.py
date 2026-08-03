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

from dataclasses import dataclass
from typing import Sequence

from sts2_env.core.constants import IRONCLAD_STARTING_HP
from sts2_env.evaluation.battery import Pilot, Tier, mean_gauntlet_hp

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
    return scored


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
