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
from sts2_env.evaluation.battery import Pilot, Tier, score_cell

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
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
) -> CandidateScore:
    """Play `deck + card` and report how it did."""
    candidate_deck = list(deck) + ([card] if card is not None else [])

    total_fights = 0
    weighted_win = 0.0
    weighted_hp = 0.0
    hp_samples = 0

    for tier in tiers:
        cell = score_cell(
            candidate_deck, tier, pilot, seeds=seeds, max_hp=max_hp
        )
        total_fights += cell.fights
        weighted_win += cell.win_rate * cell.fights
        if cell.hp_lost_on_wins == cell.hp_lost_on_wins:  # not NaN
            weighted_hp += cell.hp_lost_on_wins * cell.fights
            hp_samples += cell.fights

    win_rate = weighted_win / total_fights if total_fights else 0.0
    hp_lost = weighted_hp / hp_samples if hp_samples else float("nan")

    return CandidateScore(
        card=card,
        score=_combined(win_rate, hp_lost, max_hp),
        win_rate=win_rate,
        hp_lost=hp_lost,
        fights=total_fights,
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
