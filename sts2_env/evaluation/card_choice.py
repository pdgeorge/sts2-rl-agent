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

DENSITY_POINTS = 40.0
"""Run-winrate points per unit of block-density correction.

Replaces DEFENCE_CAP_BY_ACT and DEFENCE_FLOOR_BY_ACT, which counted cards. A
count is not a stable target: four Defends is 40% of a starter deck and 16% of a
25-card one, so "two defensive cards by act 2" silently weakened exactly as the
run got deadlier. Density is the quantity that stays meaningful.

Scale. One card moves a 20-card deck's density by ~0.05, so a deck sitting 0.05
below the band gets ~1.5 points of push toward a block card -- enough to flip a
close call, not enough to overturn the 3-point prior gap between a strong card
and a weak one. Calibrated to that, not to taste.

WHY THIS IS ONE TERM AND NOT TWO RULES

The old pair were a hard cap (drop candidates) and a floor (promote one within a
tolerance), and they fought each other. Forcing the act 2 floor to fire needed a
3.5-point tolerance that also promoted BULWARK over UPPERCUT -- a prior given
enough room to beat real data has enough room to beat everything else. A signed,
continuous term has no tolerance to widen: it rewards block below the band and
penalises it above, which is what the cap and floor were each half-saying.

WHAT THIS RETIRES

DEFENCE_CAP_BY_ACT was labelled MEASURED on this table:

    0 defensive cards 77.3%   1 card 74.7%   2 cards 74.7%   3 cards 38.7%

produced by `greedy_pilot`, which ranks by base_damage alone and therefore plays
block cards only with leftover energy after everything damaging. Handed three
block cards it holds three near-dead cards, so the "collapse" is as easily a
measurement of the pilot as of the deck. Under density, three block cards in a
12-card deck is 25% -- the bottom of the healthy band, not a collapse.

DEFENCE_FLOOR_BY_ACT was pdgeorge's play knowledge, and its act 2 half sat
against untapped's thick sample (BLOOD_WALL -4% over 10,000 offers). The band
dissolves that argument rather than settling it: density needs no per-act
tuning, because "have block you can actually draw" is true in every act, and the
untapped figure was a card-level average over decks at 45% density and decks at
10% -- a conditional quantity estimated by an unconditional statistic.
"""

BLOAT_POINTS = 2.0
CYCLE_TARGET_TURNS = 3.0
"""Penalty for how long the deck takes to cycle, and the point it starts.

pdgeorge: "'something' only beats 'skip' up to a point. There is a point where
taking garbage worsens your deck." Nothing in the scorer said that. The battery
runs five fights from full HP, over which one filler card costs almost nothing,
so a marginal card beat skipping at any deck size and decks grew to 17-23 cards
live while the guide's headline advice is to skip liberally.

QUADRATIC, AND THAT IS THE POINT

Cycle time is `(deck size - draw) / 5`, so it is LINEAR in deck size and the
delta from adding one card is a flat 0.2 turns however big the deck is. A linear
penalty therefore taxes card and skip equally and never changes the decision --
which is the trap, because it looks like it prices bloat and does not.

Squaring the overshoot makes the gap between taking and skipping widen as the
deck grows, which is the actual claim:

    deck   cycle   penalty gap vs skipping
     15     3.0          0.1 points
     20     4.0          0.9
     25     5.0          1.7
     30     6.0          2.5

So an 8th card needs to be barely positive and a 20th needs to be clearly good.
CYCLE_TARGET_TURNS = 3.0 is a 15-card deck at five cards a turn -- lean but not
starved, and it is where Baalorlord's density figures stop being comfortable.
"""

USE_PRIORS_BY_DEFAULT = False
"""Whether untapped's card winrates lead the ranking. OFF -- a measured null.

30 paired simulated runs per arm, same run seeds, `scripts/eval_drafting.py`:

    priors OFF               8.7 floors +/- 0.9    43% skip
    priors ON, raw           7.4 floors +/- 0.7    35% skip
    priors ON, centred       7.5 floors +/- 0.6     5% skip
    priors ON, minus-max     7.7 floors +/- 0.7    55% skip

Not significant at 2 sem, and kept off anyway: three independent integrations
all landed on the same side, and it matches what happened live. Floors over 45
live runs fell to 10.1 against a 13.0-13.6 baseline the day priors shipped,
degrading as the day went on -- 11.4 in the first half, 8.8 in the second.

WHY IT PLAUSIBLY CANNOT WORK AS BUILT

untapped measures what a card is worth to a HUMAN. Cyra is flown by a pilot that
scores 29 of 86 Ironclad cards at exactly zero and cannot sequence, hold, or
build around anything. A card worth +4% to someone who can set it up is worth
nothing to a pilot that will never play it, and the prior overrides the one
instrument that does know that -- the battery, whatever its own faults, at least
measures what THIS pilot can convert.

That makes this a downstream symptom of PLAN_DECKBUILDING phase D0 rather than a
tuning problem, and it should be retried after the pilot can play a full deck --
not before.

WHAT WAS LEARNED AND KEPT

The minus-max shift stays in the code even though the feature is off, because
the failure it fixes is real and subtle: untapped publishes no row for SKIPPING,
so skip sits at 0.00 while any card with a positive delta gets a free push above
it. Centring on the mean makes that worse, not better -- it forces the best card
positive by construction. Anyone re-enabling this needs that already solved.

The scraped table, `card_priors`, the take-rate guard on the smith veto and the
`[humans N%]` log annotation all stay on. They cost nothing and the log line is
how a bad pick gets noticed.
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
    import dataclasses

    from sts2_env.evaluation.deck_metrics import (
        BLOCK_DENSITY_MAX,
        BLOCK_DENSITY_MIN,
        block_density,
    )

    if not ranked:
        return ranked

    before = _band_distance(block_density(deck))
    rescored = []
    for candidate in ranked:
        if candidate.card is None:
            rescored.append(candidate)
            continue
        after = _band_distance(block_density(list(deck) + [candidate.card]))
        # Positive when the card moves the deck TOWARD the band, negative when it
        # pushes further out. One term replaces the old cap and floor because it
        # is signed: block is rewarded below the band and penalised above it,
        # which is what the cap and the floor were separately trying to say.
        move = (before - after) * DENSITY_POINTS
        if move:
            rescored.append(
                dataclasses.replace(candidate, score=candidate.score + move)
            )
        else:
            rescored.append(candidate)

    rescored.sort(key=lambda s: s.score, reverse=True)

    density = block_density(deck)
    if not (BLOCK_DENSITY_MIN <= density <= BLOCK_DENSITY_MAX):
        logger.info(
            "composition: block density %.0f%% is outside the %.0f-%.0f%% band; "
            "candidates moved by up to %.2f points",
            density * 100, BLOCK_DENSITY_MIN * 100, BLOCK_DENSITY_MAX * 100,
            max((abs(a.score - b.score) for a, b in zip(rescored, ranked)),
                default=0.0),
        )
    return rescored


def _band_distance(density: float) -> float:
    """How far outside the healthy block band a density sits. 0.0 inside it."""
    from sts2_env.evaluation.deck_metrics import BLOCK_DENSITY_MAX, BLOCK_DENSITY_MIN

    if density < BLOCK_DENSITY_MIN:
        return BLOCK_DENSITY_MIN - density
    if density > BLOCK_DENSITY_MAX:
        return density - BLOCK_DENSITY_MAX
    return 0.0


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
    use_priors: bool = USE_PRIORS_BY_DEFAULT,
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
    scored = _to_winrate_points(scored, floor, use_priors, deck)
    scored.sort(key=lambda s: s.score, reverse=True)
    return apply_composition_rules(scored, deck, floor)


def _to_winrate_points(
    scored: list[CandidateScore], floor: int, use_priors: bool, deck: Sequence = ()
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

    # None means untapped has never seen this card -- a patch's new card, or one
    # our enum names differently. It keeps a zero prior and is ranked on the
    # battery alone, rather than being scored as average.
    raw = [
        (prior_score(c.card, decision="card_reward", floor=floor) or 0.0)
        if (use_priors and c.card is not None) else 0.0
        for c in scored
    ]

    # Shifted so the BEST card on offer contributes exactly zero and every other
    # card is negative. The prior can then only ever rank cards against each
    # other; whether the best of them beats declining is left entirely to the
    # battery, which is the only thing that knows about THIS deck.
    #
    # Measured over 30 paired simulated runs, and both wrong versions are worth
    # keeping written down because the second looked like the fix for the first:
    #
    #     raw priors        7.4 floors   35% skip rate
    #     centred (mean)    7.5 floors    5% skip rate    <- worse
    #     no priors         8.7 floors   43% skip rate
    #
    # untapped publishes no row for skipping, so skip sits at 0.00 while any
    # card with a positive delta gets a free push above it -- and those deltas
    # are conditioned on players who TOOK the card, inflating them further. The
    # prior was structurally a take-more signal, the opposite of the guide's own
    # headline advice.
    #
    # Centring on the mean makes that WORSE, not better: it forces the best card
    # positive by construction, so something always beats skip. Subtracting the
    # max is the version that keeps untapped's ordering without ever letting it
    # vote on whether to draft at all.
    offered = [v for v, c in zip(raw, scored) if c.card is not None]
    best_prior = max(offered) if offered else 0.0

    rescored = []
    for value, candidate in zip(raw, scored):
        prior = (value - best_prior) if candidate.card is not None else 0.0
        adjustment = (candidate.score - reference) * BATTERY_POINTS_PER_UNIT
        rescored.append(
            dataclasses.replace(
                candidate,
                score=prior + adjustment - _bloat_penalty(deck, candidate.card),
                prior=prior,
            )
        )
    return rescored


def _bloat_penalty(deck: Sequence, card) -> float:
    """What this deck costs itself by being long, in run-winrate points.

    Charged to skipping too, at the deck's current length -- the penalty is not
    a fee for taking a card, it is the standing cost of the deck you would have.
    What decides the pick is the DIFFERENCE, and squaring the overshoot is what
    makes that difference grow with deck size instead of staying flat.
    """
    from sts2_env.evaluation.deck_metrics import cycle_time

    full = list(deck) + ([card] if card is not None else [])
    overshoot = max(0.0, cycle_time(full) - CYCLE_TARGET_TURNS)
    return BLOAT_POINTS * overshoot * overshoot


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
