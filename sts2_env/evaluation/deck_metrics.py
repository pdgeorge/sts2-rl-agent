"""Deck-level numbers, computed from a decklist. No pilot, no seeds, no noise.

From Baalorlord's core deckbuilding concepts article. These are the cheapest
useful signals in the whole evaluation stack: pure functions of a list of cards,
microseconds each, and -- unlike everything else in `evaluation/` -- they cannot
be contaminated by the pilot's blind spots, because they never play a card.

WHY DENSITY RATHER THAN A COUNT

`card_choice.DEFENCE_FLOOR_BY_ACT` counts defensive cards: one by act 1, two by
act 2. A count is not a stable target, because the same cards are a smaller and
smaller share of a growing deck:

    10-card starter, 4 Defends           40% block density
    25-card act 2 deck, same 4 Defends   16% block density

So "two defensive cards by act 2" silently weakens exactly as the run gets more
dangerous. The failure reported from live play -- "taking big hits but having no
block cards in the deck" -- is what a diluted density looks like from inside.

It also reconciles two things that looked contradictory. untapped reports act 2
block cards as NEGATIVE (BLOOD_WALL -4% over 10,000 offers). That is a card-level
average over every deck that took one: decks already at 45% block and decks at
10%, added together. Density is a deck-level quantity, and a block card is bad at
45% and good at 15%. A conditional quantity was being estimated by an
unconditional statistic; there was never a real disagreement.

THE BAND, VERIFIED

Baalorlord recommends ~33% block. Checked here rather than quoted, by exact
hypergeometric on an opening hand of 5 (not the binomial approximation):

    density   P(no block)   P(>=4 of 5 are block)
      20%         28%              0.1%
      33%          8%              3.1%
      50%          2%             15.2%

Below ~25% you brick on defence; above ~45% you draw hands that cannot kill
anything. Both failure modes are real and the floor between them is wide, which
is why this is a band and not a threshold.

WHAT IS DELIBERATELY NOT HERE

Frontload versus scaling damage. Splitting those needs to know what a card does
over time, which is exactly the judgement the pilot is currently too blind to
make -- see PLAN_DECKBUILDING.md phase D0. Adding a keyword list to fake it would
be `deck_features.py`'s mistake again, where 20 of 45 hand-written keyword
strings match no card in this game at all.
"""

from __future__ import annotations

from math import comb
from typing import Sequence

DEFAULT_HAND_SIZE = 5
"""Cards drawn per turn, used for cycle time and the hypergeometric checks."""

BLOCK_DENSITY_TARGET = 0.33
BLOCK_DENSITY_MIN = 0.25
BLOCK_DENSITY_MAX = 0.45
"""The healthy band. See the table in the module docstring for where it comes
from. Not every deck wants the same number -- a high-draw deck sees more of
itself per turn and tolerates less block -- so `block_density_penalty` is shaped
as a soft band rather than a hard rule."""

UPGRADE_DENSITY_TARGET = 0.33
"""Baalorlord puts 33-50% upgraded as reasonable for winning high-difficulty
runs. Recorded as the low end, because this agent is nowhere near it: the
measured jump from upgrading eight cards was the act 1 boss going from 13.9% to
69.4% win rate, which is what being far below the band looks like."""


def _is_block(card) -> bool:
    """Any block at all counts.

    Deliberately NOT `card_choice.HIGH_QUALITY_BLOCK` (base_block >= 8). That
    constant answers "is this card real defence against a 40-damage hit", and
    excludes Defend on purpose. Density answers "will I have block in hand",
    where a 5-block Defend counts for exactly as much as anything else. The two
    are different questions and both thresholds are right for their own.
    """
    return (getattr(card, "base_block", 0) or 0) > 0


def block_density(deck: Sequence) -> float:
    """Fraction of the deck that provides block."""
    if not deck:
        return 0.0
    return sum(1 for c in deck if _is_block(c)) / len(deck)


def upgrade_density(deck: Sequence) -> float:
    """Fraction of the deck that is upgraded."""
    if not deck:
        return 0.0
    return sum(1 for c in deck if getattr(c, "upgraded", False)) / len(deck)


def cards_drawn_by(deck: Sequence) -> int:
    """Total extra cards the deck draws per cycle.

    Reads `effect_vars["cards"]`, which is the key the game actually uses. The
    pilot's own draw term read `effect_vars["draw"]` for its entire life and no
    card in the game has ever carried that key, so it returned zero always --
    worth stating here because the same mistake is one typo away.
    """
    total = 0
    for card in deck:
        variables = getattr(card, "effect_vars", None) or {}
        total += int(variables.get("cards", 0) or 0)
    return total


def cycle_time(deck: Sequence, hand_size: int = DEFAULT_HAND_SIZE) -> float:
    """Turns to draw the whole deck once. Lower is better; adding a card raises it.

    `(deck size - card draw) / cards drawn per turn`, from the article. The
    interesting consequence is the one it states outright: a card that draws
    exactly ONE card does not improve cycle time against skipping, so it has to
    justify itself on the rest of its text alone.
    """
    if not deck or hand_size <= 0:
        return 0.0
    return max(0.0, len(deck) - cards_drawn_by(deck)) / hand_size


def p_no_block(deck: Sequence, hand_size: int = DEFAULT_HAND_SIZE) -> float:
    """Chance the opening hand contains no block at all.

    Exact hypergeometric over the real decklist rather than the binomial
    approximation, which overstates the risk on small decks -- and act 1 decks
    are small, which is where the number gets used.
    """
    n = len(deck)
    if n == 0 or hand_size <= 0:
        return 0.0
    draws = min(hand_size, n)
    blockers = sum(1 for c in deck if _is_block(c))
    if n - blockers < draws:
        return 0.0
    return comb(n - blockers, draws) / comb(n, draws)


def p_flooded(deck: Sequence, hand_size: int = DEFAULT_HAND_SIZE,
              flooded_at: int = 4) -> float:
    """Chance the opening hand is mostly block and cannot kill anything.

    The other wall of the band, and the reason more block is not simply better.
    """
    n = len(deck)
    if n == 0 or hand_size <= 0:
        return 0.0
    draws = min(hand_size, n)
    blockers = sum(1 for c in deck if _is_block(c))
    total = comb(n, draws)
    return sum(
        comb(blockers, k) * comb(n - blockers, draws - k)
        for k in range(min(flooded_at, draws), min(blockers, draws) + 1)
    ) / total


def block_density_penalty(deck: Sequence) -> float:
    """How far outside the healthy band this deck sits, 0.0 when inside it.

    Signed by which wall it failed: NEGATIVE means too little block, POSITIVE
    means too much. A caller wanting "should I take this block card" wants the
    sign, not just the distance -- at +0.1 another block card makes things worse.
    """
    density = block_density(deck)
    if density < BLOCK_DENSITY_MIN:
        return density - BLOCK_DENSITY_MIN
    if density > BLOCK_DENSITY_MAX:
        return density - BLOCK_DENSITY_MAX
    return 0.0


def describe(deck: Sequence) -> str:
    """One line for logs, so a bad deck is visible in the transcript."""
    return (
        f"{len(deck)} cards  block {block_density(deck):.0%} "
        f"(no-block hand {p_no_block(deck):.0%}, flooded {p_flooded(deck):.0%})  "
        f"upgraded {upgrade_density(deck):.0%}  cycle {cycle_time(deck):.1f} turns"
    )
