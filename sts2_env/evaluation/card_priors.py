"""What tens of thousands of real runs think of a card.

Scraped from sts2.untapped.gg by `scripts/scrape_untapped.py` into
`data/untapped_cards.json`, and read here.

WHY A PRIOR AT ALL

The battery answers "is this card good?" from eight simulated seeds flown by a
heuristic pilot. That is the wrong instrument for the question, and it was
measurably wrong on the first case anyone checked -- act 1 card reward:

    TAUNT        offered 15,000 times    picked 60%    run winrate  +1%
    FIEND_FIRE   offered    970 times    picked 67%    run winrate  -1%

    battery:     FIEND_FIRE +0.636  >>  TAUNT +0.395

A 16,000-run sample and an 8-seed simulation disagreed, and the simulation lost.
So card *quality* comes from here, and the battery is left the question it is
actually good at: whether this card fits THIS deck, at THIS HP, right now --
which untapped cannot see, because it has no idea what else you are holding.

THREE DECISIONS, NOT ONE

untapped splits by the decision being made, and the splits genuinely disagree.
Fiend Fire is roughly neutral as an act 1 draft (-1%) and clearly bad as an act 1
smith (-6% act winrate). A single "card quality" number would average those into
something true of neither, so `decision=` is required rather than defaulted.

WHAT THESE NUMBERS ARE NOT

A delta is conditioned on runs where a player PICKED the card. A card that only
gets taken when the deck already supports it, or one favoured by strong players,
reads better than it is. This is a prior on a noisy observational estimate, not a
causal effect -- which is why `prior_weight` shrinks toward zero on small samples
rather than trusting a +4% drawn from 400 offers.

MISSING IS NOT ZERO

A card with no row returns None, and callers fall back to the battery alone. A
new card from a patch has no data by definition, and scoring it as 0% would rank
it above everything untapped has measured as negative -- so an unknown card must
stay unknown rather than becoming quietly average.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TABLE = Path(__file__).resolve().parents[2] / "data" / "untapped_cards.json"

DECISIONS = ("card_reward", "shop", "smith")

CONFIDENCE_SAMPLE = 5000
"""Offers at which a delta is trusted at roughly full weight.

Untapped's own numbers are quantised to whole percent, so a +1% on 970 offers is
one part sampling noise and one part rounding. Taunt's act 1 draft row carries
15,000 offers and Fiend Fire's carries 970; treating those as equally solid is
how a thin sample wins an argument it should not. Shrinkage is
`offers / (offers + CONFIDENCE_SAMPLE)`, so 5,000 offers counts half.
"""

# Priors are returned in percentage points of run win rate -- untapped's own
# unit, kept rather than rescaled so a log line reads as the number on the site.
# `card_choice` converts the battery onto this scale, not the other way round:
# the prior is the quantity with 27,000 runs behind it, so it sets the units.


class MissingPriorTable(FileNotFoundError):
    """The scraped table is absent. Run scripts/scrape_untapped.py."""


@lru_cache(maxsize=4)
def load_table(path: str | Path = DEFAULT_TABLE) -> dict:
    path = Path(path)
    if not path.is_file():
        raise MissingPriorTable(
            f"{path} not found. Build it with:\n"
            f"    python scripts/scrape_untapped.py --character ironclad"
        )
    return json.loads(path.read_text())


def _card_key(card) -> str | None:
    card_id = getattr(card, "card_id", card)
    name = getattr(card_id, "name", None)
    if name is None and isinstance(card_id, str):
        name = card_id
    return name.upper() if isinstance(name, str) else None


def act_for_floor(floor: int) -> int:
    return 1 if floor <= 17 else (2 if floor <= 34 else 3)


def card_stats(card, *, decision: str, floor: int = 1, table: dict | None = None) -> dict | None:
    """The raw untapped row for this card, decision and act, or None."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")

    key = _card_key(card)
    if key is None:
        return None
    try:
        data = table if table is not None else load_table()
    except MissingPriorTable:
        return None

    entry = data.get("cards", {}).get(key)
    if not entry:
        return None
    return (entry.get(decision) or {}).get(str(act_for_floor(floor)))


def prior_score(
    card, *, decision: str, floor: int = 1, table: dict | None = None
) -> float | None:
    """Untapped's opinion of this card, in run-winrate points, or None.

    Shrunk toward zero by sample size, so a thin row cannot outvote a thick one.
    Prefers run winrate over act winrate: an act delta rewards a card that wins
    the act you are in and says nothing about the run it costs you later, and
    Armaments is exactly that shape -- +0% act, -1% run in act 1 drafts.
    """
    stats = card_stats(card, decision=decision, floor=floor, table=table)
    if not stats:
        return None

    delta = stats.get("run_winrate")
    if delta is None:
        delta = stats.get("act_winrate")
    if delta is None:
        return None

    offers = float(stats.get("offered", 0) or 0)
    confidence = offers / (offers + CONFIDENCE_SAMPLE) if offers > 0 else 0.0
    return float(delta) * confidence


def describe(card, *, decision: str, floor: int = 1) -> str:
    """One-line explanation, for logs. Says the sample size out loud, because a
    delta without one is how a 970-offer number gets read as settled."""
    stats = card_stats(card, decision=decision, floor=floor)
    if not stats:
        return "no untapped data"
    score = prior_score(card, decision=decision, floor=floor)
    return (
        f"run{stats.get('run_winrate', 0):+d}% over {stats.get('offered', 0):,} "
        f"offers -> {score:+.2f}pts"
    )
