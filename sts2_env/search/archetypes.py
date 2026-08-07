"""Which deck is this, and does this card belong in it.

Step 4-5 of the Phase 5 build plan. Loads the card vectors, defines the
archetypes, and keeps a run's accumulated evidence about which deck it is
building.

CARDS BELONG TO MORE THAN ONE ARCHETYPE, AND THAT IS NOT A BUG
--------------------------------------------------------------
Rupture grants Strength whenever you lose HP. It is a bloodletting card *and* a
strength card, and the vectors say so:

    RUPTURE   bloodletting +0.53   strength +0.25   strike +0.08
    BARRICADE block-scaling +0.68  strength +0.05   strike +0.02

A single-archetype card spikes on one and sits near zero elsewhere; a
dual-archetype card scores on both, in the right order. Accumulating
`score[a] += cosine(card, a)` handles this for free -- Rupture adds to
bloodletting and to strength, in proportion.

This was originally measured with leave-one-out single-label accuracy, which
scored 10/15 and looked like a failure. It was the metric that was wrong: it
insists each card has exactly one archetype, so a card that legitimately serves
two is marked incorrect for saying so. Top-2 is 13/15, and the per-card
breakdowns above are the real evidence.

PEAKEDNESS FALLS OUT OF THE SAME NUMBERS
----------------------------------------
`max(sims) - mean(sims)` is high for a card that commits you to one deck and low
for one that fits anywhere. Barricade is peaked; Iron Wave is flat (+0.55
against +0.09). Rupture sits between, which is right -- a card that works in two
decks commits you less than one that works in one.

Early picks want peaked cards, because the deck has no direction yet and a
committed card supplies one. Later picks want fit to the direction already
chosen. Both come from these vectors.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "card_embeddings.npz"

ARCHETYPE_SEEDS: dict[str, tuple[str, ...]] = {
    "strike-synergy": (
        "PERFECTED_STRIKE", "ASHEN_STRIKE", "TWIN_STRIKE",
        "POMMEL_STRIKE", "SETUP_STRIKE",
    ),
    "block-scaling": ("BARRICADE", "BODY_SLAM", "DEMONIC_SHIELD"),
    "strength": ("DEMON_FORM", "INFLAME", "LIMIT_BREAK"),
    "exhaust": ("CORRUPTION", "DARK_EMBRACE", "FIEND_FIRE"),
    "bloodletting": ("RUPTURE", "TEAR_ASUNDER", "SPITE", "INFERNO"),
}
"""Screened for fat-deck viability, because Cyra cannot remove cards.

`exhaust` survives that screen despite appearances: Corruption makes Skills free
and exhausts them, which thins the deck *during* the fight -- exactly the
thinning that cannot be arranged between fights. What is excluded is anything
needing a thin deck to function at all, like precise draw combos.

`bloodletting` is seeded on its payoffs (Rupture, Tear Asunder, Spite, Inferno)
rather than its enablers (Bloodletting, Offering, Hemokinesis). Seeding on
payoffs makes the archetype mean "built to lose HP profitably" rather than
"contains a card that costs HP".

A seed absent from the build is skipped rather than fatal -- LIMIT_BREAK is not
in this one.
"""

STARTER_CARDS = frozenset({"STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"})

MIN_CARDS_FOR_DIRECTION = 3
"""Below this there is no deck to read.

The Ironclad starts with 5 Strike, 4 Defend and Bash. A centroid over the whole
deck is ten parts starter to one part signal and reads the same for every run,
so the direction is taken over non-starter cards only and not trusted until
there are a few.
"""

COMMIT_MARGIN = 0.15
"""How far ahead the leader must be before the deck has a plan.

A margin rather than a pick count. Two Barricade-shaped cards settle it by pick
two; six diffuse ones never should. A fixed "commit at three picks" is arbitrary
in both directions.
"""


def _strip(card_id: str) -> str:
    return card_id[:-5] if card_id.endswith("_CARD") else card_id


@functools.lru_cache(maxsize=1)
def _vectors() -> tuple[dict[str, int], np.ndarray]:
    """{card_id: row} and the mean-centred unit vectors.

    CENTRING IS NOT OPTIONAL. Every card in this game is ~0.9 similar to every
    other one -- they are all short mechanics text from a single game, so the
    vectors share an enormous "this is a card" component that swamps everything
    that matters. Raw, the archetype spread is ~0.09 and a deck flips label on a
    0.007 margin. Centred, the spread is ~0.48. Same vectors, one subtraction.
    """
    blob = np.load(DATA, allow_pickle=False)
    ids = [str(x) for x in blob["ids"]]
    raw = blob["vectors"].astype(np.float32)
    centred = raw - raw.mean(axis=0)
    centred /= np.linalg.norm(centred, axis=1, keepdims=True)
    return {cid: i for i, cid in enumerate(ids)}, centred


def _row(card_id: str) -> np.ndarray | None:
    index, vectors = _vectors()
    for key in (card_id, f"{_strip(card_id)}_CARD", _strip(card_id)):
        if key in index:
            return vectors[index[key]]
    return None


@functools.lru_cache(maxsize=1)
def _archetype_matrix() -> tuple[tuple[str, ...], np.ndarray]:
    names, rows = [], []
    for name, seeds in ARCHETYPE_SEEDS.items():
        vectors = [v for v in (_row(c) for c in seeds) if v is not None]
        if not vectors:
            continue
        centroid = np.mean(vectors, axis=0)
        norm = np.linalg.norm(centroid)
        names.append(name)
        rows.append(centroid / norm if norm else centroid)
    return tuple(names), np.stack(rows)


def archetype_names() -> tuple[str, ...]:
    return _archetype_matrix()[0]


def card_affinities(card_id: str) -> dict[str, float]:
    """How much this card belongs to each archetype. Empty if unknown."""
    vector = _row(card_id)
    if vector is None:
        return {}
    names, matrix = _archetype_matrix()
    return dict(zip(names, (matrix @ vector).tolist()))


def peakedness(card_id: str) -> float:
    """How much this card commits you. `max - mean` over the affinities.

    High is deck-defining (Barricade), low is generically fine (Iron Wave).
    Zero for a card with no vector, so an unknown card never looks committed.
    """
    affinities = card_affinities(card_id)
    if not affinities:
        return 0.0
    values = np.fromiter(affinities.values(), dtype=np.float32)
    return float(values.max() - values.mean())


class DeckDirection:
    """A run's accumulated evidence about which deck it is building.

    Votes accumulate rather than a label being recomputed, so a card that serves
    two archetypes contributes to both -- which is what cards actually do.
    """

    def __init__(self, commit_margin: float = COMMIT_MARGIN):
        self.commit_margin = commit_margin
        self.scores: dict[str, float] = {name: 0.0 for name in archetype_names()}
        self.counted = 0
        self._committed: str | None = None

    def observe(self, card_id: str) -> None:
        """Count a card the deck has gained. Starters carry no direction."""
        if _strip(card_id) in STARTER_CARDS or card_id in STARTER_CARDS:
            return
        affinities = card_affinities(card_id)
        if not affinities:
            return
        for name, value in affinities.items():
            self.scores[name] += value
        self.counted += 1

    def observe_deck(self, card_ids) -> None:
        for card_id in card_ids:
            self.observe(card_id)

    @property
    def leader(self) -> tuple[str | None, float]:
        """(archetype, margin over the runner-up). (None, 0.0) with no evidence."""
        if self.counted < MIN_CARDS_FOR_DIRECTION:
            return None, 0.0
        ranked = sorted(self.scores.items(), key=lambda kv: -kv[1])
        if len(ranked) < 2:
            return ranked[0][0] if ranked else None, 0.0
        return ranked[0][0], ranked[0][1] - ranked[1][1]

    @property
    def committed(self) -> str | None:
        """The archetype this run is building, once the evidence is clear.

        Sticky: once committed it stays, because a deck that changes its mind on
        floor 12 has two half-decks. The margin can only ever grow as more
        on-archetype cards are taken.
        """
        if self._committed is not None:
            return self._committed
        name, margin = self.leader
        if name is not None and margin >= self.commit_margin:
            self._committed = name
        return self._committed

    def fit(self, card_id: str) -> float:
        """How well a card suits the direction so far.

        Before commitment this is peakedness -- with no plan, the useful card is
        the one that supplies a plan. After, it is affinity for the committed
        archetype.
        """
        committed = self.committed
        if committed is None:
            return peakedness(card_id)
        return card_affinities(card_id).get(committed, 0.0)
