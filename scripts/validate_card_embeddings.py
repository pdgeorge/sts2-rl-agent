"""Does the card-embedding archetype idea actually work? Run this before building on it.

    hf download t22000t/slay-the-spire-2-card-embeddings --repo-type dataset \
        --local-dir <dir>
    python scripts/validate_card_embeddings.py <dir>/embeddings.parquet

Four checks, in the order they can fail cheapest-first. See the Phase 5 design
note in `docs/GLM_ROADMAP_50P_ACT1.md`.

THE ONE THING THAT MATTERS MOST
------------------------------
**Mean-centre the embeddings before doing anything.** Raw cosine does not work
on this dataset. Every STS2 card is ~0.9 similar to every other STS2 card --
they are all short mechanics JSON from one game, so the vectors share a huge
common "this is a card" component that swamps the differences that matter.
Measured: raw archetype spread ~0.09, and a deck flipped label on a 0.007
margin, which is a coin toss wearing a classifier's hat.

Subtracting the pool mean removes that shared component. Spread goes to ~0.475
and leave-one-out classification goes to 11/11. Same vectors, same cosine, one
subtraction.

NO MODEL IS DOWNLOADED OR RUN
-----------------------------
Archetypes are defined by **seed cards**, not by prose. The archetype vector is
the normalised mean of its seeds' embeddings, which the dataset already ships.
So Qwen3-Embedding-0.6B never has to run -- not here, and not at decision time
either. Defining archetypes by cards rather than sentences is also better
grounded: it says "like these", not "like how I described these".

WHAT THIS DOES NOT SHOW
-----------------------
That archetype-guided picking builds *better decks*. That needs an A/B of full
simulated runs against `card_quality.py` alone. This only shows the signal is
real enough to be worth building on.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

#: Screened for fat-deck viability -- see the HARD CONSTRAINT in the Phase 5
#: design note. Cyra cannot remove cards, so archetypes that need a thin deck
#: are excluded. `exhaust` is kept deliberately as a NEGATIVE CONTROL: it needs
#: thinning to work, so it must never be selected as a target, and it is here
#: only to show the classifier can tell it apart from the viable three.
ARCHETYPE_SEEDS = {
    "strike-synergy": [
        "PERFECTED_STRIKE", "ASHEN_STRIKE", "TWIN_STRIKE",
        "POMMEL_STRIKE", "SETUP_STRIKE",
    ],
    "block-scaling": ["BARRICADE", "BODY_SLAM", "DEMONIC_SHIELD"],
    "strength": ["DEMON_FORM", "INFLAME", "LIMIT_BREAK"],
    "exhaust": ["CORRUPTION", "DARK_EMBRACE", "FIEND_FIRE"],
    "bloodletting": ["RUPTURE", "TEAR_ASUNDER", "SPITE", "INFERNO"],
}
"""Seeded on payoffs, not enablers, so `bloodletting` means "built to lose HP
profitably" rather than "contains a card that costs HP". A seed absent from the
build (LIMIT_BREAK is not in this one) is skipped, not fatal."""

STARTER_CARDS = {"STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"}
MIN_PICKED_FOR_CENTROID = 3
"""Below this, the centroid is noise.

The Ironclad starter is 5 Strike, 4 Defend and Bash. A centroid over the whole
deck is ten parts starter to one part signal and reads "attack deck" for every
run, which is why the centroid is taken over non-starter cards only and why the
first few picks cannot use it at all.
"""


def _strip_card_suffix(card_id: str) -> str:
    """The simulator names some cards `BARRICADE_CARD`; the dataset says `BARRICADE`."""
    return card_id[:-5] if card_id.endswith("_CARD") else card_id


def resolve(card_id: str, emb: dict) -> str | None:
    """Find a card in whichever id convention this embedding set uses.

    Our own vectors carry the simulator's ids (`BARRICADE_CARD`); the published
    dataset drops the suffix (`BARRICADE`). Seeds are written in the short form
    and resolved here, so the same seed list works against both.
    """
    if card_id in emb:
        return card_id
    if f"{card_id}_CARD" in emb:
        return f"{card_id}_CARD"
    stripped = _strip_card_suffix(card_id)
    return stripped if stripped in emb else None


def load_embeddings(path: str) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Return {card_id: centred unit vector} and {card_id: parsed card_text}.

    Accepts our own `.npz` (from `generate_card_embeddings.py`) or the reference
    `.parquet` from the published dataset, so the two can be compared directly.
    """
    if str(path).endswith(".npz"):
        blob = np.load(path, allow_pickle=False)
        ids = [str(x) for x in blob["ids"]]
        raw = blob["vectors"].astype(np.float32)
        meta: dict[str, dict] = {}
        text_path = Path(path).parent.parent / "output" / "card_text.json"
        if not text_path.exists():
            text_path = Path("output/card_text.json")
        if text_path.exists():
            meta = {k: v for k, v in json.loads(text_path.read_text()).items()}
        centred = raw - raw.mean(axis=0)
        centred /= np.linalg.norm(centred, axis=1, keepdims=True)
        return {cid: centred[i] for i, cid in enumerate(ids)}, meta

    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pydict()
    ids = list(table["id"])
    raw = np.stack([np.asarray(v, dtype=np.float32) for v in table["embedding"]])

    centred = raw - raw.mean(axis=0)
    centred /= np.linalg.norm(centred, axis=1, keepdims=True)

    meta: dict[str, dict] = {}
    for card_id, text in zip(ids, table["card_text"]):
        try:
            meta[card_id] = json.loads(text)
        except Exception:
            meta[card_id] = {}
    return {cid: centred[i] for i, cid in enumerate(ids)}, meta


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def archetype_matrix(emb: dict[str, np.ndarray], seeds=ARCHETYPE_SEEDS):
    names, rows = [], []
    for name, cards in seeds.items():
        vectors = [emb[resolve(c, emb)] for c in cards if resolve(c, emb)]
        if not vectors:
            continue
        names.append(name)
        rows.append(_unit(np.mean(vectors, axis=0)))
    return names, np.stack(rows)


def classify(deck_card_ids, emb, names, matrix):
    """(label, similarities) for a deck, or None when there is too little signal."""
    vectors = [
        emb[resolve(c, emb)]
        for c in deck_card_ids
        if resolve(c, emb) and c not in STARTER_CARDS
    ]
    if len(vectors) < MIN_PICKED_FOR_CENTROID:
        return None
    sims = matrix @ _unit(np.mean(vectors, axis=0))
    return names[int(sims.argmax())], sims


def peakedness(card_id: str, emb, matrix) -> float | None:
    """How much a card commits to one archetype: `max(sims) - mean(sims)`.

    High means deck-*defining* (Barricade); low means generically fine (Iron
    Wave). This is the number that separates "great card" from "card that
    decides what deck you are building", which is the distinction early picks
    need and card quality alone cannot make.
    """
    key = resolve(card_id, emb)
    if key is None:
        return None
    sims = matrix @ emb[key]
    return float(sims.max() - sims.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", help="embeddings.parquet from the HF dataset")
    parser.add_argument("--situations",
                        default="tests/fixtures/act1_combat_train_2000.json")
    args = parser.parse_args()

    emb, meta = load_embeddings(args.parquet)
    names, matrix = archetype_matrix(emb)
    print(f"{len(emb)} card embeddings, {len(names)} archetypes\n")

    print("=== 1. coverage against the simulator ===")
    import sts2_env.cards  # noqa: F401  (resolves package import order)
    from sts2_env.core.enums import CardId

    sim_ids = {c.name for c in CardId}
    matched = {s for s in sim_ids if _strip_card_suffix(s) in emb}
    print(f"  {len(matched)}/{len(sim_ids)} simulator cards have an embedding")

    print("\n=== 2. leave-one-out: a held-out seed must still find its archetype ===")
    hits = total = 0
    for name, seeds in ARCHETYPE_SEEDS.items():
        present = [c for c in seeds if resolve(c, emb)]
        for held in present:
            rest = {
                n: ([c for c in s if c != held and resolve(c, emb)] if n == name
                    else [c for c in s if resolve(c, emb)])
                for n, s in ARCHETYPE_SEEDS.items()
            }
            if len(rest[name]) < 2:
                continue
            loo_names, loo_matrix = archetype_matrix(emb, rest)
            predicted = loo_names[int((loo_matrix @ emb[resolve(held, emb)]).argmax())]
            hits += predicted == name
            total += 1
            if predicted != name:
                print(f"  MISS {held}: true={name} pred={predicted}")
    print(f"  {hits}/{total} correct")

    print("\n=== 3. peakedness separates deck-defining from generically good ===")
    for card in ("BARRICADE", "DEMON_FORM", "PERFECTED_STRIKE", "CORRUPTION",
                 "IRON_WAVE", "SHRUG_IT_OFF", "POMMEL_STRIKE"):
        score = peakedness(card, emb, matrix)
        print(f"  {card:<18} {score:+.3f}" if score is not None
              else f"  {card:<18} (absent)")

    print("\n=== 4. real harvested decks, by floor band ===")
    print("  NOTE: these decks were assembled by a walker making RANDOM card")
    print("  picks, and the ironclad pool is itself skewed, so a lopsided")
    print("  distribution here is expected and is not evidence either way.")
    from sts2_env.search.situation import load_situations

    bands: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    spreads = []
    for situation in load_situations(args.situations):
        result = classify([c.card_id for c in situation.deck], emb, names, matrix)
        if result is None:
            continue
        label, sims = result
        spreads.append(float(sims.max() - sims.min()))
        floor = situation.total_floor
        band = "1-8" if floor <= 8 else ("9-12" if floor <= 12 else "13-16")
        bands[band][label] += 1
    print(f"  mean archetype spread {np.mean(spreads):.3f} "
          f"(raw, uncentred, this is ~0.09 and unusable)")
    for band in ("1-8", "9-12", "13-16"):
        total_band = sum(bands[band].values())
        if not total_band:
            continue
        share = {k: f"{100 * v / total_band:.0f}%" for k, v in bands[band].most_common()}
        print(f"  floors {band:<6} n={total_band:<5} {share}")

    pool = collections.Counter()
    for card_id, card_meta in meta.items():
        if card_meta.get("color") == "ironclad":
            pool[names[int((matrix @ emb[card_id]).argmax())]] += 1
    pool_total = sum(pool.values())
    print("  ironclad POOL skew for comparison:",
          {k: f"{100 * v / pool_total:.0f}%" for k, v in pool.most_common()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
