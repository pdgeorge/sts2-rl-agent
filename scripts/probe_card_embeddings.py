"""Phase 1 gate: do the built vectors carry mechanical information?

    python scripts/probe_card_embeddings.py

Fits a linear probe from each card's vector to facts the simulator already knows
-- card type, cost, base damage, base block -- and scores it on held-out cards.
The point is not that a probe is a good model; it is that a *linear* readout
recovering these facts means the information is present and easy to reach, so a
policy head can use it. A linear probe is therefore a lower bound.

The comparison is against the same probe run on the HuggingFace vectors, which
were built from the game's own card text:

    card_type   96.3%   (majority-class baseline 40.4%)
    cost        R^2 0.26
    base_damage R^2 0.34
    base_block  R^2 0.21

Those numbers are the bar. Our text is code-derived rather than natural language,
so it may do better on mechanics and worse on flavour. Materially *worse* on card
type would mean the template is losing information and needs fixing before
anything is built on top.

Ground truth here is this repo's own metadata, not someone else's opinion, which
is what makes this a gate rather than a smoke test. Neighbour lists are printed
too, but only as a sanity check -- agreeing with another unvalidated system is
weak evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HF_BASELINE = {
    "card_type": 0.963,
    "cost": 0.26,
    "base_damage": 0.34,
    "base_block": 0.21,
}
MAJORITY_BASELINE = 0.404


def collect_targets(names: list[str]):
    """Ground truth from the simulator, for the cards that have a row."""
    import sts2_env.cards  # noqa: F401
    import sts2_env.powers  # noqa: F401
    from sts2_env.cards.factory import card_preview
    from sts2_env.core.enums import CardId

    rows, cost, damage, block, types = [], [], [], [], []
    for row, name in enumerate(names):
        try:
            preview = card_preview(CardId[name])
        except Exception:  # noqa: BLE001
            continue
        if preview.cost is None or preview.cost < 0:
            continue
        rows.append(row)
        cost.append(preview.cost)
        damage.append(preview.base_damage or 0)
        block.append(preview.base_block or 0)
        types.append(preview.card_type.name)
    return np.array(rows), np.array(cost), np.array(damage), np.array(block), types


def _splits(n: int, count: int, holdout: float = 0.2):
    """Repeated random splits.

    One split is not a result. A single seed reported cost at R^2 -0.261 when the
    median over twenty splits was -0.031 -- the worst draw presented as the
    finding, which is the same mistake that put `run_ppo_v4` in the ledger as an
    improvement. Everything here is averaged.
    """
    for seed in range(count):
        order = np.random.default_rng(seed).permutation(n)
        cut = int((1.0 - holdout) * n)
        yield order[:cut], order[cut:]


def _r2(design, targets, splits) -> tuple[float, float]:
    targets = np.asarray(targets, dtype=np.float64)
    scores = []
    for train, test in splits:
        weights, *_ = np.linalg.lstsq(design[train], targets[train], rcond=None)
        predicted = design[test] @ weights
        residual = ((targets[test] - predicted) ** 2).sum()
        total = ((targets[test] - targets[test].mean()) ** 2).sum()
        scores.append(1 - residual / max(total, 1e-9))
    scores = np.asarray(scores)
    return float(scores.mean()), float(scores.std())


def _accuracy(design, onehot, splits) -> tuple[float, float]:
    scores = []
    for train, test in splits:
        weights, *_ = np.linalg.lstsq(design[train], onehot[train], rcond=None)
        scores.append(
            (np.argmax(design[test] @ weights, axis=1)
             == np.argmax(onehot[test], axis=1)).mean()
        )
    scores = np.asarray(scores)
    return float(scores.mean()), float(scores.std())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path("data/card_embeddings/v1"))
    parser.add_argument("--splits", type=int, default=20,
                        help="Random 80/20 splits to average over.")
    parser.add_argument("--neighbours", action="store_true",
                        help="Also print nearest neighbours for a few known cards.")
    args = parser.parse_args()

    from sts2_env.embedding.table import load_table

    table = load_table(args.table)
    names = [None] * len(table.index)
    for name, row in table.index.items():
        names[row] = name

    rows, cost, damage, block, types = collect_targets(names)
    vectors = table.vectors[rows].astype(np.float64)
    design = np.hstack([vectors, np.ones((len(vectors), 1))])

    splits = list(_splits(len(rows), args.splits))

    print(f"table: {table.vectors.shape[0]} cards x {table.dims} dims "
          f"(model {table.manifest.get('model')})")
    print(f"probed on {len(rows)} cards, {args.splits} random 80/20 splits\n")

    labels = sorted(set(types))
    onehot = np.stack([np.array([t == label for t in types], dtype=np.float64)
                       for label in labels], axis=1)

    results = {
        "card_type": _accuracy(design, onehot, splits),
        "cost": _r2(design, cost, splits),
        "base_damage": _r2(design, damage, splits),
        "base_block": _r2(design, block, splits),
    }

    print(f"{'target':<14} {'ours':>16} {'HF text':>9} {'delta':>8}")
    print("-" * 50)
    for key, (mean, spread) in results.items():
        baseline = HF_BASELINE[key]
        note = "" if mean >= baseline - 0.05 else "   <-- worse"
        shown = f"{mean:.3f} +/- {spread:.3f}"
        print(f"{key:<14} {shown:>16} {baseline:>9.3f} {mean - baseline:>+8.3f}{note}")
    print(f"\n(card_type majority-class baseline {MAJORITY_BASELINE:.1%})")

    gate_ok = results["card_type"][0] >= HF_BASELINE["card_type"] - 0.05
    print("\nGATE:", "PASS" if gate_ok else "FAIL",
          "-- card type is the load-bearing one; the numeric targets are already "
          "carried explicitly by the scalar features beside the embedding.")

    if args.neighbours:
        unit = table.vectors / np.maximum(
            np.linalg.norm(table.vectors, axis=1, keepdims=True), 1e-9)
        similarity = unit @ unit.T
        np.fill_diagonal(similarity, -1)
        print("\nnearest neighbours (sanity check only):")
        for probe in ("BASH", "BODY_SLAM", "BARRICADE_CARD", "WHIRLWIND", "BURN"):
            if probe not in table.index:
                continue
            row = table.index[probe]
            nearest = np.argsort(-similarity[row])[:5]
            print(f"  {probe:<16} -> {[names[i] for i in nearest]}")

    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
