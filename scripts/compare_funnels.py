"""Compare funnel arms that were run on the same seeds.

    python scripts/compare_funnels.py --tags greedy,planned \
        --rows-prefix output/funnel_routing

Reports reach, boss win and clear per arm with 95% intervals, then the PAIRED
comparison against the first tag -- same seed, both arms, so seed variance
cancels. Act 1 variance between seeds dwarfs the effects being measured, which
is why the paired column is the one to believe when it disagrees with the
unpaired ones.

McNemar on the discordant pairs, because "40% against 39%" over 150 runs is
noise and the only honest way to say so is a p-value.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from math import comb
from pathlib import Path


def _load(prefix: str, tag: str) -> dict[int, dict]:
    path = Path(f"{prefix}.{tag}.rows.jsonl")
    rows: dict[int, dict] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "seed" in r:
                rows[r["seed"]] = r
    return rows


def _ci(k: int, n: int) -> str:
    if not n:
        return "     n/a"
    p = k / n
    return f"{100 * p:5.1f}% +/-{100 * 1.96 * math.sqrt(p * (1 - p) / n):4.1f}"


def _mcnemar(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(max(a, b), n + 1)) / 2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--rows-prefix", required=True)
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    data = {t: _load(args.rows_prefix, t) for t in tags}

    print()
    print("=" * 76)
    print(f"{'arm':<20}{'n':>5}{'reach boss':>16}{'win|reach':>16}{'clear':>16}")
    print("-" * 76)
    for t in tags:
        rows = list(data[t].values())
        n = len(rows)
        if not n:
            print(f"{t:<20}{0:>5}   (no rows)")
            continue
        reach = sum(1 for r in rows if r.get("boss"))
        won = sum(1 for r in rows if r.get("boss") and r.get("act", 1) >= 2)
        print(f"{t:<20}{n:>5}{_ci(reach, n):>16}{_ci(won, reach):>16}"
              f"{_ci(won, n):>16}")

    base = tags[0]
    print()
    print(f"paired against '{base}' (same seed, both arms):")
    print(f"  {'arm':<20}{'pairs':>7}{'clear +':>9}{'clear -':>9}{'net':>7}"
          f"{'McNemar p':>12}")
    print("  " + "-" * 64)
    for t in tags[1:]:
        pairs = [(data[base][s], data[t][s]) for s in data[base] if s in data[t]]
        if not pairs:
            continue
        plus = sum(1 for b, x in pairs
                   if x.get("act", 1) >= 2 and b.get("act", 1) < 2)
        minus = sum(1 for b, x in pairs
                    if b.get("act", 1) >= 2 and x.get("act", 1) < 2)
        print(f"  {t:<20}{len(pairs):>7}{plus:>9}{minus:>9}{plus - minus:>+7}"
              f"{_mcnemar(plus, minus):>12.2f}")
    print()
    print("  PREDICTION on file: offline reach 54% -> 65%. A miss is a miss.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
