"""What she was holding when she died, and what she never drank.

    python scripts/analyse_potions.py

METHOD, AND WHY IT IS FUSSY
---------------------------
A first pass at this returned numbers that could not be true: runs that lost
the boss averaged 0.75 potions held and 1.46 used. Two causes, both mine.

1. Runs were keyed by (session, run). That key COLLIDES -- the run counter
   restarts, so several real runs share one bucket and their potion uses pile
   up together. Two buckets showed 122 and 60 uses at floor 17, which is not
   possible with three slots, and those two alone moved every mean.
2. 99 journal records carry no `session` at all, so they attached to the wrong
   bucket entirely.

Runs are delimited by `run_start` here instead, which is what actually marks
one. Uses are counted only between the boss's own combat_start and the end of
that run.

DO NOT count boss fights by combat_start/combat_end pairs. There are ~300 such
pairs against 151 runs that reach the boss, because a known bridge bug splits
one fight into several segments. Counting segments says the boss is won 65% of
the time; counting runs says 29%, and 29% is the true one.

FAIRY_IN_A_BOTTLE is reported separately: it fires automatically on death, so
"held and never used" is its correct behaviour, not hoarding.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
import sys

AUTOMATIC = {"FAIRY_IN_A_BOTTLE"}


def load_runs(pattern: str) -> list[dict]:
    runs: list[dict] = []
    cur: dict | None = None
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event, floor = rec.get("event"), rec.get("floor")
                if event == "run_start":
                    if cur:
                        runs.append(cur)
                    cur = {"max_floor": 0, "held": None, "used": [], "in_boss": False,
                           "hp": None, "max_hp": None}
                if cur is None:
                    continue
                if isinstance(floor, int):
                    cur["max_floor"] = max(cur["max_floor"], floor)
                if (event == "combat_start" and rec.get("room_type") == "Boss"
                        and floor == 17):
                    cur["in_boss"] = True
                    if cur["held"] is None:
                        cur["held"] = [p for p in (rec.get("potions") or []) if p]
                        cur["hp"], cur["max_hp"] = rec.get("hp"), rec.get("max_hp")
                if event == "potion_used" and floor == 17 and cur["in_boss"]:
                    cur["used"].append(rec.get("potion"))
    if cur:
        runs.append(cur)
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journals", default="output/live_journal*.jsonl")
    args = ap.parse_args()

    runs = load_runs(args.journals)
    reached = [r for r in runs if r["held"] is not None]
    if not reached:
        print("no act 1 boss fights found")
        return 1
    won = [r for r in reached if r["max_floor"] >= 18]
    lost = [r for r in reached if r["max_floor"] < 18]

    print(f"runs {len(runs)}   reached act 1 boss {len(reached)}   "
          f"won {len(won)} = {100 * len(won) / len(reached):.0f}%")
    print()
    print(f"{'':<7}{'n':>5}{'held':>8}{'used':>8}{'entered holding':>18}"
          f"{'used nothing':>15}")
    for label, group in (("WON", won), ("LOST", lost)):
        holding = [r for r in group if r["held"]]
        idle = [r for r in holding if not r["used"]]
        print(f"{label:<7}{len(group):>5}"
              f"{statistics.mean(len(r['held']) for r in group):>8.2f}"
              f"{statistics.mean(len(r['used']) for r in group):>8.2f}"
              f"{len(holding):>18}"
              f"{len(idle):>10} = {100 * len(idle) / max(len(holding), 1):.0f}%")

    idle_potions: collections.Counter = collections.Counter()
    for r in lost:
        if r["held"] and not r["used"]:
            idle_potions.update(r["held"])
    print()
    print("held UNUSED through a LOST act 1 boss fight:")
    for pid, n in idle_potions.most_common():
        note = "   (automatic -- fires on death, not hoarding)" if pid in AUTOMATIC else ""
        print(f"  {pid:<28}{n:>4}{note}")

    hoarded = sum(n for p, n in idle_potions.items() if p not in AUTOMATIC)
    print()
    print(f"non-automatic potions that died in the belt: {hoarded}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
