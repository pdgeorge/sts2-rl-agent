"""Which deck archetype beats which act 1 boss.

    .venv/bin/python scripts/archetype_by_boss.py --tags cardprior

pd's question, and the first half of a bigger one: right now the agent FALLS
INTO an archetype -- `DeckDirection` accumulates votes from whatever the card
rewards happened to offer -- rather than choosing one. Choosing would mean
knowing the boss, and act 1 has only two variants of three bosses each
(Vantom / Ceremonial Beast / Kin, or Waterfall / Soul Fysh / Lagavulin), so it
is a three-way decision rather than an open one.

Before any of that is worth building, the effect has to exist. This measures it
from data already on disk: the archetype is recomputed from the DECK the
journal now records at the boss, so it needs no live change and works
retroactively on any session that carries deck lists.

READ THE CELL COUNTS FIRST. Six bosses times four archetypes is 24 cells, and
one session of 100 runs puts about three runs in each. Nothing here resolves at
that n -- it is an instrument that has to accumulate across sessions, and a cell
with n<10 is an anecdote with a percentage sign on it.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _archetype(deck: list[str]) -> str:
    try:
        from sts2_env.search.archetypes import DeckDirection
    except Exception:
        return "(unavailable)"
    direction = DeckDirection()
    direction.observe_deck([str(c).replace("+", "") for c in deck])
    return direction.committed or "(uncommitted)"


def _load(tags: list[str]) -> list[dict]:
    rows = []
    for tag in tags:
        path = REPO / f"output/live_journal_{tag}.jsonl"
        if not path.exists():
            print(f"  (no journal for {tag})")
            continue
        runs: dict = collections.defaultdict(dict)
        for line in path.open(encoding="utf-8", errors="replace"):
            try:
                record = json.loads(line)
            except Exception:
                continue
            if record.get("run") is None:
                continue
            key = (tag, record.get("session"), record["run"])
            run = runs[key]
            event = record.get("event")
            if (event == "combat_start" and record.get("room_type") == "Boss"
                    and (record.get("floor") or 99) <= 18):
                run["boss"] = ",".join(sorted({e.get("id") for e in
                                               (record.get("enemies") or [])}))[:26]
                run["deck"] = record.get("deck")
                run["hp"] = record.get("hp")
            elif event == "act_clear" and record.get("act_from") == 1:
                run["cleared"] = True
        for key, run in runs.items():
            if run.get("boss"):
                rows.append({
                    "boss": run["boss"],
                    "deck": run.get("deck"),
                    "cleared": bool(run.get("cleared")),
                })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--min-cell", type=int, default=5)
    args = ap.parse_args()

    rows = _load(args.tags)
    if not rows:
        raise SystemExit("no act 1 boss arrivals found")

    with_deck = [r for r in rows if r["deck"]]
    print(f"{len(rows)} act 1 boss arrivals, {len(with_deck)} carry a deck list")
    if not with_deck:
        print("\nNo decks recorded. Deck logging landed 2026-08-20; only sessions "
              "from after that\ncan answer this. Re-run against a newer tag.")
        return 0

    for row in with_deck:
        row["arch"] = _archetype(row["deck"])

    bosses = sorted({r["boss"] for r in with_deck})
    arches = sorted({r["arch"] for r in with_deck})

    print(f"\n{'boss':<26}" + "".join(f"{a[:11]:>13}" for a in arches))
    for boss in bosses:
        cells = []
        for arch in arches:
            g = [r for r in with_deck if r["boss"] == boss and r["arch"] == arch]
            if not g:
                cells.append(f"{'-':>13}")
            else:
                won = sum(1 for r in g if r["cleared"])
                cells.append(f"{100*won/len(g):>8.0f}% n{len(g):<3}")
        print(f"  {boss:<24}" + "".join(cells))

    print("\nARCHETYPE OVERALL (all bosses pooled)")
    for arch in arches:
        g = [r for r in with_deck if r["arch"] == arch]
        won = sum(1 for r in g if r["cleared"])
        half = 100 * 1.96 * math.sqrt((won/len(g))*(1-won/len(g))/len(g)) if len(g) else 0
        print(f"  {arch:<24}{100*won/len(g):>7.1f}% +/-{half:<5.1f} n={len(g)}")

    thin = sum(1 for boss in bosses for arch in arches
               if 0 < len([r for r in with_deck if r["boss"] == boss and r["arch"] == arch])
               < args.min_cell)
    print(f"\n  cells with fewer than {args.min_cell} runs: {thin} of "
          f"{len(bosses)*len(arches)}. Those are anecdotes; keep accumulating.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
