"""What she brought to the fight she lost, in one page per run.

    .venv/bin/python scripts/run_report.py --tag wednesday
    .venv/bin/python scripts/run_report.py --tag wednesday --only lost --limit 5
    .venv/bin/python scripts/run_report.py --tag wednesday --summary

WHY THIS EXISTS
---------------
Every session so far recorded a lost boss fight as `deck_size: 18`,
`relic_count: 4`. Two integers. There is no argument you can have with two
integers, which is most of why the analysis kept drifting back to combat: combat
was the only pillar whose log could be questioned.

The three pillars pd named -- pathing, deck building, combat -- and what each
one needed before it could be looked at:

  combat        `combat_options` already logged every enumerated line with its
                score, and now the leaf position each one ends in. Fine.
  deck building the bridge has ALWAYS sent the full deck (`RlRunInfo.cs`
                attaches every card as `{id, upgraded}`) and the journal wrote
                only its length. Now recorded at every floor, every elite and
                boss, and on `run_end`.
  pathing       the path walked is now carried on `run_end` itself rather than
                left to be reassembled by scanning `floor` rows backwards --
                which was possible and which nobody did.

DELIBERATELY NOT LOGGED: the routes NOT taken, and a score for each. pd's call,
and it is the right one. "I reached the boss with 0 elites and lost" already
implies "should have taken more elites"; enumerating the two elite nodes she
walked past adds no fact to that sentence. Smart routing beats more elites, and
a counterfactual route score would be a number nobody could check. The path
taken, the deck it built and the outcome are the evidence.

READ IT WITH THE ROOM TYPE. A run that ends on floor 11 in a Monster room is a
different failure from one that ends on floor 17 against the boss, and the
summary splits them for that reason.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Act 1 ends here. Anything past it is a run that already cleared once.
ACT1_BOSS_FLOOR = 17


def _load(tag: str) -> dict[tuple[str, int], dict]:
    """Everything each run needs, keyed by (session, run index).

    NOT by run index alone. A session that crashes is restarted by
    `--restart-on-crash`, and the new session numbers its runs from 1 again --
    so `wednesday` holds two sessions and 62 runs under 57 distinct indexes.
    Keying on the index alone silently welds pairs of unrelated runs together:
    two paths end to end, elite counts summed, and the LAST run's death
    attributed to the first one's floor. Every number in this file was wrong by
    that amount before the key was fixed.
    """
    path = REPO / f"output/live_journal_{tag}.jsonl"
    if not path.exists():
        raise SystemExit(f"no journal at {path}")

    runs: dict[tuple[str, int], dict] = collections.defaultdict(dict)
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except Exception:
            continue
        index, event = row.get("run"), row.get("event")
        if index is None:
            continue
        run = runs[(str(row.get("session") or ""), index)]
        if event == "run_end":
            run["end"] = row
        elif event == "act_clear":
            run.setdefault("clears", []).append(row)
        elif event == "floor":
            # The fallback path for sessions journalled before `run_end`
            # carried its own copy. Kept because the sessions already on disk
            # are the ones worth reading.
            run.setdefault("path", []).append(row)
            if row.get("deck"):
                run["deck"] = row["deck"]
            if row.get("relics"):
                run["relics"] = row["relics"]
        elif event == "combat_start" and row.get("room_type") in ("Boss", "Elite"):
            run.setdefault("big_fights", []).append(row)
            if row.get("deck"):
                run["deck"] = row["deck"]
    return dict(runs)


def _outcome(run: dict) -> str:
    end = run.get("end") or {}
    if end.get("act_cleared") or run.get("clears"):
        return "WON"
    return "LOST"


def _path_line(run: dict) -> str:
    end = run.get("end") or {}
    path = end.get("path") or run.get("path") or []
    parts = []
    for step in path:
        floor = step.get("floor")
        room = str(step.get("room_type") or "?")
        parts.append(f"{floor}:{room}")
    return " -> ".join(parts) if parts else "(not recorded)"


def _deck_line(run: dict) -> str:
    end = run.get("end") or {}
    deck = end.get("deck") or run.get("deck")
    if not deck:
        return "(not recorded -- session predates deck logging)"
    counted = collections.Counter(deck)
    return ", ".join(f"{name} x{n}" if n > 1 else name
                     for name, n in counted.most_common())


def _report(key: tuple[str, int], run: dict) -> str:
    end = run.get("end") or {}
    result = _outcome(run)
    floor = end.get("floor")
    room = end.get("room_type") or "?"
    killer = end.get("death_enemy_id")
    relics = end.get("relics") or run.get("relics") or []
    deck = end.get("deck") or run.get("deck") or []
    elites = sum(1 for f in (end.get("path") or run.get("path") or [])
                 if f.get("room_type") == "Elite" and (f.get("floor") or 0) <= ACT1_BOSS_FLOOR)

    session, index = key
    article = "an" if str(room)[:1].upper() in "AEIOU" else "a"
    where = "the act 1 boss" if room == "Boss" else f"{article} {room} room"
    head = (f"Run {index} ({session}): {where} was {result} on floor {floor}"
            + (f", killed by {killer}" if killer and result == "LOST" else ""))

    lines = [head, "=" * len(head)]
    lines.append(f"  HP at the end   : {end.get('run_hp')}/{end.get('run_max_hp')}")
    lines.append(f"  elites fought   : {elites}   (act 1)")
    lines.append(f"  gold            : {end.get('gold')}")
    lines.append(f"  relics ({len(relics)})      : {', '.join(map(str, relics)) or '(none)'}")
    lines.append(f"  deck ({len(deck)} cards) : {_deck_line(run)}")
    lines.append(f"  path            : {_path_line(run)}")
    return "\n".join(lines)


def _summary(runs: dict[tuple[str, int], dict]) -> str:
    out = []
    won = [r for r in runs.values() if _outcome(r) == "WON"]
    lost = [r for r in runs.values() if _outcome(r) == "LOST"]
    out.append(f"{len(runs)} runs: {len(won)} cleared act 1, {len(lost)} did not")

    where = collections.Counter(
        (r.get("end") or {}).get("room_type") or "?" for r in lost)
    out.append("\nwhere the lost runs ended")
    for room, count in where.most_common():
        out.append(f"  {count:4d}  {room}")

    def elites_of(run):
        return sum(1 for f in ((run.get("end") or {}).get("path") or run.get("path") or [])
                   if f.get("room_type") == "Elite"
                   and (f.get("floor") or 0) <= ACT1_BOSS_FLOOR)

    for label, group in (("cleared", won), ("lost", lost)):
        if not group:
            continue
        decks = [len((r.get("end") or {}).get("deck") or r.get("deck") or []) for r in group]
        decks = [d for d in decks if d]
        elites = [elites_of(r) for r in group]
        relics = [len((r.get("end") or {}).get("relics") or r.get("relics") or [])
                  for r in group]
        out.append(f"\n{label} (n={len(group)})")
        out.append(f"  elites in act 1 : {statistics.mean(elites):.2f}")
        out.append(f"  relics held     : {statistics.mean(relics):.2f}")
        if decks:
            out.append(f"  deck size       : {statistics.mean(decks):.1f}")
        else:
            out.append("  deck size       : (not recorded -- pre-dates deck logging)")

    # The card-level question the old logs could not be asked at all.
    def card_counts(group):
        c = collections.Counter()
        for run in group:
            deck = (run.get("end") or {}).get("deck") or run.get("deck") or []
            c.update(set(deck))
        return c

    cw, cl = card_counts(won), card_counts(lost)
    if cw or cl:
        out.append("\ncards by how often they appear in a CLEARED deck vs a lost one")
        out.append("  (presence, not copies; only cards seen in 3+ runs)")
        rows = []
        for card in set(cw) | set(cl):
            seen = cw[card] + cl[card]
            if seen < 3:
                continue
            rate = 100 * cw[card] / seen
            rows.append((rate, seen, card))
        for rate, seen, card in sorted(rows, reverse=True)[:12]:
            out.append(f"   {rate:5.0f}% cleared  (n={seen:3d})  {card}")
        if not rows:
            out.append("   (no deck lists in this session)")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="session tag, e.g. wednesday")
    ap.add_argument("--only", choices=("won", "lost", "all"), default="all")
    ap.add_argument("--limit", type=int, default=0, help="0 for every run")
    ap.add_argument("--summary", action="store_true", help="only the summary")
    args = ap.parse_args()

    runs = _load(args.tag)
    if not runs:
        raise SystemExit("no runs in that journal")

    if not args.summary:
        shown = 0
        for key in sorted(runs):
            run = runs[key]
            if args.only != "all" and _outcome(run).lower() != args.only:
                continue
            print(_report(key, run))
            print()
            shown += 1
            if args.limit and shown >= args.limit:
                break

    print(_summary(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
