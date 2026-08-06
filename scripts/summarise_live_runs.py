"""Turn a live journal into the answers you would otherwise read by hand.

    python scripts/summarise_live_runs.py output/live_journal.jsonl

A few dozen runs is thousands of events. This reduces them to the questions that
decide what to work on next:

  where runs end          which room, on which floor, at what HP
  which fights cost most  damage taken per fight, by room type
  what it takes           the cards actually picked, and how often it skipped
  what it plays           the cards played most, and the ones never played at all

The last one is worth having for its own sake. A card that is picked and then
never played is a deckbuilding error the win rate cannot show you.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load(path: str | Path) -> list[dict[str, Any]]:
    events = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _by_run(events: list[dict]) -> dict[tuple, list[dict]]:
    """Keyed by (session, run), never by run alone.

    The journal is appended across sessions and the run counter restarts at 1
    each time, so one file can hold runs 1,2,3,1,2,3. Grouping on the number
    alone merges unrelated runs and reports fewer, longer ones.
    """
    runs: dict[tuple, list[dict]] = defaultdict(list)
    for event in events:
        runs[(event.get("session", ""), event.get("run", 0))].append(event)
    return runs


def _pct(count: int, total: int) -> str:
    return f"{count / total:5.1%}" if total else "    -"


def report(events: list[dict]) -> str:
    runs = _by_run(events)
    ends = [e for e in events if e["event"] == "run_end"]
    combats = [e for e in events if e["event"] == "combat_end"]
    plays = [e for e in events if e["event"] == "card_played"]
    choices = [e for e in events if e["event"] == "choice"]

    lines: list[str] = ["", "=" * 68, f"{len(runs)} runs, {len(events)} events", ""]

    # -- where runs end ---------------------------------------------------
    if ends:
        floors = [e.get("floor") or 0 for e in ends]
        lines += [
            "WHERE RUNS END",
            f"  floors    mean {statistics.mean(floors):.1f}   "
            f"median {statistics.median(floors):.0f}   "
            f"min {min(floors)}   max {max(floors)}",
        ]
        rooms = Counter(str(e.get("room_type", "?")) for e in ends)
        for room, count in rooms.most_common():
            lines.append(f"    {room:<12} {count:>3}  {_pct(count, len(ends))}")
        # Reaching act 2, never a floor. The live act 1 boss room IS floor 17, so
        # "floor >= 17" counted every boss death as a clear and reported 20% for
        # a session that never beat the boss once. See live_eval._cleared_act_1.
        reached_boss = sum(1 for e in ends
                           if str(e.get("room_type", "")).upper() == "BOSS"
                           or (e.get("floor") or 0) >= 17)
        cleared = sum(1 for e in ends
                      if (isinstance(e.get("act"), int) and e["act"] >= 2)
                      or (not isinstance(e.get("act"), int) and (e.get("floor") or 0) > 17))
        lines.append(f"  reached the act 1 boss:   {reached_boss}/{len(ends)}  "
                     f"{_pct(reached_boss, len(ends))}")
        lines.append(f"  CLEARED act 1 (act >= 2): {cleared}/{len(ends)}  "
                     f"{_pct(cleared, len(ends))}")
        lines.append("")

    # -- which fights cost most -------------------------------------------
    if combats:
        lines.append("WHAT FIGHTS COST")
        by_room: dict[str, list[dict]] = defaultdict(list)
        for combat in combats:
            by_room[str(combat.get("room_type", "?"))].append(combat)
        for room, group in sorted(by_room.items(), key=lambda kv: -len(kv[1])):
            damage = [c["damage_taken"] for c in group
                      if isinstance(c.get("damage_taken"), (int, float))]
            turns = [c["turns"] for c in group if isinstance(c.get("turns"), int)]
            lines.append(
                f"    {room:<12} {len(group):>3} fights   "
                f"damage taken mean {statistics.mean(damage):5.1f}" if damage else
                f"    {room:<12} {len(group):>3} fights"
            )
            if turns:
                lines[-1] += f"   turns {statistics.mean(turns):4.1f}"
        worst = sorted(
            (c for c in combats if isinstance(c.get("damage_taken"), (int, float))),
            key=lambda c: -c["damage_taken"],
        )[:5]
        if worst:
            lines.append("  worst individual fights:")
            for combat in worst:
                enemies = ", ".join(
                    str(e.get("id")) for e in (combat.get("enemies") or [])
                ) or "?"
                lines.append(
                    f"    floor {str(combat.get('combat_floor', '?')):>3}  "
                    f"{combat['damage_taken']:>3} damage   "
                    f"{str(combat.get('room_type','?')):<8} {enemies}"
                )
        lines.append("")

    # -- what it takes ----------------------------------------------------
    rewards = [c for c in choices if c.get("screen") in
               ("card_reward", "card_bundle", "reward_screen")]
    if rewards:
        taken = Counter(str(c["chosen"]) for c in rewards if not c.get("skipped") and c.get("chosen"))
        skipped = sum(1 for c in rewards if c.get("skipped"))
        lines += [
            "WHAT IT PICKS",
            f"  {len(rewards)} card rewards, {skipped} skipped  ({_pct(skipped, len(rewards))})",
        ]
        for card, count in taken.most_common(12):
            lines.append(f"    {card:<28} {count:>3}")
        lines.append("")

    # -- what it plays ----------------------------------------------------
    if plays:
        played = Counter(str(p.get("card")) for p in plays)
        lines += ["WHAT IT PLAYS", f"  {len(plays)} cards played"]
        for card, count in played.most_common(12):
            lines.append(f"    {card:<28} {count:>4}")

        # Picked but never played: a deckbuilding error a win rate cannot show.
        picked = {str(c["chosen"]) for c in rewards
                  if not c.get("skipped") and c.get("chosen")}
        never = sorted(picked - set(played))
        if never:
            lines += ["", "  picked but never played once:"]
            for card in never[:12]:
                lines.append(f"    {card}")
        lines.append("")

    # -- other choices ----------------------------------------------------
    other = [c for c in choices if c.get("screen") in ("rest_site", "map_select", "event", "shop")]
    if other:
        lines.append("OTHER CHOICES")
        by_screen: dict[str, Counter] = defaultdict(Counter)
        for choice in other:
            by_screen[str(choice.get("screen"))][str(choice.get("chosen"))] += 1
        for screen, counter in by_screen.items():
            lines.append(f"  {screen}:")
            for what, count in counter.most_common(6):
                lines.append(f"    {what:<28} {count:>3}")
        lines.append("")

    lines += ["=" * 68, ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise a live run journal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("journal", nargs="?", default="output/live_journal.jsonl")
    args = parser.parse_args()

    path = Path(args.journal)
    if not path.exists():
        print(f"No journal at {path}. Run live_eval with --journal first.")
        raise SystemExit(1)

    print(report(load(path)))


if __name__ == "__main__":
    main()
