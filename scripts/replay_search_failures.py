"""Rebuild every state that made LiveSearch.decide raise, offline.

    python scripts/replay_search_failures.py

WHY THIS EXISTS
---------------
A raise inside `LiveSearch.decide` does not degrade the search. It REPLACES it:
the runner falls back to the trained combat model for the rest of that fight.
So a reconstruction bug does not cost a little accuracy, it costs the whole
searcher -- and the searcher is the agent. `MODELS.md` records the search
lifting boss win rate from 6.7% to ~20%, so a fight that falls back is a fight
played by the weaker of the two.

That makes it the leading explanation for the number this project is stuck on:

    win the boss    72% offline    28% live

Offline ALWAYS searches. It cannot fall back, because there is no model in the
loop and no bridge state to fail to rebuild.

Until now the only trace was `logger.exception` on the console, which scrolled
away with the session. `agent_runner._record_search_failure` now appends the
whole failing state, and `to_combat` is a pure function of it, so each line here
reproduces without the game running.

WHAT IT PRINTS
--------------
Failures grouped by where they raise, most common first, so the fix order is
the frequency order rather than whichever one was noticed.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--failures", default="output/live_search_failures.jsonl")
    ap.add_argument("--show", type=int, default=3,
                    help="Full tracebacks to print per distinct failure.")
    args = ap.parse_args()

    path = Path(args.failures)
    if not path.exists():
        print(f"no failure log at {path}")
        print("Run a live session with --live-search; failures are appended there.")
        return 1

    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        print("failure log is empty -- no LiveSearch failures recorded")
        return 0

    import sts2_env.cards  # noqa: F401
    from sts2_env.search.situation import CombatSituation

    print(f"{len(rows)} recorded failures from {path}\n")

    # Group by the recorded error, then re-raise each offline to confirm it
    # still reproduces against the current tree -- a failure that no longer
    # reproduces has been fixed and should stop being counted.
    groups: collections.Counter = collections.Counter()
    reproduced: collections.Counter = collections.Counter()
    examples: dict[str, dict] = {}

    for row in rows:
        key = f"{row.get('error_type')}: {row.get('error')}"
        groups[key] += 1
        examples.setdefault(key, row)
        state = row.get("state")
        if not isinstance(state, dict):
            continue
        try:
            situation = CombatSituation.from_bridge_state(state)
            situation.to_combat()
        except Exception:
            reproduced[key] += 1

    print(f"{'count':>6}  {'still':>6}  error")
    print(f"{'':>6}  {'repros':>6}")
    print("-" * 74)
    for key, n in groups.most_common():
        print(f"{n:>6}  {reproduced[key]:>6}  {key[:58]}")

    print()
    for key, _ in groups.most_common(args.show):
        row = examples[key]
        print("=" * 74)
        print(key)
        state = row.get("state") or {}
        enemies = state.get("enemies") or []
        print(f"  floor {state.get('floor')}  room {state.get('room_type')}  "
              f"enemies {[e.get('id') for e in enemies if isinstance(e, dict)]}")
        print(f"  potions {state.get('potion_slots') or state.get('potions')}")
        print(f"  relics {state.get('relics')}")
        print()
        if isinstance(state, dict):
            try:
                situation = CombatSituation.from_bridge_state(state)
                situation.to_combat()
                print("  (no longer reproduces -- fixed since it was recorded)")
            except Exception:
                print("  " + "  ".join(traceback.format_exc().splitlines(True)[-14:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
