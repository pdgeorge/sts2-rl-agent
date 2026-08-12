"""Play the boss fights the live agent played, offline, and compare.

    python scripts/replay_live_boss_fights.py --capture output/bridge_boss_fights.jsonl

THE QUESTION
------------
Offline wins about 70% of act 1 boss fights. Live wins about 36%. Everything
else in the funnel agrees -- reach rate is 63% offline against 58% live -- so
the whole discrepancy sits in one fight, and until it is explained the offline
number cannot be used to judge anything that touches the boss.

Two explanations, and they call for opposite work:

  EXECUTION   The same position is played better offline than live. Then the
              fault is in the live path -- reconstruction, the bridge, planning
              against a state that is not quite the real one -- and offline is
              telling the truth about the fight.
  ARRIVAL     Offline runs turn up at the boss in better shape: more HP, better
              deck, more relics. Then the boss model is fine and offline's
              optimism is upstream, in the routing and the economy.

WHY THIS AND NOT A SHARED SEED
------------------------------
Matching maps between the two sides is not available: the game seeds from a
12-character base-34 string, the simulator from an int, and whether its map
generator reproduces the game's for a given seed has never been tested. This
needs no map. A captured combat_action state IS the position -- deck, HP,
relics, potions, the enemies and their HP and intents -- and `to_combat`
rebuilds it exactly. Same fight, both agents.

WHAT IT CANNOT TELL YOU
-----------------------
The live outcome here is "did the run reach floor 18", taken from the journal.
A run can also die AFTER the boss on the same floor, and a capture quota that
fills mid-session means late fights are missing. Treat the live rate as
approximate and the offline rate as exact; the comparison is still the right
shape, because the offline side replays precisely the positions the live side
faced.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _pct(k: int, n: int) -> str:
    if not n:
        return "n/a"
    p = k / n
    return f"{100 * p:.0f}% +/- {100 * math.sqrt(p * (1 - p) / n):.0f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", default="output/bridge_boss_fights.jsonl")
    ap.add_argument("--max-nodes", type=int, default=20000,
                    help="Match the live SearchAgent default, not the funnel's 2000.")
    ap.add_argument("--time-budget", type=float, default=3.0,
                    help="Match the live wall clock.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import sts2_env.cards  # noqa: F401
    import sts2_env.powers  # noqa: F401
    from sts2_env.gym_env.action_space import get_action_mask
    from sts2_env.search.situation import CombatSituation
    from sts2_env.search.turn_search import SearchAgent

    rows = []
    with open(args.capture, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                state = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (isinstance(state, dict) and state.get("type") == "combat_action"
                    and state.get("room_type") == "Boss"):
                rows.append(state)

    # One state per fight. A fight sends a state per decision, so keeping every
    # one would weight long fights heavily and score the same position repeatedly.
    # Keyed on the enemies and their max HP plus the deck size, which is stable
    # within a fight and differs between runs.
    seen: dict[tuple, dict] = {}
    for state in rows:
        enemies = tuple(sorted(
            (e.get("id"), e.get("max_hp")) for e in (state.get("enemies") or [])))
        key = (enemies, len(state.get("deck") or []), state.get("run_max_hp"))
        seen.setdefault(key, state)
    fights = list(seen.values())
    if args.limit:
        fights = fights[:args.limit]

    print(f"{len(rows)} captured boss states -> {len(fights)} distinct fights\n")

    wins = 0
    per_boss: collections.Counter = collections.Counter()
    per_boss_win: collections.Counter = collections.Counter()

    for i, state in enumerate(fights, 1):
        situation = CombatSituation.from_bridge_state(state)
        combat = situation.to_combat()
        label = ",".join(sorted({e.get("id", "?") for e in (state.get("enemies") or [])}))
        agent = SearchAgent(max_nodes=args.max_nodes, time_budget=args.time_budget,
                            lookahead_turns=2)
        for _ in range(400):
            if combat.is_over:
                break
            mask = get_action_mask(combat)
            action = agent.act(combat)
            if action >= len(mask) or not mask[action]:
                break
            from sts2_env.gym_env.action_space import apply_combat_action
            if not apply_combat_action(combat, action):
                break
        won = combat.is_over and not combat.player.is_dead
        wins += int(won)
        per_boss[label] += 1
        per_boss_win[label] += int(won)
        print(f"  {i:>3}/{len(fights)}  {label:<26} "
              f"{'WON ' if won else 'lost'}  hp {combat.player.current_hp}")

    print()
    print(f"OFFLINE, replaying the live agent's own boss fights: "
          f"{wins}/{len(fights)} = {_pct(wins, len(fights))}")
    print()
    for boss, n in per_boss.most_common():
        print(f"  {boss:<28}{per_boss_win[boss]:>3}/{n:<4} {_pct(per_boss_win[boss], n)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
