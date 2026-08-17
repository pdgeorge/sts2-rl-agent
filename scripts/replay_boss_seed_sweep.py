"""Replay captured boss fights across reshuffled draw orders and compare to live.

    python scripts/replay_boss_seed_sweep.py --capture output/bridge_boss_fights_sunday.jsonl

The bridge sends only draw_pile_count, never the draw pile order: the game's
shuffle stream is run-level (`RunState.Rng.Shuffle`, advanced by every earlier
fight), so the exact deal a live fight saw is unrecoverable from the capture.
`to_combat` therefore draws from the combat_seed, and a single replay scores
one arbitrary reshuffle -- which cannot separate good play from a lucky deal.

Sweeping combat_seed replays the SAME arrival (deck, relics, HP, enemies,
opening intents) against many reshuffles. The win rate across seeds is the
honest offline number for that position, and comparing it to the live outcome
answers the Track A question without pretending the draw order is known.
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
    ap.add_argument("--capture", default="output/bridge_boss_fights_sunday.jsonl")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--max-nodes", type=int, default=20000)
    ap.add_argument("--time-budget", type=float, default=3.0)
    ap.add_argument("--journal", default=None,
                    help="live journal to match outcomes from")
    args = ap.parse_args()

    import sts2_env.cards  # noqa: F401
    import sts2_env.powers  # noqa: F401
    from sts2_env.gym_env.action_space import apply_combat_action, get_action_mask
    from sts2_env.search.situation import CombatSituation
    from sts2_env.search.turn_search import SearchAgent

    per_seed: dict[int, dict] = {}
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
                per_seed.setdefault(state.get("encounter_seed"), state)

    live_outcomes: dict[tuple, bool] = {}
    if args.journal:
        starts = []
        ends = {}
        for line in open(args.journal, encoding="utf-8"):
            try:
                rec = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "combat_start" and rec.get("room_type") == "Boss":
                starts.append(rec)
            elif rec.get("event") == "combat_end" and rec.get("room_type") == "Boss":
                ends[(rec.get("run"), rec.get("floor"))] = rec
        for st in starts:
            enemies = tuple(sorted(
                (e.get("id"), e.get("max_hp")) for e in st.get("enemies") or []))
            end = ends.get((st.get("run"), st.get("floor")))
            live_outcomes[(enemies, st.get("deck_size"), st.get("hp"), st.get("floor"))] = bool(
                end and end.get("hp_after", 0) > 0)

    base_seeds = list(per_seed)
    print(f"{len(base_seeds)} captured fights x {args.seeds} reshuffles\n")

    summary = []
    for i, seed in enumerate(base_seeds, 1):
        state = dict(per_seed[seed])
        enemies = tuple(sorted(
            (e.get("id"), e.get("max_hp")) for e in state.get("enemies") or []))
        live_key = (enemies, len(state.get("deck") or []),
                    state.get("run_hp"), state.get("floor"))
        live = live_outcomes.get(live_key)

        original_combat_seed = int(state.get("combat_seed")
                                   or state.get("encounter_seed") or 0)
        wins = 0
        hp_left = []
        for k in range(args.seeds):
            variant = dict(state)
            variant["combat_seed"] = original_combat_seed + 1_000_003 * k
            situation = CombatSituation.from_bridge_state(variant)
            combat = situation.to_combat()
            agent = SearchAgent(max_nodes=args.max_nodes,
                                time_budget=args.time_budget, lookahead_turns=2)
            rejected = 0
            while not combat.is_over and combat.turn_count < 60:
                mask = get_action_mask(combat)
                action = agent.act(combat)
                if action >= len(mask) or not mask[action]:
                    break
                if not apply_combat_action(combat, action):
                    rejected += 1
                    if rejected > 4:
                        break
                    continue
            won = combat.is_over and not combat.player.is_dead
            wins += int(won)
            hp_left.append(combat.player.current_hp)
        label = state.get("encounter") or "?"
        arrival = f"{state.get('run_hp')}/{state.get('run_max_hp')}"
        live_txt = {True: "WON", False: "lost", None: "?"}[live]
        print(f"{i:>2}/{len(base_seeds)} {label:<26} arrival {arrival:<7} "
              f"offline {_pct(wins, args.seeds):<12} live {live_txt}")
        summary.append((label, arrival, wins, args.seeds, live))

    total_wins = sum(w for _, _, w, _, _ in summary)
    total_n = sum(n for _, _, _, n, _ in summary)
    print(f"\nall fights pooled: {_pct(total_wins, total_n)}")
    live_won = sum(1 for _, _, _, _, l in summary if l)
    live_known = sum(1 for _, _, _, _, l in summary if l is not None)
    print(f"live on the same fights: {_pct(live_won, live_known)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
