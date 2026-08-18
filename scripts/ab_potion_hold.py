"""Does holding the fight-ending potions for elites and bosses pay?

    .venv/bin/python scripts/ab_potion_hold.py --runs 400 --workers 12

WHY OFFLINE, AFTER A LIVE SESSION ALREADY RAN IT
------------------------------------------------
The live session (`potion_fix_2`, n=91) put clear at 25.3% against a pooled
37.5%, p=0.041 -- and it cannot be believed, in either direction. n=91 resolves
+/-8.9 against a 12-point effect, and this project has already measured
**31.0% vs 44.0% on identical act 1 code** (`postfix` vs `boss_telemetry`,
p=0.058). Session-to-session variance is the same size as the effect. No single
live session can settle a question this fine.

Paired seeds can. Both arms walk the SAME seed, so the map, the card offers and
the shuffles are shared and the difference is the arm. 400 pairs resolves about
+/-1.2 points instead of +/-9, in hours.

WHY OFFLINE IS A VALID INSTRUMENT *HERE*
----------------------------------------
`PHASE_TWO.md` Track A restricts offline on BOSS questions -- the live/offline
gap on the boss fight is 45 points and unexplained. This is not a boss question.
The hold acts in the corridor, and its claimed payoff runs through chip damage
-> HP at the boss -> reach, which is the half offline agrees with live on
(reach 64% offline against 63% live). The arm delta on shared seeds is what is
read; the absolute rates are not.

NOTHING PATCHES A GLOBAL
------------------------
Both arms are `PolicyConfig` files -- `v001` (no hold, the shipped default) and
`v002_hold_potions`. The worker calls `set_active_policy` before its first
decision, in its own process, so the arms cannot see each other's value.
`PHASE_TWO.md` section 3.1 records what the alternative costs: a sweep once ran
400 runs with its baseline arm doing the exact opposite of its name.

KNOWN LIMIT OF THE MECHANISM COLUMNS
------------------------------------
Potions are counted by watching the belt shrink, which cannot say WHICH potion
left it. So the where-they-are-drunk columns cover every potion in the game,
while the arm only holds five of them -- the live session measured 85% -> 12%
on those five, and the same signal diluted across the whole pool reads 68% ->
65% here. The outcome comparison is unaffected (paired seeds, one difference
between arms), but this script cannot confirm the behavioural gate. Fixing it
means recording the potion id as it leaves the belt.

THE GATE, NOT ONLY THE RATE
---------------------------
The live session's lesson was that the outcome moved while the mechanism did
not do what it was built for: trash use fell 85% -> 12% and potions held at the
boss did not move at all (0.99 -> 0.97) because they were spent one room earlier
on elites. So this reports where potions are drunk and how many survive to the
boss, per arm. A rate change without a mechanism change is not an answer.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ARMS = ("v001", "v002_hold_potions")


def _walk(job) -> dict:
    seed, arm, max_nodes = job

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    import numpy as np
    import sts2_env.cards  # noqa: F401
    from sts2_env.core.enums import RoomType
    from sts2_env.gym_env.run_env import STS2RunEnv
    from sts2_env.policy_config import PolicyConfig, apply_active_policy, set_active_policy
    from sts2_env.run.run_manager import RunManager
    from sts2_env.search.turn_search import SearchAgent
    from ab_archetype_picking import _search_combat_action
    from live_policy import noncombat_action

    # In the WORKER, before the first decision. A parent-side call would be
    # inherited by every worker and collapse the experiment to one arm.
    policy = PolicyConfig.load(arm)
    set_active_policy(policy)
    apply_active_policy(policy)

    agent = SearchAgent(time_budget=60.0, lookahead_turns=2, max_nodes=max_nodes,
                        weights=policy.eval_weights)
    rng = np.random.default_rng(seed)
    env = STS2RunEnv()
    env.reset(seed=seed)

    potions_at_boss = None
    reached_boss = False
    room_of = {RoomType.MONSTER: "monster", RoomType.ELITE: "elite", RoomType.BOSS: "boss"}
    drunk: Counter = Counter()
    before = None

    for _ in range(3000):
        mgr = env._mgr
        if mgr is None:
            break
        mask = env.action_masks()
        valid = np.where(mask == 1)[0]
        if not len(valid):
            break

        player = getattr(getattr(mgr, "run_state", None), "player", None)
        held = [p for p in (getattr(player, "potions", None) or []) if p]
        room = room_of.get(getattr(mgr, "_current_room_type", None))

        if room == "boss" and not reached_boss:
            reached_boss = True
            potions_at_boss = len(held)

        # A potion left the belt this step: attribute it to the room it was
        # drunk in. Counted from the belt rather than from an event, because
        # the offline path has no journal to read.
        if before is not None and room is not None and len(held) < before:
            drunk[room] += before - len(held)
        before = len(held)

        if mgr.phase == RunManager.PHASE_COMBAT:
            action = _search_combat_action(agent, mgr, mask)
        else:
            action = noncombat_action(mgr, mgr.phase, mask, rng)
        if action is None:
            action = int(rng.choice(valid))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    mgr = env._mgr
    run_state = getattr(mgr, "run_state", None)
    return {
        "seed": seed, "arm": arm,
        "floor": int(getattr(run_state, "total_floor", 0) or 0),
        "act": int(getattr(run_state, "current_act_index", 0) or 0) + 1,
        "reached_boss": reached_boss,
        "potions_at_boss": potions_at_boss,
        "drunk_monster": drunk.get("monster", 0),
        "drunk_elite": drunk.get("elite", 0),
        "drunk_boss": drunk.get("boss", 0),
    }


def _wilson(k: int, n: int) -> float:
    if not n:
        return 0.0
    p = k / n
    return 100 * 1.96 * math.sqrt(p * (1 - p) / n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=400, help="paired seeds per arm")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-nodes", type=int, default=20000)
    ap.add_argument("--out", default="output/ab_potion_hold.rows.jsonl")
    args = ap.parse_args()

    jobs = [(seed, arm, args.max_nodes)
            for seed in range(args.runs) for arm in ARMS]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("", encoding="utf-8")

    print(f"{len(ARMS)} arms x {args.runs} paired seeds = {len(jobs)} runs, "
          f"{args.workers} workers\n")

    rows: list[dict] = []
    with out.open("a", encoding="utf-8") as fh, mp.Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_walk, jobs, chunksize=1), 1):
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in ARMS}
    print("\n" + "=" * 74)
    print(f"{'arm':<20}{'clear':>16}{'reach':>16}{'potions at boss':>18}")
    print("=" * 74)
    for arm in ARMS:
        rs = by_arm[arm]
        cleared = sum(1 for r in rs if r["act"] >= 2)
        reached = sum(1 for r in rs if r["reached_boss"])
        pab = [r["potions_at_boss"] for r in rs if r["potions_at_boss"] is not None]
        print(f"  {arm:<18}{100 * cleared / len(rs):>9.1f}% +/-{_wilson(cleared, len(rs)):<4.1f}"
              f"{100 * reached / len(rs):>9.1f}% +/-{_wilson(reached, len(rs)):<4.1f}"
              f"{statistics.mean(pab) if pab else 0:>17.2f}")

    print("\nWHERE POTIONS ARE DRUNK -- the mechanism, which is what the live run failed")
    print(f"  {'arm':<20}{'monster':>10}{'elite':>10}{'boss':>10}{'% on trash':>13}")
    for arm in ARMS:
        rs = by_arm[arm]
        m = sum(r["drunk_monster"] for r in rs)
        e = sum(r["drunk_elite"] for r in rs)
        b = sum(r["drunk_boss"] for r in rs)
        tot = m + e + b
        print(f"  {arm:<20}{m:>10}{e:>10}{b:>10}"
              f"{100 * m / tot if tot else 0:>12.0f}%")

    # Paired: same seed, both arms, so seed variance cancels. This is the number
    # to believe when the unpaired columns disagree with it.
    base = {r["seed"]: r for r in by_arm[ARMS[0]]}
    test = {r["seed"]: r for r in by_arm[ARMS[1]]}
    shared = sorted(set(base) & set(test))
    gained = sum(1 for s in shared if test[s]["act"] >= 2 and base[s]["act"] < 2)
    lost = sum(1 for s in shared if base[s]["act"] >= 2 and test[s]["act"] < 2)
    n_disc = gained + lost
    z = (gained - lost) / math.sqrt(n_disc) if n_disc else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))) if n_disc else 1.0
    net = 100 * (gained - lost) / len(shared) if shared else 0.0
    print(f"\nPAIRED over {len(shared)} shared seeds (holding vs not):")
    print(f"  the hold gains {gained}, loses {lost}, net {net:+.1f} points "
          f"(McNemar z={z:.2f}, p={p:.3f})")
    print("\nRead the paired line. The arm rates carry the full between-seed "
          "variance;\nthe paired difference does not, which is the entire reason "
          "for walking shared seeds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
