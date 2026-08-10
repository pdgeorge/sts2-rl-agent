"""Refit the map HP gate, whose basis went stale under us.

    python scripts/sweep_hp_gate.py --runs 120

`ROOM_MIN_HP_FRACTION` decides which rooms are worth entering: take an elite
only above 80% health, a monster above 40%. Those numbers were FITTED, not
guessed -- over 116 live elite fights, 0.80 was where the death curve bent (21
taken / 10% died, against 85 / 21% for the old flat 0.50 threshold).

Then the simulator changed underneath them. Nine monster HP constants and
thirteen attack damages were corrected the same week, including Phantasmal
Gardener -- the single deadliest act 1 elite for this agent -- and Bygone
Effigy, both of which fed that original fit. A threshold fitted to a world that
has since been corrected is a guess wearing a measurement's clothes.

WHY THIS AND NOT MORE COMBAT WORK
---------------------------------
Both combat-side levers came back empty. The eval weights reached nothing above
1.1 se, and archetype-aware deckbuilding scored +0.0% +/- 3.3% with 104 of 120
seeds running IDENTICALLY. Routing is the remaining area with a measured basis,
and it is the one that has never been re-examined since the simulator became
trustworthy.

The arms bracket the current value in both directions, because "we fitted 0.80
and it is still 0.80" is a real possible answer and the sweep should be able to
return it.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

#: (name, {room: fraction}) merged over the shipped ROOM_MIN_HP_FRACTION.
ARMS: list[tuple[str, dict]] = [
    ("baseline .80/.40", {}),
    ("elite .70", {"elite": 0.70}),
    ("elite .90", {"elite": 0.90}),
    ("monster .55", {"monster": 0.55, "unknown": 0.55, "event": 0.55}),
    ("boss .70", {"boss": 0.70}),
]


def _walk(args) -> tuple[str, int, int, int]:
    arm_name, overrides, seed, time_budget, max_nodes = args

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    import sts2_env.cards  # noqa: F401
    from sts2_env.bridge import agent_runner
    from sts2_env.gym_env.run_env import STS2RunEnv
    from sts2_env.run.run_manager import RunManager
    from sts2_env.search.turn_search import SearchAgent
    from ab_archetype_picking import _pick_card_reward, _search_combat_action
    from harvest_combat_benchmark import _noncombat_action

    # Patched in the WORKER, so each arm gets its own process-local table and
    # arms cannot leak into one another through a shared module.
    if overrides:
        agent_runner.ROOM_MIN_HP_FRACTION = {
            **agent_runner.ROOM_MIN_HP_FRACTION, **overrides}

    agent = SearchAgent(time_budget=time_budget, lookahead_turns=2,
                        max_nodes=max_nodes)
    rng = np.random.default_rng(seed)
    env = STS2RunEnv()
    env.reset(seed=seed)
    for _ in range(3000):
        mgr = env._mgr
        if mgr is None:
            break
        mask = env.action_masks()
        valid = np.where(mask == 1)[0]
        if not len(valid):
            break
        action = None
        if mgr.phase == RunManager.PHASE_COMBAT:
            action = _search_combat_action(agent, mgr, mask)
        elif mgr.phase == RunManager.PHASE_CARD_REWARD:
            action = _pick_card_reward(mgr, mask, rng, True)
            if action is None:
                action = _noncombat_action(mgr, mgr.phase, mask, rng)
        else:
            action = _noncombat_action(mgr, mgr.phase, mask, rng)
        if action is None:
            action = int(rng.choice(valid))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    mgr = env._mgr
    rs = getattr(mgr, "run_state", None)
    out = (arm_name, seed, int(getattr(rs, "total_floor", 0) or 0),
           int(getattr(rs, "current_act_index", 0) or 0) + 1)
    env.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=31000)
    parser.add_argument("--time-budget", type=float, default=60.0)
    parser.add_argument("--max-nodes", type=int, default=2000,
                        help="Deterministic cost bound; see sweep_eval_weights.py")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", default="output/sweep_hp_gate.txt")
    args = parser.parse_args()

    workers = args.workers or max(1, (mp.cpu_count() or 2) - 2)
    seeds = [args.seed + i for i in range(args.runs)]
    jobs = [(n, ov, s, args.time_budget, args.max_nodes)
            for n, ov in ARMS for s in seeds]
    print(f"{len(ARMS)} arms x {args.runs} paired seeds = {len(jobs)} runs, "
          f"{workers} workers", flush=True)

    rows_path = Path(args.out).with_suffix(".rows.jsonl")
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_fh = rows_path.open("a", encoding="utf-8")

    results: dict[str, dict[int, tuple[int, int]]] = {n: {} for n, _ in ARMS}
    started = time.monotonic()
    with mp.Pool(workers) as pool:
        for i, (arm, seed, floor, act) in enumerate(
                pool.imap_unordered(_walk, jobs, chunksize=1), 1):
            results[arm][seed] = (floor, act)
            rows_fh.write(json.dumps(
                {"arm": arm, "seed": seed, "floor": floor, "act": act}) + "\n")
            rows_fh.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}  "
                      f"({(time.monotonic() - started) / 60:.1f} min)", flush=True)
    rows_fh.close()

    base = results["baseline .80/.40"]
    lines = ["", "=" * 74,
             f"HP GATE SWEEP  {args.runs} paired seeds/arm, "
             f"{(time.monotonic() - started) / 60:.1f} min", "",
             f"{'arm':<20}{'clear':>8}{'mean floor':>12}"
             f"   {'vs baseline (paired)':>26}", "-" * 74]
    for name, _ in ARMS:
        got = results[name]
        shared = [s for s in seeds if s in got and s in base]
        clears = [1 if got[s][1] >= 2 else 0 for s in shared]
        floors = [got[s][0] for s in shared]
        rate = sum(clears) / len(clears) if clears else 0.0
        if name.startswith("baseline"):
            delta = ""
        else:
            paired = [(1 if got[s][1] >= 2 else 0) - (1 if base[s][1] >= 2 else 0)
                      for s in shared]
            md = statistics.mean(paired) if paired else 0.0
            se = (statistics.stdev(paired) / math.sqrt(len(paired))
                  if len(paired) > 1 else float("nan"))
            sg = md / se if se and se == se and se > 0 else 0.0
            delta = f"{100 * md:+5.1f}% +/- {100 * se:4.1f}%  ({sg:+.1f} se)"
        lines.append(f"{name:<20}{100 * rate:>7.0f}%"
                     f"{statistics.mean(floors) if floors else 0:>12.1f}"
                     f"   {delta:>26}")
    lines += ["", "=" * 74, ""]
    report = "\n".join(lines)
    print(report)
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
