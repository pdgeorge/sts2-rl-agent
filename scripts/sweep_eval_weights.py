"""Fit the search's evaluation weights against outcomes instead of arguing about them.

    python scripts/sweep_eval_weights.py --runs 60

`EvalWeights` has never been measured. Its numbers are reasoned defaults -- a
point of your own HP priced at four times a point of the enemy's, block that
went unused priced slightly negative -- and reasoning is how this project has
been wrong all day. Until `sim_vs_live` showed simulated runs predicting live
ones, there was no way to check them that cost less than an hour per question.
Now there is.

WHY THIS IS THE LEVER
---------------------
The search is not compute-limited: it spends about 3% of its time budget and
enumerates the turn essentially exhaustively, and deeper lookahead changes
nothing. A complete search over a position is only as good as the number it
scores leaves with, so the evaluation IS the combat policy. Everything else in
`turn_search.py` is machinery for finding the line this file prefers.

The target is the boss. Live runs strip 55-90% of a ~250 HP boss and die with it
in the red -- a race lost by 10-30%, not a rout -- so the interesting question is
which way the weights should lean to close that.

PAIRED ON SEEDS
---------------
Every arm walks the SAME seeds, so a comparison is within-pair rather than
between two independent samples. Run-to-run variance in this game is enormous --
the same configuration has produced 35% and 70% on consecutive live sessions --
and pairing removes the part of it that comes from the map and the card offers
rather than from the weights.

The reported error bar is on the paired DIFFERENCE for that reason. An arm's own
clear rate carries the full between-seed variance and is much noisier than the
difference against baseline, which is what the sweep is actually asking about.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


#: The arms. Each is (name, {field: value}) applied over DEFAULT_WEIGHTS.
#:
#: One field at a time, deliberately. A grid over several fields at n=60 an arm
#: costs hours and cannot say which field moved the result; single changes
#: against a shared baseline can, and the interesting ones can be combined
#: afterwards once they are individually known.
ARMS: list[tuple[str, dict]] = [
    ("baseline", {}),
    # She loses the boss race by 10-30% of its HP. Damage may simply be
    # under-priced for a fight that is a race rather than a grind.
    ("enemy_hp 0.35", {"enemy_hp": 0.35}),
    ("enemy_hp 0.50", {"enemy_hp": 0.50}),
    # The penalty that pushed the searcher off blocking the Waterfall eruption.
    # Unused block is waste, but pricing it negative also prices SURVIVING it.
    ("block_unused 0", {"block_unused": 0.0}),
    # A boss is 8-12 turns. A per-turn penalty tuned to stop hallway dithering
    # may be rushing the fight that actually needs the turns.
    ("turn -0.005", {"turn": -0.005}),
]


def _walk(args) -> tuple[str, int, int, int]:
    """One full simulated run. Returns (arm, seed, floor, act)."""
    arm_name, overrides, seed, time_budget = args

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    import sts2_env.cards  # noqa: F401
    from sts2_env.gym_env.run_env import STS2RunEnv
    from sts2_env.run.run_manager import RunManager
    from sts2_env.search.evaluate import DEFAULT_WEIGHTS
    from sts2_env.search.turn_search import SearchAgent
    from ab_archetype_picking import _pick_card_reward, _search_combat_action
    from harvest_combat_benchmark import _noncombat_action

    weights = dataclasses.replace(DEFAULT_WEIGHTS, **overrides)
    agent = SearchAgent(time_budget=time_budget, lookahead_turns=2, weights=weights)

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
    run_state = getattr(mgr, "run_state", None)
    floor = int(getattr(run_state, "total_floor", 0) or 0)
    act = int(getattr(run_state, "current_act_index", 0) or 0) + 1
    env.close()
    return arm_name, seed, floor, act


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--time-budget", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=0,
                        help="0 = cpu_count - 2")
    parser.add_argument("--out", default="output/sweep_eval_weights.txt")
    args = parser.parse_args()

    workers = args.workers or max(1, (mp.cpu_count() or 2) - 2)
    seeds = [args.seed + i for i in range(args.runs)]
    jobs = [(name, ov, s, args.time_budget)
            for name, ov in ARMS for s in seeds]

    print(f"{len(ARMS)} arms x {args.runs} paired seeds = {len(jobs)} runs, "
          f"{workers} workers")
    started = time.monotonic()

    results: dict[str, dict[int, tuple[int, int]]] = {n: {} for n, _ in ARMS}
    with mp.Pool(workers) as pool:
        for i, (arm, seed, floor, act) in enumerate(
                pool.imap_unordered(_walk, jobs, chunksize=1), 1):
            results[arm][seed] = (floor, act)
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}  "
                      f"({(time.monotonic() - started) / 60:.1f} min)", flush=True)

    elapsed = time.monotonic() - started
    base = results["baseline"]

    lines = ["", "=" * 74,
             f"EVAL WEIGHT SWEEP  {args.runs} paired seeds/arm, "
             f"{elapsed / 60:.1f} min",
             "",
             f"{'arm':<18}{'clear':>8}{'mean floor':>12}"
             f"   {'vs baseline (paired)':>26}",
             "-" * 74]

    for name, _ in ARMS:
        got = results[name]
        floors = [got[s][0] for s in seeds if s in got]
        clears = [1 if got[s][1] >= 2 else 0 for s in seeds if s in got]
        n = len(clears)
        rate = sum(clears) / n if n else 0.0

        if name == "baseline":
            delta = ""
        else:
            paired = [
                (1 if got[s][1] >= 2 else 0) - (1 if base[s][1] >= 2 else 0)
                for s in seeds if s in got and s in base
            ]
            mean_d = statistics.mean(paired) if paired else 0.0
            se = (statistics.stdev(paired) / math.sqrt(len(paired))
                  if len(paired) > 1 else float("nan"))
            sigma = mean_d / se if se and se == se and se > 0 else 0.0
            delta = f"{100 * mean_d:+5.1f}% +/- {100 * se:4.1f}%  ({sigma:+.1f} se)"

        lines.append(f"{name:<18}{100 * rate:>7.0f}%"
                     f"{statistics.mean(floors) if floors else 0:>12.1f}"
                     f"   {delta:>26}")

    lines += ["",
              "Error bars are on the PAIRED difference, which is what the sweep",
              "asks about. An arm's own rate carries the full between-seed",
              "variance and is far noisier -- the same configuration has given",
              "35% and 70% on consecutive live sessions.",
              "=" * 74, ""]

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
