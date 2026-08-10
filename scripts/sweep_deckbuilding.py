"""Does archetype-aware card picking build better decks? Answered properly this time.

    python scripts/sweep_deckbuilding.py --runs 240

The question Phase 5 exists to answer, and which has never had a trustworthy
answer. `ab_archetype_picking.py` asked it first and could not settle it: it ran
combat with the frozen PPO, which wins 6.7% of bosses against the live stack's
~50%, so "a better card" meant something different there than it does in a real
run. Whichever picker won that comparison, it won a different game.

`sim_vs_live` removed that objection. Simulated runs now use the live stack --
turn search for combat, the same heuristics elsewhere -- and reproduce live
outcomes. So the comparison is finally between two card-pickers rather than
between two fictions.

WHY THIS AND NOT MORE WEIGHT TUNING
-----------------------------------
The eval-weight sweep came back a clean negative: nothing reached 1.1 se pooled
at n=120, so the search's evaluation is approximately right and is not where the
boss gap lives. Combat quality decides how well a deck is played; the DECK
decides the ceiling on what any play can achieve. Live runs strip 55-90% of a
250 HP boss and die with it in the red -- a race lost by 10-30% -- and that is a
deficit of damage available, not of decisions made.

n=240, NOT n=60
---------------
Learned the hard way on the weight sweep. At n=60 paired the difference resolves
to about +/-4% while arms move +/-5% BETWEEN replications, so a 5-point effect is
undetectable and a noise spike reads as a finding: `turn -0.005` scored +3.3% on
one seed set and -5.0% on another. n=240 an arm is roughly 80 minutes at 14
workers and resolves a 5-point effect, which is still far cheaper than the four
live hours it would take to say less.

Paired on seeds, and the error bar is on the paired difference, for the same
reason as the weight sweep: absolute clear rate wanders about 10 points between
seed sets of 60, so only the within-pair comparison means anything.
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


def _walk(args) -> tuple[str, int, int, int]:
    """One full simulated run. Returns (arm, seed, floor, act)."""
    arm_name, use_archetype, seed, time_budget, max_nodes = args

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    import sts2_env.cards  # noqa: F401
    from sts2_env.gym_env.run_env import STS2RunEnv
    from sts2_env.run.run_manager import RunManager
    from sts2_env.search.turn_search import SearchAgent
    from ab_archetype_picking import _pick_card_reward, _search_combat_action
    from harvest_combat_benchmark import _noncombat_action

    agent = SearchAgent(time_budget=time_budget, lookahead_turns=2, max_nodes=max_nodes)
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
            # The ONE line that differs between arms.
            action = _pick_card_reward(mgr, mask, rng, use_archetype)
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


ARMS = [
    ("quality only", False),      # rank_cards(offered, deck)
    ("archetype-aware", True),    # rank_cards(offered, deck, direction)
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument(
        "--time-budget", type=float, default=60.0,
        help="Offline this must never bind -- see sweep_eval_weights.py.")
    parser.add_argument(
        "--max-nodes", type=int, default=2000,
        help=(
            "DETERMINISTIC cost bound, replacing the wall-clock one offline. "
            "A time budget cannot be used here: under worker contention it "
            "truncates the search and the run stops depending only on its seed. "
            "But removing it entirely removes the cost bound too -- the node "
            "default is 20000, and one sweep run explored enough of them to "
            "spend three hours while thirteen workers sat idle. "
            "2000 is calibrated to what live reaches inside its 3s: measured "
            "positions run 33-680 nodes, and the widest artificial one 680, so "
            "this does not bind on anything act 1 produces while bounding the "
            "pathological tail. Same decisions, bounded cost, no clock."
        ))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", default="output/sweep_deckbuilding.txt")
    args = parser.parse_args()

    workers = args.workers or max(1, (mp.cpu_count() or 2) - 2)
    seeds = [args.seed + i for i in range(args.runs)]
    jobs = [(name, flag, s, args.time_budget, args.max_nodes) for name, flag in ARMS for s in seeds]

    print(f"{len(ARMS)} arms x {args.runs} paired seeds = {len(jobs)} runs, "
          f"{workers} workers")
    started = time.monotonic()

    results: dict[str, dict[int, tuple[int, int]]] = {n: {} for n, _ in ARMS}

    # Every finished run is written the moment it lands. The first version held
    # all of them in the parent and wrote once at the end, so a single pathological
    # run -- 20000 nodes deep while thirteen workers sat idle -- put three hours of
    # completed work behind one process nobody could kill without losing it.
    rows_path = Path(args.out).with_suffix(".rows.jsonl") if args.out else None
    rows_fh = None
    if rows_path is not None:
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        rows_fh = rows_path.open("a", encoding="utf-8")

    with mp.Pool(workers) as pool:
        for i, (arm, seed, floor, act) in enumerate(
                pool.imap_unordered(_walk, jobs, chunksize=1), 1):
            results[arm][seed] = (floor, act)
            if rows_fh is not None:
                rows_fh.write(json.dumps(
                    {"arm": arm, "seed": seed, "floor": floor, "act": act}) + "\n")
                rows_fh.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}  "
                      f"({(time.monotonic() - started) / 60:.1f} min)", flush=True)

    if rows_fh is not None:
        rows_fh.close()

    elapsed = time.monotonic() - started
    base = results["quality only"]
    arch = results["archetype-aware"]
    shared = [s for s in seeds if s in base and s in arch]

    def rate(got):
        v = [1 if got[s][1] >= 2 else 0 for s in shared]
        return sum(v) / len(v) if v else 0.0

    paired = [(1 if arch[s][1] >= 2 else 0) - (1 if base[s][1] >= 2 else 0)
              for s in shared]
    floor_d = [arch[s][0] - base[s][0] for s in shared]
    mean_d = statistics.mean(paired) if paired else 0.0
    se = (statistics.stdev(paired) / math.sqrt(len(paired))
          if len(paired) > 1 else float("nan"))
    fmean = statistics.mean(floor_d) if floor_d else 0.0
    fse = (statistics.stdev(floor_d) / math.sqrt(len(floor_d))
           if len(floor_d) > 1 else float("nan"))

    lines = [
        "", "=" * 70,
        f"DECKBUILDING  {len(shared)} paired seeds/arm, {elapsed / 60:.1f} min",
        "",
        f"  quality only      cleared act 1 {100 * rate(base):5.1f}%   "
        f"mean floor {statistics.mean([base[s][0] for s in shared]):.1f}",
        f"  archetype-aware   cleared act 1 {100 * rate(arch):5.1f}%   "
        f"mean floor {statistics.mean([arch[s][0] for s in shared]):.1f}",
        "",
        f"  paired difference (archetype - quality):",
        f"    act 1 clear   {100 * mean_d:+5.1f}% +/- {100 * se:4.1f}%  "
        f"({mean_d / se if se and se == se and se > 0 else 0:+.1f} se)",
        f"    mean floor    {fmean:+5.2f}  +/- {fse:4.2f}  "
        f"({fmean / fse if fse and fse == fse and fse > 0 else 0:+.1f} se)",
        "",
        f"  archetype arm won {sum(1 for d in paired if d > 0)}, "
        f"lost {sum(1 for d in paired if d < 0)}, "
        f"tied {sum(1 for d in paired if d == 0)}",
        "",
        "  A 5-point effect needs about n=240; at n=60 an arm moves +/-5% between",
        "  replications, which is how the weight sweep produced a -1.8 se result",
        "  that reversed sign on other seeds.",
        "=" * 70, "",
    ]
    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
