"""Do simulated runs predict live ones? Run this before trusting an offline A/B.

    python scripts/sim_vs_live.py --runs 60

WHY THIS IS THE MOST IMPORTANT SCRIPT HERE
------------------------------------------
Every decision about this agent is currently made from live sessions, and a
20-run live session costs an hour and resolves to about +/- 10% on reach rate.
That is not enough precision to tell a real 10-point gain from noise, and it has
already produced two wrong calls: a per-room search budget that measurement
later showed could not help (the search uses 3% of its budget), and a rollout
policy called "harmful" off a single session.

If a simulated run predicts a live one, the same questions cost minutes at
+/- 2%, and the things that are currently unfalsifiable -- deckbuilding
archetypes, the evaluation weights, the 0.80 HP gate -- become measurable. That
is the whole return on the parity work, and this is what checks whether it has
arrived.

IT RUNS THE LIVE STACK, NOT A TRAINING STACK
--------------------------------------------
Turn search for combat, and the same `agent_runner` heuristics the bridge uses
for map, rest and card rewards. Comparing anything else would measure a
different agent: the trained combat model is NOT what plays live, so an offline
number produced with it says nothing about a live number.

WHAT AGREEMENT WOULD MEAN, AND WHAT IT WOULD NOT
------------------------------------------------
Agreement on mean floor and act-1 clear rate says the simulator reproduces a
run's ARC well enough to rank changes against each other. It does not say the
two are identical -- the map generator, the card pools and the shop are still
this side's own -- so a large offline effect should still be confirmed live
before it is believed. The point is to stop spending an hour on questions whose
answer is "no difference".

FLOOR NUMBERING DIFFERS, DELIBERATELY REPORTED RAW
--------------------------------------------------
The live game puts the act 1 boss on floor 17 and the simulator on 16, which is
recorded in `live_eval.ACT1_BOSS_FLOOR`. Rather than paper over that with an
offset, both numbers are printed and the act-2 rate is given as the comparison
that does not depend on the convention at all.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def walk(seed: int, agent, rng) -> tuple[int, int]:
    """One full simulated run with the live stack. Returns (floor, act)."""
    from sts2_env.gym_env.run_env import STS2RunEnv
    from sts2_env.run.run_manager import RunManager

    sys.path.insert(0, str(REPO / "scripts"))
    from ab_archetype_picking import _pick_card_reward, _search_combat_action
    from harvest_combat_benchmark import _noncombat_action

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
    # `current_act_index`, 0-based, and NOT `act` or `act_number` -- neither
    # exists. The first version of this asked for those with a default of 1, so
    # every run reported act 1 and the clear rate came out 0/60 while the floor
    # distribution reached 45. A run on floor 45 is deep in act 3; the number was
    # measuring the default, not the run.
    act = int(getattr(run_state, "current_act_index", 0) or 0) + 1
    env.close()
    return floor, act


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument(
        "--time-budget", type=float, default=60.0,
        help=(
            "Seconds per turn. Live uses 3.0, but measurement shows the search "
            "spends ~0.08s on a boss turn and never exhausts its budget in a "
            "position act 1 can produce, so a smaller number here changes the "
            "decisions not at all and the wall clock a great deal."
        ))
    parser.add_argument("--out", default=None, help="Append a summary line here")
    args = parser.parse_args()

    import sts2_env.cards  # noqa: F401
    from sts2_env.search.turn_search import SearchAgent

    agent = SearchAgent(time_budget=args.time_budget, lookahead_turns=2)
    floors: list[int] = []
    acts: list[int] = []

    started = time.monotonic()
    for i in range(args.runs):
        seed = args.seed + i
        floor, act = walk(seed, agent, np.random.default_rng(seed))
        floors.append(floor)
        acts.append(act)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.runs}  mean floor "
                  f"{statistics.mean(floors):.1f}", flush=True)

    elapsed = time.monotonic() - started
    cleared = sum(1 for a in acts if a >= 2)
    n = len(floors)
    se = (cleared / n * (1 - cleared / n) / n) ** 0.5 if n else 0.0

    lines = [
        "",
        "=" * 58,
        f"SIMULATED  {n} runs in {elapsed / 60:.1f} min "
        f"({elapsed / max(n, 1):.1f}s per run)",
        "",
        f"  mean floor     {statistics.mean(floors):.1f}   "
        f"median {statistics.median(floors):.0f}   "
        f"min {min(floors)}   max {max(floors)}",
        f"  cleared act 1  {cleared}/{n} = {100 * cleared / n:.0f}% "
        f"+/- {100 * se:.0f}%",
        "",
        "  LIVE, same stack, most recent session (n=13):",
        "    mean floor     15.6   median 17",
        "    cleared act 1  23%  +/- 12%",
        "=" * 58,
        "",
    ]
    report = "\n".join(lines)
    print(report)
    # Per-run rows, not just the summary. The first version wrote only the
    # aggregate, so when the act figure turned out to be measuring a default
    # rather than a run, there was nothing to recompute from and the whole 40
    # minutes had to be spent again.
    print("per-run (floor, act): "
          + ", ".join(f"{f}/{a}" for f, a in zip(floors, acts)))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(report)
            fh.write("per-run (floor, act): "
                     + ", ".join(f"{f}/{a}" for f, a in zip(floors, acts)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
