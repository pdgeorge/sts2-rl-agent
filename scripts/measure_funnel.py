"""Measure the act 1 funnel offline: reach the boss, then win it.

    python scripts/measure_funnel.py --runs 400

WHY A FUNNEL AND NOT A CLEAR RATE
---------------------------------
Clear rate is a product of two independent things, and reporting only the
product hid which one was moving. Split across live sessions it is stark:

    reach boss   win boss   clear
    ~10-50%      0-20%      0-10%     early era (Aug 5-7)
    63%          ~40%       26%       recent (Aug 9-11)

Reach rate went from ~10% to 63% as the parity work landed. Boss win rate never
moved. At a 63% reach rate, 50% clear needs a 79% boss win rate, so the boss
fight is the whole remaining gap and the number worth tracking on its own.

BOTH ACT 1 VARIANTS
-------------------
Defaults to `random`, matching the game's act-1 dropdown and how every live run
was generated. Until recently the simulator only ever played Overgrowth, so
57% of real act 1 boss fights were against a boss it could not roll; a funnel
measured on Overgrowth alone is not measuring the act the agent actually meets.

COST BOUND
----------
`max_nodes`, not a wall clock. A wall-clock budget under N-worker contention
truncates the search non-deterministically and silently, which voided two
earlier sweeps. Rows are written incrementally, because a previous run lost 479
finished results by keeping them only in the parent's memory.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def rs_live(mgr):
    """The live RunState, or a stand-in, so a torn-down manager cannot raise."""
    return getattr(mgr, "run_state", None) or object()


def _walk(job) -> dict:
    seed, variant, max_nodes, time_budget = job

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    import sts2_env.cards  # noqa: F401
    from sts2_env.gym_env.run_env import STS2RunEnv
    from sts2_env.run.run_manager import RunManager
    from sts2_env.search.turn_search import SearchAgent
    from ab_archetype_picking import _search_combat_action
    from live_policy import noncombat_action

    agent = SearchAgent(time_budget=time_budget, lookahead_turns=2,
                        max_nodes=max_nodes)
    rng = np.random.default_rng(seed)
    env = STS2RunEnv(act1_variant=variant)
    env.reset(seed=seed)

    boss = None
    bosses: dict[int, str] = {}
    for _ in range(3000):
        mgr = env._mgr
        if mgr is None:
            break
        mask = env.action_masks()
        valid = np.where(mask == 1)[0]
        if not len(valid):
            break
        if mgr.phase == RunManager.PHASE_COMBAT:
            last = mgr._last_encounter
            # The act 1 boss is the only boss reachable inside act 1; recording
            # the first one seen keeps a later act's boss out of the numbers.
            if boss is None and last and "boss" in last[0]:
                boss = last[0]
            # EVERY act's boss, by act. Without this the only measurable
            # milestone is "cleared act N", and reach-versus-win cannot be
            # separated for act 2 or 3 -- which is the whole point of a funnel.
            # `floor >= 32` does NOT stand in for reaching the act 2 boss: the
            # floor only passes 32 after that fight is already won, so it scores
            # a 100% win rate by construction.
            if last and "boss" in last[0]:
                act_index = int(getattr(rs_live(mgr), "current_act_index", 0) or 0)
                bosses.setdefault(act_index, last[0])
            action = _search_combat_action(agent, mgr, mask)
        else:
            action = noncombat_action(mgr, mgr.phase, mask, rng)
        if action is None:
            action = int(rng.choice(valid))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    rs = env._mgr.run_state
    row = {
        "seed": seed,
        "variant": env._mgr._act1_variant,
        "floor": int(getattr(rs, "total_floor", 0) or 0),
        "act": int(getattr(rs, "current_act_index", 0) or 0) + 1,
        "boss": boss,
        # {act_index: setup_name} for every boss fought, so reach and win can be
        # separated per act rather than only for act 1.
        "bosses": {str(k): v for k, v in bosses.items()},
    }
    env.close()
    return row


def _pct(k: int, n: int) -> str:
    if not n:
        return "   n/a"
    p = k / n
    return f"{100 * p:4.0f}% +/-{100 * math.sqrt(p * (1 - p) / n):4.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--variant", default="random",
                        choices=("random", "overgrowth", "underdocks"))
    parser.add_argument("--max-nodes", type=int, default=2000)
    parser.add_argument("--time-budget", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--resume", action="store_true",
                        help="Skip seeds already present in the rows file.")
    parser.add_argument("--out", default="output/funnel.txt")
    args = parser.parse_args()

    workers = args.workers or max(1, (mp.cpu_count() or 2) - 2)
    rows_path = Path(args.out).with_suffix(f".{args.tag}.rows.jsonl")
    rows_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume, because a killed run leaves its finished seeds on disk and they
    # cost roughly 20 seconds of wall clock each. Completions arrive out of
    # order, so what is missing is NOT the tail -- an interrupted 400-run job
    # left seeds scattered from 50206 to 50399 undone. Read what is there and
    # ask for the difference.
    rows: list[dict] = []
    if args.resume and rows_path.exists():
        with rows_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    done = {r["seed"] for r in rows}

    jobs = [(args.seed + i, args.variant, args.max_nodes, args.time_budget)
            for i in range(args.runs) if (args.seed + i) not in done]
    print(f"{args.runs} runs, variant={args.variant}, {workers} workers, "
          f"max_nodes={args.max_nodes}"
          + (f"  [resuming: {len(done)} done, {len(jobs)} to go]" if done else ""),
          flush=True)

    rows_fh = rows_path.open("a", encoding="utf-8")
    started = time.monotonic()
    with mp.Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_walk, jobs, chunksize=1), 1):
            rows.append(row)
            rows_fh.write(json.dumps(row) + "\n")
            rows_fh.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}  "
                      f"({(time.monotonic() - started) / 60:.1f} min)", flush=True)
    rows_fh.close()

    def funnel(sel: list[dict]) -> tuple[int, int, int]:
        reached = [r for r in sel if r["boss"]]
        won = [r for r in reached if r["act"] >= 2]
        return len(sel), len(reached), len(won)

    lines = ["", "=" * 78,
             f"ACT 1 FUNNEL  tag={args.tag}  {len(rows)} runs  "
             f"({len(jobs)} this session, "
             f"{(time.monotonic() - started) / 60:.1f} min)", "",
             "  win-boss is CONDITIONAL: the share of runs that GOT THERE.",
             "  reach x win-boss = clear, e.g. 65% x 72% = 47%.", "",
             f"{'group':<14}{'n':>5}{'reach boss':>14}{'win|reached':>14}"
             f"{'clear (all)':>14}{'floor':>8}", "-" * 78]
    groups = [("ALL", rows)]
    for v in ("overgrowth", "underdocks"):
        groups.append((v, [r for r in rows if r["variant"] == v]))
    for name, sel in groups:
        n, reach, won = funnel(sel)
        floor = np.mean([r["floor"] for r in sel]) if sel else 0.0
        lines.append(f"{name:<14}{n:>5}{_pct(reach, n):>14}"
                     f"{_pct(won, reach):>14}{_pct(won, n):>14}{floor:>8.1f}")
    # Per-act milestones. Reach and win separated, which "cleared act N" alone
    # cannot do -- and which is what stops a change that buys act 1 while
    # quietly costing act 2 from looking like progress.
    lines += ["", "per-act milestones (reach the boss, then win it):",
              f"  {'act':<6}{'reached':>12}{'won|reached':>14}{'cleared':>12}"]
    for act_index in (0, 1, 2):
        reached = [r for r in rows if str(act_index) in (r.get("bosses") or {})]
        cleared = [r for r in rows if r["act"] >= act_index + 2]
        lines.append(
            f"  {act_index + 1:<6}{_pct(len(reached), len(rows)):>12}"
            f"{_pct(len(cleared), len(reached)):>14}"
            f"{_pct(len(cleared), len(rows)):>12}")

    lines += ["", "per boss (fights / wins):"]
    per = collections.Counter(r["boss"] for r in rows if r["boss"])
    wins = collections.Counter(r["boss"] for r in rows if r["boss"] and r["act"] >= 2)
    for b, n in per.most_common():
        label = b.replace("setup_", "").replace("_boss", "")
        lines.append(f"  {label:<26}{wins[b]:>4}/{n:<4} {_pct(wins[b], n)}")
    lines += ["", "=" * 78, ""]
    report = "\n".join(lines)
    print(report)
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
