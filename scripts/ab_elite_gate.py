"""Does the elite HP gate cost us the run?

    python scripts/ab_elite_gate.py --runs 150 --workers 12

THE OBSERVATION
---------------
`ROOM_MIN_HP_FRACTION["elite"]` is 0.80. Across 496 live runs the agent fights
1.28 elites and 39% of runs fight NONE at all. She arrives at rest sites at a
median 42% of max HP, so an 80% gate is a door that is mostly shut -- the same
shape as the card-skip bug (`can_skip:false`, 0 of 366 skips) and the smith gate
(fires on 11% of rest visits).

Three independent things point at this being expensive:

  1. RELICS ARE WORTH 22 POINTS. Controlled, not correlational: the same live
     decks at the same 80% HP against the same six bosses win 46% with their
     real relics and 27% with relics stripped (n=841 / n=243). Elites are one of
     only four relic sources, and the only one the agent's routing controls.

  2. LIVE RUNS THAT FIGHT ELITES CLEAR MORE. 0 elites -> 7% clear, 1 -> 9%,
     2+ -> 23%; relics at end 3.0 / 3.9 / 5.2.

  3. THE GAME'S OWN BEGINNER GUIDE SAYS SO. "Fight Elites when your deck is
     strong enough -- they drop game-changing relics", and warns specifically
     against dodging fights to preserve health, because hallway fights are the
     primary source of card rewards, gold and potions.

WHY OBSERVATION 2 IS NOT EVIDENCE, AND THIS SCRIPT IS
-----------------------------------------------------
That correlation is confounded end to end, and in the direction that flatters
the conclusion. A run that is going well is healthy, and a healthy run is
exactly the one the 0.80 gate lets through -- so "fought 2 elites" is partly a
*measurement of* the run already going well. Reverse the causation and the same
table appears with the gate doing nothing.

Only intervening separates them. Same seeds, same everything, one number
different. Paired on seed, because act 1 variance between seeds dwarfs the
effect being measured.

WHAT WOULD MAKE THIS A NULL
---------------------------
Lowering the gate buys elites and loses the runs that took them at 45% HP. That
is the failure the 0.80 was fitted to prevent, and the fit was real: measured
elite death rate is 0% at 70-79 HP against 18-29% at 40-59. So the honest prior
is that SOME gate is right and 0.80 is merely untested, not that gates are bad.

This reports elites fought, relics carried, reach, and boss win separately, so a
gate that buys relics while losing runs is visible as such rather than hiding
inside a single clear rate.

THE GATE IS 1:1 ALREADY
-----------------------
`live_policy.noncombat_action` routes offline through `_pick_map_node`, the same
function the bridge calls, so this measures the shipping policy rather than a
sweep-only copy. That was not true four days ago -- offline picked map nodes at
random -- and it is the reason this question can be asked offline at all.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _walk(job) -> dict:
    seed, elite_gate, monster_gate, variant, max_nodes = job

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    import sts2_env.cards  # noqa: F401
    from sts2_env.bridge import agent_runner as live
    from sts2_env.core.enums import RoomType
    from sts2_env.gym_env.run_env import STS2RunEnv
    from sts2_env.run.run_manager import RunManager
    from sts2_env.search.turn_search import SearchAgent
    from ab_archetype_picking import _search_combat_action
    from live_policy import noncombat_action

    # Set in the WORKER, before the first decision. Each worker is a fresh
    # process holding one arm for its whole life, so the two arms cannot see
    # each other's value -- a parent-side patch would be inherited by every
    # worker and silently collapse the experiment to one arm.
    live.ROOM_MIN_HP_FRACTION["elite"] = elite_gate
    live.ROOM_MIN_HP_FRACTION["monster"] = monster_gate
    live.ROOM_MIN_HP_FRACTION["unknown"] = monster_gate

    agent = SearchAgent(time_budget=60.0, lookahead_turns=2, max_nodes=max_nodes)
    rng = np.random.default_rng(seed)
    env = STS2RunEnv(act1_variant=variant)
    env.reset(seed=seed)

    boss = None
    boss_relics: list = []
    boss_hp = boss_max_hp = 0
    elites = 0
    seen_encounters: set = set()

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
            # Count each fight once. `_last_encounter` persists for every step
            # of the combat, so counting it raw would score a long elite fight
            # as thirty elites.
            key = (id(last), getattr(mgr, "_current_room_type", None),
                   int(getattr(getattr(mgr, "run_state", None), "total_floor", 0) or 0))
            if key not in seen_encounters:
                seen_encounters.add(key)
                if mgr._current_room_type == RoomType.ELITE:
                    elites += 1
            if boss is None and last and "boss" in last[0]:
                boss = last[0]
                player = getattr(getattr(mgr, "run_state", None), "player", None)
                boss_hp = int(getattr(player, "current_hp", 0) or 0)
                boss_max_hp = int(getattr(player, "max_hp", 0) or 0)
                boss_relics = [(r.name if hasattr(r, "name") else str(r))
                               for r in (getattr(player, "relics", None) or [])]
            action = _search_combat_action(agent, mgr, mask)
        else:
            action = noncombat_action(mgr, mgr.phase, mask, rng)
        if action is None:
            action = int(rng.choice(valid))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    rs = env._mgr.run_state
    player = getattr(rs, "player", None)
    row = {
        "seed": seed, "elite_gate": elite_gate, "monster_gate": monster_gate,
        "floor": int(getattr(rs, "total_floor", 0) or 0),
        "act": int(getattr(rs, "current_act_index", 0) or 0) + 1,
        "boss": boss, "elites": elites,
        "boss_hp": boss_hp, "boss_max_hp": boss_max_hp,
        "boss_relics": boss_relics, "n_boss_relics": len(boss_relics),
        "end_relics": len(getattr(player, "relics", None) or []),
        "deck_size": len(getattr(player, "deck", None) or []),
    }
    env.close()
    return row


def _pct(k: int, n: int) -> str:
    if not n:
        return "   n/a"
    p = k / n
    return f"{100 * p:4.0f}% +/-{100 * math.sqrt(p * (1 - p) / n):3.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=70000)
    ap.add_argument("--variant", default="random",
                    choices=("random", "overgrowth", "underdocks"))
    ap.add_argument("--max-nodes", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--gates", default="0.80,0.60,0.45",
                    help="Elite HP gates to compare. 0.80 is what ships.")
    ap.add_argument("--monster-gate", type=float, default=0.40,
                    help="Held fixed across arms so only the elite gate moves.")
    ap.add_argument("--out", default="output/ab_elite_gate.jsonl")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from sts2_env.eval.seeds import is_holdout

    gates = [float(g) for g in args.gates.split(",")]
    workers = args.workers or max(1, (mp.cpu_count() or 2) - 2)

    rows: list[dict] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out.exists():
        with out.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    done = {(r["seed"], r["elite_gate"]) for r in rows}

    jobs = [(args.seed + i, g, args.monster_gate, args.variant, args.max_nodes)
            for i in range(args.runs) for g in gates
            if (args.seed + i, g) not in done]
    print(f"{len(gates)} arms x {args.runs} paired seeds = {len(jobs)} runs, "
          f"{workers} workers, max_nodes={args.max_nodes}", flush=True)

    started = time.monotonic()
    with out.open("a", encoding="utf-8") as fh, mp.Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_walk, jobs, chunksize=1), 1):
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}  "
                      f"({(time.monotonic() - started) / 60:.1f} min)", flush=True)

    lines = ["", "=" * 84,
             f"ELITE GATE A/B  {len(rows)} runs  "
             f"({(time.monotonic() - started) / 60:.1f} min)", "",
             "  Paired on seed. 0.80 is what ships today.",
             "  win|reached is CONDITIONAL on getting there.", ""]

    for half, keep in (("tuning", lambda s: not is_holdout(s)),
                       ("HOLDOUT", is_holdout)):
        sel_all = [r for r in rows if keep(r["seed"])]
        if not sel_all:
            continue
        lines += [f"{half}:",
                  f"  {'gate':<7}{'n':>5}{'elites':>9}{'0-elite':>9}"
                  f"{'relics@boss':>13}{'reach':>12}{'win|reach':>12}"
                  f"{'clear':>12}", "  " + "-" * 78]
        for g in gates:
            sel = [r for r in sel_all if r["elite_gate"] == g]
            if not sel:
                continue
            n = len(sel)
            reached = [r for r in sel if r["boss"]]
            won = [r for r in reached if r["act"] >= 2]
            zero = sum(1 for r in sel if not r["elites"])
            rel = (statistics.mean(r["n_boss_relics"] for r in reached)
                   if reached else 0.0)
            lines.append(
                f"  {g:<7.2f}{n:>5}{statistics.mean(r['elites'] for r in sel):>9.2f}"
                f"{100 * zero / n:>8.0f}%{rel:>13.1f}"
                f"{_pct(len(reached), n):>12}{_pct(len(won), len(reached)):>12}"
                f"{_pct(len(won), n):>12}")
        lines.append("")

    # The paired view: same seed, both arms, so seed variance cancels. This is
    # the number to believe when the unpaired columns disagree with it.
    base = gates[0]
    by_seed: dict[int, dict] = collections.defaultdict(dict)
    for r in rows:
        by_seed[r["seed"]][r["elite_gate"]] = r
    lines += ["paired against the shipping gate (same seed, both arms):",
              f"  {'gate':<7}{'pairs':>7}{'clears +':>10}{'clears -':>10}"
              f"{'net':>8}", "  " + "-" * 44]
    for g in gates[1:]:
        pairs = [(v[base], v[g]) for v in by_seed.values()
                 if base in v and g in v]
        if not pairs:
            continue
        plus = sum(1 for b, x in pairs if x["act"] >= 2 and b["act"] < 2)
        minus = sum(1 for b, x in pairs if b["act"] >= 2 and x["act"] < 2)
        lines.append(f"  {g:<7.2f}{len(pairs):>7}{plus:>10}{minus:>10}"
                     f"{plus - minus:>+8}")

    lines += ["", "=" * 84, ""]
    report = "\n".join(lines)
    print(report)
    with open(str(out).replace(".jsonl", ".txt"), "a", encoding="utf-8") as fh:
        fh.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
