"""Does breaking exact evaluator ties toward a concentrated board pay?

    .venv/bin/python scripts/ab_tie_break.py --runs 400 --workers 12
    .venv/bin/python scripts/ab_tie_break.py --gate 40 --workers 12   # gate only

WHAT IS ACTUALLY BEING TESTED
-----------------------------
`evaluate.py` returns one float, and when two lines return the SAME float the
searcher keeps whichever the enumerator emitted first. That is not a rare
corner: across the `wednesday` live session, **3,457 of 10,556 combat decisions
(32.7%) had the chosen line exactly tied with another, and 100% of them were
settled by iteration order**. Offline reproduces it at 28.0% (854 of 3,048),
which is what makes it measurable here at all.

WHY THIS IS PRE-REGISTERED AS A NULL
------------------------------------
Because the leaf snapshot says most of that 33% is not a defect. Of 854 offline
ties, **89.0% end in a genuinely IDENTICAL position** -- same HP, same block,
same per-enemy HP, same board after the lookahead -- and scoring those equal is
correct, not sloppy. Against a 5 HP Seapunk, `STRIKE, DISMANTLE` and `DEFEND`
really do end the fight in the same place.

Only **11.0%** are different boards, and 89 of those 94 are the SAME POOLED
enemy HP with the damage spread differently. That is `evaluate.py`'s blind spot
stated precisely: enemy HP is scored as one pooled fraction, so setting up a
kill and smearing the same damage across two enemies are the same number to it.

**0 of the 854 differed in enemies killed, and 0 differed in player HP by 3 or
more, within the horizon.** So the payoff -- if there is one -- lands beyond the
2-turn lookahead as a kill arriving a turn sooner, and the exposure is about 3%
of combat decisions. Prediction 11 already measured what a 2.9%-exposure arm
does to a rate: nothing.

THE GATE IS THE POINT, NOT THE RATE
-----------------------------------
Prediction 11's lesson was that an arm can move an outcome while doing nothing
it was built to do, and the only defence is to measure the behaviour separately.
So `--gate` walks positions and runs the search TWICE on each -- once per arm,
same state, same seed -- and reports how often the tie-break actually changed
the chosen line. A rate change without that number is not attributable.

NOTHING PATCHES A GLOBAL
------------------------
Both arms are `PolicyConfig` files, `v001` (enumeration, shipped) and
`v003_tiebreak_focus`, and the value reaches the search as a `SearchAgent`
constructor argument in the worker's own process. `PHASE_TWO.md` 3.1 records the
alternative: 400 runs whose baseline arm did the opposite of its name.

WHY OFFLINE IS A VALID INSTRUMENT HERE
--------------------------------------
`audit_dynamics.py` puts offline and live at ~97% identical turn resolution and
99.2% move-state agreement, so in-fight mechanism questions transfer. This is an
in-fight question. The arm delta on shared seeds is what is read; the absolute
rates are not.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ARMS = ("v001", "v003_tiebreak_focus")


def _setup(arm: str):
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    import sts2_env.cards  # noqa: F401
    from sts2_env.policy_config import PolicyConfig, apply_active_policy, set_active_policy

    policy = PolicyConfig.load(arm)
    set_active_policy(policy)
    apply_active_policy(policy)
    return policy


def _walk(job) -> dict:
    seed, arm, max_nodes = job
    policy = _setup(arm)

    import numpy as np
    from ab_archetype_picking import _search_combat_action
    from live_policy import noncombat_action
    from sts2_env.gym_env.run_env import STS2RunEnv
    from sts2_env.run.run_manager import RunManager
    from sts2_env.search.turn_search import SearchAgent

    agent = SearchAgent(
        time_budget=60.0, lookahead_turns=2, max_nodes=max_nodes,
        weights=policy.eval_weights,
        # The one difference between the arms.
        tie_break=policy.tie_break,
    )
    rng = np.random.default_rng(seed)
    env = STS2RunEnv()
    env.reset(seed=seed)

    # Gate metric: in fights that START with 3+ enemies, how many player turns
    # pass before the first corpse? Concentrating damage should shorten it, and
    # if it does not then the arm is not doing what it was built to do,
    # whatever the clear rate says.
    multi_fights = 0
    turns_to_first_kill: list[int] = []
    fight_start_count = None
    fight_turns = 0
    recorded = False

    for _ in range(3000):
        mgr = env._mgr
        if mgr is None:
            break
        mask = env.action_masks()
        valid = np.where(mask == 1)[0]
        if not len(valid):
            break

        combat = getattr(mgr, "combat", None) or getattr(mgr, "_combat", None)
        in_combat = mgr.phase == RunManager.PHASE_COMBAT and combat is not None

        if in_combat:
            alive = sum(1 for e in combat.enemies if e.is_alive)
            if fight_start_count is None:
                fight_start_count = alive
                fight_turns = 0
                recorded = False
                if alive >= 3:
                    multi_fights += 1
            turn = int(getattr(combat, "turn_count", 0) or 0)
            if turn > fight_turns:
                fight_turns = turn
            if (not recorded and fight_start_count >= 3
                    and alive < fight_start_count):
                turns_to_first_kill.append(max(1, fight_turns))
                recorded = True
            action = _search_combat_action(agent, mgr, mask)
        else:
            fight_start_count = None
            action = noncombat_action(mgr, mgr.phase, mask, rng)

        if action is None:
            action = int(rng.choice(valid))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    run_state = getattr(env._mgr, "run_state", None)
    return {
        "seed": seed, "arm": arm,
        "floor": int(getattr(run_state, "total_floor", 0) or 0),
        "act": int(getattr(run_state, "current_act_index", 0) or 0) + 1,
        "hp": int(getattr(getattr(run_state, "player", None), "current_hp", 0) or 0),
        "multi_fights": multi_fights,
        "ttfk_sum": sum(turns_to_first_kill),
        "ttfk_n": len(turns_to_first_kill),
    }


def _gate(job) -> dict:
    """Same state, both arms, one process: how often does the pick change?

    This is the number prediction 13 is graded on first. It cannot be inferred
    from the two arms' runs -- they diverge after the first differing decision,
    so any later disagreement is the diverged run, not the tie-break.
    """
    seed, max_nodes = job
    policy = _setup(ARMS[0])

    import numpy as np
    from ab_archetype_picking import _search_combat_action
    from live_policy import noncombat_action
    from sts2_env.gym_env.run_env import STS2RunEnv
    from sts2_env.run.run_manager import RunManager
    from sts2_env.search.turn_search import search_turn

    counts: Counter = Counter()
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

        combat = getattr(mgr, "combat", None) or getattr(mgr, "_combat", None)
        if mgr.phase == RunManager.PHASE_COMBAT and combat is not None \
                and combat.pending_choice is None:
            kw = dict(weights=policy.eval_weights, max_nodes=max_nodes,
                      time_budget=3.0, lookahead_turns=2)
            base = search_turn(combat, tie_break="enumeration", **kw)
            focus = search_turn(combat, tie_break="focus", **kw)
            counts["decisions"] += 1
            if base.score != focus.score:
                # Must never happen: `focus` may only pick between equals.
                counts["SCORE_CHANGED"] += 1
            if tuple(base.actions) != tuple(focus.actions):
                counts["changed"] += 1
            top = base.considered[0][0] if base.considered else None
            if top is not None and sum(
                    1 for s, _ in base.considered if abs(s - top) < 1e-9) > 1:
                counts["ties"] += 1
            action = _search_combat_action(None, mgr, mask)
        else:
            action = noncombat_action(mgr, mgr.phase, mask, rng)

        if action is None:
            action = int(rng.choice(valid))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return dict(counts)


def _wilson(k: int, n: int) -> float:
    if not n:
        return 0.0
    p = k / n
    return 100 * 1.96 * math.sqrt(p * (1 - p) / n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=400, help="paired seeds per arm")
    ap.add_argument("--gate", type=int, default=0,
                    help="run only the behavioural gate, over this many seeds")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-nodes", type=int, default=20000)
    ap.add_argument("--out", default="output/ab_tie_break.rows.jsonl")
    args = ap.parse_args()

    if args.gate:
        print(f"BEHAVIOURAL GATE: both arms on the same state, {args.gate} seeds\n")
        total: Counter = Counter()
        with mp.Pool(args.workers) as pool:
            for i, c in enumerate(
                    pool.imap_unordered(_gate,
                                        [(s, args.max_nodes) for s in range(args.gate)]), 1):
                total.update(c)
                if i % 5 == 0:
                    print(f"  {i}/{args.gate} seeds", flush=True)
        d = total["decisions"]
        print(f"\n  combat decisions           {d}")
        print(f"  exact ties among them      {total['ties']}  "
              f"({100 * total['ties'] / max(1, d):.1f}%)")
        print(f"  tie-break CHANGED the pick {total['changed']}  "
              f"({100 * total['changed'] / max(1, d):.1f}%)")
        print(f"  score changed (must be 0)  {total['SCORE_CHANGED']}")
        print("\n  Gate as written in prediction 13: 2-5% of decisions changed.")
        return 0

    jobs = [(seed, arm, args.max_nodes) for seed in range(args.runs) for arm in ARMS]
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
    print("\n" + "=" * 78)
    print(f"{'arm':<24}{'clear':>17}{'turns to 1st kill':>22}{'(3+ enemy fights)':>15}")
    print("=" * 78)
    for arm in ARMS:
        rs = by_arm[arm]
        cleared = sum(1 for r in rs if r["act"] >= 2)
        n_k = sum(r["ttfk_n"] for r in rs)
        ttfk = sum(r["ttfk_sum"] for r in rs) / n_k if n_k else 0.0
        print(f"  {arm:<22}{100 * cleared / len(rs):>9.1f}% +/-{_wilson(cleared, len(rs)):<4.1f}"
              f"{ttfk:>19.2f}{n_k:>15d}")

    base = {r["seed"]: r for r in by_arm[ARMS[0]]}
    test = {r["seed"]: r for r in by_arm[ARMS[1]]}
    shared = sorted(set(base) & set(test))
    identical = sum(1 for s in shared if base[s]["floor"] == test[s]["floor"])
    gained = sum(1 for s in shared if test[s]["act"] >= 2 and base[s]["act"] < 2)
    lost = sum(1 for s in shared if base[s]["act"] >= 2 and test[s]["act"] < 2)
    n_disc = gained + lost
    z = (gained - lost) / math.sqrt(n_disc) if n_disc else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))) if n_disc else 1.0
    net = 100 * (gained - lost) / len(shared) if shared else 0.0
    print(f"\nPAIRED over {len(shared)} shared seeds (focus vs enumeration):")
    print(f"  gains {gained}, loses {lost}, net {net:+.1f} points "
          f"(McNemar z={z:.2f}, p={p:.3f})")
    print(f"  {identical}/{len(shared)} seeds ended on the IDENTICAL floor "
          f"({100 * identical / max(1, len(shared)):.0f}%) -- the exposure, "
          f"which is what prediction 11 was killed by")
    print("\nRead the paired line and the exposure together. A null with 90% of "
          "seeds\nidentical is not evidence the mechanism is wrong, only that it "
          "barely fired.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
