"""Paired-seed A/B over any number of PolicyConfig arms, with the gates that matter.

    .venv/bin/python scripts/ab_card_prior.py --runs 400 --workers 14

WHAT THE ARMS ARE
-----------------
`v001` (prior off, the shipped agent) against `v004_card_prior` (weight 1.0).
The single difference is `card_prior_weight`, read through `PolicyConfig` in
the worker's own process, so neither arm can see the other's value.

WHY THE DECKBUILDER AND NOT THE COMBAT EVALUATOR
------------------------------------------------
Five straight combat predictions -- 6, 7, 8, 10, 13 -- came back null or miss,
and the only change that ever moved the number was a bug fix. The deckbuilder
has never been tested, and it is measurably bad: over 1,478 act 1 card rewards
the agent took cards averaging **-0.28** winrate delta against a best-available
**+3.37** and a RANDOM pick of **-0.79**. It drafts barely better than chance
and takes the best-rated option 40.2% of the time. Exposure is ~7 decisions per
run on 100% of runs, where prediction 13 reached 2% of decisions and died.

THE GATES ARE THE POINT
-----------------------
Prediction 11 moved a rate while doing nothing it was built for, so the
behaviour is measured separately from the outcome:

  picking     mean prior-rating of cards taken, and how often the best-rated
              option was taken. If these do not move, the arm did not fire and
              the clear rate is telling us about something else.

  BLOAT       pd's check, and a real risk rather than a formality. The prior is
              a POSITIVE term for good cards, and the take/skip rule is
              `100 * score / QUALITY_BAR_SCALE > deck_size` -- so raising
              scores loosens the bar and she takes more. Deck size at the act 1
              boss and cards-drawn-per-deck-card are reported per arm. A clear
              win bought with a deck too big to cycle is not a win, it is a
              knob (`quality_bar_scale`) that needs turning, and this is how we
              would know which of the two happened.

WHAT WOULD MAKE THIS A HONEST NEGATIVE
--------------------------------------
The prior is human play. This searcher looks two turns ahead and plans nothing
across fights, so cards whose value comes from a plan a human makes may be
rated for a player who is not her. If the picking gates pass and clear does not
move, the answer is that the prior does not transfer -- which is worth knowing,
because it says the deckbuilder needs Cyra's own valuation and no borrowed one
will do.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARMS: tuple[str, ...] = ("v001",)  # replaced by --arms at runtime


def _room_map():
    from sts2_env.core.enums import RoomType
    return {RoomType.MONSTER: "monster", RoomType.ELITE: "elite",
            RoomType.BOSS: "boss"}
ACT1_BOSS_FLOOR = 17


#: Per-worker address-space cap. A leaked clone chain can take a single run to
#: 15 GB (measured, seed 372), and an OOM kill does not merely lose that run --
#: multiprocessing.Pool then waits forever for a result that can never arrive.
#: That has now cost three separate A/B nights: 10 hours of a hung pool on
#: 2026-08-21 with two workers killed at 12.8 GB and 17.5 GB.
#:
#: With a soft cap the worker raises MemoryError instead, the job is caught
#: below, and the pool carries on. One lost row rather than a lost night. The
#: cap is deliberately generous -- a healthy run peaks far under it, so
#: anything that trips this was never going to finish usefully.
WORKER_MEMORY_CAP_GB = 6


def _walk(job) -> dict:
    seed, arm, max_nodes = job

    import resource
    try:
        cap = WORKER_MEMORY_CAP_GB * 1024 ** 3
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except Exception:  # noqa: BLE001 - a cap we cannot set is not fatal
        pass

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    import numpy as np
    import sts2_env.cards  # noqa: F401
    from ab_archetype_picking import _search_combat_action
    from live_policy import noncombat_action
    from sts2_env.core.enums import RoomType
    from sts2_env.gym_env.run_env import STS2RunEnv
    from sts2_env.policy_config import PolicyConfig, apply_active_policy, set_active_policy
    from sts2_env.run.run_manager import RunManager
    from sts2_env.search.turn_search import SearchAgent

    policy = PolicyConfig.load(arm)
    set_active_policy(policy)
    apply_active_policy(policy)

    # The prior table, read directly, so the gate measures the SAME quantity
    # the scoreboard row was written against -- the raw winrate delta -- rather
    # than the scaled term the arm happens to use.
    try:
        prior = json.loads(
            (REPO / "data/untapped/act1_card_reward_prior.json").read_text()
        )["cards"]
    except Exception:
        prior = {}
    from sts2_env.search.situation import resolve_card_id

    def rating(card_id: str):
        cid = resolve_card_id(str(card_id).replace("+", ""))
        entry = prior.get(cid.name) if cid else None
        return float(entry["wr"]) if entry else None

    agent = SearchAgent(time_budget=60.0, lookahead_turns=2, max_nodes=max_nodes,
                        weights=policy.eval_weights)
    from sts2_env.core.enums import RoomType
    room_of = {RoomType.MONSTER: "monster", RoomType.ELITE: "elite",
               RoomType.BOSS: "boss"}

    ROOM_OF = _room_map()
    rng = np.random.default_rng(seed)
    env = STS2RunEnv()
    env.reset(seed=seed)

    taken_ratings: list[float] = []
    best_ratings: list[float] = []
    took_best = 0
    rated_decisions = 0
    skipped_offers = 0
    deck_at_boss = None
    drawn = 0
    deck_card_slots = 0

    # THE GATES PREDICTIONS 16 AND 17 ACTUALLY NAMED. This file was generalised
    # from the card-prior harness and kept ITS gates -- picking and bloat -- so
    # the first three-arm run produced outcomes with no mechanism behind them,
    # which is precisely what prediction 11 taught us not to trust. Split by
    # room because 16 claims boss fights get shorter and 17 claims CORRIDOR
    # damage falls, and those are different rooms.
    dmg = {"monster": 0, "elite": 0, "boss": 0}
    fights = {"monster": 0, "elite": 0, "boss": 0}
    boss_turns = 0
    arrival_hp = None
    arrival_max_hp = None
    in_fight = False
    fight_room = None
    fight_start_hp = 0

    for _ in range(3000):
        mgr = env._mgr
        if mgr is None:
            break
        mask = env.action_masks()
        valid = np.where(mask == 1)[0]
        if not len(valid):
            break

        run_state = getattr(mgr, "run_state", None)
        player = getattr(run_state, "player", None)
        # `run_state.deck` does not exist; the deck hangs off the PLAYER. The
        # first version read the wrong attribute and every deck column came
        # back 0.0 -- which the smoke test caught, and a 6.5 hour run would
        # have reported as a confident zero.
        deck_now = list(getattr(player, "deck", None) or [])
        floor = int(getattr(run_state, "total_floor", 0) or 0)

        if getattr(mgr, "_current_room_type", None) == RoomType.BOSS \
                and deck_at_boss is None and floor <= ACT1_BOSS_FLOOR + 1:
            deck_at_boss = len(deck_now)

        # -- the gates -------------------------------------------------------
        # Damage TAKEN per fight, by room, and how long the act 1 boss fight
        # ran. Measured by watching HP across the fight rather than summing
        # events, for the same reason `evaluate.py` scores a state instead of
        # predicting one: the difference is the truth whatever produced it.
        room = ROOM_OF.get(getattr(mgr, "_current_room_type", None))
        hp_now = int(getattr(player, "current_hp", 0) or 0)
        combat_now = getattr(mgr, "_combat", None)
        fighting = mgr.phase == RunManager.PHASE_COMBAT and combat_now is not None

        if fighting and not in_fight and room:
            in_fight, fight_room, fight_start_hp = True, room, hp_now
            fights[room] = fights.get(room, 0) + 1
            if room == "boss" and arrival_hp is None and floor <= ACT1_BOSS_FLOOR + 1:
                arrival_hp = hp_now
                arrival_max_hp = int(getattr(player, "max_hp", 0) or 0)
        elif fighting and in_fight and fight_room == "boss" \
                and floor <= ACT1_BOSS_FLOOR + 1:
            boss_turns = max(boss_turns, int(getattr(combat_now, "turn_count", 0) or 0))
        elif in_fight and not fighting:
            if fight_room:
                dmg[fight_room] = dmg.get(fight_room, 0) + max(0, fight_start_hp - hp_now)
            in_fight, fight_room = False, None

        # The picking gate. Read BEFORE the action, from the same offer the
        # chooser is about to rank, so it describes the decision actually made.
        # The picking gate. The offer set is captured BEFORE the action, and
        # scored against whatever the arm actually chose afterwards -- decoded
        # from the returned action index rather than re-ranked here. Re-ranking
        # would have measured a chooser nobody plays: the live path applies the
        # archetype fit and can skip, and neither was in the copy.
        pending_offers = None
        if mgr.phase == RunManager.PHASE_CARD_REWARD and floor <= ACT1_BOSS_FLOOR:
            picks = [a for a in (mgr.get_available_actions() or [])
                     if a.get("action") == "pick_card" and a.get("card_id")]
            if len(picks) >= 2:
                pending_offers = [a["card_id"] for a in picks]

        if mgr.phase == RunManager.PHASE_COMBAT:
            combat = getattr(mgr, "_combat", None)
            if combat is not None:
                # Dilution: how much of the deck she actually sees. A prior that
                # buys a rate by stuffing the deck would show up right here.
                hand = len(getattr(combat, "hand", []) or [])
                if hand:
                    drawn += hand
                    deck_card_slots += len(deck_now) or 1
            action = _search_combat_action(agent, mgr, mask)
        else:
            action = noncombat_action(mgr, mgr.phase, mask, rng)
        if action is None:
            action = int(rng.choice(valid))

        if pending_offers is not None:
            from sts2_env.gym_env.run_env import _CARD_RWD_START
            index = int(action) - _CARD_RWD_START
            rated = [rating(c) for c in pending_offers]
            if all(v is not None for v in rated):
                best_value = max(rated)
                rated_decisions += 1
                best_ratings.append(best_value)
                if 0 <= index < len(pending_offers):
                    chosen = rated[index]
                    taken_ratings.append(chosen)
                    if chosen == best_value:
                        took_best += 1
                else:
                    # A skip. Counted as taking nothing rather than dropped:
                    # skipping a bad offer is a legitimate answer and the arm
                    # is allowed to do more of it, so hiding it would flatter
                    # whichever arm skips most.
                    skipped_offers += 1
                    taken_ratings.append(0.0)

        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    run_state = getattr(env._mgr, "run_state", None)
    return {
        "seed": seed, "arm": arm,
        "dmg_monster": dmg["monster"], "fights_monster": fights["monster"],
        "dmg_elite": dmg["elite"], "fights_elite": fights["elite"],
        "dmg_boss": dmg["boss"], "fights_boss": fights["boss"],
        "boss_turns": boss_turns,
        "arrival_hp": arrival_hp, "arrival_max_hp": arrival_max_hp,
        "floor": int(getattr(run_state, "total_floor", 0) or 0),
        "act": int(getattr(run_state, "current_act_index", 0) or 0) + 1,
        "deck_size": len(getattr(run_state, "deck", None) or []),
        "deck_at_boss": deck_at_boss,
        "rated_decisions": rated_decisions,
        "taken_sum": sum(taken_ratings),
        "best_sum": sum(best_ratings),
        "took_best": took_best,
        "skipped_offers": skipped_offers,
        "drawn": drawn,
        "deck_slots": deck_card_slots,
    }


def _walk_safe(job) -> dict | None:
    """`_walk`, but a run that blows its memory cap costs one row, not the night.

    Returned as None rather than raised: `imap_unordered` propagates an
    exception by tearing down the pool, which is the same lost night by another
    route.
    """
    try:
        return _walk(job)
    except MemoryError:
        seed, arm, _ = job
        return {"seed": seed, "arm": arm, "failed": "MemoryError"}
    except Exception as exc:  # noqa: BLE001
        seed, arm, _ = job
        import traceback
        return {"seed": seed, "arm": arm, "failed": type(exc).__name__,
                "detail": f"{exc}", "where": traceback.format_exc().strip().split("\n")[-3:]}


def _wilson(k: int, n: int) -> float:
    if not n:
        return 0.0
    p = k / n
    return 100 * 1.96 * math.sqrt(p * (1 - p) / n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=400)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--max-nodes", type=int, default=20000)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="policy versions; the FIRST is the baseline everything "
                         "is paired against")
    ap.add_argument("--out", default="output/ab_arms.rows.jsonl")
    args = ap.parse_args()

    global ARMS
    ARMS = tuple(args.arms)
    jobs = [(seed, arm, args.max_nodes)
            for seed in range(args.runs) for arm in ARMS]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("", encoding="utf-8")
    print(f"{len(ARMS)} arms x {args.runs} paired seeds = {len(jobs)} runs, "
          f"{args.workers} workers\n")

    rows: list[dict] = []
    with out.open("a", encoding="utf-8") as fh, mp.Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_walk_safe, jobs, chunksize=1), 1):
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    failed = [r for r in rows if r.get("failed")]
    if failed:
        import collections as _c
        print(f"\n{len(failed)} runs FAILED and are excluded: "
              f"{dict(_c.Counter(r['failed'] for r in failed))}")
        print("  A capped worker loses one row instead of hanging the pool. "
              "Excluded rather than\n  counted as a loss, because a crashed run "
              "is not a lost run.")
    rows = [r for r in rows if not r.get("failed")]
    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in ARMS}
    print("\n" + "=" * 78)
    print(f"{'arm':<20}{'clear':>18}{'deck at boss':>15}{'deck end':>11}")
    print("=" * 78)
    for arm in ARMS:
        rs = by_arm[arm]
        cleared = sum(1 for r in rs if r["act"] >= 2)
        dab = [r["deck_at_boss"] for r in rs if r["deck_at_boss"]]
        print(f"  {arm:<18}{100 * cleared / len(rs):>10.1f}% +/-{_wilson(cleared, len(rs)):<4.1f}"
              f"{statistics.mean(dab) if dab else 0:>14.1f}"
              f"{statistics.mean([r['deck_size'] for r in rs]):>11.1f}")

    print("\nGATES -- prediction 16 says boss fights get SHORTER and corridor")
    print("damage falls; prediction 17 says corridor damage falls and more")
    print("arrivals land at 90-100% HP. Read these before the clear rate.")
    print(f"  {'arm':<22}{'dmg/monster fight':>19}{'dmg/elite':>11}"
          f"{'boss turns':>12}{'arrive HP':>11}{'>=90% HP':>10}")
    for arm in ARMS:
        rs = by_arm[arm]
        if not rs:
            continue
        fm = sum(r.get("fights_monster", 0) for r in rs)
        fe = sum(r.get("fights_elite", 0) for r in rs)
        dm = sum(r.get("dmg_monster", 0) for r in rs)
        de = sum(r.get("dmg_elite", 0) for r in rs)
        bt = [r["boss_turns"] for r in rs if r.get("boss_turns")]
        hp = [(r["arrival_hp"], r["arrival_max_hp"]) for r in rs
              if r.get("arrival_hp") and r.get("arrival_max_hp")]
        healthy = sum(1 for h, m in hp if h / m >= 0.9)
        print(f"  {arm:<22}{dm / fm if fm else 0:>19.2f}{de / fe if fe else 0:>11.1f}"
              f"{statistics.mean(bt) if bt else 0:>12.1f}"
              f"{statistics.mean([h for h, _ in hp]) if hp else 0:>11.1f}"
              f"{100 * healthy / len(hp) if hp else 0:>9.0f}%")
    print("\n  baselines to beat: corridor 4.6 HP/fight, boss arrivals 46% at "
          "90-100%,\n  and act 1 boss fights that losses drag to 13.4 turns.")

    base = {r["seed"]: r for r in by_arm[ARMS[0]]}
    print(f"\nPAIRED against {ARMS[0]}. Discordant pairs set the resolution, not n.")
    for arm in ARMS[1:]:
        test = {r["seed"]: r for r in by_arm[arm]}
        shared = sorted(set(base) & set(test))
        if not shared:
            continue
        identical = sum(1 for s in shared if base[s]["floor"] == test[s]["floor"])
        gained = sum(1 for s in shared if test[s]["act"] >= 2 and base[s]["act"] < 2)
        lost = sum(1 for s in shared if base[s]["act"] >= 2 and test[s]["act"] < 2)
        n_disc = gained + lost
        z = (gained - lost) / math.sqrt(n_disc) if n_disc else 0.0
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))) if n_disc else 1.0
        net = 100 * (gained - lost) / len(shared) if shared else 0.0
        print(f"  {arm:<22} gains {gained:>3}, loses {lost:>3}, net {net:+5.1f} pts "
              f"(z={z:+.2f}, p={p:.3f})  {n_disc:>3} discordant  "
              f"{100 * identical / len(shared):.0f}% identical floor")
    print("\nTwo arms that each look flat may be CANCELLING rather than inert: "
          "prediction 16\nwants her to finish fights faster and 17 makes her "
          "block more. Read them apart.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
