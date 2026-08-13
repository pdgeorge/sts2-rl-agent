"""Does trading HP for upgrades actually pay?

    python scripts/upgrades_vs_hp.py --workers 6

THE DECISION THIS INFORMS
-------------------------
Live decks reach the act 1 boss with 1.5 upgraded cards out of 20 (7%), because
`_pick_rest_option` smiths only at `hp >= 0.8 * max_hp` and she arrives at rest
sites at a median 42% -- so 11% of rest visits can upgrade at all.

The obvious fix is to lower that threshold away from a boss. But smithing is not
free: every rest spent upgrading is a rest not spent healing, so she arrives at
the boss with MORE upgrades and LESS HP. Whether that trade is worth taking is
the entire question the threshold encodes, and nobody has measured it.

This measures it. Real live decks, upgraded by 0/2/4/6 cards, fought at 80/65/50%
HP. The surface says whether upgrades outrun the HP they cost, and roughly where
the crossover is.

WHY MEASURE THIS BEFORE SWEEPING THE THRESHOLD
----------------------------------------------
A threshold sweep costs ~2 hours and answers "did this number help". This costs
less and answers "is there anything here to find, and how big". If four upgrades
do not beat the HP they cost, the rest-site plan is dead before it is built --
which is the same service the quality-bar sweep performed by coming back
negative, only cheaper.

WHICH CARDS GET UPGRADED
------------------------
Non-basic first, matching `_pick_card_select_indexes`, whose upgrade path is
"basics last, never a curse" -- a deck lists Strike, Strike, Strike, so taking
the first card put every rest-site upgrade into a basic Strike. Simulating
uniform-random upgrades would measure a policy we do not run.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from deck_or_play import ACT1_BOSSES, _live_boss_decks  # noqa: E402

BASIC = {"STRIKE_IRONCLAD", "DEFEND_IRONCLAD"}


def _with_upgrades(deck: list[dict], extra: int) -> list[dict]:
    """`extra` more upgraded cards, non-basics first, as the smith policy does."""
    out = [dict(c) for c in deck]
    order = sorted(
        range(len(out)),
        key=lambda i: (out[i]["id"] in BASIC, bool(out[i].get("upgraded")), i),
    )
    given = 0
    for i in order:
        if given >= extra:
            break
        if not out[i].get("upgraded"):
            out[i]["upgraded"] = True
            given += 1
    return out


def _fight(job) -> dict:
    entry, boss, trial, extra, hp_pct, max_nodes, time_budget = job

    sys.path.insert(0, str(REPO))
    import sts2_env.cards  # noqa: F401
    import sts2_env.powers  # noqa: F401
    from sts2_env.gym_env.action_space import apply_combat_action, get_action_mask
    from sts2_env.search.situation import CardRef, CombatSituation
    from sts2_env.search.turn_search import SearchAgent

    deck = _with_upgrades(entry["deck"], extra)
    situation = CombatSituation(
        situation_id=f"{boss}-{extra}-{hp_pct}-{trial}",
        character_id="Ironclad",
        current_hp=max(1, round(entry["max_hp"] * hp_pct / 100)),
        max_hp=entry["max_hp"],
        deck=tuple(CardRef(card_id=c["id"], upgraded=bool(c.get("upgraded")))
                   for c in deck),
        encounter=boss,
        encounter_seed=1000 + trial,
        combat_seed=2000 + trial,
        relics=tuple(entry["relics"]),
        room_type="BOSS",
        act_floor=17,
        total_floor=17,
    )
    try:
        combat = situation.to_combat()
        agent = SearchAgent(max_nodes=max_nodes, time_budget=time_budget,
                            lookahead_turns=2)
        for _ in range(400):
            if combat.is_over:
                break
            mask = get_action_mask(combat)
            action = agent.act(combat)
            if action >= len(mask) or not mask[action]:
                break
            if not apply_combat_action(combat, action):
                break
        won = combat.is_over and not combat.player.is_dead
    except Exception as exc:
        # One unclonable position must not take down a 2000-fight grid. The
        # previous harness died at 841/1020 on `CloneError: a turn-setup
        # callback is pending`, losing the rest of the run.
        return {"extra": extra, "hp_pct": hp_pct, "boss": boss,
                "won": False, "error": type(exc).__name__}
    return {"extra": extra, "hp_pct": hp_pct, "boss": boss, "won": bool(won),
            "error": None}


def _pct(k: int, n: int) -> str:
    if not n:
        return "  n/a"
    p = k / n
    return f"{100 * p:4.0f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-decks", type=int, default=30)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-nodes", type=int, default=6000)
    ap.add_argument("--time-budget", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="output/upgrades_vs_hp.jsonl")
    args = ap.parse_args()

    upgrades = (0, 2, 4, 6)
    hp_levels = (80, 65, 50)

    decks = _live_boss_decks(15, 17)[:args.max_decks]
    jobs = [(d, boss, t, extra, hp, args.max_nodes, args.time_budget)
            for d in decks for boss in ACT1_BOSSES
            for extra in upgrades for hp in hp_levels
            for t in range(args.trials)]
    print(f"{len(decks)} live decks x {len(ACT1_BOSSES)} bosses x "
          f"{len(upgrades)} upgrade levels x {len(hp_levels)} hp levels "
          f"= {len(jobs)} fights, {args.workers} workers", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with out.open("a", encoding="utf-8") as fh, mp.Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_fight, jobs, chunksize=1), 1):
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    grid: dict[tuple, list] = collections.defaultdict(list)
    for r in rows:
        grid[(r["extra"], r["hp_pct"])].append(r["won"])

    print()
    print("=" * 62)
    print("BOSS WIN RATE: extra upgrades (down) against HP on arrival (across)")
    print()
    print(f"  {'upgrades':<12}" + "".join(f"{h}% HP".rjust(10) for h in hp_levels))
    print("  " + "-" * 46)
    for extra in upgrades:
        cells = []
        for hp in hp_levels:
            v = grid.get((extra, hp), [])
            cells.append(_pct(sum(v), len(v)).rjust(10))
        label = f"+{extra}" + (" (live)" if extra == 0 else "")
        print(f"  {label:<12}" + "".join(cells))
    print()
    errs = sum(1 for r in rows if r.get("error"))
    if errs:
        kinds = collections.Counter(r["error"] for r in rows if r.get("error"))
        print(f"  {errs} fights errored and are counted as losses: {dict(kinds)}")
    print()
    print("  live today sits at +0 upgrades and ~80% HP.")
    print("  the rest-site fix moves RIGHT (less HP) and DOWN (more upgrades).")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
