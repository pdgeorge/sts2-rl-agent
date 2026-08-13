"""At a shop, is a removal worth more than a relic?

    python scripts/removal_vs_relic.py --workers 6

THE DECISION THIS INFORMS
-------------------------
`_pick_shop_action` buys removal ONLY to delete a curse, and ranks it LAST,
behind relics, cards and potions. The reason is written at the call site and is
a capability gap rather than a preference: "there is no way to tell from it
whether that Defend was the deck's only block." `rank_cards` could tell it, so
the question is whether it is worth wiring up.

Two candidate policies, and they need comparing in the same units:

  REMOVE   thin the deck by its worst card (a basic, in a 43%-basic deck)
  RELIC    spend the same gold on the relic instead

"Weigh relic against thinning" is genuinely high-tier play when done by feel. It
stops being complicated the moment both sides are priced in one currency, which
is what this measures. If a removal is worth 6 points and a marginal relic 4,
the shop priority is arithmetic and `SHOP_PURCHASE_ACTION_PRIORITY` just gets
reordered.

WHY THE EXISTING EVIDENCE DOES NOT ALREADY ANSWER IT
----------------------------------------------------
`deck_or_play.py` found live and offline decks winning identically
relic-for-relic, and that has been read as "deck composition is not the
problem". It does not say that. It says composition does not explain the
offline/live DIFFERENCE -- both sides carry exactly 9.0 basics and both are
mediocre. Improving both is still open, and removal is the only lever that
touches the nine basics, which neither skipping nor upgrading can.

So the prior here is genuinely open, unlike the rest-site question, where the
30%-of-max-HP heal made the answer arithmetic before the sweep.

WHAT IT MEASURES
----------------
Real live boss decks, crossed:

  removals   0 / 1 / 2 / 3 basics deleted, worst first (Strike before Defend)
  relics     as carried, and with one taken away

The relic column gives the MARGINAL value of a relic, which is the number the
shop decision actually needs. The 22-point figure from `deck_or_play.py` is the
value of ALL 4.7 relics at once and must not be divided by 4.7 and used here --
relic value is not linear, and the shop is choosing one.

HP IS HELD AT 80%
-----------------
Live's mean at boss entry. Removal does not trade against HP the way smithing
does -- it is bought with gold -- so there is no reason to sweep HP here, and
holding it fixed keeps the grid small enough to be conclusive per cell.
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

#: Removal order for a deck that is 43% basics. Strike before Defend because a
#: deck that cannot block dies faster than one that cannot kill, and the whole
#: risk in loosening the shop rule is deleting the last Defend.
REMOVAL_ORDER = ("STRIKE_IRONCLAD", "DEFEND_IRONCLAD")

#: Never thin below this. A deck with two Defends left is not a lean deck, it is
#: a broken one, and measuring removals past this point would price a policy
#: nobody would ship.
MIN_KEEP_PER_BASIC = 2


def _with_removals(deck: list[dict], count: int) -> list[dict]:
    """`count` worst cards deleted, unupgraded basics first, floor-protected."""
    out = [dict(c) for c in deck]
    for _ in range(count):
        victim = None
        for basic in REMOVAL_ORDER:
            same = [i for i, c in enumerate(out) if c["id"] == basic]
            if len(same) <= MIN_KEEP_PER_BASIC:
                continue
            # An unupgraded copy first: deleting a Strike+ throws away a smith.
            plain = [i for i in same if not out[i].get("upgraded")]
            victim = (plain or same)[0]
            break
        if victim is None:
            break
        out.pop(victim)
    return out


def _fight(job) -> dict:
    entry, boss, trial, removals, drop_relic, max_nodes = job

    sys.path.insert(0, str(REPO))
    import sts2_env.cards  # noqa: F401
    import sts2_env.powers  # noqa: F401
    from sts2_env.gym_env.action_space import apply_combat_action, get_action_mask
    from sts2_env.search.situation import CardRef, CombatSituation
    from sts2_env.search.turn_search import SearchAgent

    deck = _with_removals(entry["deck"], removals)
    relics = list(entry["relics"])
    if drop_relic and relics:
        relics = relics[:-1]

    situation = CombatSituation(
        situation_id=f"{boss}-{removals}-{drop_relic}-{trial}",
        character_id="Ironclad",
        current_hp=max(1, round(entry["max_hp"] * 0.80)),
        max_hp=entry["max_hp"],
        deck=tuple(CardRef(card_id=c["id"], upgraded=bool(c.get("upgraded")))
                   for c in deck),
        encounter=boss,
        encounter_seed=1000 + trial,
        combat_seed=2000 + trial,
        relics=tuple(relics),
        room_type="BOSS",
        act_floor=17,
        total_floor=17,
    )
    try:
        combat = situation.to_combat()
        agent = SearchAgent(max_nodes=max_nodes, time_budget=3.0,
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
        # CloneError has taken down two grids at 841/1020 and 201/360. One
        # unclonable position must not cost the other 2000 fights.
        return {"removals": removals, "drop_relic": drop_relic, "boss": boss,
                "won": False, "size": len(deck), "error": type(exc).__name__}
    return {"removals": removals, "drop_relic": drop_relic, "boss": boss,
            "won": bool(won), "size": len(deck), "error": None}


def _cell(v: list[bool]) -> str:
    if not v:
        return "n/a".rjust(13)
    p = sum(v) / len(v)
    return f"{100 * p:.0f}% +/-{100 * math.sqrt(p * (1 - p) / len(v)):.0f}".rjust(13)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-decks", type=int, default=30)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--max-nodes", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="output/removal_vs_relic.jsonl")
    args = ap.parse_args()

    removals = (0, 1, 2, 3)
    decks = _live_boss_decks(15, 17)[:args.max_decks]
    jobs = [(d, boss, t, r, drop, args.max_nodes)
            for d in decks for boss in ACT1_BOSSES
            for r in removals for drop in (False, True)
            for t in range(args.trials)]
    print(f"{len(decks)} live decks x {len(ACT1_BOSSES)} bosses x "
          f"{len(removals)} removal levels x 2 relic levels x {args.trials} "
          f"trials = {len(jobs)} fights, {args.workers} workers", flush=True)

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
    size: dict[int, list] = collections.defaultdict(list)
    for r in rows:
        grid[(r["removals"], r["drop_relic"])].append(r["won"])
        size[r["removals"]].append(r["size"])

    print()
    print("=" * 70)
    print("BOSS WIN: basics removed (down) x relics carried (across)")
    print()
    print(f"  {'removed':<10}{'deck':>7}{'all relics':>15}{'one fewer':>15}")
    print("  " + "-" * 47)
    for r in removals:
        mean_size = sum(size[r]) / len(size[r]) if size[r] else 0
        print(f"  {('-' + str(r)) if r else '0 (live)':<10}{mean_size:>7.1f}"
              f"{_cell(grid.get((r, False), []))}{_cell(grid.get((r, True), []))}")
    print()
    base = grid.get((0, False), [])
    if base:
        b = sum(base) / len(base)
        one = grid.get((1, False), [])
        rel = grid.get((0, True), [])
        if one:
            print(f"  one removal is worth {100 * (sum(one) / len(one) - b):+.0f} points")
        if rel:
            print(f"  one marginal relic is worth "
                  f"{100 * (b - sum(rel) / len(rel)):+.0f} points")
        print()
        print("  compare: one UPGRADE is worth ~5 points (upgrades_vs_hp.py),")
        print("           1% of max HP is worth ~1 point.")
    errs = collections.Counter(r["error"] for r in rows if r.get("error"))
    if errs:
        print(f"\n  errored and counted as losses: {dict(errs)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
