"""Is it the deck, or is it the play?

    python scripts/deck_or_play.py --runs-per-deck 3

THE QUESTION
------------
Live wins 29% of act 1 boss fights. Offline wins 74% of them. Same policy, same
search, same simulator. Something differs, and it is one of two things:

  THE DECK    she arrives at the boss with a worse deck live than offline
  THE PLAY    the decks are comparable and the live path plays them worse

Those call for opposite work, and no amount of staring at win rates separates
them.

THE EXPERIMENT
--------------
Take the decks LIVE actually brought to the act 1 boss -- read out of the
captured protocol, with the HP and relics she actually had -- and fight the act 1
bosses with them offline, under the same search that plays live.

  search wins ~74% with live's decks   ->  the decks are fine. The live path is
                                           losing fights the same agent wins on
                                           the same material. Look at the bridge,
                                           the reconstruction, the execution.
  search wins ~29% with live's decks   ->  the decks are the problem. Offline's
                                           advantage was built before the fight
                                           started, in what it drafted and what
                                           HP it arrived with.

Anything in between splits the difference and says how much each contributes.

WHY NOT JUST REPLAY THE CAPTURED FIGHTS
---------------------------------------
Only two act 1 boss fights are reconstructable turn-for-turn -- the capture quota
bucketed all combat together until yesterday, so bosses were crowded out by
floor-1 monsters. n=2 decides nothing. Deck snapshots are plentiful (3,262 across
all floors, 154 at floor 17) because they ride on every state, so this trades
exact position for sample size, which is the right trade at this stage.

HP IS CONTROLLED, DELIBERATELY
------------------------------
Each deck fights at a FIXED fraction of max HP, not at whatever HP its snapshot
happened to carry. A first version used the snapshot's own HP and drew decks
averaging 28% health -- snapshots taken mid-fight and near death -- so every deck
lost and the result measured nothing except that a boss beats you at 28%.

Fixing HP isolates the question. "Can this deck beat this boss from a normal
starting position" is answerable; "did this run die" is already known. The live
mean at boss entry is about 80%, which is the default, and `--hp-percent` sweeps
it to separate deck quality from arrival condition.

WHAT IT DOES NOT CONTROL
------------------------
The opening shuffle, and the exact turn the position was captured at. Each deck
is fought several times against each boss to average over the draw. This measures
"can this deck beat this boss with good play", not "would this exact turn have
gone differently".
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import multiprocessing as mp
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: The six act 1 bosses, by encounter setup name.
ACT1_BOSSES = (
    "setup_vantom_boss",
    "setup_ceremonial_beast_boss",
    "setup_the_kin_boss",
    "setup_waterfall_giant_boss",
    "setup_soul_fysh_boss",
    "setup_lagavulin_matriarch_boss",
)


def _live_boss_decks(min_floor: int, max_floor: int) -> list[dict]:
    """Decks live actually carried into the act 1 boss, with HP and relics."""
    seen: dict[tuple, dict] = {}
    patterns = ("output/bridge_protocol*.jsonl", "output/bridge_boss_fights*.jsonl")
    for pattern in patterns:
        for path in glob.glob(str(REPO / pattern)):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        state = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(state, dict):
                        continue
                    floor = state.get("floor")
                    deck = state.get("deck")
                    if not isinstance(floor, int) or not isinstance(deck, list) or not deck:
                        continue
                    if not (min_floor <= floor <= max_floor):
                        continue
                    hp = state.get("run_hp")
                    max_hp = state.get("run_max_hp")
                    if not isinstance(hp, int) or not isinstance(max_hp, int) or max_hp <= 0:
                        continue
                    # HP from the snapshot is NOT used to fight; see the module
                    # docstring. It is only required to exist so that max_hp is
                    # real, because max_hp does grow over a run and a deck should
                    # fight at the health pool it actually had.
                    cards = tuple(sorted(
                        str(c.get("id") if isinstance(c, dict) else c) for c in deck))
                    key = (cards, max_hp)
                    seen.setdefault(key, {
                        "deck": [
                            {"id": str(c.get("id") if isinstance(c, dict) else c),
                             "upgraded": bool(c.get("upgraded")) if isinstance(c, dict) else False}
                            for c in deck
                        ],
                        "hp": hp, "max_hp": max_hp,
                        "relics": [str(r) for r in (state.get("relics") or [])],
                        "floor": floor,
                    })
    return list(seen.values())


def _offline_boss_decks(path: Path) -> list[dict]:
    """Decks the OFFLINE agent carried into the act 1 boss, from the funnel."""
    seen: dict[tuple, dict] = {}
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            deck = row.get("boss_deck")
            max_hp = row.get("boss_max_hp")
            if not deck or not max_hp:
                continue
            key = (tuple(sorted(c["id"] for c in deck)), max_hp)
            seen.setdefault(key, {
                "deck": [{"id": c["id"], "upgraded": bool(c.get("upgraded"))}
                         for c in deck],
                "hp": row.get("boss_hp", 0), "max_hp": max_hp,
                "relics": [], "floor": 17,
            })
    return list(seen.values())


def _fight(job) -> dict:
    entry, boss, trial, max_nodes, time_budget, hp_percent = job

    sys.path.insert(0, str(REPO))
    import sts2_env.cards  # noqa: F401
    import sts2_env.powers  # noqa: F401
    from sts2_env.gym_env.action_space import apply_combat_action, get_action_mask
    from sts2_env.search.situation import CardRef, CombatSituation
    from sts2_env.search.turn_search import SearchAgent

    situation = CombatSituation(
        situation_id=f"{boss}-{trial}",
        character_id="Ironclad",
        current_hp=max(1, round(entry["max_hp"] * hp_percent / 100)),
        max_hp=entry["max_hp"],
        deck=tuple(CardRef(card_id=c["id"], upgraded=c["upgraded"])
                   for c in entry["deck"]),
        encounter=boss,
        # Vary the roll per trial so enemy HP and the opening draw differ, and
        # a deck is judged on several fights rather than one lucky shuffle.
        encounter_seed=1000 + trial,
        combat_seed=2000 + trial,
        relics=tuple(entry["relics"]),
        room_type="BOSS",
        act_floor=17,
        total_floor=17,
    )
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
    return {"boss": boss, "trial": trial, "won": bool(won),
            "deck_size": len(entry["deck"]), "hp": entry["hp"],
            "max_hp": entry["max_hp"], "floor": entry["floor"],
            "hp_left": combat.player.current_hp}


def _pct(k: int, n: int) -> str:
    if not n:
        return "n/a"
    p = k / n
    return f"{100 * p:.0f}% +/-{100 * math.sqrt(p * (1 - p) / n):.0f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strip-relics", action="store_true",
                    help=("Fight with no relics. Used to measure how much the "
                          "relic-less offline capture distorts a deck "
                          "comparison -- the funnel records boss_deck but not "
                          "relics, so offline decks fight bare."))
    ap.add_argument("--decks-from", default="live",
                    choices=("live", "offline"),
                    help=("live reads the captured protocol; offline reads the "
                          "funnel's boss_deck capture. Running both at the SAME "
                          "--hp-percent is the only way to separate deck quality "
                          "from arrival HP -- offline naturally arrives at 93%% "
                          "and live at 81%%, so an uncontrolled comparison "
                          "attributes an HP difference to the deck."))
    ap.add_argument("--offline-rows",
                    default="output/funnel.deckcapture.rows.jsonl")
    ap.add_argument("--min-floor", type=int, default=15)
    ap.add_argument("--max-floor", type=int, default=17)
    ap.add_argument("--runs-per-deck", type=int, default=2,
                    help="Trials per (deck, boss) pair, to average over the draw.")
    ap.add_argument("--max-decks", type=int, default=0)
    ap.add_argument("--hp-percent", type=float, default=80.0,
                    help="Fight at this %% of max HP. Live's mean at boss entry "
                         "is ~80%%. Sweep it to separate deck from arrival HP.")
    ap.add_argument("--max-nodes", type=int, default=20000,
                    help="The LIVE budget, so play quality matches live's.")
    ap.add_argument("--time-budget", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="output/deck_or_play.jsonl")
    args = ap.parse_args()

    if args.decks_from == "offline":
        decks = _offline_boss_decks(Path(args.offline_rows))
    else:
        decks = _live_boss_decks(args.min_floor, args.max_floor)
    if args.max_decks:
        decks = decks[:args.max_decks]
    if not decks:
        print("no live decks found in that floor range")
        return 1

    sizes = [len(d["deck"]) for d in decks]
    hps = [d["max_hp"] for d in decks]
    print(f"{len(decks)} distinct live decks from floors "
          f"{args.min_floor}-{args.max_floor}")
    print(f"   deck size: mean {statistics.mean(sizes):.1f}, "
          f"median {statistics.median(sizes):.0f}, "
          f"range {min(sizes)}-{max(sizes)}")
    print(f"   max hp: mean {statistics.mean(hps):.0f}; "
          f"fighting at {args.hp_percent:.0f}% of it")
    print()

    if args.strip_relics:
        for d in decks:
            d["relics"] = []
    jobs = [(d, boss, t, args.max_nodes, args.time_budget, args.hp_percent)
            for d in decks for boss in ACT1_BOSSES
            for t in range(args.runs_per_deck)]
    workers = args.workers or max(1, (mp.cpu_count() or 2) - 2)
    print(f"{len(jobs)} fights, {workers} workers", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with out.open("a", encoding="utf-8") as fh, mp.Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_fight, jobs, chunksize=1), 1):
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 100 == 0:
                won = sum(r["won"] for r in rows)
                print(f"  {i}/{len(jobs)}  running win rate "
                      f"{100 * won / len(rows):.0f}%", flush=True)

    won = sum(r["won"] for r in rows)
    print()
    print("=" * 70)
    print(f"LIVE DECKS, PLAYED OFFLINE BY THE SEARCH: {_pct(won, len(rows))}")
    print()
    print("  compare:  live boss win 29%   offline boss win 74%")
    print()
    per = collections.Counter(r["boss"] for r in rows)
    win = collections.Counter(r["boss"] for r in rows if r["won"])
    for boss, n in per.most_common():
        label = boss.replace("setup_", "").replace("_boss", "")
        print(f"  {label:<24}{win[boss]:>4}/{n:<5} {_pct(win[boss], n)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
