"""How good does a card have to be before it is worth adding?

    python scripts/sweep_skip_threshold.py --runs 400

WHY A QUALITY BAR AND NOT A DECK SIZE
-------------------------------------
The obvious lever looks like deck size -- act 1 boss decks were 21 and 22 cards
with nine basic Strike/Defend still in them, against 173-222 HP bosses. It is
the wrong lever. Deck size is not intrinsically bad: a cycling deck wants cards,
a strength deck does not, and a flat cap tells a cycling deck to stop building
the thing that makes it work.

Size should fall out of quality. A cycling deck clears a quality bar thirty
times and ends up with thirty cards; a strength deck clears it twelve times.
That is the same rule producing two different decks, which is what a
deckbuilding policy is supposed to do.

WHAT WE KNOW THE POLICY DOES AND DOES NOT DO
--------------------------------------------
Measured over 366 real card-reward screens from the captured protocol:

  ranking      works. DeckDirection commits on 191 of them (strike-synergy 105,
               bloodletting 29, strength 25, block-scaling 17, exhaust 15) and
               changes the top pick on 40 of those 191 -- 21%. It is steering.

  declining    has never happened. The best card on offer scored 1.00 to 5.90
               and SKIP_THRESHOLD was 0.0, so the skip never fired once. The mod
               could not click Skip either, which is fixed, but the policy still
               would not ask.

So the policy ranks correctly and cannot decline, and a perfect ranker that
cannot decline still builds a pile.

THE ARMS
--------
The observed score distribution is min 1.00, median 2.50, max 5.90. A threshold
at 2.5 declines about half of all offers, which is aggressive; 1.5 declines the
weakest. The baseline is included so "taking everything was right all along" can
win, and the deck-size rule is disabled in every arm so this measures the
quality bar alone.

SCORED ON ACT 2 AS WELL
-----------------------
A threshold tuned only on act 1 clear is exactly the change that could buy act 1
by gutting act 2 -- a thinner deck that beats one boss and starves later. Act 2
reach is reported beside it so that shows up here rather than three weeks later.
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

#: (name, QUALITY_BAR_SCALE). The bar is `100*score/scale > deck_size`, so a
#: LARGER scale is stricter: it stops the median offer (2.50) sooner.
#:
#:     scale  6  median card taken up to 41 cards
#:     scale  8  ...up to 31
#:     scale 10  ...up to 24
#:     scale 12  ...up to 20
#:
#: `take everything` is the current behaviour and is included so that "taking
#: every card was right all along" is allowed to win. It is expressed as a scale
#: so small that nothing is ever refused, rather than as a special case, so every
#: arm runs the same code path.
#:
#: THE DIRECTION IS EASY TO GET BACKWARDS AND I DID. This arm was first written
#: as 1e9 "large enough that nothing is refused", which is the strictest possible
#: value: `100*score/1e9` is ~0, so it refused EVERY card. It ran 400 seeds that
#: way before the deck size gave it away -- mean deck 10.5, the starter deck
#: untouched, and 7.0% clear against the 47% the same agent gets normally. The
#: mean deck size column is in the report precisely so an arm that is not doing
#: what its name says cannot pass unnoticed.
ARMS: list[tuple[str, float]] = [
    # Three arms, bracketing pd's stated optimum of 25-35 cards. The median
    # offer (2.50) stops the deck at 100*2.5/scale:
    #   take everything  never refuses          (today's behaviour)
    #   scale 8          stops the median at 31 (inside the stated optimum)
    #   scale 12         stops it at 20         (tighter than today's 21-22)
    #
    # The baseline is not a formality. A stricter bar does NOT remove the nine
    # basic Strike/Defend already in the deck, so a smaller deck is a MORE basic
    # deck -- 9 of 15 is 60% against 9 of 21 at 43%. Stricter could be actively
    # worse, and this is the arm that would show it.
    ("take everything", 1e-9),
    ("scale 8", 8.0),
    ("scale 12", 12.0),
]

#: Above any score the game produces, so the deck-size rule cannot fire and each
#: arm measures its quality bar alone.
DECK_SIZE_RULE_OFF = 999


def _walk(args) -> dict:
    arm_name, scale, seed, max_nodes, time_budget = args

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    import sts2_env.cards  # noqa: F401
    from sts2_env.bridge import agent_runner
    from sts2_env.gym_env.run_env import STS2RunEnv
    from sts2_env.run.run_manager import RunManager
    from sts2_env.search.turn_search import SearchAgent
    from ab_archetype_picking import _search_combat_action
    from live_policy import noncombat_action

    # Patched in the WORKER so each arm gets its own process-local value and
    # arms cannot leak into one another through a shared module. `agent_runner`
    # imported the name, so patching card_quality alone would not take.
    agent_runner.QUALITY_BAR_SCALE = scale
    agent_runner.CARD_REWARD_LARGE_DECK_SIZE = DECK_SIZE_RULE_OFF

    agent = SearchAgent(time_budget=time_budget, lookahead_turns=2,
                        max_nodes=max_nodes)
    rng = np.random.default_rng(seed)
    env = STS2RunEnv(act1_variant="random")
    env.reset(seed=seed)

    boss = None
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
            if boss is None and last and "boss" in last[0]:
                boss = last[0]
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
        "arm": arm_name, "seed": seed,
        "floor": int(getattr(rs, "total_floor", 0) or 0),
        "act": int(getattr(rs, "current_act_index", 0) or 0) + 1,
        "boss": boss,
        "deck_size": len(getattr(getattr(rs, "player", None), "deck", []) or []),
    }
    env.close()
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=60000)
    parser.add_argument("--max-nodes", type=int, default=2000)
    parser.add_argument("--time-budget", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", default="output/sweep_skip_threshold.txt")
    args = parser.parse_args()

    workers = args.workers or max(1, (mp.cpu_count() or 2) - 2)
    seeds = [args.seed + i for i in range(args.runs)]
    rows_path = Path(args.out).with_suffix(".rows.jsonl")
    rows_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, int]] = set()
    rows: list[dict] = []
    if rows_path.exists():
        with rows_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(row)
                done.add((row["arm"], row["seed"]))

    jobs = [(name, scale, s, args.max_nodes, args.time_budget)
            for name, scale in ARMS for s in seeds if (name, s) not in done]
    print(f"{len(ARMS)} arms x {args.runs} paired seeds; {len(jobs)} to run, "
          f"{workers} workers", flush=True)

    fh = rows_path.open("a", encoding="utf-8")
    started = time.monotonic()
    with mp.Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_walk, jobs, chunksize=1), 1):
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}  "
                      f"({(time.monotonic() - started) / 60:.1f} min)", flush=True)
    fh.close()

    by_arm: dict[str, dict[int, dict]] = {name: {} for name, _ in ARMS}
    for row in rows:
        if row["arm"] in by_arm:
            by_arm[row["arm"]][row["seed"]] = row

    base = by_arm[ARMS[0][0]]
    lines = ["", "=" * 82,
             f"SKIP THRESHOLD SWEEP  {args.runs} paired seeds/arm  "
             f"({(time.monotonic() - started) / 60:.1f} min this session)", "",
             "  the deck-size rule is OFF in every arm; this is the quality bar alone", "",
             f"{'arm':<18}{'clear':>8}{'act2 reach':>12}{'deck':>7}"
             f"{'   vs take-everything (paired)':>32}", "-" * 82]
    for name, _ in ARMS:
        got = by_arm[name]
        shared = [s for s in seeds if s in got and s in base]
        if not shared:
            continue
        clears = [1 if got[s]["act"] >= 2 else 0 for s in shared]
        act2 = [1 if got[s]["floor"] >= 32 else 0 for s in shared]
        decks = [got[s].get("deck_size", 0) for s in shared]
        if name == ARMS[0][0]:
            delta = ""
        else:
            paired = [(1 if got[s]["act"] >= 2 else 0)
                      - (1 if base[s]["act"] >= 2 else 0) for s in shared]
            md = statistics.mean(paired)
            se = (statistics.stdev(paired) / math.sqrt(len(paired))
                  if len(paired) > 1 else float("nan"))
            sig = md / se if se and se == se and se > 0 else 0.0
            delta = f"{100 * md:+5.1f}% +/- {100 * se:4.1f}%  ({sig:+.1f} se)"
        lines.append(
            f"{name:<18}{100 * statistics.mean(clears):>7.1f}%"
            f"{100 * statistics.mean(act2):>11.1f}%"
            f"{statistics.mean(decks):>7.1f}{delta:>32}")
    lines += ["", "=" * 82, ""]
    report = "\n".join(lines)
    print(report)
    with open(args.out, "a", encoding="utf-8") as out:
        out.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
