"""What a card is worth to Cyra, estimated from the offers she was given.

    .venv/bin/python scripts/analyse_card_offers.py --tags wednesday tuesday
    .venv/bin/python scripts/analyse_card_offers.py --tags wednesday --min-offers 20

THE DESIGN, AND WHY THE OBVIOUS VERSION IS WRONG
------------------------------------------------
The obvious analysis is "runs where card X was taken did better than runs where
it was declined". It does not work here, and the reason is measured rather than
suspected: across 1,866 scored card rewards the agent took its own ranker's
top-scored card **1,866 times**. Picking is deterministic given the scores.

So "declined" never means "she passed on it". It means "something else in that
same offer scored higher" -- and that is exactly a statement about the OTHER
cards. X is taken when the co-offers were weak and declined when they were
strong, so the declined group is systematically the group that got a better
card instead. Comparing them directly measures the strength of the alternative,
not the value of X, and it biases every card the same way.

Two estimators are reported instead, and they fail differently:

  ITT (offered vs never offered, INSIDE A FIXED FLOOR WINDOW)
      What the game rolled is random, so whether X was offered is unconfounded
      by the ranker. The window is not decoration: the first version of this
      compared "ever offered" across whole runs and reported 60 of 77 cards as
      significant, every one of them positive. That is survivorship, not card
      value -- a run that dies on floor 5 sees two card rewards and a run that
      clears sees eight, so "was offered X" was largely measuring "lived long
      enough to be offered anything". Restricting to offers below a fixed floor
      AND to runs that reached that floor gives every run in the sample a
      comparable number of chances. The residual difference in chances is
      printed rather than assumed away.

  TAKEN vs DECLINED, with the offer strength reported alongside
      Higher power, and biased. The bias is not left implicit: the mean score of
      the card actually taken is printed for both groups, so the size and the
      direction of the confound are visible in the same table. If the taken and
      declined groups have similar winning scores, the bias is small for that
      card; if they do not, the comparison is not to be believed.

WHAT THIS IS FOR
----------------
It is the correction term for a card-rating prior, and the validation of one.
The scorer that currently ranks cards is rarity + output-per-energy + cheapness,
which gives OFFERING, INFLAME and DEMON_FORM the identical score of 4.200 and
WHIRLWIND, SWORD_BOOMERANG and POMMEL_STRIKE an identical 1.800. A prior can
break those ties; this file says whether a given prior's ordering agrees with
what actually happens in our runs.

READ THE INTERVALS BEFORE THE RANKING. At 167 runs and a ~39% clear rate, a
card seen in 50 decks resolves its clear rate to about +/-14 points. Almost
nothing will be significant, and that is itself the finding: it sets the number
of runs needed before a learned adjustment may touch a live policy.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ACT1_BOSS_FLOOR = 17


def _load(tags: list[str]) -> tuple[dict, list[dict]]:
    """(runs by key, one row per card-reward decision).

    Keyed on (tag, session, run). NOT on the run index: `--restart-on-crash`
    starts a new session numbering from 1, so a single journal holds several
    runs under the same index and keying on it welds them together.
    """
    runs: dict[tuple, dict] = collections.defaultdict(
        lambda: {"cleared": False, "floor": 0, "boss_hp": None})
    decisions: list[dict] = []

    for tag in tags:
        path = REPO / f"output/live_journal_{tag}.jsonl"
        if not path.exists():
            print(f"  (no journal for {tag}, skipped)")
            continue
        for line in path.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("run") is None:
                continue
            key = (tag, row.get("session"), row["run"])
            event = row.get("event")

            if event == "run_end":
                runs[key]["floor"] = int(row.get("floor") or 0)
                if row.get("act_cleared"):
                    runs[key]["cleared"] = True
            elif event == "act_clear" and row.get("act_from") == 1:
                runs[key]["cleared"] = True
            elif event == "combat_start" and row.get("room_type") == "Boss" \
                    and (row.get("floor") or 99) <= ACT1_BOSS_FLOOR + 1:
                runs[key]["boss_hp"] = row.get("hp")
            elif event == "card_reward_options":
                options = row.get("options") or []
                if not options:
                    continue
                chosen = row.get("chosen")
                taken = next((o for o in options
                              if o.get("index") == chosen), None)
                decisions.append({
                    "key": key,
                    "floor": row.get("floor"),
                    "skipped": bool(row.get("skipped")),
                    "offered": [str(o.get("card")) for o in options],
                    "taken": str(taken.get("card")) if taken else None,
                    # The score of the card that WON the offer. This is the
                    # confound made visible: a card is declined precisely when
                    # this number is high.
                    "winning_score": (taken or {}).get("score"),
                })
    return dict(runs), decisions


def _wilson(k: int, n: int) -> tuple[float, float]:
    """(rate, half-width) in percent. Wilson, so a 0/5 does not read as 0+/-0."""
    if not n:
        return 0.0, 0.0
    p = k / n
    return 100 * p, 100 * 1.96 * math.sqrt(p * (1 - p) / n)


def _two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float:
    if not n1 or not n2:
        return 0.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--min-offers", type=int, default=15,
                    help="cards seen fewer times than this are not reported")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--window", type=int, default=9,
                    help="only offers on floors below this, and only runs that "
                         "reached it -- this is what controls survivorship")
    args = ap.parse_args()

    runs, decisions = _load(args.tags)
    if not decisions:
        raise SystemExit("no card-reward decisions found in those journals")

    # Survivorship control. Every run in the sample got a comparable number of
    # chances to be offered anything, because every one of them reached the
    # same floor, and only offers from before that floor count.
    window = args.window
    reached = {k for k, v in runs.items() if (v.get("floor") or 0) >= window}
    decisions = [d for d in decisions
                 if d["key"] in reached and (d["floor"] or 99) < window]
    runs = {k: v for k, v in runs.items() if k in reached}
    if not decisions:
        raise SystemExit(f"no card rewards below floor {window} in runs that reached it")

    cleared = {k: v["cleared"] for k, v in runs.items()}
    n_runs = len(runs)
    base_k = sum(1 for v in cleared.values() if v)
    base_rate, base_half = _wilson(base_k, n_runs)

    print(f"window: floors < {window}, restricted to the {n_runs} runs that "
          f"reached floor {window}")
    print(f"{len(decisions)} card rewards inside the window, "
          f"baseline act 1 clear {base_rate:.1f}% +/-{base_half:.1f}\n")

    # -- per card ----------------------------------------------------------
    offered_in: dict[str, set] = collections.defaultdict(set)
    taken_in: dict[str, set] = collections.defaultdict(set)
    win_score_taken: dict[str, list] = collections.defaultdict(list)
    win_score_declined: dict[str, list] = collections.defaultdict(list)
    offers_seen: collections.Counter = collections.Counter()
    chances: collections.Counter = collections.Counter()

    for d in decisions:
        chances[d["key"]] += 1
    for d in decisions:
        for card in set(d["offered"]):
            offers_seen[card] += 1
            offered_in[card].add(d["key"])
            if d["winning_score"] is not None:
                if card == d["taken"]:
                    win_score_taken[card].append(d["winning_score"])
                else:
                    win_score_declined[card].append(d["winning_score"])
        if d["taken"]:
            taken_in[d["taken"]].add(d["key"])

    all_keys = set(runs)
    rows = []
    for card, seen in offers_seen.items():
        if seen < args.min_offers:
            continue
        # ITT: offer is a game roll, so this comparison is unconfounded.
        off = offered_in[card] & all_keys
        never = all_keys - off
        k_off = sum(1 for r in off if cleared.get(r))
        k_never = sum(1 for r in never if cleared.get(r))
        itt_rate, itt_half = _wilson(k_off, len(off))
        itt_z = _two_proportion_z(k_off, len(off), k_never, len(never))
        # Residual exposure: if the offered group still saw more rewards than
        # the never-offered group, the window has not fully controlled it and
        # the z is still partly survivorship.
        ex_off = statistics.mean([chances[r] for r in off]) if off else 0.0
        ex_never = statistics.mean([chances[r] for r in never]) if never else 0.0

        # Taken vs declined -- higher power, confounded, bias printed alongside.
        tk = taken_in[card] & all_keys
        dc = off - tk
        k_tk = sum(1 for r in tk if cleared.get(r))
        k_dc = sum(1 for r in dc if cleared.get(r))
        tk_rate, tk_half = _wilson(k_tk, len(tk))
        dc_rate, _ = _wilson(k_dc, len(dc))
        td_z = _two_proportion_z(k_tk, len(tk), k_dc, len(dc))

        bias = None
        if win_score_taken[card] and win_score_declined[card]:
            bias = (statistics.mean(win_score_declined[card])
                    - statistics.mean(win_score_taken[card]))

        rows.append({
            "card": card, "seen": seen,
            "n_off": len(off), "itt": itt_rate, "itt_half": itt_half, "itt_z": itt_z,
            "ex_gap": ex_off - ex_never,
            "n_tk": len(tk), "tk": tk_rate, "tk_half": tk_half,
            "n_dc": len(dc), "dc": dc_rate, "td_z": td_z, "bias": bias,
        })

    rows.sort(key=lambda r: -r["itt_z"])

    print("ITT -- runs where the card was EVER OFFERED vs never. The offer is a")
    print("game roll, so this is the unconfounded estimate. Diluted by the take rate.")
    print(f"  {'card':<26}{'offers':>7}{'runs':>6}{'clear':>9}{'+/-':>7}{'z':>7}{'exp gap':>9}")
    for r in rows[:args.top]:
        print(f"  {r['card']:<26}{r['seen']:>7}{r['n_off']:>6}"
              f"{r['itt']:>8.1f}%{r['itt_half']:>7.1f}{r['itt_z']:>7.2f}"
              f"{r['ex_gap']:>+9.2f}")

    print("\nTAKEN vs DECLINED -- more power, and biased. `bias` is how much")
    print("stronger the winning card was when this one was DECLINED; a large")
    print("positive number means the declined group simply got a better card,")
    print("and the comparison for that row is not to be believed.")
    print(f"  {'card':<26}{'n taken':>8}{'clear':>8}{'n decl':>8}{'clear':>8}{'z':>7}{'bias':>8}")
    for r in sorted(rows, key=lambda r: -r["td_z"])[:args.top]:
        bias = f"{r['bias']:+.2f}" if r["bias"] is not None else "  --"
        print(f"  {r['card']:<26}{r['n_tk']:>8}{r['tk']:>7.1f}%"
              f"{r['n_dc']:>8}{r['dc']:>7.1f}%{r['td_z']:>7.2f}{bias:>8}")

    # -- what would it take to resolve anything? ---------------------------
    print("\nRESOLUTION -- what these n actually buy")
    survivors = [r for r in rows if abs(r["itt_z"]) >= 1.96]
    print(f"  cards reported                      : {len(rows)}")
    print(f"  significant on ITT at 95%           : {len(survivors)}"
          f"  (expected by chance alone: {0.05 * len(rows):.1f})")
    if rows:
        med = statistics.median(r["itt_half"] for r in rows)
        print(f"  median 95% half-width on ITT clear  : +/-{med:.1f} points")
        p = base_k / n_runs if n_runs else 0.39
        for effect in (5, 10):
            need = math.ceil(2 * (1.96 + 0.84) ** 2 * p * (1 - p) / (effect / 100) ** 2)
            print(f"  runs-with-the-card needed for {effect:>2}pts : ~{need}"
                  f" per arm (80% power)")
    print("\n  A card significant here at n=167 is a lead to TEST, not a finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
