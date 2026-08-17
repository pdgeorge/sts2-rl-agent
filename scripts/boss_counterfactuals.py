"""The boss fights she LOST, replayed under counterfactuals: what would have won?

    python scripts/boss_counterfactuals.py --capture output/bridge_boss_fights_<tag>.jsonl \
        --journal output/live_journal_<tag>.jsonl

THE QUESTION, WHICH IS pd's
---------------------------
Take the positions where a run reached the act 1 boss and lost it. She is above
50% HP and her deck was good enough to get her there. So: what could have been
done differently? Each arm below is one answer, and the arm that converts a
position names the lever that position needed.

This is the only measurement that BOUNDS how much is available from better play.
`SCOREBOARD.md` records that at >=90% of max HP on arrival she still loses 42%
of act 1 bosses, and that perfect HP economy therefore caps clear at about 39%.
What it cannot say is whether those 42% were losable at all. If a twenty-times
compute budget converts them, the ceiling is play and it is reachable. If
nothing converts them, the lever is upstream -- deck, relics, arrival HP -- and
no amount of searching the fight will find it.

WHY ARMS AND NOT A SINGLE REPLAY
--------------------------------
`replay_boss_seed_sweep.py` already answers "does the shipped agent win this
position", across reshuffles, which is the right unit. This asks the next
question: *which change* wins it. Same positions, same reshuffle seeds, several
configurations -- so a difference between arms is the configuration and nothing
else.

PAIRED ON THE DEAL, WHICH IS THE WHOLE DISCIPLINE
-------------------------------------------------
The bridge does not send draw pile ORDER (`docs/` open item: the shuffle stream
is run-level, so the exact deal is unrecoverable). Every replay therefore
reshuffles, and one replay of one position is a coin toss dressed as a result.

So each position is played over N reshuffles, and **every arm sees the same N
deals**. The unit of comparison is a win RATE per position, and the comparison
is within-position, within-deal. Run-to-run variance in this game is enormous
and pairing removes the part of it that comes from the shuffle rather than from
the arm.

Offline is a known-optimistic instrument on boss questions -- `PHASE_TWO.md`
Track A puts the live/offline gap at 45 points. That does not invalidate this:
the ABSOLUTE rates here are suspect and the DIFFERENCES between arms on
identical positions are not, which is the same footing every paired sweep in
this repo stands on. Read the arm deltas, not the arm rates.

WHAT THIS CANNOT TELL YOU
-------------------------
Whether the live agent would have found the winning line in 3 seconds on the
real bridge. It tells you the line exists and what it costs to find. Those are
different claims and only the first is measured here.
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
sys.path.insert(0, str(REPO))

#: A prime stride, so the reshuffle seeds an arm sees are spread rather than
#: adjacent -- adjacent seeds on a small RNG can correlate, and a sweep whose
#: "independent" deals are neighbours measures its own stride.
SEED_STRIDE = 1_000_003

#: Turn cap per replay. A boss fight is ~10 turns live; 60 is well past any real
#: fight and stops a stalling configuration from running forever.
MAX_TURNS = 60


def _wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    z, p, d = 1.96, k / n, 1 + 1.96 ** 2 / n
    c = (p + 1.96 ** 2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + 1.96 ** 2 / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


# ---------------------------------------------------------------------------
# The arms. Each is (name, mutate_state, agent_kwargs, note).
#
# `mutate_state` edits the captured bridge state -- the POSITION she arrived in.
# `agent_kwargs` edits the searcher -- how she PLAYS it. Keeping the two kinds
# separate is the point of the whole script: one says "she needed a better
# arrival", the other says "she needed to play it better", and a grid that mixed
# them could not tell you which.
# ---------------------------------------------------------------------------

def _bump_hp(delta: int):
    def mutate(state: dict) -> dict:
        player = dict(state.get("player") or {})
        cap = player.get("max_hp") or state.get("run_max_hp") or 80
        player["hp"] = max(1, min(cap, (player.get("hp") or 0) + delta))
        state["player"] = player
        state["run_hp"] = player["hp"]
        return state
    return mutate


def _drop_basics(count: int):
    """Remove up to `count` unupgraded Strike/Defend, Strike first.

    The same order `SHOP_PURCHASE_ACTION_PRIORITY` removes in, so the arm
    prices the removal the shipped policy would actually have bought rather
    than an idealised one. `PHASE_TWO.md` Track C measured a removal at ~3.3
    boss-win points on a grid; this checks it against real lost positions.
    """
    def mutate(state: dict) -> dict:
        deck = list(state.get("deck") or [])
        for target in ("STRIKE_IRONCLAD", "DEFEND_IRONCLAD"):
            i = 0
            while i < len(deck) and count > 0:
                card = deck[i]
                same = card.get("id") == target and not card.get("upgraded")
                remaining = sum(1 for c in deck if c.get("id") == target)
                if same and remaining > 2:
                    deck.pop(i)
                    state["_removed"] = state.get("_removed", 0) + 1
                    if state["_removed"] >= count:
                        state["deck"] = deck
                        return state
                    continue
                i += 1
        state["deck"] = deck
        return state
    return mutate


ARMS: list[tuple[str, object, dict, str]] = [
    ("baseline", None, {}, "the shipped agent at live settings"),
    # -- how she PLAYS it -------------------------------------------------
    ("think_10x", None, {"time_budget": 30.0, "max_nodes": 200_000},
     "same position, ten times the thinking. Converts => the line was there"),
    ("lookahead_4", None, {"lookahead_turns": 4},
     "twice the horizon. MODELS.md says 4 scored WORSE than 2; retested here"),
    ("rollouts_on", None, {"top_k": 5},
     "DEFAULT_TOP_K is 0 after a null; a boss race is where it should pay"),
    ("no_potions", None, {"include_potions": False},
     "control: how much of the current rate is the potion rules"),
    # -- what she ARRIVED with --------------------------------------------
    ("hp_plus_15", _bump_hp(15), {}, "the chip-damage lever, priced on real losses"),
    ("minus_2_basics", _drop_basics(2), {}, "the removal lever, same"),
]


def _load_positions(capture: Path, journal: Path | None) -> list[dict]:
    """Captured act 1 boss arrivals, keyed per fight, newest state per fight.

    The FIRST state of a fight is the arrival, which is what a counterfactual
    has to start from -- a state from turn six has already spent the resources
    the counterfactual is about.
    """
    first: dict[object, dict] = {}
    with capture.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                state = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(state, dict) or state.get("type") != "combat_action":
                continue
            if state.get("room_type") != "Boss":
                continue
            key = state.get("encounter_seed")
            if key is not None and key not in first:
                first[key] = state
    positions = list(first.values())

    if journal is not None:
        lost = _lost_fights(journal)
        tagged = []
        for state in positions:
            enemies = tuple(sorted(
                (e.get("id") or "") for e in state.get("enemies") or []))
            state["_live_lost"] = lost.get(enemies)
            tagged.append(state)
        positions = tagged
    return positions


def _lost_fights(journal: Path) -> dict[tuple, bool]:
    """Per boss lineup: did the live agent lose it more often than not?

    Deliberately coarse. Matching a captured fight to its journal row exactly
    needs a key the capture and the journal share and they do not have one, so
    this is a lineup-level hint for the report and nothing is filtered on it.
    """
    out: dict[tuple, list[bool]] = {}
    with journal.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "combat_end" or rec.get("room_type") != "Boss":
                continue
            enemies = tuple(sorted(
                (e.get("id") or "") for e in rec.get("enemies") or []))
            out.setdefault(enemies, []).append((rec.get("hp_after") or 0) > 0)
    return {k: (sum(v) / len(v)) < 0.5 for k, v in out.items() if v}


def _play_job(job):
    """Module-level so it can be pickled to a worker."""
    state, seed, agent_kwargs, tag = job
    return tag, _play(state, seed, agent_kwargs)


def _play(state: dict, seed: int, agent_kwargs: dict) -> bool | None:
    """One replay. True/False for won/lost, None if the position would not build."""
    from sts2_env.gym_env.action_space import apply_combat_action, get_action_mask
    from sts2_env.search.situation import CombatSituation
    from sts2_env.search.turn_search import SearchAgent

    variant = json.loads(json.dumps(state))  # deep copy; arms must not leak
    variant["combat_seed"] = int(
        variant.get("combat_seed") or variant.get("encounter_seed") or 0
    ) + SEED_STRIDE * seed
    try:
        combat = CombatSituation.from_bridge_state(variant).to_combat()
    except Exception:
        return None

    kwargs = {"max_nodes": 20_000, "time_budget": 3.0, "lookahead_turns": 2}
    kwargs.update(agent_kwargs)
    agent = SearchAgent(**kwargs)

    rejected = 0
    while not combat.is_over and combat.turn_count < MAX_TURNS:
        mask = get_action_mask(combat)
        try:
            action = agent.act(combat)
        except Exception:
            return None
        if action >= len(mask) or not mask[action]:
            break
        if not apply_combat_action(combat, action):
            rejected += 1
            if rejected > 4:
                break
            continue
    return bool(combat.player.current_hp > 0 and combat.is_over
                and getattr(combat, "player_won", False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--journal", default=None,
                    help="live journal, to mark which lineups she lost live")
    ap.add_argument("--seeds", type=int, default=12,
                    help="reshuffles per position per arm; every arm sees the same ones")
    ap.add_argument("--limit", type=int, default=0, help="cap positions, for a smoke test")
    ap.add_argument("--arms", default="",
                    help="comma-separated arm names; default is all of them")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel replays. DEFAULT 1, on purpose: the live "
                         "searcher is TIME-budgeted (measured on a real boss "
                         "turn: 609 of ~1447 nodes in its 3s), so saturating "
                         "the cores while a session is running makes the agent "
                         "under-search and changes the very number the session "
                         "is measuring. Raise it only when nothing is playing.")
    args = ap.parse_args()

    import sts2_env.cards  # noqa: F401
    import sts2_env.powers  # noqa: F401

    positions = _load_positions(Path(args.capture),
                                Path(args.journal) if args.journal else None)
    if args.limit:
        positions = positions[:args.limit]
    if not positions:
        print("no captured act 1 boss arrivals in that file -- nothing to replay.")
        print("if the session crashed, check the capture survived the restarts.")
        return 1

    chosen = [a for a in ARMS if not args.arms or a[0] in args.arms.split(",")]
    total = len(positions) * len(chosen) * args.seeds
    print(f"{len(positions)} captured boss arrivals x {len(chosen)} arms "
          f"x {args.seeds} reshuffles = {total} replays\n")

    # Every (position, arm, deal) up front, so the pool sees one flat work list
    # and every arm gets the SAME deals -- the pairing is in the seed, not in
    # the scheduling.
    jobs = []
    for p_index, state in enumerate(positions):
        for name, mutate, kwargs, _note in chosen:
            arm_state = json.loads(json.dumps(state))
            if mutate is not None:
                arm_state = mutate(arm_state)
            for seed in range(args.seeds):
                jobs.append((arm_state, seed, kwargs, (name, p_index)))

    raw: dict[tuple, list] = {}
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for i, (tag, outcome) in enumerate(
                    pool.imap_unordered(_play_job, jobs, chunksize=1), 1):
                raw.setdefault(tag, []).append(outcome)
                if i % 25 == 0:
                    print(f"  {i}/{len(jobs)} replays", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            tag, outcome = _play_job(job)
            raw.setdefault(tag, []).append(outcome)
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)} replays", flush=True)

    results: dict[str, dict[int, list[bool]]] = {a[0]: {} for a in chosen}
    for (name, p_index), outcomes in raw.items():
        results[name][p_index] = [o for o in outcomes if o is not None]

    for p_index, state in enumerate(positions):
        lineup = "+".join(sorted((e.get("id") or "?") for e in state.get("enemies") or []))
        hp = (state.get("player") or {}).get("hp")
        print(f"\n[{p_index + 1}/{len(positions)}] {lineup[:44]:<46} hp {hp} "
              f"deck {len(state.get('deck') or [])}"
              f"{'   (lost live)' if state.get('_live_lost') else ''}")
        for name, _m, _k, _note in chosen:
            kept = results[name].get(p_index) or []
            print(f"      {name:<16} {sum(kept)}/{len(kept)}" if kept
                  else f"      {name:<16} unbuildable")

    print("\n" + "=" * 72)
    print("ARM SUMMARY -- read the deltas, not the rates (offline is optimistic)")
    print("=" * 72)
    base = results.get("baseline", {})
    base_rate = {i: (sum(v) / len(v)) for i, v in base.items() if v}
    for name, _m, _k, note in chosen:
        per = {i: (sum(v) / len(v)) for i, v in results[name].items() if v}
        if not per:
            continue
        k = sum(sum(v) for v in results[name].values())
        n = sum(len(v) for v in results[name].values())
        lo, hi = _wilson(k, n)
        line = f"  {name:<16} {100 * k / n:5.1f}%  ({lo:.0f}-{hi:.0f})  n={n:<5}"
        if name != "baseline" and base_rate:
            shared = [i for i in per if i in base_rate]
            diffs = [per[i] - base_rate[i] for i in shared]
            if diffs:
                mean = statistics.mean(diffs)
                se = (statistics.stdev(diffs) / math.sqrt(len(diffs))
                      if len(diffs) > 1 else 0.0)
                better = sum(1 for d in diffs if d > 0)
                worse = sum(1 for d in diffs if d < 0)
                line += (f" paired {100 * mean:+5.1f} +/- {100 * 1.96 * se:4.1f}"
                         f"  better {better} worse {worse} tied {len(diffs) - better - worse}")
        print(line + f"\n{'':<18}{note}")

    # The actual question: which positions does each arm rescue?
    print("\n" + "=" * 72)
    print("POSITIONS THE BASELINE LOSES OUTRIGHT (0 wins across every reshuffle)")
    print("=" * 72)
    dead = [i for i, v in base.items() if v and sum(v) == 0]
    if not dead:
        print("  none -- the baseline wins at least one reshuffle everywhere.")
    for i in dead:
        rescuers = [name for name, _m, _k, _n in chosen
                    if name != "baseline" and results[name].get(i)
                    and sum(results[name][i]) > 0]
        lineup = "+".join(sorted(
            (e.get("id") or "?") for e in positions[i].get("enemies") or []))
        print(f"  {lineup[:44]:<46} rescued by: "
              f"{', '.join(rescuers) if rescuers else 'NOTHING -- lost on arrival'}")
    print("\nA position nothing rescues was decided before the fight began, and it "
          "belongs to the\nreach/HP half of the funnel. One that `think_10x` rescues "
          "was a line she had and did\nnot find. Those are different bugs with "
          "different fixes, and this is the only\ninstrument that separates them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
