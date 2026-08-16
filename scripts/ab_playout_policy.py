"""Paired A/B of the lookahead's playout policy, on the act 1 combat benchmark.

    python scripts/ab_playout_policy.py --limit 120 --time-budget 1.0

WHAT THIS MEASURES AND WHY IT IS ALLOWED TO BE OFFLINE
-----------------------------------------------------
`SCOREBOARD.md` says offline results do not count as progress, and that stands:
this script does NOT report a clear rate and nothing here is a win. What it
measures is the searcher's BEHAVIOUR -- how often it plays a Power, and whether
that rate rises with the length of the fight -- and behaviour is a property of
the agent rather than of the run, so it is legitimately observable offline.

The specific claim under test, from the live 2026-08-15 session (n=100 runs,
13,251 card plays):

    fight length   plays   power   attack   skill
    1-2 turns      2416    2.32%   77.2%    20.4%
    3-4 turns      5456    3.06%   66.0%    30.2%
    5-7 turns      3247    3.60%   58.6%    36.6%
    8+ turns       2132    2.25%   54.6%    41.0%

The power-play rate is FLAT in fight length, and the agent's whole answer to a
long fight is to block more. `DEFAULT_TOP_K` in turn_search.py records the same
shape and the failed attempt to fix it with depth. If the new playout is doing
what it was written to do, the last column of this script's output rises with
fight length; if it does not, the change is not worth taking live.

Both arms run in ONE process over the same situations with the same seeds, so
the comparison is paired and the only difference is the policy.
"""

from __future__ import annotations

import argparse
import collections
import math

import numpy as np

import sts2_env.cards  # noqa: F401  (registers the card effects)
import sts2_env.powers  # noqa: F401
from sts2_env.core.constants import ACTION_END_TURN
from sts2_env.core.enums import CardType
from sts2_env.gym_env.action_space import (
    action_to_card_and_target,
    apply_combat_action,
    get_action_mask,
)
from sts2_env.search import turn_search
from sts2_env.search.situation import load_situations
from sts2_env.search.turn_search import SearchAgent

DEFAULT_BENCHMARK = "tests/fixtures/act1_combat_benchmark.json"


def _old_heuristic_playout_action(combat, actions, turns_remaining=1):
    """The policy as it stood before 2026-08-16, kept verbatim as the baseline.

    Copied rather than imported because the point is to compare against what
    actually produced every number in MODELS.md, and a shared helper would drift
    away from that over time.
    """
    need_block = turn_search._incoming_damage(combat) > combat.player.block
    best_action, best_value = None, -1.0
    for action in actions:
        hand_index, _ = action_to_card_and_target(action)
        if hand_index is None or hand_index >= len(combat.hand):
            continue
        card = combat.hand[hand_index]
        block = card.base_block or 0
        damage = card.base_damage or 0
        value = float(block if need_block and block else damage)
        if value <= 0 and turn_search._is_power_card(card):
            value = 0.5
        if value > best_value:
            best_action, best_value = action, value
    if best_action is None or best_value <= 0:
        return None
    return best_action


def _holds_a_power(situation) -> bool:
    from sts2_env.cards.factory import create_card
    from sts2_env.search.situation import resolve_card_id

    for ref in situation.deck:
        resolved = resolve_card_id(str(ref.card_id))
        if resolved is None:
            continue
        if create_card(resolved).card_type == CardType.POWER:
            return True
    return False


def _play(situation, *, time_budget: float, max_turns: int) -> dict:
    """Play one benchmark fight to the end and report what happened in it."""
    combat = situation.to_combat()
    agent = SearchAgent(time_budget=time_budget)
    start_hp = combat.player.current_hp
    plays = collections.Counter()

    turns = 0
    while not combat.is_over and turns < max_turns:
        for _ in range(turn_search.MAX_PLAYOUT_ACTIONS_PER_TURN + 4):
            if combat.is_over:
                break
            action = agent.act(combat)
            if action == ACTION_END_TURN:
                break
            hand_index, _ = action_to_card_and_target(action)
            if hand_index is not None and hand_index < len(combat.hand):
                card_type = combat.hand[hand_index].card_type
                plays[card_type.name.lower()] += 1
                plays["n"] += 1
            if not apply_combat_action(combat, action):
                break
        if combat.is_over:
            break
        combat.end_player_turn()
        turns += 1

    return {
        "turns": max(1, turns),
        "damage": max(0, start_hp - combat.player.current_hp),
        "won": bool(combat.is_over and combat.player.current_hp > 0),
        "plays": plays,
    }


def _report(label: str, fights: list[dict]) -> None:
    by_length: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for fight in fights:
        turns = fight["turns"]
        bucket = ("1-2" if turns <= 2 else "3-4" if turns <= 4
                  else "5-7" if turns <= 7 else "8+")
        by_length[bucket].update(fight["plays"])

    total_turns = sum(f["turns"] for f in fights)
    total_damage = sum(f["damage"] for f in fights)
    wins = sum(1 for f in fights if f["won"])
    print(f"\n=== {label} ===")
    # Damage per FIGHT is the number that matters -- damage per turn falls just
    # by taking longer, and the chip damage that decides a run is per fight.
    print(f"  fights {len(fights)}  won {wins} ({100*wins/len(fights):.1f}%)  "
          f"damage/fight {total_damage/len(fights):.2f}  "
          f"damage/turn {total_damage/max(1,total_turns):.2f}  "
          f"turns/fight {total_turns/len(fights):.2f}")
    print(f"  {'length':>8s} {'plays':>7s} {'power':>8s} {'attack':>8s} {'skill':>8s}")
    for bucket in ("1-2", "3-4", "5-7", "8+"):
        counter = by_length.get(bucket)
        if not counter or not counter["n"]:
            continue
        n = counter["n"]
        print(f"  {bucket:>8s} {n:7d} {100*counter['power']/n:7.2f}% "
              f"{100*counter['attack']/n:7.1f}% {100*counter['skill']/n:7.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--time-budget", type=float, default=1.0)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--powers-only", action="store_true",
                        help="Keep only situations whose deck holds a Power. "
                             "110 of the first 150 benchmark decks hold NONE, "
                             "so the full benchmark cannot test a claim about "
                             "how Powers are played -- it dilutes it to "
                             "nothing. Live decks average 1.4 Powers (138 "
                             "taken across the 100 runs of 2026-08-15), so "
                             "this subset is the closer analogue as well.")
    args = parser.parse_args()

    situations = load_situations(args.benchmark)
    if args.powers_only:
        situations = [s for s in situations if _holds_a_power(s)]
    if args.limit:
        situations = situations[:args.limit]
    print(f"{len(situations)} situations, {args.time_budget}s per turn")

    new_policy = turn_search._heuristic_playout_action
    results = {}
    for label, policy in (("OLD playout (block-then-damage, Power at 0.5)",
                           _old_heuristic_playout_action),
                          ("NEW playout (everything priced in HP)",
                           new_policy)):
        turn_search._heuristic_playout_action = policy
        results[label] = [
            _play(s, time_budget=args.time_budget, max_turns=args.max_turns)
            for s in situations
        ]
    turn_search._heuristic_playout_action = new_policy

    for label, fights in results.items():
        _report(label, fights)

    # PAIRED, because both arms played the same situations in the same order.
    # An eyeballed "3% less damage" across two runs is exactly the shape of the
    # false positives this repo keeps producing; a sign test over the pairs
    # says whether there is anything there.
    old, new = results.values()
    deltas = [n["damage"] - o["damage"] for o, n in zip(old, new)]
    better = sum(1 for d in deltas if d < 0)
    worse = sum(1 for d in deltas if d > 0)
    same = sum(1 for d in deltas if d == 0)
    decided = better + worse
    mean = sum(deltas) / len(deltas)
    # Two-sided sign test against a fair coin, normal approximation.
    if decided:
        z = (better - decided / 2) / (0.25 * decided) ** 0.5
        p = math.erfc(abs(z) / math.sqrt(2))
    else:
        z = p = float("nan")
    print(f"\n=== PAIRED, new minus old, damage per fight ===")
    print(f"  mean delta {mean:+.2f} HP   new better {better}, worse {worse}, "
          f"tied {same}")
    print(f"  sign test z={z:.2f}  p={p:.3f}"
          f"{'  -- not resolvable' if not (p < 0.05) else ''}")


if __name__ == "__main__":
    main()
