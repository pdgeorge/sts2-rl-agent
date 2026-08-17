"""Replay one captured boss fight turn by turn and show what the search decides.

    python scripts/trace_waterfall_fight.py --capture output/bridge_boss_fights_sunday.jsonl

The live agent lost Waterfall Giant twice on 2026-08-16 arriving at 80/80 HP
(runs 7 and 24), and the same positions replayed won by the offline search. A
position that arrives at full HP and still loses live is the cleanest possible
site for a live-path defect, because arrival economy cannot be the explanation.

This probe prints, per player turn:

  - the hand the search sees, cross-checked against the hand the game actually
    dealt at that decision point (the bridge captures a hand per decision).
    A mismatch there means the reconstructed draw order disagrees with the
    game's, which would also contaminate the paired replay comparison.
  - every card the search plays, and the enemy state it played it against.

Default fight is run 7's Waterfall Giant; --seed selects another capture.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _card_ids(cards) -> list[str]:
    out = []
    for c in cards or []:
        if isinstance(c, dict):
            out.append(str(c.get("id")))
        else:
            out.append(str(c))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", default="output/bridge_boss_fights_sunday.jsonl")
    ap.add_argument("--seed", type=int, default=None,
                    help="encounter_seed of the fight to trace; default: the "
                         "first Waterfall Giant fight in the capture")
    ap.add_argument("--max-nodes", type=int, default=20000)
    ap.add_argument("--time-budget", type=float, default=3.0)
    ap.add_argument("--max-turns", type=int, default=40)
    args = ap.parse_args()

    import sts2_env.cards  # noqa: F401
    import sts2_env.powers  # noqa: F401
    from sts2_env.core.constants import ACTION_END_TURN
    from sts2_env.gym_env.action_space import (
        action_to_card_and_target,
        apply_combat_action,
        get_action_mask,
        is_potion_action,
    )
    from sts2_env.search.situation import CombatSituation
    from sts2_env.search.turn_search import SearchAgent

    per_seed: dict[int, list[dict]] = {}
    with open(args.capture, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                state = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (isinstance(state, dict) and state.get("type") == "combat_action"
                    and state.get("room_type") == "Boss"):
                continue
            per_seed.setdefault(state.get("encounter_seed"), []).append(state)

    seed = args.seed
    if seed is None:
        for s, states in per_seed.items():
            if any(e.get("id") == "WATERFALL_GIANT"
                   for st in states for e in (st.get("enemies") or [])):
                seed = s
                break
    if seed is None or seed not in per_seed:
        print("fight not found in capture", file=sys.stderr)
        return 1

    states = per_seed[seed]
    start = states[0]
    captured_hand_by_round: dict[int, list[str]] = {}
    for st in states:
        rnd = st.get("round")
        if rnd is not None and rnd not in captured_hand_by_round:
            captured_hand_by_round[rnd] = sorted(x.removesuffix("_CARD") for x in _card_ids(st.get("hand")))

    encounter = start.get("encounter")
    print(f"tracing {encounter}, encounter_seed={seed}, "
          f"arrival {start.get('run_hp')}/{start.get('run_max_hp')} HP, "
          f"deck {len(start.get('deck') or [])}, relics {start.get('relics')}\n")

    situation = CombatSituation.from_bridge_state(start)
    # mid-fight overlay: the captured round-1 state IS the opening position --
    # hand, HP, enemies exactly as the game reported them. to_combat() alone
    # would redraw the opening hand from combat_seed, which is a reshuffle, and
    # the whole point of this trace is to start from what live actually saw.
    combat = situation.to_combat_mid_fight(start)
    agent = SearchAgent(max_nodes=args.max_nodes, time_budget=args.time_budget,
                        lookahead_turns=2)

    last_turn = combat.turn_count
    rejected = 0
    while not combat.is_over and combat.turn_count < args.max_turns:
        turn = combat.turn_count
        if turn != last_turn:
            last_turn = turn
        player = combat.player
        enemies = [e for e in combat.enemies if not e.is_dead]
        hand_sim = sorted(str(c.card_id.name).removesuffix("_CARD") for c in combat.hand)
        # round is 1-based in the bridge; turn_count is 0-based at first turn
        captured = captured_hand_by_round.get(turn)
        parity = "?" if captured is None else ("OK" if captured == hand_sim else "DIVERGED")
        print(f"--- turn {turn}: hp {player.current_hp} block {player.block} "
              f"hand-parity {parity}")
        if parity == "DIVERGED":
            print(f"    sim hand:      {hand_sim}")
            print(f"    captured hand: {captured}")
        for e in enemies:
            mid = getattr(e, "monster_id", None)
            mid_name = getattr(mid, "name", str(mid))
            print(f"    {mid_name:<22} hp {e.current_hp}/{e.max_hp} block {e.block}")
        while combat.turn_count == turn and not combat.is_over:
            action = agent.act(combat)
            if action == ACTION_END_TURN:
                apply_combat_action(combat, action)
                break
            name = "?"
            if is_potion_action(action):
                name = f"potion#{action}"
            else:
                hand_idx, target_idx = action_to_card_and_target(action)
                if hand_idx is not None and hand_idx < len(combat.hand):
                    name = combat.hand[hand_idx].card_id.name
            ok = apply_combat_action(combat, action)
            if not ok:
                rejected += 1
                if rejected > 5:
                    print("    too many rejected actions, stopping", file=sys.stderr)
                    return 2
                continue
            tgt = ""
            if not is_potion_action(action):
                _, t = action_to_card_and_target(action)
                tgt = f" -> {getattr(getattr(combat.enemies[t], 'monster_id', None), 'name', '?')}" if t is not None and t < len(combat.enemies) else ""
            print(f"    played {name}{tgt}   "
                  f"(hp {combat.player.current_hp}, enemies "
                  f"{[(getattr(getattr(e, 'monster_id', None), 'name', '?'), e.current_hp) for e in combat.enemies if not e.is_dead]})")

    won = combat.is_over and not combat.player.is_dead
    print(f"\nRESULT: {'WON' if won else 'lost'} "
          f"(hp {combat.player.current_hp}, turns {combat.turn_count})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
