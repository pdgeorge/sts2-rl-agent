"""Print a simulated fight turn by turn, so it can be eyeballed against the game.

    python scripts/show_fight.py                       # act 1 normal, starter deck
    python scripts/show_fight.py --act 1 --kind elite --seed 3
    python scripts/show_fight.py --encounter 4 --deck strong

Exists because the battery says a starter deck loses ~34 HP per act 1 hallway
fight, which implies dying around floor 6, while live runs reach floor 13-17. One
of those is wrong, and a per-turn trace is the cheapest way for someone who knows
the real game to see which.
"""

from __future__ import annotations

import argparse

from sts2_env.core.constants import ACTION_END_TURN, IRONCLAD_STARTING_HP


def _intents(combat) -> str:
    parts = []
    for enemy in combat.enemies:
        if not enemy.is_alive:
            continue
        ai = combat.enemy_ais.get(enemy.combat_id)
        text = "?"
        if ai is not None:
            move = ai.current_move
            bits = []
            for intent in (move.intents or ()):
                kind = getattr(getattr(intent, "intent_type", None), "name", "?")
                if getattr(intent, "is_attack", False):
                    hits = max(1, intent.hits)
                    per = intent.damage
                    bits.append(f"{kind} {per}x{hits}={per * hits}" if hits > 1
                                else f"{kind} {per}")
                else:
                    bits.append(kind)
            text = ",".join(bits) or "?"
        name = getattr(enemy, "monster_id", None) or "enemy"
        block = f" blk{enemy.block}" if enemy.block else ""
        parts.append(f"{name}[{enemy.current_hp}/{enemy.max_hp}{block}] -> {text}")
    return " | ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act", type=int, default=1)
    parser.add_argument("--kind", default="normal",
                        choices=["weak", "normal", "elite", "boss"])
    parser.add_argument("--encounter", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deck", default="starter", choices=["starter", "strong"])
    parser.add_argument("--max-hp", type=int, default=IRONCLAD_STARTING_HP)
    args = parser.parse_args()

    from sts2_env.cards.base import CardInstance
    from sts2_env.cards.factory import create_card
    from sts2_env.cards.ironclad import create_ironclad_starter_deck
    from sts2_env.core.combat import CombatState
    from sts2_env.core.enums import CardId
    from sts2_env.core.rng import Rng
    from sts2_env.evaluation.battery import Tier, encounters_for
    from sts2_env.evaluation.pilots import greedy_pilot
    from sts2_env.gym_env.action_space import (
        action_to_card_and_target,
        apply_action,
        get_action_mask,
    )

    deck = create_ironclad_starter_deck()
    if args.deck == "strong":
        deck = deck + [
            create_card(CardId[n])
            for n in ("IRON_WAVE", "ANGER", "HEMOKINESIS", "BODY_SLAM", "ARMAMENTS")
        ]

    encounters = encounters_for(Tier(args.act, args.kind))
    if not encounters:
        print(f"no encounters for act{args.act} {args.kind}")
        return 1
    setup = encounters[args.encounter % len(encounters)]

    combat = CombatState(
        player_hp=args.max_hp, player_max_hp=args.max_hp,
        deck=[c.clone(i) for i, c in enumerate(deck)],
        rng_seed=args.seed, character_id="Ironclad",
    )
    setup(combat, Rng(args.seed))
    combat.start_combat()

    print(f"act{args.act} {args.kind} #{args.encounter % len(encounters)}  "
          f"seed {args.seed}  deck={args.deck} ({len(deck)} cards)")
    print(f"enemies: {_intents(combat)}")
    print()

    turn = combat.turn_count
    hp_at_turn_start = combat.player.current_hp
    print(f"--- turn {turn} --- hp {combat.player.current_hp}/{args.max_hp} "
          f"blk {combat.player.block} energy {combat.energy}")
    print(f"    facing: {_intents(combat)}")
    print(f"    hand: {', '.join(c.card_id.name for c in combat.hand)}")

    for _ in range(400):
        if combat.is_over:
            break
        if not get_action_mask(combat).any():
            break

        action = greedy_pilot(combat)
        hand_index, _ = action_to_card_and_target(int(action))
        played = (combat.hand[hand_index].card_id.name
                  if hand_index is not None and hand_index < len(combat.hand)
                  else None)

        enemy_hp_before = [e.current_hp for e in combat.enemies]
        hp_before = combat.player.current_hp

        if not apply_action(combat, action):
            print("    (engine refused the action)")
            break

        if action == ACTION_END_TURN:
            taken = hp_before - combat.player.current_hp
            print(f"    END TURN -> took {taken} damage")
            if combat.is_over:
                break
            turn = combat.turn_count
            print()
            print(f"--- turn {turn} --- hp {combat.player.current_hp}/{args.max_hp} "
                  f"blk {combat.player.block} energy {combat.energy}")
            print(f"    facing: {_intents(combat)}")
            print(f"    hand: {', '.join(c.card_id.name for c in combat.hand)}")
        elif played:
            dealt = sum(max(0, b - e.current_hp)
                        for b, e in zip(enemy_hp_before, combat.enemies))
            note = f" -> {dealt} dmg" if dealt else ""
            print(f"    play {played}{note}  (blk {combat.player.block}, "
                  f"energy {combat.energy})")

    print()
    outcome = "WON" if combat.player_won else "LOST"
    print(f"{outcome} after {combat.turn_count} turns, "
          f"hp {combat.player.current_hp}/{args.max_hp} "
          f"(lost {args.max_hp - combat.player.current_hp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
