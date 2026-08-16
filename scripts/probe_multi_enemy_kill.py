"""Does the searcher take an available kill when there is more than one enemy?

    .venv/bin/python scripts/probe_multi_enemy_kill.py

WHY THIS AND NOT ANOTHER WEIGHT SWEEP
-------------------------------------
`WEEKEND_DECISIONS.md` §1 records that the kill is verified in the SINGLE-enemy
case -- 8 positions out of 8 -- and states plainly that "the multi-enemy case,
where the agent must choose *which* enemy to kill, has never been tested". It is
43% of the encounter table (35 of 81), it includes all three act 1 bosses, and
the most expensive fight in the `run100_2` journal was a three-body boss
(KIN_FOLLOWER, KIN_FOLLOWER, KIN_PRIEST, 90 damage).

`PHASE_TWO.md` §2.1 is the reason this is a probe rather than a sweep. Four wins
came from removing things the agent could not do; six nulls came from adjusting
numbers that were already roughly right. This asks which of the two the multi-
enemy case is, before anything is built.

THE POSITION
------------
One enemy is put in reach of the hand, and it is the one telegraphing the most
damage. Killing it is unambiguously correct: it removes that damage this turn
and every turn after, where blocking removes it once. The probe plays the
searcher's own line out and asks whether the corpse is on the floor.

The same position is then re-run with the other enemies removed, which is the
control -- same hand, same victim, same telegraphed damage, one body instead of
several. A gap between the two arms is about enemy COUNT and nothing else.

WHAT WOULD MAKE THIS A NULL
---------------------------
The kill taken >= 90% of the time with no trend in enemy count. Written down
here before the run, because a probe whose failure condition is decided
afterwards is not a probe.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import sts2_env.cards  # noqa: F401,E402
from sts2_env.cards.factory import create_card  # noqa: E402
from sts2_env.core.enums import CardId  # noqa: E402
from sts2_env.gym_env.action_space import apply_combat_action  # noqa: E402
from sts2_env.search.cloning import clone_combat  # noqa: E402
from sts2_env.search.situation import (  # noqa: E402
    CardRef,
    CombatSituation,
    encounter_registry,
)
from sts2_env.search.turn_search import (  # noqa: E402
    _incoming_damage_by_enemy,
    search_turn,
)

#: The hand every position is given. Three Strikes is 18 base damage across three
#: energy, so a victim inside that range is killable and the rest of the board is
#: not -- which is what makes the choice a choice rather than a sweep of the
#: table. Two Defends so that blocking is always an option worth considering.
HAND = (
    CardId.STRIKE_IRONCLAD,
    CardId.STRIKE_IRONCLAD,
    CardId.STRIKE_IRONCLAD,
    CardId.DEFEND_IRONCLAD,
    CardId.DEFEND_IRONCLAD,
)

#: Where the victim's HP is set. Below one Strike, so a single card ends it and
#: the searcher can still afford everything else it might want to do. Anything
#: larger and a refusal to kill could be an energy argument rather than a
#: valuation one.
VICTIM_HP = 5

DECK = tuple([CardRef("STRIKE_IRONCLAD")] * 5 + [CardRef("DEFEND_IRONCLAD")] * 5)


def _build(encounter: str, seed: int):
    return CombatSituation(
        situation_id="probe",
        character_id="Ironclad",
        current_hp=60,
        max_hp=80,
        deck=DECK,
        encounter=encounter,
        encounter_seed=99,
        combat_seed=seed,
        relics=("BURNING_BLOOD",),
    ).to_combat()


def _set_hand(combat, energy: int) -> None:
    combat.hand.clear()
    for card_id in HAND:
        combat.hand.append(create_card(card_id))
    combat.energy = energy


def _victim(combat):
    """The living enemy telegraphing the most damage this turn.

    Ties go to nobody: a position where two enemies threaten the same amount
    does not have one correct target, so it is not evidence either way and is
    dropped rather than scored.
    """
    per_enemy = _incoming_damage_by_enemy(combat)
    alive = [e for e in combat.enemies if e.is_alive]
    if len(alive) < 2:
        return None, 0
    ranked = sorted(alive, key=lambda e: -per_enemy.get(e.combat_id, 0))
    top = per_enemy.get(ranked[0].combat_id, 0)
    if top <= 0:
        return None, 0  # nothing is attacking; there is no damage to remove
    if per_enemy.get(ranked[1].combat_id, 0) == top:
        return None, 0
    return ranked[0], top


def _kills_the_victim(combat, victim_id: int) -> bool:
    """Play the searcher's own line on a copy and look at the body."""
    result = search_turn(combat)
    state = clone_combat(combat)
    for action in result.actions:
        apply_combat_action(state, action)
    for enemy in state.enemies:
        if enemy.combat_id == victim_id:
            return not enemy.is_alive
    return True  # removed from the table entirely, which is deader still


def _solo(combat, victim_id: int):
    """The control: the same victim, the same threat, on its own.

    The other enemies are removed rather than killed, so the position the
    searcher sees differs in exactly one respect -- how many bodies are on the
    table -- and `len(killable)` is the only term that can move.
    """
    solo = clone_combat(combat)
    keep = [e for e in solo.enemies if e.combat_id == victim_id]
    for enemy in solo.enemies:
        if enemy.combat_id != victim_id:
            solo.enemy_ais.pop(enemy.combat_id, None)
    solo.enemies[:] = keep
    return solo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=6,
                        help="combat seeds per encounter; each rolls different intents")
    parser.add_argument("--energy", type=int, default=3,
                        help="1 makes the kill and the block mutually exclusive, which is "
                             "the decision; 3 lets the searcher do both and only asks "
                             "whether it notices a free kill")
    args = parser.parse_args()

    registry = encounter_registry()
    rows = []
    for encounter in sorted(registry):
        for seed in range(1, args.seeds + 1):
            try:
                combat = _build(encounter, seed * 1013)
            except Exception:
                break  # unbuildable encounter, not a finding

            victim, threat = _victim(combat)
            if victim is None:
                continue

            victim.current_hp = min(victim.current_hp, VICTIM_HP)
            _set_hand(combat, args.energy)

            n_alive = sum(1 for e in combat.enemies if e.is_alive)
            solo = _solo(combat, victim.combat_id)
            _set_hand(solo, args.energy)

            try:
                took = _kills_the_victim(combat, victim.combat_id)
                took_solo = _kills_the_victim(solo, victim.combat_id)
            except Exception as exc:  # a position that cannot be searched is a bug, not a score
                print(f"  !! {encounter} seed {seed}: {type(exc).__name__}: {exc}")
                continue

            rows.append((encounter, n_alive, threat, took, took_solo))

    if not rows:
        print("no scorable positions -- the probe found nothing to ask about")
        return 1

    took_n = sum(1 for r in rows if r[3])
    solo_n = sum(1 for r in rows if r[4])
    print("=" * 74)
    print(f"MULTI-ENEMY KILL PROBE   {args.energy} energy, {len(rows)} positions, "
          f"{len({r[0] for r in rows})} encounters")
    print("=" * 74)
    print(f"  kill taken, several enemies on the table : "
          f"{took_n}/{len(rows)}  {100*took_n/len(rows):.1f}%")
    print(f"  kill taken, same victim alone (control)  : "
          f"{solo_n}/{len(rows)}  {100*solo_n/len(rows):.1f}%")

    by_count: dict[int, list] = defaultdict(list)
    for _, n_alive, _, took, _ in rows:
        by_count[n_alive].append(took)
    print("\n  by number of living enemies")
    print("    enemies   positions   kill taken")
    for n_alive in sorted(by_count):
        hits = by_count[n_alive]
        print(f"    {n_alive:^7}   {len(hits):^9}   "
              f"{sum(hits)}/{len(hits)}  {100*sum(hits)/len(hits):.1f}%")

    missed = [r for r in rows if not r[3]]
    if missed:
        print(f"\n  positions where the kill was on the table and was not taken "
              f"({len(missed)})")
        seen = set()
        for encounter, n_alive, threat, _, took_solo in missed:
            if encounter in seen:
                continue
            seen.add(encounter)
            control = "kills it alone" if took_solo else "misses it alone too"
            print(f"    {encounter:<42} {n_alive} enemies, "
                  f"{threat} incoming -- {control}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
