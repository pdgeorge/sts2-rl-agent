"""Intent types for monster moves.

AN INTENT IS A PROMISE ABOUT DAMAGE, AND IT USED TO LIE

`damage` is the move's declared base value. It is NOT what the player will take:
Strength, Weak, Vulnerable and relic hooks all apply at strike time. Reading the
field directly under-reports, and the gap grows exactly when it matters most --
against an enemy that spends its early turns buffing.

Measured on Cubex Construct, act 1:

    enemy strength  0: intent says 7, player takes  9
    enemy strength  3: intent says 7, player takes 12
    enemy strength  6: intent says 7, player takes 15
    enemy strength 10: intent says 7, player takes 19

Everything that tried to defend used the stated number, blocked against 7 while
19 was incoming, and looked like blocking does not work. Three separate
defensive-pilot experiments were invalidated by this, and the policy's own
observation carried the same wrong figure.

So use `effective_total(...)`. `damage` remains the base, because that is what the
move declares and what the parity suites compare against.
"""

from __future__ import annotations

from dataclasses import dataclass

from sts2_env.core.enums import IntentType


@dataclass
class Intent:
    """A single intent shown to the player."""

    intent_type: IntentType
    damage: int = 0
    hits: int = 1

    @property
    def total_damage(self) -> int:
        return self.damage * self.hits

    @property
    def is_attack(self) -> bool:
        return self.intent_type in (IntentType.ATTACK, IntentType.MULTI_ATTACK)

    def effective_damage(self, dealer, target, combat) -> int:
        """Per-hit damage this intent would actually deal, right now.

        Runs the declared base through the same pipeline the attack will use, so
        the number shown and the number taken agree.
        """
        if not self.is_attack or not self.damage:
            return 0
        from sts2_env.core.damage import calculate_damage
        from sts2_env.core.enums import ValueProp

        try:
            return int(calculate_damage(
                self.damage, dealer, target, ValueProp.MOVE, combat))
        except Exception:  # noqa: BLE001 -- never let a preview break a fight
            return int(self.damage)

    def effective_total(self, dealer, target, combat) -> int:
        """All hits combined. This is the number to block against."""
        return self.effective_damage(dealer, target, combat) * max(1, self.hits)


def attack_intent(damage: int) -> Intent:
    return Intent(IntentType.ATTACK, damage=damage, hits=1)


def multi_attack_intent(damage: int, hits: int) -> Intent:
    return Intent(IntentType.MULTI_ATTACK, damage=damage, hits=hits)


def defend_intent() -> Intent:
    return Intent(IntentType.DEFEND)


def buff_intent() -> Intent:
    return Intent(IntentType.BUFF)


def debuff_intent() -> Intent:
    return Intent(IntentType.DEBUFF)


def strong_debuff_intent() -> Intent:
    return Intent(IntentType.DEBUFF_STRONG)


def sleep_intent() -> Intent:
    return Intent(IntentType.SLEEP)


def status_intent() -> Intent:
    return Intent(IntentType.STATUS_CARD)


def incoming_damage(combat, target=None) -> int:
    """Total damage every living enemy is about to deal to `target`.

    One definition, used by the observation encoder and by any pilot that wants
    to defend. Two implementations of this is precisely the bug class this repo
    keeps hitting -- and the last one cost three invalidated experiments.
    """
    target = target if target is not None else getattr(combat, "player", None)
    if target is None:
        return 0

    total = 0
    for enemy in getattr(combat, "enemies", []) or []:
        if not getattr(enemy, "is_alive", False):
            continue
        ai = getattr(combat, "enemy_ais", {}).get(getattr(enemy, "combat_id", None))
        if ai is None:
            continue
        move = ai.current_move
        for intent in (getattr(move, "intents", None) or ()):
            total += intent.effective_total(enemy, target, combat)
    return total
