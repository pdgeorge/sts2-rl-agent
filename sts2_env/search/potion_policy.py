"""Rules of thumb for drinking potions, where the search has no opinion.

OVERRIDE THESE IF SOMETHING IS PROVEN BETTER. They are a starting position from
a human who plays the game, not a measured optimum, and nothing here should
survive a paired A/B that says otherwise. Each rule below records what it costs
to be wrong, so the experiment that replaces it knows what to look for.

WHY THE SEARCH NEEDS HELP AT ALL
--------------------------------
`evaluate.py` has no potion term -- not one mention. A potion in reserve is
worth nothing to it, so drinking one is FREE and any board improvement makes
the line score better. That is why "use it if it saves you real HP" already
works without a rule: Block Potion at 15 hp against the giant's 20-damage
pressure gun gets drunk, verified.

What that same blind spot gets wrong is a potion whose payoff is not on the
board at end of turn. A card generator hands you a cost-0 card; the leaf is
scored after the turn ends, when an unplayed card in hand is worth exactly
nothing, and playing it costs depth the search would rather spend elsewhere.
So the generators never look worth it, and across the live journals 39 of the
49 non-automatic potions that died in the belt during a LOST act 1 boss fight
were exactly those: Skill 10, Attack 10, Colorless 7, Power 6, Duplicator 6.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
"Use any other potion if it saves 10+ HP" is not implemented, because the
search already does it and does it better -- it knows the actual incoming
damage, the actual block, and what else the turn could spend energy on. A rule
would only overrule that with a worse estimate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sts2_env.core.constants import POTION_ACTION_START, POTION_TARGET_OPTIONS
from sts2_env.core.enums import PotionTargetType, RoomType

if TYPE_CHECKING:
    from sts2_env.core.combat import CombatState

#: Junk that occupies a slot. A Token-rarity rock dealing 15 unpowered damage
#: (`PotionShapedRock`, `DamageVar(15m, Unpowered)`), which is never worth
#: saving: it does not scale, it cannot be improved, and while it sits there it
#: is the reason a real potion gets declined at a reward screen.
#: Wrong if: 15 damage held back would have been the exact lethal later. Cheap.
DRINK_ON_SIGHT = frozenset({"PotionShapedRock"})

#: Potions whose payoff is a card, not a board change. Held for a "better
#: moment" that never arrives, because the search cannot see their value at all.
#: Wrong if: the cards drawn are dead weight in the fight that matters, in which
#: case this spends a potion for nothing on every elite and boss.
CARD_GENERATORS = frozenset({
    "SkillPotion",
    "AttackPotion",
    "PowerPotion",
    "ColorlessPotion",
    "Duplicator",
})

#: Where a card generator is worth spending. Normal fights are won without it;
#: the fights that end runs are elites and bosses, and turn 1 is when a cost-0
#: card still has a whole fight to pay off in.
BIG_FIGHTS = frozenset({RoomType.ELITE, RoomType.BOSS})

#: Potions kept OUT of hallway fights, so they are still in the belt when a
#: fight that can end the run starts. pd's call, 2026-08-17.
#:
#: FORCING IS ONLY HALF A POLICY, and the measurement says so. `CARD_GENERATORS`
#: above forces a drink on turn 1 of a big fight but does nothing to stop one
#: earlier, and `evaluate.py` has no potion term -- a potion in reserve is worth
#: zero, so drinking is FREE and any board improvement makes the line score
#: better. Over the 200 runs since the power-hook fix:
#:
#:     PowderedDemise         31 monster /  4 elite / 0 boss   89% on trash
#:     DistilledChaos         10 /  1 / 0                      83%
#:     OrobicAcid              4 /  0 / 0                      80%
#:     GigantificationPotion   5 /  2 / 1                      56%
#:     Duplicator             17 /  0 / 1                      94%
#:
#: against 12% for the four the force rule already covers. The split is not
#: about the rule, it is about what the searcher can SEE: Skill/Attack/Power/
#: Colorless hand you a card, which scores nothing at end of turn, so the search
#: never drinks them on its own and they survive to the elite. These five change
#: the board, so the search drinks them the moment they help -- and a hallway
#: monster is where they help first. Duplicator is in `CARD_GENERATORS` already
#: and is still 94% trash, which is the proof that forcing alone does not hold.
#:
#: Wrong if: the chip damage these prevent in the corridor is worth more than
#: the damage they deal at the boss. Prediction 10 priced 15 HP at +4.6
#: boss-win points, so that is a real possibility and it is why this is
#: measured rather than assumed.
#: EMPTY, which is the shipped behaviour. `apply_active_policy` overwrites this
#: from `PolicyConfig.hold_potions_for_big_fights`, so the A/B arm turns it on
#: without anything patching a global.
#:
#: MEASURED AND REVERTED, 2026-08-18. Live for one session as a hard-coded set.
#: The mechanism worked and the hypothesis did not: trash use of the five fell
#: 85% -> 12%, and potions held entering the act 1 boss did not move (0.99 ->
#: 0.97). They were not saved for the boss, they were spent one room earlier on
#: elites -- group elite use went 7 -> 13 in half as many runs. Clear fell 12
#: points at p=0.041, which this project cannot separate from session variance:
#: two sessions on identical act 1 code have already differed by 13 points.
HOLD_FOR_BIG_FIGHTS: frozenset[str] = frozenset()


def _telegraphed_damage(combat: "CombatState") -> int:
    """What the living enemies have announced for this turn.

    Read off the intents the simulator already holds rather than estimated, for
    the same reason `evaluate.py` scores a state instead of predicting one.
    """
    total = 0
    for enemy in combat.enemies:
        if getattr(enemy, "is_dead", False) or not getattr(enemy, "is_alive", True):
            continue
        ai = (getattr(combat, "enemy_ais", None) or {}).get(enemy.combat_id)
        move = getattr(ai, "current_move", None)
        for intent in getattr(move, "intents", None) or ():
            total += (getattr(intent, "damage", 0) or 0) * max(1, getattr(intent, "hits", 1) or 1)
    return total


def should_hold(combat: "CombatState", potion) -> bool:
    """Keep this potion for a fight that can end the run?

    NEVER HELD THROUGH A LETHAL TURN. If what the enemies have telegraphed is
    at least the player's remaining HP, the hold is released and the search may
    drink -- a potion saved for a boss the run never reaches is worth nothing,
    and this is the failure mode a blanket hold would introduce. The test is
    the board's own declared damage, so it needs no tuned threshold.
    """
    if str(getattr(potion, "potion_id", "")) not in HOLD_FOR_BIG_FIGHTS:
        return False
    if _room_type(combat) in BIG_FIGHTS:
        return False
    player = combat.player
    unblocked = max(0, _telegraphed_damage(combat) - (getattr(player, "block", 0) or 0))
    if unblocked >= max(0, getattr(player, "current_hp", 0)):
        return False
    return True


def _room_type(combat: "CombatState") -> RoomType | None:
    """`combat.room` is a CombatRoom, not a RoomType -- and it is unhashable, so
    testing it against a frozenset raises rather than simply missing. Some call
    sites (and the search's own clones) carry a bare RoomType instead, and some
    carry nothing at all, so all three are accepted here.
    """
    room = getattr(combat, "room", None)
    if room is None:
        return None
    if isinstance(room, RoomType):
        return room
    return getattr(room, "room_type", None)


def _potion_action(slot: int, target_offset: int) -> int:
    """Slot-major, matching `bridge/state_adapter` and the action space."""
    return POTION_ACTION_START + slot * POTION_TARGET_OPTIONS + target_offset


def _target_offset(combat: "CombatState", potion) -> int | None:
    """0 for self/all, else 1 + enemy index. None when nothing is targetable.

    Targets the weakest living enemy, so a rock that can finish something does.
    """
    model = getattr(potion, "model", None)
    target_type = getattr(model, "target_type", None)
    if target_type != PotionTargetType.ANY_ENEMY:
        return 0
    alive = [(i, e) for i, e in enumerate(combat.enemies) if not e.is_dead]
    if not alive:
        return None
    return 1 + min(alive, key=lambda pair: pair[1].current_hp)[0]


def forced_potion_action(combat: "CombatState", legal: set[int]) -> int | None:
    """The potion this position should drink regardless of what the search says.

    Returns an action index that is present in `legal`, or None to leave the
    decision to the search -- which is the answer for most positions.
    """
    if combat.pending_choice is not None:
        return None

    first_turn = int(getattr(combat, "round_number", 1) or 1) <= 1
    big_fight = _room_type(combat) in BIG_FIGHTS

    for slot, potion in enumerate(combat.potions or []):
        if potion is None:
            continue
        pid = str(getattr(potion, "potion_id", ""))
        if pid in DRINK_ON_SIGHT:
            pass
        elif pid in CARD_GENERATORS and first_turn and big_fight:
            pass
        else:
            continue

        offset = _target_offset(combat, potion)
        if offset is None:
            continue
        action = _potion_action(slot, offset)
        if action in legal:
            return action
    return None
