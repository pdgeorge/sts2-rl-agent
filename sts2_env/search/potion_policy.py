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
