"""The potion rules of thumb, and the room-type trap that broke them once."""

import numpy as np
import pytest

import sts2_env.cards  # noqa: F401
import sts2_env.potions  # noqa: F401
import sts2_env.powers  # noqa: F401
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import RoomType
from sts2_env.core.rng import Rng
from sts2_env.gym_env.action_space import get_action_mask, is_potion_action
from sts2_env.monsters.act1_weak import create_shrinker_beetle
from sts2_env.potions.base import create_potion
from sts2_env.run.rooms import CombatRoom
from sts2_env.search.potion_policy import forced_potion_action
from sts2_env.search.turn_search import SearchAgent


def _combat(potion_id: str | None, room, *, hp: int = 70) -> CombatState:
    potions = [create_potion(potion_id, slot=0)] if potion_id else []
    combat = CombatState(
        player_hp=hp, player_max_hp=80, deck=create_ironclad_starter_deck(),
        rng_seed=3, character_id="Ironclad", room=room, potions=potions,
    )
    creature, ai = create_shrinker_beetle(Rng(3))
    combat.add_enemy(creature, ai)
    combat.start_combat()
    return combat


def _forced(combat: CombatState):
    legal = {int(a) for a in np.where(get_action_mask(combat) == 1)[0]}
    return forced_potion_action(combat, legal)


@pytest.mark.parametrize("room_type", [RoomType.MONSTER, RoomType.ELITE, RoomType.BOSS])
def test_the_rock_is_drunk_in_any_room(room_type):
    """It is a Token-rarity rock for 15 damage. There is no better moment."""
    assert _forced(_combat("POTION_SHAPED_ROCK", CombatRoom(room_type))) is not None


@pytest.mark.parametrize("potion_id", [
    "SKILL_POTION", "ATTACK_POTION", "POWER_POTION", "COLORLESS_POTION",
])
def test_card_generators_are_drunk_only_in_big_fights(potion_id):
    assert _forced(_combat(potion_id, CombatRoom(RoomType.MONSTER))) is None
    assert _forced(_combat(potion_id, CombatRoom(RoomType.ELITE))) is not None
    assert _forced(_combat(potion_id, CombatRoom(RoomType.BOSS))) is not None


def test_card_generators_are_a_turn_one_rule_only():
    combat = _combat("SKILL_POTION", CombatRoom(RoomType.BOSS))
    assert _forced(combat) is not None
    combat.end_player_turn()
    assert _forced(combat) is None


def test_a_room_may_be_a_CombatRoom_a_RoomType_or_absent():
    """`combat.room` is a CombatRoom in play, and CombatRoom is UNHASHABLE, so
    testing it against a frozenset raises TypeError rather than missing. That
    took out nine tests across live_search and search_turn; the bare RoomType
    and None forms are here because other call sites really do pass them.
    """
    assert _forced(_combat("POTION_SHAPED_ROCK", CombatRoom(RoomType.BOSS))) is not None
    assert _forced(_combat("POTION_SHAPED_ROCK", RoomType.BOSS)) is not None
    assert _forced(_combat("POTION_SHAPED_ROCK", None)) is not None

    # A card generator needs the room to know it is a big fight, so an unknown
    # room must decline rather than raise.
    assert _forced(_combat("SKILL_POTION", None)) is None


def test_an_empty_belt_forces_nothing():
    assert _forced(_combat(None, CombatRoom(RoomType.BOSS))) is None


def test_the_agent_acts_on_the_rule():
    """End to end: SearchAgent.act returns the potion, not a card play."""
    combat = _combat("SKILL_POTION", CombatRoom(RoomType.BOSS))
    assert is_potion_action(SearchAgent(max_nodes=4000).act(combat))

    quiet = _combat("SKILL_POTION", CombatRoom(RoomType.MONSTER))
    assert not is_potion_action(SearchAgent(max_nodes=4000).act(quiet))


def test_the_rule_is_skipped_when_potions_are_disabled():
    combat = _combat("POTION_SHAPED_ROCK", CombatRoom(RoomType.BOSS))
    agent = SearchAgent(max_nodes=4000, include_potions=False)
    assert not is_potion_action(agent.act(combat))
