"""The telegraphed-damage estimate must match what the turn actually deals."""

import pytest

import sts2_env.cards  # noqa: F401
import sts2_env.powers  # noqa: F401
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import IntentType, PowerId
from sts2_env.core.rng import Rng
from sts2_env.monsters.act1_weak import create_shrinker_beetle
from sts2_env.monsters.intents import Intent
from sts2_env.search.turn_search import _incoming_damage

CHOMP = "CHOMP_MOVE"


def _combat(enemy_powers=None, player_powers=None):
    combat = CombatState(
        player_hp=200, player_max_hp=200, deck=create_ironclad_starter_deck(),
        rng_seed=1, character_id="Ironclad",
    )
    creature, ai = create_shrinker_beetle(Rng(1))
    combat.add_enemy(creature, ai)
    combat.start_combat()
    combat.set_enemy_state(creature, CHOMP)
    for power, amount in (enemy_powers or {}).items():
        creature.apply_power(power, amount, applier=creature)
    for power, amount in (player_powers or {}).items():
        combat.player.apply_power(power, amount, applier=creature)
    return combat, creature, ai


def _actual(combat, creature, ai):
    combat.player.block = 0
    before = combat.player.current_hp
    ai.states[CHOMP].effect_fn(combat)
    return before - combat.player.current_hp


@pytest.mark.parametrize("enemy_powers,player_powers", [
    ({}, {}),
    ({PowerId.STRENGTH: 3}, {}),
    ({PowerId.WEAK: 2}, {}),
    ({}, {PowerId.VULNERABLE: 2}),
    ({PowerId.STRENGTH: 3}, {PowerId.VULNERABLE: 2}),
])
def test_estimate_matches_what_lands(enemy_powers, player_powers):
    """The estimate drives the rollout's block decision; a raw intent under-reads
    every buffed enemy, and the enemies that matter are the ones that scale."""
    combat, creature, ai = _combat(enemy_powers, player_powers)
    estimate = _incoming_damage(combat)
    combat2, creature2, ai2 = _combat(enemy_powers, player_powers)
    assert estimate == _actual(combat2, creature2, ai2)


def test_a_bridge_intent_is_not_modified_twice():
    """The live game telegraphs the FINAL number, Strength already in it.

    Applying modifiers on top would inflate every buffed enemy on the live path,
    which is the opposite error and just as wrong.
    """
    combat, creature, ai = _combat({PowerId.STRENGTH: 3}, {})
    ai.current_move.intents = [
        Intent(IntentType.ATTACK, damage=10, hits=1, pre_modified=True)
    ]
    assert _incoming_damage(combat) == 10


def test_multi_hit_intents_scale_per_hit():
    combat, creature, ai = _combat({PowerId.STRENGTH: 2}, {})
    ai.current_move.intents = [Intent(IntentType.MULTI_ATTACK, damage=5, hits=3)]
    assert _incoming_damage(combat) == (5 + 2) * 3
