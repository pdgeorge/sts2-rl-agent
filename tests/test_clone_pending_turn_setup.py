"""A combat stopped on a pending choice must still be searchable.

THE RUN THIS COST
-----------------
Live, 2026-08-14, floor 11, Punch Construct in a `?` room. The reconstruction
was perfect -- 61 HP, 3 energy, the right hand, the right enemy carrying
Artifact -- and the fight opened with "choose cards to discard". `clone_combat`
refused to copy a combat holding a pending turn-setup callback, `SearchAgent`
caught the `CloneError` and ended the turn, and the agent played ZERO cards
across 8 turns while taking 61 damage and dying.

The journal recorded `searches: 9, search_failures: 0`. The search reported
success while standing perfectly still, which is the worst failure mode
available: silent, and indistinguishable from working. The same error had
already killed two offline harness runs and been written off.

WHY REFUSING WAS THE WRONG SAFE
-------------------------------
The refusal was correct about the hazard. `copy.deepcopy` returns functions by
reference, so a copied combat holds a callback closing over the ORIGINAL, and
resuming the copy drives the real fight. But "refuse" is only safe if the caller
does something sensible with the refusal, and this caller ended the turn.

The fix records the callback as data (`_pending_turn_setup_spec`) so a clone can
rebuild it bound to itself. These tests pin BOTH halves: that such a state can
now be cloned, and that the clone is genuinely independent -- because a clone
that shares state with the live fight is far worse than one that never happened.
"""

from __future__ import annotations

import pytest

import sts2_env.cards  # noqa: F401
import sts2_env.powers  # noqa: F401
from sts2_env.search.cloning import CloneError, can_clone, clone_combat
from sts2_env.search.situation import CardRef, CombatSituation


def _combat():
    deck = tuple([CardRef(card_id="STRIKE_IRONCLAD", upgraded=False)] * 5
                 + [CardRef(card_id="DEFEND_IRONCLAD", upgraded=False)] * 5)
    return CombatSituation(
        situation_id="clone", character_id="Ironclad", current_hp=60,
        max_hp=80, deck=deck, encounter="setup_shrinker_beetle_weak",
        encounter_seed=5, combat_seed=5, relics=(), room_type="MONSTER",
        act_floor=1, total_floor=1,
    ).to_combat()


def test_a_pending_player_turn_setup_no_longer_blocks_cloning():
    combat = _combat()
    combat._pending_turn_setup = combat._continue_player_turn_setup
    combat._pending_turn_setup_spec = ("player", 0, "block")
    assert can_clone(combat), (
        "a pending turn-setup callback with a rebuild spec must be cloneable; "
        "refusing here is what made the agent stand still for 8 turns"
    )
    assert clone_combat(combat) is not None


def test_the_clone_does_not_share_the_original_fight():
    """The hazard the refusal existed to prevent must still be prevented."""
    combat = _combat()
    combat._pending_turn_setup = combat._continue_player_turn_setup
    combat._pending_turn_setup_spec = ("player", 0, "block")

    clone = clone_combat(combat)
    clone.player.current_hp = 1
    assert combat.player.current_hp != 1, "clone shares player state"

    # The rebuilt callback must be bound to the CLONE. Comparing closure cells
    # rather than calling it, because calling advances the turn.
    assert clone._pending_turn_setup is not combat._pending_turn_setup


def test_a_callback_with_no_spec_is_still_refused():
    """No spec means no safe rebuild, and silence would be the old bug again."""
    combat = _combat()
    combat._pending_turn_setup = combat._continue_player_turn_setup
    combat._pending_turn_setup_spec = None
    assert not can_clone(combat)
    with pytest.raises(CloneError):
        clone_combat(combat)


def test_an_enemy_move_spec_rebinds_to_the_clones_enemy():
    """Matching by combat_id, never by object identity.

    Binding to the original's creature is precisely the sharing bug: the clone
    would resolve its pending enemy move against a monster in the live fight.
    """
    combat = _combat()
    enemy = combat.enemies[0]
    combat._pending_turn_setup = lambda: None
    combat._pending_turn_setup_spec = ("enemy", enemy.combat_id, 0)

    clone = clone_combat(combat)
    assert clone._pending_turn_setup is not None
    assert clone.enemies[0] is not enemy
    assert clone.enemies[0].combat_id == enemy.combat_id
