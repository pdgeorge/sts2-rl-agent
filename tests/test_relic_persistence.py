"""Relics obtained during a run survive into the next combat.

This is a regression suite for a bug that was invisible for a long time and cost
more than any of the training work built on top of it.

`RunState.__init__` aliased `self.relics = self.player.relics`, and
`CombatState._build_player_state` rebound `player_state.relics` to a fresh list
on every combat. After the first fight the two names pointed at different lists.
Relics obtained later went to the player's list; `RunManager._enter_combat` fed
`run_state.relics` -- the stale one -- into the next combat, which then
overwrote the player's list with it. The relic vanished, silently, with no error
and no log line.

The visible consequence was that the simulator never held more than the starting
relic, while the real game hands out five or six by the end of act 1. Every model
trained here learned the wrong game.
"""

from __future__ import annotations

import pytest

from sts2_env.core.enums import RoomType
from sts2_env.run.run_manager import RunManager


@pytest.fixture
def manager() -> RunManager:
    return RunManager(seed=7, character_id="Ironclad")


def test_run_state_relics_are_the_players_relics(manager: RunManager) -> None:
    assert manager.run_state.relics == manager.run_state.player.relics


def test_relic_obtained_after_a_combat_survives_the_next_one(manager: RunManager) -> None:
    player = manager.run_state.player
    manager._enter_combat(RoomType.MONSTER)

    player.obtain_relic("BRONZE_SCALES")
    assert "BRONZE_SCALES" in manager.run_state.relics

    manager._enter_combat(RoomType.MONSTER)

    assert "BRONZE_SCALES" in player.relics, "the relic was dropped entering combat"
    assert "BRONZE_SCALES" in manager.run_state.relics


def test_the_relics_effects_are_live_in_that_combat(manager: RunManager) -> None:
    """Present in the list is not the same as active in the fight."""
    player = manager.run_state.player
    manager._enter_combat(RoomType.MONSTER)
    player.obtain_relic("BRONZE_SCALES")
    manager._enter_combat(RoomType.MONSTER)

    active = {type(r).__name__ for r in manager.get_combat_state().current_player_state.relics}
    assert "BronzeScales" in active


def test_relics_accumulate_across_several_combats(manager: RunManager) -> None:
    player = manager.run_state.player
    manager._enter_combat(RoomType.MONSTER)
    player.obtain_relic("BRONZE_SCALES")
    manager._enter_combat(RoomType.MONSTER)
    player.obtain_relic("ODDLY_SMOOTH_STONE")
    manager._enter_combat(RoomType.ELITE)

    assert set(manager.run_state.relics) == {
        "BURNING_BLOOD", "BRONZE_SCALES", "ODDLY_SMOOTH_STONE",
    }


def test_run_state_and_player_cannot_come_apart(manager: RunManager) -> None:
    """The specific mechanism that broke: rebinding one of the two names.

    A property has no second list to fall out of step, so appending through
    either name is visible through both -- which is what every reader assumed
    all along.
    """
    manager._enter_combat(RoomType.MONSTER)
    manager.run_state.player.relics.append("BRONZE_SCALES")
    assert "BRONZE_SCALES" in manager.run_state.relics

    manager.run_state.relics = ["BURNING_BLOOD", "GOLDEN_PEARL"]
    assert manager.run_state.player.relics == ["BURNING_BLOOD", "GOLDEN_PEARL"]
