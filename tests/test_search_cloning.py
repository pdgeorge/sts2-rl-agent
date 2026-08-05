"""A cloned combat is independent of the one it came from.

Everything search concludes is read off a clone. If a clone shares anything
mutable with the original, search does not merely think badly -- it corrupts the
fight it was supposed to be reasoning about, and does it without raising.
"""

from __future__ import annotations

import pytest

from sts2_env.gym_env.action_space import apply_combat_action, get_action_mask
from sts2_env.search.cloning import CloneError, can_clone, clone_combat
from sts2_env.search.situation import CardRef, CombatSituation

import numpy as np


def _situation() -> CombatSituation:
    return CombatSituation(
        situation_id="clone-test",
        character_id="Ironclad",
        current_hp=60,
        max_hp=80,
        deck=tuple(
            [CardRef("STRIKE_IRONCLAD")] * 5
            + [CardRef("DEFEND_IRONCLAD")] * 4
            + [CardRef("BASH")]
        ),
        encounter="setup_shrinker_beetle_weak",
        encounter_seed=99,
        combat_seed=1001,
        relics=("BURNING_BLOOD",),
    )


def _play_out(combat, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(200):
        if combat.is_over:
            return
        valid = np.where(get_action_mask(combat) == 1)[0]
        if not len(valid):
            return
        apply_combat_action(combat, int(rng.choice(valid)))


def test_a_fresh_combat_can_be_cloned() -> None:
    assert can_clone(_situation().to_combat())


def test_playing_the_clone_leaves_the_original_untouched() -> None:
    original = _situation().to_combat()
    before = (
        original.player.current_hp,
        original.turn_count,
        [c.card_id.name for c in original.hand],
        [e.current_hp for e in original.enemies],
    )

    _play_out(clone_combat(original))

    assert (
        original.player.current_hp,
        original.turn_count,
        [c.card_id.name for c in original.hand],
        [e.current_hp for e in original.enemies],
    ) == before


def test_the_clone_does_not_share_the_runs_rng() -> None:
    """Shared streams would make the real fight depend on how much search did.

    The enemies' moves and the player's shuffles are drawn from run-level
    streams; if a clone advanced them, thinking harder would change the game.
    """
    original = _situation().to_combat()
    run_state = original.current_player_state.player_state.run_state
    assert run_state is not None

    clone = clone_combat(original)
    clone_run_state = clone.current_player_state.player_state.run_state
    assert clone_run_state is not run_state
    assert clone.shuffle_rng is not run_state.rng.shuffle
    assert clone.monster_ai_rng is not run_state.rng.monster_ai


def test_two_clones_of_one_state_play_out_identically() -> None:
    original = _situation().to_combat()
    a, b = clone_combat(original), clone_combat(original)
    _play_out(a, seed=5)
    _play_out(b, seed=5)
    assert a.player.current_hp == b.player.current_hp
    assert [e.current_hp for e in a.enemies] == [e.current_hp for e in b.enemies]


def test_mutating_the_clones_piles_does_not_reach_the_original() -> None:
    original = _situation().to_combat()
    clone = clone_combat(original)
    hand_before = len(original.hand)

    clone.hand.clear()
    clone.draw_pile.clear()
    clone.player.current_hp = 1
    if clone.enemies:
        clone.enemies[0].current_hp = 1

    assert len(original.hand) == hand_before
    assert original.player.current_hp == 60
    assert original.enemies[0].current_hp > 1


def test_relics_are_copied_not_shared() -> None:
    original = _situation().to_combat()
    clone = clone_combat(original)
    clone.current_player_state.player_state.relics.append("BRONZE_SCALES")
    assert "BRONZE_SCALES" not in original.current_player_state.player_state.relics


def test_a_pending_turn_setup_refuses_to_clone() -> None:
    """The one case deepcopy gets wrong, and gets wrong silently.

    deepcopy returns functions by identity, so the copy would hold a callback
    bound to the original combat.
    """
    combat = _situation().to_combat()
    combat._pending_turn_setup = lambda: None

    assert not can_clone(combat)
    with pytest.raises(CloneError, match="turn-setup callback is pending"):
        clone_combat(combat)


def test_cloning_is_fast_enough_to_search_with() -> None:
    """Not a micro-benchmark for its own sake. Enumerating a turn is hundreds of
    clones, so a regression here turns a 200 ms search into an unusable one."""
    import time

    combat = _situation().to_combat()
    start = time.perf_counter()
    for _ in range(100):
        clone_combat(combat)
    per_clone_ms = (time.perf_counter() - start) / 100 * 1000

    assert per_clone_ms < 10.0, f"{per_clone_ms:.1f} ms per clone is too slow to search with"


# -- the invariant that actually matters -------------------------------------

def _all_situations():
    from sts2_env.search.situation import load_situations
    return load_situations("tests/fixtures/act1_combat_benchmark.json")


def test_a_clone_plays_out_exactly_as_the_original_would() -> None:
    """The real contract, and the one the first version of this file missed.

    Object-level checks passed while the searcher was still planning against a
    fiction: monster moves are closures over their own Creature, and deepcopy
    returns functions by identity, so a copied fight's monsters went on acting
    on the *original's* creatures. Rollouts buffed the real enemy -- a Nibbit
    reached 140 block from search alone -- while the copy's enemies did nothing,
    so search believed incoming damage was a fraction of the truth.

    Playing the same actions on both and demanding the same result is what
    catches that, and it catches the next one of its kind too.
    """
    for situation in _all_situations()[:30]:
        original = situation.to_combat()
        clone = clone_combat(original)

        rng = np.random.default_rng(0)
        actions = []
        for _ in range(200):
            if original.is_over:
                break
            valid = np.where(get_action_mask(original) == 1)[0]
            if not len(valid):
                break
            action = int(rng.choice(valid))
            actions.append(action)
            apply_combat_action(original, action)

        for action in actions:
            apply_combat_action(clone, action)

        assert clone.player.current_hp == original.player.current_hp, (
            f"{situation.situation_id}: clone ended at {clone.player.current_hp} HP, "
            f"original at {original.player.current_hp}"
        )
        assert [e.current_hp for e in clone.enemies] == [
            e.current_hp for e in original.enemies
        ], f"{situation.situation_id}: enemies diverged"


def test_no_monster_move_still_points_at_the_originals_creature() -> None:
    """The mechanism itself, named, so a regression says what broke."""
    for situation in _all_situations()[:30]:
        original = situation.to_combat()
        clone = clone_combat(original)

        for enemy in clone.enemies:
            ai = clone.enemy_ais.get(enemy.combat_id)
            if ai is None:
                continue
            for state in ai.states.values():
                effect = getattr(state, "effect_fn", None)
                for cell in getattr(effect, "__closure__", None) or ():
                    captured = cell.cell_contents
                    assert not any(captured is e for e in original.enemies), (
                        f"{situation.situation_id}: {state.state_id} still acts on "
                        f"the original's creature"
                    )


def test_rollouts_do_not_buff_the_real_enemy() -> None:
    """The symptom, stated in the terms it was found in."""
    situation = _all_situations()[0]
    combat = situation.to_combat()
    before = [(e.current_hp, e.block) for e in combat.enemies]

    for _ in range(30):
        rollout = clone_combat(combat)
        rollout.end_player_turn()

    assert [(e.current_hp, e.block) for e in combat.enemies] == before
