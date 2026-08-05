"""Each term of the evaluation, on its own.

The evaluation is the searcher's judgement. A term with the wrong sign does not
produce an error, it produces an agent that confidently plays badly -- so every
term is pinned here individually rather than trusted as part of a total.
"""

from __future__ import annotations

import pytest

from sts2_env.core.enums import PowerId
from sts2_env.search.evaluate import (
    DEFAULT_WEIGHTS,
    EvalWeights,
    evaluate,
    evaluate_components,
    explain,
)
from sts2_env.search.situation import CardRef, CombatSituation


def _combat(**overrides):
    base = dict(
        situation_id="eval-test",
        character_id="Ironclad",
        current_hp=60,
        max_hp=80,
        deck=tuple([CardRef("STRIKE_IRONCLAD")] * 5 + [CardRef("DEFEND_IRONCLAD")] * 5),
        encounter="setup_shrinker_beetle_weak",
        encounter_seed=99,
        combat_seed=1001,
        relics=("BURNING_BLOOD",),
    )
    base.update(overrides)
    return CombatSituation(**base).to_combat()


# -- the player's HP dominates ----------------------------------------------

def test_more_player_hp_scores_higher() -> None:
    hurt, healthy = _combat(), _combat()
    hurt.player.current_hp = 20
    healthy.player.current_hp = 70
    assert evaluate(healthy) > evaluate(hurt)


def test_player_hp_is_scored_as_a_fraction_of_max() -> None:
    """So the same weight means the same thing at 80 HP and at 500."""
    small, large = _combat(), _combat(current_hp=250, max_hp=500)
    small.player.current_hp = 40   # half
    large.player.current_hp = 250  # half
    assert evaluate_components(small)["player_hp"] == pytest.approx(
        evaluate_components(large)["player_hp"]
    )


def test_a_point_of_player_hp_outweighs_a_point_of_enemy_hp() -> None:
    """Enemy HP resets every fight; yours does not. A line that trades your HP
    for speed is borrowing against the next elite."""
    assert DEFAULT_WEIGHTS.player_hp > DEFAULT_WEIGHTS.enemy_hp


# -- the enemies -------------------------------------------------------------

def test_damaging_an_enemy_scores_higher() -> None:
    before = _combat()
    after = _combat()
    after.enemies[0].current_hp = max(1, after.enemies[0].current_hp // 2)
    assert evaluate(after) > evaluate(before)


def test_a_dead_enemy_beats_the_same_damage_spread_around() -> None:
    """A dead enemy stops attacking. That is worth more than the arithmetic."""
    combat = _combat(encounter="setup_slimes_weak")
    if len(combat.enemies) < 2:
        pytest.skip("needs a multi-enemy encounter")

    killed = _combat(encounter="setup_slimes_weak")
    total = sum(e.current_hp for e in killed.enemies)
    killed.enemies[0].current_hp = 0
    removed = total - sum(e.current_hp for e in killed.enemies)

    spread = _combat(encounter="setup_slimes_weak")
    per_enemy = removed // len(spread.enemies)
    for enemy in spread.enemies:
        enemy.current_hp = max(1, enemy.current_hp - per_enemy)

    assert evaluate(killed) > evaluate(spread)


# -- terminal states ---------------------------------------------------------

def test_winning_scores_far_above_not_winning() -> None:
    won = _combat()
    won.is_over = True
    won.player_won = True
    assert evaluate(won) > evaluate(_combat()) + 1.0


def test_dying_scores_far_below_everything() -> None:
    lost = _combat()
    lost.is_over = True
    lost.player_won = False
    lost.player.current_hp = 0
    assert evaluate(lost) < evaluate(_combat()) - 5.0


def test_a_loss_is_worse_than_a_win_is_good() -> None:
    """A run survives many missed wins and exactly one death."""
    assert abs(DEFAULT_WEIGHTS.loss) > DEFAULT_WEIGHTS.win


def test_zero_hp_counts_as_lost_even_before_the_flag_is_set() -> None:
    dying = _combat()
    dying.player.current_hp = 0
    assert evaluate_components(dying)["terminal"] == DEFAULT_WEIGHTS.loss


# -- powers ------------------------------------------------------------------

def test_strength_on_the_player_is_good() -> None:
    plain, strong = _combat(), _combat()
    strong.player.apply_power(PowerId.STRENGTH, 3)
    assert evaluate(strong) > evaluate(plain)


def test_vulnerable_on_the_player_is_bad() -> None:
    plain, weakened = _combat(), _combat()
    weakened.player.apply_power(PowerId.VULNERABLE, 2)
    assert evaluate(weakened) < evaluate(plain)


def test_vulnerable_on_an_enemy_is_good() -> None:
    plain, marked = _combat(), _combat()
    marked.enemies[0].apply_power(PowerId.VULNERABLE, 2)
    assert evaluate(marked) > evaluate(plain)


def test_powers_on_a_dead_enemy_do_not_count() -> None:
    combat = _combat()
    combat.enemies[0].current_hp = 0
    scored = evaluate_components(combat)["powers"]
    combat.enemies[0].apply_power(PowerId.VULNERABLE, 5)
    assert evaluate_components(combat)["powers"] == pytest.approx(scored)


# -- tempo -------------------------------------------------------------------

def test_taking_longer_scores_slightly_worse() -> None:
    quick, slow = _combat(), _combat()
    slow.turn_count = quick.turn_count + 10
    assert evaluate(slow) < evaluate(quick)


def test_the_turn_cost_does_not_swamp_a_point_of_hp() -> None:
    """Otherwise a setup turn always looks worse than a bad attack."""
    one_turn = abs(DEFAULT_WEIGHTS.turn)
    one_hp_at_80 = DEFAULT_WEIGHTS.player_hp / 80
    assert one_turn < one_hp_at_80 * 3


# -- the whole thing ---------------------------------------------------------

def test_components_sum_to_the_score() -> None:
    combat = _combat()
    assert evaluate(combat) == pytest.approx(sum(evaluate_components(combat).values()))


def test_weights_can_be_overridden() -> None:
    combat = _combat()
    doubled = EvalWeights(player_hp=2.0)
    assert evaluate_components(combat, doubled)["player_hp"] == pytest.approx(
        2 * evaluate_components(combat)["player_hp"]
    )


def test_explain_names_every_term() -> None:
    text = explain(_combat())
    for term in ("player_hp", "enemy_hp", "kill", "terminal", "powers", "turn"):
        assert term in text
    assert "TOTAL" in text


def test_evaluation_does_not_mutate_the_combat() -> None:
    combat = _combat()
    before = (combat.player.current_hp, combat.turn_count,
              [e.current_hp for e in combat.enemies])
    evaluate(combat)
    assert (combat.player.current_hp, combat.turn_count,
            [e.current_hp for e in combat.enemies]) == before


def test_a_power_cannot_outweigh_what_happened_to_hp() -> None:
    """The bug that made the searcher refuse to attack.

    Bygone Effigy gains tens of Strength when woken. Uncapped, that scored -1.2
    -- five times the enemy-HP term's whole range -- so every line that dealt
    damage looked worse than passing, and the searcher passed until it died.
    """
    combat = _combat()
    combat.enemies[0].apply_power(PowerId.STRENGTH, 40)
    powers = evaluate_components(combat)["powers"]
    assert abs(powers) <= DEFAULT_WEIGHTS.powers_cap

    # And the cap has to sit below the term that measures real consequences.
    assert DEFAULT_WEIGHTS.powers_cap < DEFAULT_WEIGHTS.player_hp


def test_the_cap_binds_in_both_directions() -> None:
    good, bad = _combat(), _combat()
    good.player.apply_power(PowerId.STRENGTH, 40)
    bad.player.apply_power(PowerId.VULNERABLE, 40)
    assert evaluate_components(good)["powers"] == pytest.approx(DEFAULT_WEIGHTS.powers_cap)
    assert evaluate_components(bad)["powers"] == pytest.approx(-DEFAULT_WEIGHTS.powers_cap)


def test_small_power_readings_are_still_a_live_tiebreaker() -> None:
    """Capping must not flatten the ordinary case it exists for."""
    plain, marked = _combat(), _combat()
    marked.enemies[0].apply_power(PowerId.VULNERABLE, 2)
    assert 0 < evaluate_components(marked)["powers"] < DEFAULT_WEIGHTS.powers_cap
    assert evaluate(marked) > evaluate(plain)
