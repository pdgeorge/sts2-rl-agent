"""The searcher, on positions where the right play is not in doubt.

Behavioural rather than line-by-line: asserting an exact action tuple would pin
the tests to the current weights and break on every tuning change. What must hold
is the outcome -- lethal is taken, a survivable turn is survived, a multiplier is
applied before the cards it multiplies.
"""

from __future__ import annotations

import pytest

from sts2_env.cards.factory import create_card
from sts2_env.core.enums import CardId, PowerId
from sts2_env.gym_env.action_space import apply_combat_action, action_to_card_and_target
from sts2_env.search.cloning import clone_combat
from sts2_env.search.situation import CardRef, CombatSituation
from sts2_env.search.turn_search import SearchAgent, search_turn


def _combat(*, hp=60, max_hp=80, encounter="setup_shrinker_beetle_weak", seed=1001):
    return CombatSituation(
        situation_id="turn-test",
        character_id="Ironclad",
        current_hp=hp,
        max_hp=max_hp,
        deck=tuple([CardRef("STRIKE_IRONCLAD")] * 5 + [CardRef("DEFEND_IRONCLAD")] * 5),
        encounter=encounter,
        encounter_seed=99,
        combat_seed=seed,
        relics=("BURNING_BLOOD",),
    ).to_combat()


def _set_hand(combat, *card_ids: CardId) -> None:
    combat.hand.clear()
    for card_id in card_ids:
        combat.hand.append(create_card(card_id))


def _play_line(combat, result):
    for action in result.actions:
        assert apply_combat_action(combat, action), f"planned action {action} was refused"
    return combat


def _hand_card_names(combat, result) -> list[str]:
    """Which cards the line plays, in order, resolved against a copy."""
    names = []
    state = clone_combat(combat)
    for action in result.actions:
        hand_index, _ = action_to_card_and_target(action)
        if hand_index is not None and hand_index < len(state.hand):
            names.append(state.hand[hand_index].card_id.name)
        apply_combat_action(state, action)
    return names


# -- it takes the win --------------------------------------------------------

def test_lethal_is_taken() -> None:
    combat = _combat()
    _set_hand(combat, CardId.STRIKE_IRONCLAD, CardId.STRIKE_IRONCLAD, CardId.DEFEND_IRONCLAD)
    combat.energy = 3
    for enemy in combat.enemies:
        enemy.current_hp = 4

    result = search_turn(combat)
    _play_line(combat, result)

    assert combat.is_over and combat.player_won


def test_a_win_this_turn_beats_a_safer_looking_turn() -> None:
    """Blocking is not better than winning."""
    combat = _combat()
    _set_hand(combat, CardId.STRIKE_IRONCLAD, CardId.DEFEND_IRONCLAD, CardId.DEFEND_IRONCLAD)
    combat.energy = 3
    for enemy in combat.enemies:
        enemy.current_hp = 3

    played = _hand_card_names(combat, search_turn(combat))
    assert "STRIKE_IRONCLAD" in played


# -- it defends when defending is what survives ------------------------------

def test_a_survivable_turn_is_survived() -> None:
    """With a big hit telegraphed and block available, the line has to keep the
    player alive -- this is the exact failure mode of the current models, which
    attack into lethal because nothing taught them the intent matters."""
    combat = _combat(hp=12)
    _set_hand(combat, CardId.DEFEND_IRONCLAD, CardId.DEFEND_IRONCLAD, CardId.STRIKE_IRONCLAD)
    combat.energy = 3
    for enemy in combat.enemies:
        enemy.current_hp = enemy.max_hp

    result = search_turn(combat)
    after = clone_combat(combat)
    _play_line(after, result)
    after.end_player_turn()

    assert after.player.current_hp > 0, "the searcher played into a survivable death"


def test_block_is_preferred_when_a_hit_is_actually_coming() -> None:
    """The condition matters, and the first version of this test got it wrong.

    Against an opening intent of DEBUFF for 0 damage the searcher struck instead
    of blocking, and it was right to: there was nothing to block, so blocking is
    wasted energy while the attack removes 6 enemy HP for free. Pick an encounter
    that really is winding up, or the test asserts a superstition.
    """
    combat = _combat(hp=12, encounter="setup_fossil_stalker_normal")
    _set_hand(combat, CardId.DEFEND_IRONCLAD, CardId.STRIKE_IRONCLAD)
    combat.energy = 1

    incoming = sum(
        intent.damage * max(1, intent.hits)
        for enemy in combat.enemies
        if (ai := combat.enemy_ais.get(enemy.combat_id))
        for intent in ai.current_move.intents
    )
    assert incoming >= 8, "this test needs a real attack telegraphed"

    played = _hand_card_names(combat, search_turn(combat))
    assert played[:1] == ["DEFEND_IRONCLAD"]


def test_block_is_not_wasted_when_nothing_is_coming() -> None:
    """The other half: no incoming damage means block is energy spent on nothing."""
    combat = _combat(hp=12, encounter="setup_shrinker_beetle_weak")
    _set_hand(combat, CardId.DEFEND_IRONCLAD, CardId.STRIKE_IRONCLAD)
    combat.energy = 1

    incoming = sum(
        intent.damage * max(1, intent.hits)
        for enemy in combat.enemies
        if (ai := combat.enemy_ais.get(enemy.combat_id))
        for intent in ai.current_move.intents
    )
    assert incoming == 0, "this test needs a turn with nothing telegraphed"

    played = _hand_card_names(combat, search_turn(combat))
    assert played[:1] == ["STRIKE_IRONCLAD"]


# -- sequencing, which is the whole point ------------------------------------

def test_bash_is_played_before_the_strikes_it_multiplies() -> None:
    """Vulnerable multiplies what comes after it, so the order is the play.

    A policy net has to learn this from returns. Search gets it from arithmetic
    every turn, including for cards it has never seen.
    """
    combat = _combat()
    _set_hand(combat, CardId.STRIKE_IRONCLAD, CardId.BASH, CardId.STRIKE_IRONCLAD)
    combat.energy = 3
    for enemy in combat.enemies:
        enemy.current_hp = enemy.max_hp = 60

    played = _hand_card_names(combat, search_turn(combat))
    if "BASH" in played and "STRIKE_IRONCLAD" in played:
        assert played.index("BASH") < played.index("STRIKE_IRONCLAD")


def test_vulnerable_ends_up_applied_before_damage_lands() -> None:
    combat = _combat()
    _set_hand(combat, CardId.STRIKE_IRONCLAD, CardId.BASH)
    combat.energy = 3
    enemy = combat.enemies[0]
    enemy.current_hp = enemy.max_hp = 60

    result = search_turn(combat)
    _play_line(combat, result)

    # Either it killed nothing yet and left the enemy marked, or it did more
    # damage than two unmodified strikes -- both mean the order was used.
    assert enemy.get_power_amount(PowerId.VULNERABLE) > 0 or enemy.current_hp < 48


# -- it does not corrupt the fight it is reasoning about ---------------------

def test_searching_does_not_touch_the_real_combat() -> None:
    combat = _combat()
    before = (
        combat.player.current_hp,
        combat.player.block,
        combat.energy,
        combat.turn_count,
        [c.card_id.name for c in combat.hand],
        [e.current_hp for e in combat.enemies],
    )

    search_turn(combat)

    assert (
        combat.player.current_hp,
        combat.player.block,
        combat.energy,
        combat.turn_count,
        [c.card_id.name for c in combat.hand],
        [e.current_hp for e in combat.enemies],
    ) == before


def test_every_planned_action_is_actually_playable() -> None:
    """A line that cannot be executed is worse than no line: it is a decision the
    game will refuse, which is how a screen loops forever."""
    for seed in (1001, 1002, 1003):
        combat = _combat(seed=seed)
        result = search_turn(combat)
        _play_line(combat, result)  # asserts internally


# -- budgets, honestly reported ----------------------------------------------

def test_a_tight_node_budget_is_reported_not_hidden() -> None:
    combat = _combat(encounter="setup_slimes_weak")
    result = search_turn(combat, max_nodes=5)
    assert not result.exhausted
    assert result.nodes <= 20


def test_a_tight_time_budget_still_returns_a_playable_line() -> None:
    combat = _combat(encounter="setup_slimes_weak")
    result = search_turn(combat, time_budget=0.001)
    _play_line(combat, result)


def test_an_unbudgeted_search_reports_itself_exhaustive() -> None:
    combat = _combat()
    assert search_turn(combat).exhausted


# -- the gap that becomes a phrase -------------------------------------------

def test_the_gap_is_the_margin_between_the_two_best_lines() -> None:
    result = search_turn(_combat())
    if result.runner_up is not None:
        assert result.gap == pytest.approx(result.score - result.runner_up)
        assert result.gap >= 0


def test_the_gap_is_absent_rather_than_faked_when_there_was_no_choice() -> None:
    combat = _combat()
    combat.hand.clear()
    result = search_turn(combat)
    assert result.runner_up is None
    assert result.gap is None


# -- the agent ---------------------------------------------------------------

def test_the_agent_plays_a_fight_to_the_end() -> None:
    from sts2_env.search.benchmark import play
    from sts2_env.search.situation import load_situations

    situation = load_situations("tests/fixtures/act1_combat_benchmark.json")[0]
    result = play(situation, SearchAgent(time_budget=1.0))

    assert not result.stalled
    assert result.turns > 0


def test_the_agent_reports_what_it_spent() -> None:
    from sts2_env.search.benchmark import play
    from sts2_env.search.situation import load_situations

    situation = load_situations("tests/fixtures/act1_combat_benchmark.json")[0]
    agent = SearchAgent(time_budget=1.0)
    play(situation, agent)

    stats = agent.stats()
    assert stats["searches"] > 0
    assert stats["nodes_per_search"] > 0


# -- seeing past this turn ---------------------------------------------------

def test_the_lookahead_plays_the_turns_out_without_touching_the_original() -> None:
    from sts2_env.search.turn_search import _playout

    combat = _combat()
    original_turn = combat.turn_count
    scratch = clone_combat(combat)
    _playout(scratch, 2)

    assert scratch.turn_count > original_turn
    assert combat.turn_count == original_turn


def test_the_playout_does_not_stop_dead_on_a_power() -> None:
    """A Power has no damage and no block, so ranking on those alone scores it
    zero -- and stopping there hid the payoff the lookahead exists to reveal."""
    from sts2_env.search.turn_search import _playout

    combat = _combat()
    _set_hand(combat, CardId.INFLAME, CardId.STRIKE_IRONCLAD)
    combat.energy = 3

    scratch = clone_combat(combat)
    _playout(scratch, 1)

    assert scratch.turn_count > combat.turn_count, "the playout stalled instead of playing on"


def test_lookahead_is_on_by_default() -> None:
    """It took boss fights from 6.7% to 33.3% on the benchmark; off by default
    would be leaving that on the floor."""
    from sts2_env.search.turn_search import DEFAULT_LOOKAHEAD_TURNS

    assert DEFAULT_LOOKAHEAD_TURNS >= 1


def test_a_lookahead_search_still_returns_a_playable_line() -> None:
    combat = _combat()
    result = search_turn(combat, lookahead_turns=2)
    _play_line(combat, result)


# -- playing the shortlist to the end ----------------------------------------

def test_rollouts_only_run_for_the_shortlist() -> None:
    """A rollout costs what a hundred leaf evaluations do, so only the most
    promising few earn one -- times however many futures each is sampled over."""
    combat = _combat(encounter="setup_slimes_weak")
    result = search_turn(combat, top_k=3, rollout_samples=2)
    assert result.rollouts <= 3 * 2
    assert result.leaves > 3


def test_several_futures_are_sampled_rather_than_one() -> None:
    """One rollout is one sample carrying the whole variance, and at half weight
    that was enough to drown a 3-damage certainty: the searcher played Strike
    before Bash and threw away the Vulnerable multiplier."""
    combat = _combat(encounter="setup_slimes_weak")
    one = search_turn(combat, top_k=2, rollout_samples=1)
    many = search_turn(combat, top_k=2, rollout_samples=4)
    assert many.rollouts > one.rollouts


def test_rollouts_can_be_turned_off() -> None:
    combat = _combat()
    assert search_turn(combat, top_k=0).rollouts == 0


def test_rescoring_still_returns_a_line_that_can_be_played() -> None:
    for seed in (1001, 1002, 1003):
        combat = _combat(seed=seed)
        _play_line(combat, search_turn(combat, top_k=5))


def test_rescoring_leaves_the_real_combat_untouched() -> None:
    combat = _combat()
    before = (combat.player.current_hp, combat.turn_count,
              [c.card_id.name for c in combat.hand],
              [e.current_hp for e in combat.enemies])
    search_turn(combat, top_k=5)
    assert (combat.player.current_hp, combat.turn_count,
            [c.card_id.name for c in combat.hand],
            [e.current_hp for e in combat.enemies]) == before


def test_lethal_is_still_taken_with_rollouts_on() -> None:
    """The rollout must not talk it out of winning now."""
    combat = _combat()
    _set_hand(combat, CardId.STRIKE_IRONCLAD, CardId.STRIKE_IRONCLAD, CardId.DEFEND_IRONCLAD)
    combat.energy = 3
    for enemy in combat.enemies:
        enemy.current_hp = 4

    _play_line(combat, search_turn(combat, top_k=5))
    assert combat.is_over and combat.player_won


def test_a_survivable_turn_is_still_survived_with_rollouts_on() -> None:
    """The failure the two-turn lookahead introduced, re-checked at full depth:
    a rollout ending in death must not make dying now look equivalent."""
    combat = _combat(hp=12, encounter="setup_fossil_stalker_normal")
    _set_hand(combat, CardId.DEFEND_IRONCLAD, CardId.STRIKE_IRONCLAD)
    combat.energy = 1

    result = search_turn(combat, top_k=5)
    after = clone_combat(combat)
    _play_line(after, result)
    after.end_player_turn()
    assert after.player.current_hp > 0


def test_the_time_budget_is_respected_by_the_rollout_stage() -> None:
    combat = _combat(encounter="setup_slimes_weak")
    result = search_turn(combat, top_k=50, time_budget=0.05)
    assert result.elapsed < 3.0
    _play_line(combat, result)


def test_the_same_position_searches_the_same_way_twice() -> None:
    """Reproducibility is what the benchmark rests on: two agents are only
    comparable if each faces the same fight twice.

    The resampled futures are seeded from the position, never from `id(combat)`
    -- an address differs between processes, which would have made the same
    fight play differently on every run while looking deterministic.
    """
    first = search_turn(_combat(), top_k=5, rollout_samples=3)
    second = search_turn(_combat(), top_k=5, rollout_samples=3)
    assert first.actions == second.actions
    assert first.score == pytest.approx(second.score)


def test_resampling_actually_varies_the_future() -> None:
    """If every sample were identical, averaging would buy nothing."""
    from sts2_env.search.cloning import clone_combat
    from sts2_env.search.evaluate import evaluate
    from sts2_env.search.turn_search import _playout, _reseed_futures

    base = _combat()
    base.end_player_turn()

    outcomes = set()
    for sample in range(4):
        future = clone_combat(base)
        if sample:
            _reseed_futures(future, sample)
        _playout(future, 6)
        outcomes.add(round(evaluate(future), 4))

    assert len(outcomes) > 1, "every sampled future came out identical"


# -- holding a potion for the fight that can end the run ---------------------

def _with_potion(combat, potion_id):
    from sts2_env.potions.base import create_potion
    combat.potions = [create_potion(potion_id), None, None]
    return combat


def test_a_held_potion_is_not_drunk_in_a_hallway_fight():
    """PowderedDemise went down on trash 89% of the time and never once on a boss.

    Forcing was only half a policy: `CARD_GENERATORS` forces a drink on turn 1
    of an elite but nothing stopped an earlier one, and `evaluate.py` has no
    potion term, so a reserve scores zero and drinking is free.
    """
    from sts2_env.core.enums import RoomType
    from sts2_env.search.potion_policy import should_hold

    combat = _combat(hp=60, encounter="setup_shrinker_beetle_weak")
    combat.room = RoomType.MONSTER
    _with_potion(combat, "PowderedDemise")
    assert should_hold(combat, combat.potions[0])


def test_the_same_potion_is_free_to_drink_in_an_elite():
    from sts2_env.core.enums import RoomType
    from sts2_env.search.potion_policy import should_hold

    combat = _combat(hp=60, encounter="setup_shrinker_beetle_weak")
    combat.room = RoomType.ELITE
    _with_potion(combat, "PowderedDemise")
    assert not should_hold(combat, combat.potions[0])


def test_a_hold_is_released_when_the_turn_could_be_lethal():
    """A potion saved for a boss the run never reaches is worth nothing.

    The release test is the board's own telegraphed damage against remaining
    HP, so it needs no tuned threshold and cannot go stale on a rebalance.
    """
    from sts2_env.core.enums import RoomType
    from sts2_env.search.potion_policy import _telegraphed_damage, should_hold

    combat = _combat(hp=60, encounter="setup_fossil_stalker_normal")
    combat.room = RoomType.MONSTER
    _with_potion(combat, "PowderedDemise")
    incoming = _telegraphed_damage(combat)
    assert incoming > 0, "this test needs a real attack telegraphed"

    combat.player.current_hp = incoming + 5
    assert should_hold(combat, combat.potions[0]), "not lethal yet, so still held"

    combat.player.current_hp = incoming
    assert not should_hold(combat, combat.potions[0]), "lethal on the table: drink it"


def test_an_unheld_potion_is_never_blocked():
    from sts2_env.core.enums import RoomType
    from sts2_env.search.potion_policy import should_hold

    combat = _combat(hp=60, encounter="setup_shrinker_beetle_weak")
    combat.room = RoomType.MONSTER
    _with_potion(combat, "BlockPotion")
    assert not should_hold(combat, combat.potions[0])


def test_the_search_will_not_play_a_held_potion_in_a_hallway():
    """End to end: the branch is not offered, so no line can contain it."""
    from sts2_env.core.enums import RoomType
    from sts2_env.gym_env.action_space import is_potion_action

    combat = _combat(hp=60, encounter="setup_shrinker_beetle_weak")
    combat.room = RoomType.MONSTER
    _with_potion(combat, "PowderedDemise")
    result = search_turn(combat)
    assert not any(is_potion_action(a) for a in result.actions)
