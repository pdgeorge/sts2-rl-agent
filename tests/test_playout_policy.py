"""The playout that scores the lookahead, and the three things it could not see.

WHY THIS FILE EXISTS
--------------------
`search_turn` scores a candidate line by ending the turn on a copy, letting the
enemies reply, and then playing `lookahead_turns` more turns with a cheap
policy. That policy is the lookahead's eyesight: a line's future is only ever as
visible as the policy playing it out.

The old one scored every card as `block if a hit is coming else damage`, and any
Power at a flat 0.5. Three consequences, all measured over the 13,251 live card
plays of the 2026-08-15 session (n=100 runs):

  - Block and damage were compared as though they shared a unit, so a 6-damage
    Strike beat a 5-block Defend on a turn that needed block. 6 > 5.
  - A Power ranked below the worst attack in the deck, so the playout whose
    entire job was to reveal a scaling card's payoff never played one. The live
    power-play rate came out FLAT in fight length -- 2.32% in 1-2 turn fights
    against 2.25% in 8+ turn fights, where the long fights are what a Power is
    for.
  - Nothing separated killing an enemy from chipping it, so the agent blocked
    its way through 5.2-turn elite fights at 7.2 damage a turn.

`DEFAULT_TOP_K` and `MODELS.md:97` record the attempt to fix this with depth
instead: +0.5% +/- 1.1% win rate, and the power rate just as flat, because a
deeper rollout of a policy that never plays a Power still never plays a Power.

The new policy scores every card in ONE unit -- HP -- so the comparisons above
are possible at all. These tests pin the three behaviours that unit buys, not
the numbers that produce them.
"""

from __future__ import annotations

import numpy as np

from sts2_env.cards.factory import create_card
from sts2_env.core.enums import CardId
from sts2_env.gym_env.action_space import action_to_card_and_target, get_action_mask
from sts2_env.core.constants import ACTION_END_TURN
from sts2_env.search.situation import CardRef, CombatSituation
from sts2_env.search.turn_search import _heuristic_playout_action


def _combat(*, hp=60, max_hp=80, encounter="setup_shrinker_beetle_weak", seed=1001):
    return CombatSituation(
        situation_id="playout-test",
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


def _chosen_card(combat, turns_remaining: int) -> str | None:
    """The card the playout policy would play here, by name."""
    mask = get_action_mask(combat)
    actions = [int(a) for a in np.where(mask == 1)[0] if a != ACTION_END_TURN]
    action = _heuristic_playout_action(combat, actions, turns_remaining)
    if action is None:
        return None
    hand_index, _ = action_to_card_and_target(action)
    if hand_index is None or hand_index >= len(combat.hand):
        return None
    return combat.hand[hand_index].card_id.name


# -- the unit bug: 6 damage is not bigger than 5 block ------------------------


def test_it_blocks_rather_than_attacking_when_a_hit_is_coming():
    """The old policy played Strike here because 6 > 5, comparing damage to
    block as though they were the same quantity."""
    combat = _combat()
    _set_hand(combat, CardId.STRIKE_IRONCLAD, CardId.DEFEND_IRONCLAD)
    combat.player.block = 0
    # A telegraphed hit that Defend can meaningfully reduce.
    incoming = sum(
        intent.damage or 0
        for enemy in combat.enemies
        if enemy.is_alive
        for intent in combat.enemy_ais[enemy.combat_id].current_move.intents
    )
    if incoming <= 0:
        return  # this seed telegraphed no attack; the case does not apply

    assert _chosen_card(combat, turns_remaining=2) == "DEFEND_IRONCLAD", (
        "with damage incoming the playout must block; scoring 6 damage against "
        "5 block as though they were one unit is what it used to do"
    )


def test_block_past_the_telegraphed_hit_is_worth_nothing():
    """Block already covering the hit means the next card should not be more
    block. Same rule as `EvalWeights.block_unused`."""
    combat = _combat()
    _set_hand(combat, CardId.STRIKE_IRONCLAD, CardId.DEFEND_IRONCLAD)
    combat.player.block = 999  # everything incoming is already covered

    assert _chosen_card(combat, turns_remaining=2) != "DEFEND_IRONCLAD", (
        "block beyond the incoming hit saves no HP, so it must not outrank an "
        "attack"
    )


# -- Powers are worth what the turns left will collect ------------------------


def test_a_power_is_played_when_there_are_turns_left_to_collect_it():
    combat = _combat()
    _set_hand(combat, CardId.STRIKE_IRONCLAD, CardId.INFLAME)
    combat.player.block = 999  # take block out of the comparison
    combat.current_player_state.energy = 3

    assert _chosen_card(combat, turns_remaining=5) == "INFLAME", (
        "over five turns +2 Strength beats one Strike; the old policy scored "
        "every Power at 0.5 and so never played one at any horizon"
    )


def test_the_same_power_loses_on_the_last_turn_of_the_horizon():
    """The mirror image, and the reason this is not just 'powers are good'."""
    combat = _combat()
    _set_hand(combat, CardId.STRIKE_IRONCLAD, CardId.INFLAME)
    combat.player.block = 999
    combat.current_player_state.energy = 3

    assert _chosen_card(combat, turns_remaining=1) == "STRIKE_IRONCLAD", (
        "a Power played on the last turn buys nothing -- if it wins here the "
        "value is not being scaled by the turns left to collect it"
    )


# -- killing the thing that is about to hit you -------------------------------


def test_it_takes_the_kill_over_the_block():
    """pd's thesis, priced.

    An enemy on low HP intending damage is worth its damage EVERY remaining
    turn if killed, and once if blocked. The old policy had no term for this at
    all -- a killing blow scored exactly its damage number, the same as a swing
    that left the enemy at 1 HP and still swinging.
    """
    combat = _combat()
    _set_hand(combat, CardId.STRIKE_IRONCLAD, CardId.DEFEND_IRONCLAD)
    combat.player.block = 0

    alive = [e for e in combat.enemies if e.is_alive]
    if len(alive) != 1:
        return  # the single-enemy case is what this pins; see WEEKEND_DECISIONS 1
    enemy = alive[0]
    enemy.current_hp = 1  # a Strike kills it outright
    enemy.block = 0

    assert _chosen_card(combat, turns_remaining=3) == "STRIKE_IRONCLAD", (
        "killing the enemy removes its damage for every turn that follows; "
        "blocking removes it once"
    )


def test_it_returns_none_when_nothing_is_worth_playing():
    """A policy that plays a worthless card burns the energy a later turn
    needed. None means 'stop', and `_playout` reads it that way."""
    combat = _combat()
    _set_hand(combat, CardId.DEFEND_IRONCLAD)
    combat.player.block = 999  # the only card in hand saves nothing

    assert _chosen_card(combat, turns_remaining=1) is None
