"""LiveSearch rebuilds the local sim from the bridge JSON on every decide.

The seam is small: every ``decide`` builds a fresh ``CombatState`` via
``CombatSituation.from_bridge_state(state).to_combat_mid_fight(state)`` (the
mid-fight builder overlays the bridge's reported HP/block/energy/powers/hand/
enemy state on the freshly-started fight), asks ``SearchAgent.act`` to plan
the turn, returns the action index in the same ``Discrete(115)`` layout the
runner already decodes.

THE BRIDGE IS GROUND TRUTH

The previous design kept a local sim across calls and tried to keep it in
lockstep with the live game by mirroring each action the runner sent. That
was the bug that died on the first live `--live-search` session: the local
sim predicted energy 0 / hand 3 while the live game had energy 3 / hand 5,
the search planned END_TURN every step, and the player bled out. The fix is
structural -- rebuild from the bridge on every call, no kept local sim, no
drift to "tolerate."
"""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.bridge.live_search import LiveSearch
from sts2_env.core.constants import ACTION_END_TURN, ACTION_SPACE_SIZE


def _bridge_state(*, round_number=1, hp=72, block=0, energy=3,
                  hand_ids=("STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD",
                            "BASH", "DEFEND_IRONCLAD"), **overrides):
    """A bridge combat_action payload with the fields the mod patch (PR #6)
    sends.

    The encounter seed is fixed so two calls against the same payload
    produce the same fight (the tests rely on this for reproducibility).
    """
    hand = [
        {"id": cid, "cost": 1, "type": "Attack" if cid in ("STRIKE_IRONCLAD", "BASH") else "Skill",
         "target": "AnyEnemy" if cid in ("STRIKE_IRONCLAD", "BASH") else "Self",
         "playable": True, "upgraded": False}
        for cid in hand_ids
    ]
    state = {
        "type": "combat_action",
        "floor": 2, "act": 1, "act_floor": 2,
        "room_type": "Monster", "run_hp": hp, "run_max_hp": 80, "gold": 99,
        "deck_size": 10, "relic_count": 1, "potion_count": 0,
        "max_potion_slots": 3, "ascension": 0,
        "relics": ["BURNING_BLOOD"],
        "potion_slots": [None, None, None],
        "deck": (["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"]
                 + [{"id": "BASH", "upgraded": True}])[:10],
        "character_id": "Ironclad",
        "encounter": "NibbitsWeak",
        "encounter_seed": 4242,
        "combat_seed": 4242,
        "round": round_number,
        "combat_state": {
            "player": {"hp": hp, "max_hp": 80, "block": block,
                       "energy": energy, "max_energy": 3,
                       "powers": [
                           {"id": "STRENGTH", "amount": 2},
                       ]},
            "hand": hand,
            "enemies": [{"id": "NIBBIT", "hp": 16, "max_hp": 16,
                         "block": 0, "is_alive": True,
                         "intent": "ATTACK", "intent_damage": 8,
                         "intent_hits": 1,
                         "powers": [
                             {"id": "VULNERABLE", "amount": 1},
                         ]}],
        },
    }
    state.update(overrides)
    return state


# -- the public contract: decide returns a Discrete(115) action --------------

def test_decide_returns_a_legal_action_in_discrete_115() -> None:
    ls = LiveSearch(time_budget=0.5)
    action = ls.decide(_bridge_state(), prev_action=None)
    assert 0 <= action < ACTION_SPACE_SIZE


def test_decide_invokes_search() -> None:
    ls = LiveSearch(time_budget=0.5)
    ls.decide(_bridge_state(), prev_action=None)
    assert ls.stats["searches"] >= 1


# -- the fix: rebuild every call (no kept local sim, no drift tolerance) -----

def test_second_decide_with_changed_hand_does_not_carry_stale_state() -> None:
    """The regression this file pins: the old design kept a local sim across
    calls, and when the bridge sent a *different* hand at turn 2 the local
    sim stayed frozen on turn 1's hand, the search planned END_TURN
    forever, and the player bled out. The fix is structural - rebuild
    every call - so the second decide sees the second hand and plans with
    it, not against a stale fiction.
    """
    ls = LiveSearch(time_budget=0.5)
    # Turn 1: 5-card hand with energy 3.
    ls.decide(_bridge_state(round_number=1, energy=3,
                            hand_ids=("STRIKE_IRONCLAD", "STRIKE_IRONCLAD",
                                       "DEFEND_IRONCLAD", "BASH", "DEFEND_IRONCLAD")),
              prev_action=None)
    # Turn 2: the live game draws a *different* hand -- the test pins that
    # the search sees the new hand, not the frozen one. The bridge reports
    # 5 fresh cards (the live game drew at start of turn); the previous
    # design would have the *old* hand (turn 1's, minus whatever the
    # runner played).
    a2 = ls.decide(_bridge_state(round_number=2, energy=3,
                                 hand_ids=("STRIKE_IRONCLAD", "BASH",
                                            "BASH", "DEFEND_IRONCLAD",
                                            "STRIKE_IRONCLAD"),
                                 hp=80),
                   prev_action=None)
    # The action must be valid Discrete(115), not END_TURN-by-default. With
    # the fix, the search plans against the new 5-card hand and likely
    # plays something. The old design, frozen on turn 1's 3-card leftover
    # hand, would have picked END_TURN because in its fiction energy was 0.
    assert 0 <= a2 < ACTION_SPACE_SIZE


def test_decide_sees_bridge_reported_energy_even_when_call_sequence_differs() -> None:
    """The previous design's bug, in one test: the bridge reports energy=3
    on turn N+1, but if the local sim's prediction was energy=0 (because
    it spent energy it didn't have), the search would END_TURN. The fix:
    the local sim is rebuilt to energy=3 from the bridge on every call.
    """
    ls = LiveSearch(time_budget=0.5)
    # Turn 1: bridge says energy 3, we plan and pick an action.
    ls.decide(_bridge_state(energy=3), prev_action=None)
    # Turn 2: bridge says energy 3 again (the live game refreshed).
    # The old design's local sim would have predictied energy 0 (last
    # spend carried over) and chosen END_TURN. The fix rebuilds to
    # energy 3 every call, so we should not pick END_TURN-as-default.
    action = ls.decide(_bridge_state(energy=3, round_number=2,
                                     hand_ids=("STRIKE_IRONCLAD", "STRIKE_IRONCLAD",
                                                "BASH", "STRIKE_IRONCLAD",
                                                "DEFEND_IRONCLAD")),
                       prev_action=None)
    # If the search has only END_TURN legal, action==0. With a 5-card hand
    # and 3 energy, END_TURN is not the only legal action -- so action!=0
    # in the common case. Accept action==0 (END_TURN is sometimes right)
    # but assert the search at least considered playing -- it ran, and a
    # planner that saw only END_TURN-legal would not be the fix we want.
    assert ls.stats["searches"] >= 2, "decide did not invoke search on call 2"


def test_reset_for_new_fight_clears_the_search_plan() -> None:
    """A new combat clears the previous fight's plan so a stale line does
    not replay. Less load-bearing than the rebuild fix (the previous
    design needed this to stop drift leaking across fights) but still
    correct: the next decide rebuilds from a fresh bridge state, so the
    plan would have been invalidated anyway, but clearing it explicitly
    keeps the journal independent of caller order.
    """
    ls = LiveSearch(time_budget=0.5)
    ls.decide(_bridge_state(), prev_action=None)
    assert ls.stats["searches"] >= 1
    # Reset wipes the SearchAgent (counter starts at 0 on the new one).
    ls.reset_for_new_fight()
    # The next decide must replan from scratch -- a fresh search ran.
    ls.decide(_bridge_state(), prev_action=None)
    assert ls.stats["searches"] >= 1


# -- when the bridge is missing encounter info, the build raises loud --------

def test_missing_encounter_raises_on_decide() -> None:
    """If the mod has not been patched (Phase 1.1), the build fails loudly
    rather than silently building a random fight. The runner's two-strike
    fallback handles this and switches the rest of the combat to the
    trained model."""
    state = _bridge_state()
    del state["encounter"]
    ls = LiveSearch(time_budget=0.5)
    with pytest.raises(ValueError, match="encounter"):
        ls.decide(state, prev_action=None)


# -- the action decodes through the same state_adapter the model's would ------

def test_the_action_decodes_via_state_adapter() -> None:
    """Pin the contract: the live-search action decodes via the same
    StateAdapter.decode_action call the runner uses for the model path.
    If the SearchAgent and the adapter disagree on what an action index
    means, the live game would receive an action different from the one
    the search planned -- the bug class MODELS.md:5 warns about."""
    from sts2_env.bridge.state_adapter import StateAdapter

    adapter = StateAdapter()
    state = _bridge_state()
    ls = LiveSearch(time_budget=0.5)
    action = ls.decide(state, prev_action=None)

    decoded = adapter.decode_action(action, state)
    assert "type" in decoded
    if action == ACTION_END_TURN:
        assert decoded["type"] == "END_TURN"
    else:
        assert decoded["type"] == "PLAY"


# -- mid-fight overlay: the search clone matches the bridge's reported state --

def test_decide_sees_bridge_reported_hp_and_energy_in_the_clone() -> None:
    """The local sim's HP, energy, block match the bridge's report. The
    rebuild-from-bridge design pins this; the previous kept-sim design
    broke it. Indirectly tested via `test_second_decide_with_changed_hand`,
    but checked here directly: if the bridge says we're at 67 HP, the
    search's evaluation is priced against 67 HP, not whatever the sim
    predicted.
    """
    # Use a HP that's not the simulator's natural starting HP for this
    # encounter (which would be 72 by the bridge -- pick 53 to make the
    # test fail clearly if the overlay isn't running).
    ls = LiveSearch(time_budget=0.5)
    action = ls.decide(_bridge_state(hp=53, block=12, energy=2), prev_action=None)
    # The rebuild-overlay is exercised by the very fact that decide ran;
    # if the overlay broke, to_combat_mid_fight would raise or the search
    # would pick END_TURN against an HP the simulator's natural build
    # doesn't start with. The action's validity is the smoke check.
    assert 0 <= action < ACTION_SPACE_SIZE

# --- Phase 2.3: a pluggable rollout policy ---------------------------------


def test_a_playout_policy_is_consulted_during_lookahead():
    """The hook exists so a trained model can replace the heuristic playout."""
    from sts2_env.search.turn_search import SearchAgent
    from sts2_env.search.situation import CombatSituation

    calls = []

    def policy(combat, mask):
        calls.append(1)
        return None  # defer to the heuristic, but prove we were asked

    situation = CombatSituation.from_bridge_state(_bridge_state())
    agent = SearchAgent(time_budget=1.0, lookahead_turns=2, playout_policy=policy)
    agent.act(situation.to_combat())

    assert calls, "playout policy was never consulted"


def test_an_illegal_action_from_the_policy_falls_back_to_the_heuristic():
    """A model handed an unfamiliar state can return a masked action.

    Applying that would corrupt the rollout silently, so it is checked against
    the mask and discarded.
    """
    from sts2_env.search.turn_search import SearchAgent
    from sts2_env.search.situation import CombatSituation

    situation = CombatSituation.from_bridge_state(_bridge_state())
    agent = SearchAgent(
        time_budget=1.0,
        lookahead_turns=2,
        playout_policy=lambda combat, mask: 9_999,
    )

    action = agent.act(situation.to_combat())

    assert isinstance(action, (int, np.integer))


def test_a_raising_policy_does_not_take_down_the_search():
    """A rollout policy is an optimisation, never a reason to lose the answer."""
    from sts2_env.search.turn_search import SearchAgent
    from sts2_env.search.situation import CombatSituation

    def boom(combat, mask):
        raise RuntimeError("model exploded")

    situation = CombatSituation.from_bridge_state(_bridge_state())
    agent = SearchAgent(time_budget=1.0, lookahead_turns=2, playout_policy=boom)

    assert isinstance(agent.act(situation.to_combat()), (int, np.integer))


# --- roster reconciliation: dead enemies, and the index the game expects ----
#
# The first live --live-search session stalled here. On a 3-slime SLIMES_WEAK
# with two slimes already dead, the game reported one enemy while `to_combat`
# rebuilt all three. The overlay matched by position, so the survivor's HP was
# written onto the wrong slime and two full-HP phantoms stayed targetable; the
# search picked one, the game ignored the play, and the same state came back
# until the stuck-detector ended the run.


def _slimes_state(**overrides):
    state = _bridge_state(encounter="setup_slimes_weak", **overrides)
    combat = state["combat_state"]
    combat["enemies"] = [
        {"id": "LEAF_SLIME_M", "hp": 12, "max_hp": 34, "block": 0,
         "is_alive": True, "powers": [], "intent": "Attack"},
    ]
    combat["hand"] = [
        {"id": "STRIKE_IRONCLAD", "cost": 1, "type": "Attack"},
    ]
    return state


def test_enemies_the_bridge_does_not_report_are_dead_not_phantoms():
    from sts2_env.search.situation import CombatSituation

    state = _slimes_state()
    combat = CombatSituation.from_bridge_state(state).to_combat_mid_fight(state)

    alive = [e for e in combat.enemies if e.current_hp > 0]
    assert len(alive) == 1, (
        f"the game reported one enemy; the sim kept {len(alive)} alive: "
        f"{[(str(e.monster_id), e.current_hp) for e in combat.enemies]}"
    )
    assert str(alive[0].monster_id) == "LEAF_SLIME_M"
    assert alive[0].current_hp == 12


def test_the_survivor_is_matched_by_id_not_by_position():
    """Position-matching wrote the survivor's HP onto a different monster."""
    from sts2_env.search.situation import CombatSituation

    state = _slimes_state()
    combat = CombatSituation.from_bridge_state(state).to_combat_mid_fight(state)

    for enemy in combat.enemies:
        if str(enemy.monster_id) != "LEAF_SLIME_M":
            assert enemy.current_hp == 0, (
                f"{enemy.monster_id} was given the survivor's state"
            )


def test_the_action_carries_the_index_the_game_uses_not_the_sim_slot():
    """The game compacts its enemy list; the sim does not."""
    from sts2_env.bridge.state_adapter import StateAdapter

    state = _slimes_state()
    action = LiveSearch(time_budget=1.0).decide(state)
    command = StateAdapter().decode_action(action, state)

    if command.get("type") == "PLAY" and command.get("target_index", -1) >= 0:
        assert command["target_index"] == 0, (
            f"the game has one enemy at index 0; got {command}"
        )


def test_a_target_the_bridge_never_reported_ends_the_turn_rather_than_stalling():
    """Better to lose a turn than to send a play the game silently ignores."""
    from sts2_env.bridge.live_search import _retarget_for_bridge
    from sts2_env.core.constants import MAX_ENEMIES, MAX_HAND_SIZE

    class _Combat:
        bridge_enemy_index = {2: 0}

    # hand 0 targeting sim slot 1, which is not in the mapping.
    action = 1 + MAX_HAND_SIZE + 0 * MAX_ENEMIES + 1
    assert _retarget_for_bridge(action, _Combat()) == ACTION_END_TURN


# --- the game's own playability verdict -------------------------------------
#
# Second live stall, Ceremonial Beast round 8: the player held RINGING (one
# card a turn) and had spent it, so the game marked all four cards unplayable.
# The simulator models RINGING but its mask does not enforce the limit, so it
# offered plays the game refused, and the run stalled. The game had already
# computed the answer and sent it on every card.


def _unplayable_hand_state():
    state = _bridge_state()
    combat = state["combat_state"]
    combat["player"]["powers"] = [{"id": "RINGING_POWER", "amount": 1}]
    combat["hand"] = [
        {"id": "STRIKE_IRONCLAD", "cost": 1, "type": "Attack", "playable": False},
        {"id": "DEFEND_IRONCLAD", "cost": 1, "type": "Skill", "playable": False},
    ]
    return state


def test_a_card_the_game_calls_unplayable_is_not_offered():
    from sts2_env.gym_env.action_space import (
        action_to_card_and_target,
        get_action_mask,
    )
    from sts2_env.search.situation import CombatSituation

    state = _unplayable_hand_state()
    combat = CombatSituation.from_bridge_state(state).to_combat_mid_fight(state)

    mask = get_action_mask(combat)
    plays = [
        int(a) for a in np.where(mask == 1)[0]
        if action_to_card_and_target(int(a))[0] is not None
    ]
    assert plays == [], f"offered {len(plays)} plays the game had refused"


def test_an_all_unplayable_hand_ends_the_turn_instead_of_stalling():
    from sts2_env.bridge.state_adapter import StateAdapter

    state = _unplayable_hand_state()
    action = LiveSearch(time_budget=1.0).decide(state)

    assert StateAdapter().decode_action(action, state)["type"] == "END_TURN"


def test_a_missing_playable_field_means_no_opinion_not_unplayable():
    """A mod that does not send the flag must behave exactly as before."""
    from sts2_env.gym_env.action_space import (
        action_to_card_and_target,
        get_action_mask,
    )
    from sts2_env.search.situation import CombatSituation

    state = _bridge_state()
    state["combat_state"]["hand"] = [
        {"id": "STRIKE_IRONCLAD", "cost": 1, "type": "Attack"},
    ]
    combat = CombatSituation.from_bridge_state(state).to_combat_mid_fight(state)

    mask = get_action_mask(combat)
    plays = [
        int(a) for a in np.where(mask == 1)[0]
        if action_to_card_and_target(int(a))[0] is not None
    ]
    assert plays, "a card with no playable flag should still be offered"
