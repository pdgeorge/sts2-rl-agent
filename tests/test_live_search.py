"""LiveSearch builds a local mirror of the live fight and asks the SearchAgent
to play it. The seam is small: bridge JSON -> CombatSituation.from_bridge_state
-> to_combat -> SearchAgent.act -> action index (same Discrete(115) layout
the runner already decodes), and the local sim mirrors the live game's
actions on every subsequent call.

Tests use the same mock-bridge-payload shape as test_combat_situation_from_bridge
(the Phase 1.1 spec from PR #6), so the live path this exercises will match
what the real bridge sends once the C# mod patch is compiled in.
"""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.bridge.live_search import LiveSearch
from sts2_env.core.constants import ACTION_END_TURN, ACTION_SPACE_SIZE


# Reuse a fixture-payload helper in the shape the Phase 1.1 spec defines. The
# deck is the actual Ironclad starter so the SearchAgent has real cards to
# plan with. encounter_seed is fixed so two runs of a test pick the same HP.
def _bridge_state(*, round_number=1, hp=72, block=0, energy=3,
                  hand_size=5, **overrides):
    deck = (["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"]
            + [{"id": "BASH", "upgraded": True}])[:10]
    hand_ids = ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "STRIKE_IRONCLAD",
                "BASH", "DEFEND_IRONCLAD"][:hand_size]
    hand = [
        {"id": cid, "cost": 1, "type": "Attack" if "STRIKE" in cid else "Skill",
         "target": "AnyEnemy" if "STRIKE" in cid or cid == "BASH" else "Self",
         "playable": True}
        for cid in hand_ids
    ]
    state = {
        "type": "combat_action",
        "floor": 2,
        "act": 1,
        "act_floor": 2,
        "room_type": "Monster",
        "run_hp": hp,
        "run_max_hp": 80,
        "gold": 99,
        "deck_size": 10,
        "relic_count": 1,
        "potion_count": 0,
        "max_potion_slots": 3,
        "ascension": 0,
        "relics": ["BURNING_BLOOD"],
        "potion_slots": [None, None, None],
        "deck": deck,
        "character_id": "Ironclad",
        "encounter": "NibbitsWeak",
        "encounter_seed": 4242,
        "combat_seed": 4242,
        "round": round_number,
        "combat_state": {
            "player": {"hp": hp, "max_hp": 80, "block": block,
                       "energy": energy, "max_energy": 3},
            "hand": hand,
            "enemies": [{"id": "NIBBIT", "hp": 16, "max_hp": 16,
                         "is_alive": True}],
        },
    }
    state.update(overrides)
    return state


# -- the first decide: build local sim and pick a legal action ---------------

def test_first_decide_returns_a_legal_action_in_discrete_115() -> None:
    ls = LiveSearch(time_budget=0.5)  # short for test speed
    action = ls.decide(_bridge_state(), prev_action=None)
    assert 0 <= action < ACTION_SPACE_SIZE


def test_first_decide_invokes_search_at_least_once() -> None:
    ls = LiveSearch(time_budget=0.5)
    ls.decide(_bridge_state(), prev_action=None)
    assert ls.stats["searches"] >= 1
    assert ls.stats["rebuild_count"] == 1


# -- subsequent decide: mirror and pick another legal action -----------------

def test_subsequent_decides_also_return_legal_actions() -> None:
    ls = LiveSearch(time_budget=0.5)
    a1 = ls.decide(_bridge_state(), prev_action=None)
    a2 = ls.decide(_bridge_state(), prev_action=a1)
    assert 0 <= a2 < ACTION_SPACE_SIZE
    # No rebuild on subsequent calls -- same fight, same local sim.
    assert ls.stats["rebuild_count"] == 1


def test_search_keeps_planning_across_calls_within_a_turn() -> None:
    """The SearchAgent plans once per turn and replays. We mirror that:
    the first decide replans (1 search), subsequent calls within the same
    turn typically replay the plan (0 new searches)."""
    ls = LiveSearch(time_budget=0.5)
    ls.decide(_bridge_state(), prev_action=None)
    searches_after_first = ls.stats["searches"]
    ls.decide(_bridge_state(), prev_action=None)
    # The second call may or may not replan depending on the turn -- one
    # search per turn is the SearchAgent's design. The assertion is that
    # the planner was used at most twice over two decide calls.
    assert ls.stats["searches"] <= searches_after_first + 1


# -- per-fight lifecycle -----------------------------------------------------

def test_reset_for_new_fight_clears_the_local_sim() -> None:
    ls = LiveSearch(time_budget=0.5)
    ls.decide(_bridge_state(), prev_action=None)
    assert ls.stats["rebuild_count"] == 1
    ls.reset_for_new_fight()
    ls.decide(_bridge_state(), prev_action=None)
    # A new fight forces a fresh build.
    assert ls.stats["rebuild_count"] == 2


# -- drift detection logs but does not crash ---------------------------------

def test_drift_is_logged_but_does_not_break_decide() -> None:
    """HP drift beyond tolerance is logged, not raised. The SearchAgent
    continues with the local sim; the live game's mask is the final
    authority on legality."""
    ls = LiveSearch(time_budget=0.5)
    ls.decide(_bridge_state(hp=72), prev_action=None)
    # The next bridge state reports a wildly different HP -- drift.
    action = ls.decide(_bridge_state(hp=30), prev_action=None)
    assert 0 <= action < ACTION_SPACE_SIZE
    assert ls.stats["drift_count"] == 1


# -- when the bridge is missing encounter info, the build raises loud -------

def test_missing_encounter_raises_on_first_decide() -> None:
    """If the mod has not been patched (Phase 1.1), the build should fail
    loudly rather than silently building a random fight."""
    state = _bridge_state()
    del state["encounter"]
    ls = LiveSearch(time_budget=0.5)
    with pytest.raises(ValueError, match="encounter"):
        ls.decide(state, prev_action=None)


# -- the action is decoded by state_adapter just like the model's would ------

def test_the_action_decodes_via_state_adapter() -> None:
    """Pin the contract: the live-search action is decoded by the same
    StateAdapter.decode_action call the runner uses for the model path.
    If the SearchAgent and the adapter disagree on what an action index
    means, the live game would receive an action different from the one
    the search planned -- the bug class MODELS.md:5 warns about."""
    from sts2_env.bridge.state_adapter import StateAdapter

    adapter = StateAdapter()
    state = _bridge_state()
    ls = LiveSearch(time_budget=0.5)
    action = ls.decide(state, prev_action=None)

    # The adapter must accept the action without raising, and produce a
    # bridge-action dict with a `type` key the client knows.
    decoded = adapter.decode_action(action, state)
    assert "type" in decoded
    # ACTION_END_TURN (0) decodes to END_TURN; a card-play decodes to PLAY.
    if action == ACTION_END_TURN:
        assert decoded["type"] == "END_TURN"
    else:
        assert decoded["type"] == "PLAY"