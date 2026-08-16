"""Tests for bridge agent non-combat choices."""

from sts2_env.bridge.agent_runner import (
    TERMINAL_PHASES,
    _phase_for_state,
    _pick_boss_relic_option,
    _pick_card_bundle_index,
    _pick_card_reward_index,
    _pick_card_select_indexes,
    _pick_crystal_sphere_option,
    _pick_map_node,
    _pick_reward_screen_option,
    _pick_rest_option,
    _pick_shop_option,
    _pick_treasure_option,
)
from sts2_env.bridge.protocol import BridgeStateType


def test_phase_mapping_treats_run_complete_as_terminal() -> None:
    phase = _phase_for_state({"type": BridgeStateType.RUN_COMPLETE, "result": "victory"})

    assert phase == BridgeStateType.RUN_COMPLETE
    assert phase in TERMINAL_PHASES


def test_map_policy_prefers_rest_when_hp_is_low() -> None:
    state = {
        "player": {"hp": 20, "max_hp": 80},
        "nodes": [
            {"index": 0, "type": "Elite"},
            {"index": 1, "type": "RestSite"},
            {"index": 2, "type": "Monster"},
        ],
    }

    assert _pick_map_node(state) == 1


def test_map_policy_takes_the_monster_over_the_elite_even_when_healthy() -> None:
    """The reverse of what this test used to assert, on 100 live runs.

    Act 1, pre-boss: 82 elite entries ended 19 runs (23%) at a mean 31.8 chip;
    705 monster entries ended 14 (2.0%) at 6.3. An elite is 11x more likely to
    end the run and costs five times the HP, and boss win falls off a cliff at
    80% HP on arrival -- 3% below it (1/31) against 54% at or above (19/35).

    It used to prefer the elite for the relic. Relics were then measured and are
    not the gap: offline carries FEWER relics (2.8 vs 4.7) and wins more.
    """
    state = {
        "player": {"hp": 70, "max_hp": 80},
        "nodes": [
            {"index": 0, "type": "RestSite"},
            {"index": 1, "type": "Monster"},
            {"index": 2, "type": "Elite"},
        ],
    }

    assert _pick_map_node(state) == 1


def test_card_reward_policy_prefers_power_and_skips_large_decks() -> None:
    state = {
        "cards": [
            {"index": 0, "id": "STRIKE_IRONCLAD", "type": "Attack"},
            {"index": 1, "id": "INFLAME", "type": "Power"},
        ],
        "can_skip": True,
        "run_state": {"deck": ["card"] * 10},
    }

    assert _pick_card_reward_index(state) == 1

    state["run_state"] = {"deck": ["card"] * 31}
    assert _pick_card_reward_index(state) is None


def test_card_select_policy_uses_required_card_count() -> None:
    state = {
        "cards": [{"index": 3}, {"index": 5}, {"index": 8}],
        "min_select": 2,
        "max_select": 3,
    }

    assert _pick_card_select_indexes(state) == [3, 5]


def test_reward_screen_policy_picks_rewards_before_proceeding() -> None:
    state = {
        "options": [
            {"index": 0, "action": "proceed", "enabled": True},
            {"index": 1, "action": "pick_reward", "enabled": True},
        ],
    }

    assert _pick_reward_screen_option(state) == 1


def test_card_bundle_policy_picks_enabled_bundle_by_action() -> None:
    state = {
        "bundles": [
            {"index": 0, "action": "inspect", "enabled": True},
            {"index": 3, "action": "pick_card_bundle", "enabled": True},
        ],
    }

    assert _pick_card_bundle_index(state) == 3


def test_crystal_sphere_policy_clicks_cells_before_proceeding() -> None:
    state = {
        "options": [
            {"index": 0, "action": "proceed", "enabled": True},
            {"index": 7, "action": "divine_cell", "x": 5, "y": 6, "enabled": True},
        ],
    }

    assert _pick_crystal_sphere_option(state) == 7


def test_rest_policy_uses_option_ids_not_order() -> None:
    state = {
        "player": {"hp": 70, "max_hp": 80},
        "options": [
            {"index": 0, "id": "HEAL", "enabled": True},
            {"index": 1, "id": "SMITH", "enabled": True},
        ],
    }

    assert _pick_rest_option(state) == 1

    state["player"] = {"hp": 20, "max_hp": 80}
    assert _pick_rest_option(state) == 0


def test_shop_policy_buys_before_leaving() -> None:
    state = {
        "options": [
            {"index": 0, "action": "leave_shop", "enabled": True},
            {"index": 1, "action": "buy_card", "enabled": True},
            {"index": 2, "action": "buy_relic", "enabled": True},
        ],
    }

    assert _pick_shop_option(state) == 2


def test_treasure_and_boss_relic_policy_use_action_labels() -> None:
    treasure = {
        "options": [
            {"index": 4, "action": "collect", "enabled": True},
        ],
    }
    boss_relic = {
        "options": [
            {"index": 2, "action": "pick_relic", "enabled": True},
        ],
    }

    assert _pick_treasure_option(treasure) == 4
    assert _pick_boss_relic_option(boss_relic) == 2


# ---------------------------------------------------------------------------
# HP economy. The old policy had one 50%-of-max threshold governing every room,
# which authorised exactly the fights that ended runs: 32 of 56 recorded elite
# choices were made at 40-59 HP, where the measured death rate is 18-29%.
# ---------------------------------------------------------------------------


def test_map_policy_refuses_an_elite_in_the_dangerous_band() -> None:
    """60% health cleared the old threshold and is where elites kill her."""
    state = {
        "player": {"hp": 48, "max_hp": 80},
        "nodes": [
            {"index": 0, "type": "Elite"},
            {"index": 1, "type": "Monster"},
        ],
    }

    assert _pick_map_node(state) == 1


def test_map_policy_routes_to_rest_when_a_room_is_unaffordable() -> None:
    state = {
        "player": {"hp": 48, "max_hp": 80},
        "nodes": [
            {"index": 0, "type": "Elite"},
            {"index": 1, "type": "RestSite"},
        ],
    }

    assert _pick_map_node(state) == 1


def test_map_policy_replays_the_floor_45_death() -> None:
    """The decision that actually ended the deepest live run.

    Floor 42 at 76/97 -- 78% health, comfortably "healthy" under the old 50%
    threshold. It took the act 3 elite, lost 58 HP, and died on floor 45 at 21
    with no rest site between. The elite must lose to the Unknown here.
    """
    state = {
        "floor": 42,
        "act": 3,
        "run_hp": 76,
        "run_max_hp": 97,
        "nodes": [
            {"index": 0, "type": "Unknown"},
            {"index": 1, "type": "Elite"},
        ],
    }

    assert _pick_map_node(state) == 0


def test_map_policy_still_takes_the_elite_when_it_is_the_only_fight() -> None:
    """Demoted, not deleted -- and this is the case that matters most.

    58 of the 87 live elite picks had NO non-elite alternative on the adjacent
    nodes; only 29 did. So demotion recovers about a third of the elites, and
    the rest need a route planner that can see the whole graph several floors
    ahead. Refusing to move is not an option the game offers.
    """
    state = {
        "player": {"hp": 93, "max_hp": 93},
        "nodes": [
            {"index": 0, "type": "RestSite"},
            {"index": 1, "type": "Elite"},
        ],
    }

    assert _pick_map_node(state) == 1


def test_map_policy_treats_unknown_as_a_fight() -> None:
    """Unknown nodes resolve to combat often enough to cost HP like one."""
    state = {
        "player": {"hp": 8, "max_hp": 80},
        "nodes": [
            {"index": 0, "type": "Unknown"},
            {"index": 1, "type": "Shop"},
        ],
    }

    assert _pick_map_node(state) == 1


def test_map_policy_still_moves_when_nothing_is_affordable() -> None:
    """Refusing to move is not an option the game offers.

    On a map with only fights and no HP to pay for them, the run has to take the
    cheapest one rather than stall -- a stalled run is the failure mode this
    whole file exists to avoid.
    """
    state = {
        "player": {"hp": 6, "max_hp": 80},
        "nodes": [
            {"index": 0, "type": "Elite"},
            {"index": 1, "type": "Monster"},
        ],
    }

    assert _pick_map_node(state) == 1


def test_rest_policy_heals_before_an_act_boss() -> None:
    """Smithing before a boss happened 17 times, at a median 49 HP.

    The measured death rate entering a boss at 40-49 HP is 88%, so the upgrade
    was being bought with the run.
    """
    state = {
        "floor": 16,
        "run_hp": 49,
        "run_max_hp": 80,
        "deck": [{"id": "ANGER"}],
        "options": [
            {"index": 0, "id": "HEAL", "enabled": True},
            {"index": 1, "id": "SMITH", "enabled": True},
        ],
    }

    assert _pick_rest_option(state) == 0

    # Healthy enough to fight it: the upgrade is worth more than the HP.
    state["run_hp"] = 70
    assert _pick_rest_option(state) == 1
