"""The non-combat choices that end runs on their own.

Two live runs died to these on 2026-08-05, neither of them in a fight. The
journal caught the first as it happened:

    floor 4  "Linger" over "Exit Baths"   hp 68 -> 66 -> 63 -> 59 -> 54 -> 48 -> 41

Hot Baths re-presents the same screen and charges more each time. Its labels are
`Linger` and `Exit Baths`, which contain no warning of any kind, so nothing that
reads words alone can save a run from it.
"""

from __future__ import annotations

import pytest

from sts2_env.bridge.agent_runner import (
    _is_basic_card,
    _pick_card_select_indexes,
    _pick_event_option,
    _pick_rest_option,
    _worth_upgrading,
)


def _event(options, hp=80, max_hp=80, event_id="TEST_EVENT"):
    return {
        "type": "event",
        "event_id": event_id,
        "run_hp": hp,
        "run_max_hp": max_hp,
        "options": [
            {"action": "event_choice", "id": "event_choice", "enabled": True,
             "index": i, "event_id": event_id, **o}
            for i, o in enumerate(options)
        ],
    }


# -- reading the option at all -----------------------------------------------

def test_the_cost_is_read_from_the_description_where_it_lives() -> None:
    """`id` is always the literal `event_choice`; the numbers are in
    `description` and the words are in `label`. The previous version read
    neither, so nothing it looked at ever mentioned HP."""
    state = _event(
        [
            {"label": "This", "description": "Lose [red]40[/red] HP. Gain [blue]43[/blue] Gold."},
            {"label": "That", "description": "Add [red]Clumsy[/red] to your Deck."},
        ],
        hp=50, max_hp=80,
    )
    assert _pick_event_option(state) == 1


def test_a_cheap_cost_at_full_health_is_still_taken() -> None:
    """The guard must not turn into a refusal to ever engage with an event."""
    state = _event(
        [
            {"label": "This", "description": "Lose [red]6[/red] HP. Gain 43 Gold."},
            {"label": "That", "description": "Add Clumsy to your Deck."},
        ],
        hp=80, max_hp=80,
    )
    assert _pick_event_option(state) == 0


# -- the ones that end runs --------------------------------------------------

def test_an_option_that_would_drop_below_the_floor_is_refused() -> None:
    state = _event(
        [
            {"label": "Offer blood", "description": "Lose [red]30[/red] HP."},
            {"label": "Leave", "description": ""},
        ],
        hp=45, max_hp=80,
    )
    assert _pick_event_option(state) == 1


def test_setting_max_hp_to_one_is_refused() -> None:
    """Not a death, which is the problem: the run continues and cannot survive
    its next fight."""
    state = _event(
        [
            {"label": "Accept", "description": "Your Max HP becomes [red]1[/red]."},
            {"label": "Decline", "description": ""},
        ],
        hp=70, max_hp=80,
    )
    assert _pick_event_option(state) == 1


def test_losing_a_large_slice_of_max_hp_is_refused() -> None:
    state = _event(
        [
            {"label": "Bathe", "description": "Lose [red]30[/red] Max HP."},
            {"label": "Leave", "description": ""},
        ],
        hp=80, max_hp=80,
    )
    assert _pick_event_option(state) == 1


def test_when_every_option_is_harmful_the_cheapest_is_taken() -> None:
    """And never the first one merely because it was first, which is what the
    old fall-through did after explicitly rejecting it."""
    state = _event(
        [
            {"label": "Much", "description": "Lose [red]40[/red] HP."},
            {"label": "Less", "description": "Lose [red]12[/red] HP."},
        ],
        hp=45, max_hp=80,
    )
    assert _pick_event_option(state) == 1


# -- the one that words cannot catch -----------------------------------------

def test_a_screen_that_charges_again_is_only_paid_once() -> None:
    """Hot Baths, exactly as it killed run 1.

    No description, no warning word: `Linger` and `Exit Baths`. What gives it
    away is that the same screen keeps coming back.
    """
    seen: dict[str, int] = {}
    options = [{"label": "Linger", "description": ""},
               {"label": "Exit Baths", "description": ""}]

    chosen = [
        _pick_event_option(_event(options, hp=hp, max_hp=80, event_id="HOT_BATHS"), seen)
        for hp in (68, 66, 63, 59, 54)
    ]

    assert chosen[0] == 0, "engaging once is a decision, not a mistake"
    assert chosen[-1] == 1, "it must eventually take the exit"
    assert chosen.count(0) <= 2, f"paid the same screen {chosen.count(0)} times"


def test_different_events_do_not_share_a_repeat_count() -> None:
    seen: dict[str, int] = {}
    first = _event([{"label": "Take", "description": ""}], event_id="EVENT_A")
    second = _event([{"label": "Take", "description": ""}], event_id="EVENT_B")
    _pick_event_option(first, seen)
    _pick_event_option(first, seen)
    assert _pick_event_option(second, seen) == 0


def test_an_event_with_one_option_is_still_answerable() -> None:
    state = _event([{"label": "Proceed", "description": ""}])
    assert _pick_event_option(state) == 0


def test_no_options_does_not_raise() -> None:
    assert _pick_event_option({"type": "event", "options": []}) == 0


# -- upgrading ---------------------------------------------------------------

@pytest.mark.parametrize("name,basic", [
    ("STRIKE_IRONCLAD", True),
    ("DEFEND_IRONCLAD", True),
    ("STRIKE_SILENT", True),
    ("DEFEND_NECROBINDER", True),
    ("PERFECTED_STRIKE", False),
    ("BLIGHT_STRIKE", False),
    ("BASH", False),
])
def test_which_cards_count_as_basic(name, basic) -> None:
    assert _is_basic_card({"id": name}) is basic


def test_an_upgrade_goes_to_a_real_card_not_a_strike() -> None:
    """A deck is listed Strike, Strike, Strike..., and taking the first card
    meant every rest-site upgrade went into a basic."""
    state = {
        "type": "card_select",
        "cards": [
            {"id": "STRIKE_IRONCLAD"}, {"id": "STRIKE_IRONCLAD"},
            {"id": "DEFEND_IRONCLAD"}, {"id": "BASH"}, {"id": "INFLAME"},
        ],
        "min_select": 1, "max_select": 1,
    }
    assert _pick_card_select_indexes(state) == [3]


def test_an_already_upgraded_card_is_not_chosen_again() -> None:
    state = {
        "type": "card_select",
        "cards": [
            {"id": "STRIKE_IRONCLAD"},
            {"id": "BASH", "upgraded": True},
            {"id": "INFLAME"},
        ],
        "min_select": 1, "max_select": 1,
    }
    assert _pick_card_select_indexes(state) == [2]


def test_with_only_basics_it_still_answers_the_screen() -> None:
    """Refusing is not an option once the screen is open -- leaving it unanswered
    is what makes a run sit forever on one room."""
    state = {
        "type": "card_select",
        "cards": [{"id": "STRIKE_IRONCLAD"}, {"id": "DEFEND_IRONCLAD"}],
        "min_select": 1, "max_select": 1,
    }
    assert _pick_card_select_indexes(state) == [0]


def test_a_deck_worth_upgrading_smiths() -> None:
    state = {
        "type": "rest_site",
        "run_hp": 70, "run_max_hp": 80,
        "deck": ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH", "INFLAME"],
        "options": [
            {"id": "HEAL", "action": "rest_option", "enabled": True, "index": 0},
            {"id": "SMITH", "action": "rest_option", "enabled": True, "index": 1},
        ],
    }
    assert _pick_rest_option(state) == 1


def test_a_deck_of_only_basics_rests_instead() -> None:
    """No upgrade here changes a fight, so the HP is worth more."""
    state = {
        "type": "rest_site",
        "run_hp": 70, "run_max_hp": 80,
        "deck": ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4,
        "options": [
            {"id": "HEAL", "action": "rest_option", "enabled": True, "index": 0},
            {"id": "SMITH", "action": "rest_option", "enabled": True, "index": 1},
        ],
    }
    assert _pick_rest_option(state) == 0


def test_a_deck_whose_real_cards_are_all_upgraded_rests() -> None:
    state = {
        "type": "rest_site",
        "run_hp": 70, "run_max_hp": 80,
        "deck": ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH+", "INFLAME+"],
        "options": [
            {"id": "HEAL", "action": "rest_option", "enabled": True, "index": 0},
            {"id": "SMITH", "action": "rest_option", "enabled": True, "index": 1},
        ],
    }
    assert _pick_rest_option(state) == 0


def test_being_hurt_still_beats_smithing() -> None:
    state = {
        "type": "rest_site",
        "run_hp": 20, "run_max_hp": 80,
        "deck": ["BASH", "INFLAME"],
        "options": [
            {"id": "HEAL", "action": "rest_option", "enabled": True, "index": 0},
            {"id": "SMITH", "action": "rest_option", "enabled": True, "index": 1},
        ],
    }
    assert _pick_rest_option(state) == 0


def test_worth_upgrading_is_the_rule_both_places_share() -> None:
    assert _worth_upgrading({"id": "INFLAME"})
    assert not _worth_upgrading({"id": "INFLAME", "upgraded": True})
    assert not _worth_upgrading({"id": "STRIKE_IRONCLAD"})
    assert not _worth_upgrading("BASH+")


# -- reading HP outside combat -----------------------------------------------

def test_hp_is_readable_on_a_screen_with_no_player_block() -> None:
    """The bug underneath the other two.

    Outside combat the bridge sends no `player` block, only `run_hp` /
    `run_max_hp`. `_read_hp_ratio` looked for `hp`/`max_hp` and returned None at
    every rest site, shop, event and map screen -- so every "when low on HP"
    branch in this module was unreachable in the only places it applied.
    """
    from sts2_env.bridge.agent_runner import _read_hp_ratio

    rest_site = {"type": "rest_site", "run_hp": 20, "run_max_hp": 80}
    assert _read_hp_ratio(rest_site) == pytest.approx(0.25)


def test_combat_states_still_read_from_the_player_block() -> None:
    from sts2_env.bridge.agent_runner import _read_hp_ratio

    combat = {"type": "combat_action", "player": {"hp": 40, "max_hp": 80}}
    assert _read_hp_ratio(combat) == pytest.approx(0.5)


def test_routing_avoids_elites_when_hurt() -> None:
    """The consequence that costs runs: the low-HP route was unreachable, so a
    hurt agent walked into the elite anyway."""
    from sts2_env.bridge.agent_runner import _pick_map_node

    state = {
        "type": "map_select",
        "run_hp": 15, "run_max_hp": 80,
        "nodes": [
            {"index": 0, "type": "Elite", "row": 1, "col": 1},
            {"index": 1, "type": "RestSite", "row": 1, "col": 2},
        ],
    }
    assert _pick_map_node(state) == 1
