"""Endless Conveyor, and the class of event it belongs to.

WHAT THIS COST, THREE TIMES
---------------------------
2026-08-14: the agent answered this event 83 times and the run ended on floor 4
at FULL HEALTH -- 84/84, no death, just a run that could not leave a room.
2026-08-14, again: 120 gold fed to the belt in one room, because only HP was
parsed out of the option text.
2026-08-16: floor 11 at 250 gold, one grab, and the run ended 30 seconds later
at full HP. Three rounds of guards, and the event still decided a run.

WHY THEY ALL MISSED
-------------------
Every guard was reactive -- `repeats >= 1`, `EVENT_MAX_REPEATS`,
`EVENT_HARD_CAP` -- and this event is decided on the FIRST answer. The gold
guard could never fire because the mod sends

    label = "Grab Suspicious Condiment off the Belt"

with the 40 gold nowhere in the text, so `_event_gold_cost` reads 0, both
options score as safe, and `safe[0][0]` returns index 0: the grab.

WHAT THE GAME ACTUALLY DOES
---------------------------
From `decompiled/MegaCrit.Sts2.Core.Models.Events/EndlessConveyor.cs`:

    GenerateInitialOptions()      -> [ Grab, ObserveChef ]
    GrabSomethingOffTheBelt()     -> LoseGold(40), roll a dish, re-open with
                                     [ Grab, Leave ]. It never ends on its own.
    ObserveChef()                 -> upgrade a random card, SetEventFinished()
    once Gold < 40, the grab option's action is NULL and clicking does nothing

So index 0 is always the grab, index 1 is always the way out, and the right
play is never to grab: Observe is a free upgrade (~5 points of act 1 boss win
in this repo) and 40 gold is over half a shop removal (~3.3 points).

These tests pin the decision, and the general guard that would have caught it
without knowing anything about this event.
"""

from __future__ import annotations

import pytest

from sts2_env.bridge import agent_runner
from sts2_env.bridge.agent_runner import _pick_event_option


@pytest.fixture(autouse=True)
def _clean_gold_learning():
    """The learned set is process-global by design; isolate the tests from it."""
    agent_runner._event_options_that_charged_gold.clear()
    agent_runner._reset_event_gold_learning()
    yield
    agent_runner._event_options_that_charged_gold.clear()
    agent_runner._reset_event_gold_learning()


def _option(index, label, *, description="", enabled=True, event_id="EndlessConveyor"):
    """The payload shape `RlEventRoomHandler` really sends.

    `id` and `action` are both the literal `event_choice`, the human text is in
    `label`, and `event_id` rides on every option rather than on the state.
    """
    return {
        "index": index,
        "id": "event_choice",
        "action": "event_choice",
        "label": label,
        "description": description,
        "enabled": enabled,
        "event_id": event_id,
    }


def _conveyor_state(*, dish="Suspicious Condiment", gold=250, first_visit=True):
    exit_label = "Observe the Chef" if first_visit else "Leave"
    return {
        "type": "event",
        "run_hp": 91, "run_max_hp": 91, "gold": gold, "floor": 11, "act": 1,
        "options": [
            _option(0, f"Grab {dish} off the Belt"),
            _option(1, exit_label),
        ],
    }


# -- the decision ------------------------------------------------------------


def test_it_leaves_the_belt_on_the_very_first_visit():
    """The one that matters. Every previous guard fired too late to help."""
    assert _pick_event_option(_conveyor_state(), {}) == 1


def test_it_leaves_the_belt_when_rich():
    """Gold is not the question. 250 gold buys three shop removals, and the
    grab is still a worse deal than a free upgrade."""
    assert _pick_event_option(_conveyor_state(gold=999), {}) == 1


def test_it_takes_leave_on_the_second_page():
    """After a grab the game replaces Observe the Chef with Leave. Both end the
    event, so the rule is 'anything that is not a grab'."""
    assert _pick_event_option(_conveyor_state(first_visit=False), {}) == 1


@pytest.mark.parametrize("dish", [
    "Suspicious Condiment", "Caviar", "Clam Roll", "Golden Fysh",
    "Jelly Liver", "Fried Eel", "Spicy Snappy", "Seapunk Salad",
])
def test_every_dish_is_declined(dish):
    """RollDish changes the label every time; the tail does not. A guard that
    matched the dish name would work once and then quietly stop."""
    assert _pick_event_option(_conveyor_state(dish=dish), {}) == 1


def test_the_event_id_alone_is_enough_without_the_english_label():
    """`label` is localised, `event_id` is not. On a non-English client the
    label fallback is gone and the id has to carry it."""
    state = {
        "type": "event", "run_hp": 91, "run_max_hp": 91, "gold": 250,
        "options": [
            _option(0, "Foerderband-Gericht nehmen"),
            _option(1, "Dem Koch zusehen"),
        ],
    }
    # Neither label contains "off the belt", so only the id can decide, and the
    # first non-grab option is index 0 -- which is why the id path must not
    # simply fall through to the label scan.
    assert agent_runner._is_endless_conveyor("EndlessConveyor", state["options"])


def test_a_locked_grab_is_never_chosen():
    """Once Gold < 40 the grab's action is NULL and the mod reports it
    disabled. Clicking it does nothing and the room never ends -- the hang."""
    state = _conveyor_state(gold=10)
    state["options"][0]["enabled"] = False
    assert _pick_event_option(state, {}) == 1


# -- the general guard, which needs to know nothing about this event ----------


def test_an_option_that_silently_took_gold_is_not_taken_again():
    """The mechanism that would have caught the conveyor on its own.

    `_event_gold_cost` can only read a cost the text admits to. The gold total
    cannot lie, so a choice is priced against the next state's gold.
    """
    seen: dict[str, int] = {}
    state = {
        "type": "event", "run_hp": 80, "run_max_hp": 80, "gold": 200,
        "options": [
            _option(0, "Sample the wares", event_id="SomeUnknownEvent"),
            _option(1, "Move along", event_id="SomeUnknownEvent"),
        ],
    }
    assert _pick_event_option(state, seen) == 0  # nothing known yet

    # The event re-presents itself and 40 gold has gone missing.
    charged = dict(state, gold=160)
    _pick_event_option(charged, seen)

    assert ("SOMEUNKNOWNEVENT", "Sample the wares") in \
        agent_runner._event_options_that_charged_gold


def test_a_free_option_is_preferred_over_a_known_paying_one_on_visit_one():
    """Learning is worthless if it is only ever applied on a repeat."""
    agent_runner._event_options_that_charged_gold.add(("SOMEEVENT", "Pay the toll"))
    state = {
        "type": "event", "run_hp": 80, "run_max_hp": 80, "gold": 200,
        "options": [
            _option(0, "Pay the toll", event_id="SomeEvent"),
            _option(1, "Take the long way", event_id="SomeEvent"),
        ],
    }
    assert _pick_event_option(state, {}) == 1


def test_gold_spent_in_a_different_event_is_not_blamed_on_this_one():
    """Gold moves for many reasons. Only a drop within the SAME event is
    attributable, or the guard would learn nonsense from a shop visit."""
    seen: dict[str, int] = {}
    first = {
        "type": "event", "run_hp": 80, "run_max_hp": 80, "gold": 200,
        "options": [_option(0, "Look around", event_id="EventA"),
                    _option(1, "Leave", event_id="EventA")],
    }
    _pick_event_option(first, seen)

    second = {
        "type": "event", "run_hp": 80, "run_max_hp": 80, "gold": 20,
        "options": [_option(0, "Look around", event_id="EventB"),
                    _option(1, "Leave", event_id="EventB")],
    }
    _pick_event_option(second, seen)

    assert not agent_runner._event_options_that_charged_gold, (
        "180 gold went missing between two DIFFERENT events; blaming the first "
        "event's option for it is how a guard learns superstitions"
    )
