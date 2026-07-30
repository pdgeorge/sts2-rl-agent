"""Only the four moments, and never a number.

In cyra_brain a milestone bypasses the reaction cooldown -- it forces Cyra to
speak, and the cooldown exists so she does not talk over herself. A run has ~400
decisions, so the cost of getting this wrong is her narrating every card play.

The other rule these pin: no raw probability ever reaches her. A softmax over
action logits is not a calibrated confidence, so 0.75 does not mean "75% sure".
Hand her the number and she will say it, authoritatively and wrongly.
"""

from __future__ import annotations

import pytest

from sts2_env.bridge.milestones import (
    MilestoneWatcher,
    OBVIOUS_GAP,
    TORN_GAP,
    gut_phrase,
)


@pytest.fixture
def watcher():
    return MilestoneWatcher()


def _state(**kw):
    base = {"type": "map_select", "act": 1, "floor": 1}
    base.update(kw)
    return base


# --- the gut line -----------------------------------------------------------

def test_a_torn_decision_says_so():
    assert "either way" in gut_phrase(0.01)


def test_an_easy_decision_says_so():
    assert "obvious" in gut_phrase(0.9)


def test_no_margin_still_answers():
    """The margin can fail to read; she must still have something to say."""
    assert gut_phrase(None)


@pytest.mark.parametrize("gap", [0.0, 0.01, TORN_GAP, 0.3, OBVIOUS_GAP, 1.0, None])
def test_no_phrasing_ever_contains_a_number(gap):
    """The whole point: values are not calibrated, so she must not quote them."""
    phrase = gut_phrase(gap)
    assert not any(ch.isdigit() for ch in phrase), phrase
    for banned in ("%", "percent", "probability", "confidence"):
        assert banned not in phrase.lower()


# --- run start --------------------------------------------------------------

def test_run_start_fires_once(watcher):
    first = watcher.observe(_state())
    second = watcher.observe(_state())
    assert len(first) == 1 and first[0]["kind"] == "sts2_run_start"
    assert second == [], "a run starts once"


def test_reset_allows_the_next_run_to_start(watcher):
    watcher.observe(_state())
    watcher.reset()
    assert watcher.observe(_state()), "run 2 must announce itself"


# --- elite and boss ---------------------------------------------------------

def test_beating_an_elite_is_a_milestone(watcher):
    watcher.observe(_state(room_type="Elite", type="combat_action"))
    events = watcher.observe(_state(type="reward_screen", run_hp=40, run_max_hp=80))
    kinds = [e["kind"] for e in events]
    assert "sts2_elite" in kinds


def test_beating_a_boss_says_which_act(watcher):
    watcher.observe(_state(room_type="Boss", type="combat_action", act=1))
    events = watcher.observe(_state(type="reward_screen", act=1,
                                    run_hp=60, run_max_hp=80))
    boss = [e for e in events if e["kind"] == "sts2_boss"]
    assert boss and "act 1" in boss[0]["text"]


def test_a_close_win_is_described_as_close(watcher):
    watcher.observe(_state(room_type="Elite", type="combat_action"))
    events = watcher.observe(_state(type="reward_screen", run_hp=8, run_max_hp=80))
    assert "Barely" in events[0]["text"] or "Barely" in events[-1]["text"]


def test_an_ordinary_fight_is_not_a_milestone(watcher):
    """Most fights are Monster rooms; those are autopilot."""
    watcher.observe(_state(room_type="Monster", type="combat_action"))
    events = watcher.observe(_state(type="reward_screen", run_hp=40, run_max_hp=80))
    assert [e for e in events if e["kind"] in ("sts2_elite", "sts2_boss")] == []


def test_an_elite_win_fires_only_once(watcher):
    """reward_screen and card_reward both arrive after one fight."""
    watcher.observe(_state(room_type="Elite", type="combat_action"))
    first = watcher.observe(_state(type="reward_screen", run_hp=40, run_max_hp=80))
    second = watcher.observe(_state(type="card_reward", run_hp=40, run_max_hp=80))
    assert len([e for e in first if e["kind"] == "sts2_elite"]) == 1
    assert [e for e in second if e["kind"] == "sts2_elite"] == []


# --- the Ancient ------------------------------------------------------------

def test_choosing_an_ancient_is_a_milestone(watcher):
    state = _state(nodes=[{"type": "Ancient"}, {"type": "Monster"}])
    event = watcher.map_choice(state, 0, 0.02)
    assert event is not None
    assert "Ancient" in event["text"]
    assert "either way" in event["text"], "the gut phrasing rides along"


def test_choosing_an_ordinary_node_is_not(watcher):
    state = _state(nodes=[{"type": "Ancient"}, {"type": "Monster"}])
    assert watcher.map_choice(state, 1, 0.9) is None


def test_an_out_of_range_choice_is_ignored(watcher):
    state = _state(nodes=[{"type": "Ancient"}])
    assert watcher.map_choice(state, 7, 0.5) is None


# --- payload shape ----------------------------------------------------------

def test_every_event_is_a_milestone_with_text(watcher):
    """cyra_brain reads exactly `text` and `tier`; both must be present."""
    watcher.observe(_state(room_type="Boss", type="combat_action"))
    events = watcher.observe(_state(type="reward_screen", run_hp=60, run_max_hp=80))
    events += watcher.observe(_state())
    for e in events:
        assert e["tier"] == "milestone"
        assert e["text"] and isinstance(e["text"], str)
