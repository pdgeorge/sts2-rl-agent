"""A boss death is not a clear.

On 2026-08-05 a 30-run live session reported "CLEARED act 1: 6/30, 20.0%". All
six were floor 17, room Boss, 0 HP, act 1: the boss was never beaten once. The
metric counted reaching floor 17 as a clear because the simulator puts the act 1
boss on floor 16 -- but in the live game the boss room IS floor 17, so dying to it
scores the same as walking past it.

The whole point of these runs is deciding what to build next. A metric that turns
0% into 20% points that decision at the wrong thing.
"""

from __future__ import annotations

from sts2_env.bridge.live_eval import (
    LiveEvalRecorder,
    _cleared_act_1,
    _reached_act_1_boss,
)


def _run(**fields):
    base = {"floor": 8, "act": 1, "room_type": "Monster", "result": "terminated"}
    base.update(fields)
    return base


# -- the exact runs that were misreported ------------------------------------

def test_dying_on_the_act_1_boss_is_not_a_clear() -> None:
    died = _run(floor=17, act=1, room_type="Boss", run_hp=0)
    assert _reached_act_1_boss(died)
    assert not _cleared_act_1(died)


def test_reaching_act_2_is_a_clear() -> None:
    cleared = _run(floor=18, act=2, room_type="Monster")
    assert _cleared_act_1(cleared)


def test_the_whole_session_scores_zero() -> None:
    """The six runs as they were actually logged."""
    recorder = LiveEvalRecorder(None, "test")
    for _ in range(6):
        recorder(_run(floor=17, act=1, room_type="Boss", run_hp=0))
    for _ in range(24):
        recorder(_run(floor=8, act=1, room_type="Monster", run_hp=0))

    report = recorder.report()
    assert "reached the act 1 boss        6/30" in report.replace("  ", " ").replace("   ", " ") or "6/30" in report
    assert "0/30" in report, "a session with no clear must report no clear"
    assert "0.0%" in report


# -- the fallback, for a summary with no act ---------------------------------

def test_without_an_act_field_the_boss_floor_alone_is_not_enough() -> None:
    assert not _cleared_act_1({"floor": 17, "room_type": "Boss"})


def test_without_an_act_field_going_beyond_the_boss_counts() -> None:
    assert _cleared_act_1({"floor": 18, "room_type": "Monster"})


# -- reaching it is still worth reporting ------------------------------------

def test_the_boss_room_counts_as_reaching_it_whatever_the_floor() -> None:
    assert _reached_act_1_boss({"floor": 16, "room_type": "Boss"})


def test_an_ordinary_death_reached_nothing() -> None:
    assert not _reached_act_1_boss(_run(floor=9, room_type="Elite"))


def test_reached_and_died_are_reported_separately() -> None:
    """The distinction the old report could not draw at all."""
    recorder = LiveEvalRecorder(None, "test")
    recorder(_run(floor=17, act=1, room_type="Boss"))     # reached, died
    recorder(_run(floor=19, act=2, room_type="Monster"))  # reached, cleared

    report = recorder.report()
    assert "died to it" in report
