"""The detector that makes the simulator's parity gaps visible from a live run.

`situation.py` overwrites enemy state with the bridge's every decision, so a
wrong constant is silently corrected at the root and the run continues. Only the
root gets corrected -- the search's lookahead rolls forward on the simulator's
own numbers -- so these gaps cost real fights while leaving no trace.

Three were found by hand in one afternoon (Waterfall Giant, Phantasmal Gardener,
an unslotted summon). Hand-checking does not scale to 83 monsters.
"""

from __future__ import annotations

import pytest

from sts2_env.search.parity import (
    check_max_hp,
    disparity_summary,
    report_disparity,
    reset_disparities,
    simulator_hp_range,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_disparities()
    yield
    reset_disparities()


def test_a_value_outside_the_simulators_range_is_reported() -> None:
    """A value the simulator cannot roll must be named.

    Deliberately synthetic rather than a real monster's real disparity, which is
    what this asserted first: it used Phantasmal Gardener's 26 against a
    simulator that said 28-32, and then passed only until that constant was
    fixed. A regression test that dies when the bug it describes is fixed tests
    the bug, not the detector.
    """
    span = simulator_hp_range("CORPSE_SLUG")
    assert span is not None

    check_max_hp("CORPSE_SLUG", span[1] + 50)

    found = disparity_summary()
    assert len(found) == 1
    assert "CORPSE_SLUG" in found[0]


def test_a_value_inside_the_range_is_silent() -> None:
    """Two RNG streams disagreeing is not a modelling error.

    Corpse Slug rolls 25-27 in both, and the live game reported all three
    values across 490 fights. Reporting those would bury the real findings.
    """
    span = simulator_hp_range("CORPSE_SLUG")
    assert span is not None
    for hp in range(span[0], span[1] + 1):
        check_max_hp("CORPSE_SLUG", hp)

    assert disparity_summary() == []


def test_fixed_monsters_stay_silent() -> None:
    """Regression guard on the two already corrected.

    Waterfall Giant was 250 in the simulator against the game's 240, and Fogmog
    was always right. If either starts reporting, a constant has drifted back.
    """
    check_max_hp("WATERFALL_GIANT", 240)
    check_max_hp("FOGMOG", 74)

    assert disparity_summary() == []


def test_repeats_are_counted_but_logged_once(caplog) -> None:
    """A disparity recurs every decision of every fight; the log must not."""
    with caplog.at_level("WARNING"):
        for _ in range(50):
            report_disparity("max_hp", "SOME_MONSTER", "10-10", 99)

    assert sum("DISPARITY" in r.message for r in caplog.records) == 1
    assert disparity_summary()[0].endswith("x50")


def test_an_unbuildable_monster_is_not_reported_here() -> None:
    """Silent by design: that gap is reported where the build fails, and
    guessing at a range for a monster we cannot construct would be noise."""
    check_max_hp("NOT_A_REAL_MONSTER", 42)

    assert disparity_summary() == []
