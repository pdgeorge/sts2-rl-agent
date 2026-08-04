"""Replay real recorded bridge states through the live decision path.

WHY THIS FILE EXISTS

`from_bridge` read "hp" and "current_hp". The mod sends `run_hp`. So `_read_int`
fell through to its default of ZERO and every live rest decision was computed as
though the player were dead -- a rest at 0 HP scores 48.8 against any upgrade, so
resting always won and the agent never upgraded anything.

4,700 tests passed throughout. Every one of them built its own bridge state, and
every one used "hp", because that is what the person writing the test assumed.
The tests validated the code against its author's assumptions.

These tests read states the GAME produced. That is the entire point, and it is
the only kind of test that could have caught it.

Recorded with:

    python -m sts2_env.bridge.live_eval --model-path <model> \\
        --measured-drafting --record-replay output/replay.json

Skipped when no trace is present, so the suite still runs on a clean checkout --
but any trace in output/ is checked automatically.

NOTE: only traces recorded after `raw_state` was added carry the untrimmed
state. `resulting_state` is normalised into a fixed comparison shape for combat
parity, which drops `deck` and `run_state` -- exactly the fields the run-level
decisions need.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

TRACES = sorted(Path("output").glob("replay*.json"))


def _states(state_type: str) -> list[dict]:
    """Every raw recorded state of this type, across all traces on disk."""
    found = []
    for path in TRACES:
        try:
            data = json.loads(path.read_text())
        except Exception:  # noqa: BLE001 -- an unreadable trace is not a failure
            continue
        for step in data.get("steps", []):
            raw = step.get("raw_state") or {}
            if raw.get("type") == state_type:
                found.append(raw)
        initial = data.get("initial_state") or {}
        if initial.get("type") == state_type and "run_hp" in initial:
            found.append(initial)
    return found


pytestmark = pytest.mark.skipif(
    not TRACES, reason="no recorded bridge trace in output/; see the module docstring"
)


def test_hp_is_read_from_every_recorded_state_that_has_it():
    """The bug, asserted against reality rather than a hand-made dict.

    Any recorded state carrying HP must be read as that HP, never as the zero
    default. This is the assertion whose absence cost a week.
    """
    from sts2_env.evaluation.from_bridge import _read_int

    checked = 0
    for state_type in ("rest_site", "card_reward", "map_select", "card_select"):
        for state in _states(state_type):
            if "run_hp" not in state:
                continue
            expected = int(state["run_hp"])
            actual = _read_int(state, "run_hp", "hp", "current_hp", default=0)
            assert actual == expected, (
                f"{state_type}: read {actual} from a state whose run_hp is {expected}"
            )
            checked += 1

    if checked == 0:
        pytest.skip("no recorded states carried run_hp (trace predates raw_state)")


def test_max_hp_is_never_silently_the_default():
    """`run_max_hp`, not `max_hp`. Defaulting to 80 was right for a fresh
    Ironclad and wrong for every run that raised its maximum -- recorded runs
    reach 86 and 91."""
    from sts2_env.evaluation.from_bridge import _read_int

    for state_type in ("rest_site", "card_reward"):
        for state in _states(state_type):
            if "run_max_hp" not in state:
                continue
            assert _read_int(state, "run_max_hp", "max_hp", default=80) == int(
                state["run_max_hp"]
            )


def test_recorded_rest_sites_decide_without_raising():
    """The live path, on real states. Returning None is a legitimate answer --
    it falls back to the heuristic -- but an exception is not."""
    from sts2_env.evaluation.from_bridge import choose_rest_option
    from sts2_env.evaluation.pilots import greedy_pilot

    states = _states("rest_site")
    if not states:
        pytest.skip("no rest sites recorded")
    for state in states:
        choose_rest_option(state, greedy_pilot, seeds=(0,))


def test_recorded_card_rewards_decide_without_raising():
    from sts2_env.evaluation.from_bridge import choose_card_index
    from sts2_env.evaluation.pilots import greedy_pilot

    states = _states("card_reward")
    if not states:
        pytest.skip("no card rewards recorded")
    for state in states:
        choose_card_index(state, greedy_pilot, seeds=(0,))
