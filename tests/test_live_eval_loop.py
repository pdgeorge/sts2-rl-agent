"""run_agent must keep playing after a run ends.

This is the behaviour the live-eval harness exists for, and the one thing that
was actually missing: the mod already starts runs back to back, but this side
broke out of its loop on the first terminal message. So every live session ever
recorded was one run long, and there is still no live measurement to put beside
the simulator's "clears act 1 one time in a hundred".

Driven through a fake client rather than a socket, so the loop's own control flow
is what is under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.bridge import agent_runner


class _FakeClient:
    """Replays a scripted list of states and records what was sent back."""

    def __init__(self, states):
        self._states = list(states)
        self.sent: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def receive_state(self):
        if not self._states:
            # Nothing scripted left: end the session the way a real disconnect
            # would, rather than blocking a test forever.
            raise ConnectionError("no more scripted states")
        return self._states.pop(0)

    def send_action(self, action):
        self.sent.append(action)

    def choose(self, index):
        self.sent.append({"action": "choose", "index": index})

    def skip(self):
        self.sent.append({"action": "skip"})

    def end_turn(self):
        self.sent.append({"action": "end_turn"})

    def ping(self):
        return True


class _FakeModel:
    """Always picks the first legal action; the policy is not what is under test."""

    def __init__(self, obs_dims):
        self.observation_space = type("S", (), {"shape": (obs_dims,)})()

    def predict(self, obs, action_masks=None, deterministic=True):
        idx = int(np.argmax(action_masks)) if action_masks is not None else 0
        return idx, None


def _terminal(floor, result="died"):
    return {"type": "run_complete", "floor": floor, "act": 1,
            "result": result, "run_hp": 0}


def _card_reward(floor):
    return {"type": "card_reward", "floor": floor, "act": 1, "can_skip": False,
            "cards": [{"id": "BASH", "cost": 1, "type": "Attack"}],
            "run_hp": 50, "run_max_hp": 80, "deck_size": 10, "gold": 100}


@pytest.fixture
def patched(monkeypatch):
    from sts2_env.gym_env.run_env import RUN_OBS_SIZE

    def _fake_load(path):
        return _FakeModel(RUN_OBS_SIZE)

    monkeypatch.setattr(agent_runner, "load_model", _fake_load)
    return monkeypatch


def _play(patched, states, max_runs):
    client = _FakeClient(states)
    patched.setattr(agent_runner, "STS2GameClient",
                    lambda *a, **k: client)
    seen: list[dict] = []
    try:
        agent_runner.run_agent(model_path="fake.zip", max_runs=max_runs,
                               on_run_end=seen.append)
    except ConnectionError:
        pass
    return seen, client


def test_a_death_does_not_end_the_session(patched):
    """The regression: this used to stop after the first terminal message."""
    seen, _ = _play(patched, [
        _card_reward(3), _terminal(3),
        _card_reward(9), _terminal(9),
        _card_reward(20), _terminal(20),
    ], max_runs=3)

    assert len(seen) == 3, "all three runs must be recorded, not just the first"
    assert [r["floor"] for r in seen] == [3, 9, 20]


def test_max_runs_stops_the_session(patched):
    seen, _ = _play(patched, [
        _card_reward(3), _terminal(3),
        _card_reward(9), _terminal(9),
    ], max_runs=1)

    assert len(seen) == 1, "max_runs=1 must stop after one run"


def test_each_run_is_numbered_and_timed(patched):
    seen, _ = _play(patched, [_terminal(4), _terminal(11)], max_runs=2)
    assert [r["run"] for r in seen] == [1, 2]
    assert all("seconds" in r for r in seen)


def test_the_floor_is_remembered_from_earlier_states(patched):
    """A terminal message without run-level fields must not record floor 0.

    game_over does not always carry them. Reading the floor off the final
    message alone would score exactly the runs that ended in death as floor 0 --
    wrong rather than missing, and wrong in the direction that flatters nothing.
    """
    seen, _ = _play(patched, [
        _card_reward(14),
        {"type": "run_complete", "result": "died"},   # no floor at all
    ], max_runs=1)

    assert seen[0]["floor"] == 14, "the last floor seen must carry through"


def test_progress_does_not_leak_between_runs(patched):
    """Run 2 must not inherit run 1's floor if run 2 reports none."""
    seen, _ = _play(patched, [
        _card_reward(18), _terminal(18),
        {"type": "run_complete", "result": "died"},
    ], max_runs=2)

    assert seen[0]["floor"] == 18
    assert seen[1].get("floor", 0) != 18, "stale floor carried into the next run"
