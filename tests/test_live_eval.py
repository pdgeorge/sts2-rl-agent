"""A live session must survive a death and keep playing.

Every live measurement so far has been a single run, because run_agent exited on
the first terminal message even though the mod starts runs back to back. These
tests pin the two things that makes a difference to: the loop continues, and each
finished run is recorded with the floor it actually ended on.

The floor one is less obvious than it looks. game_over does not always carry the
run-level fields, so reading the floor off the terminal message alone records 0
for exactly the runs that ended in a death -- which is most of them, and would
make the headline number silently wrong rather than absent.
"""

from __future__ import annotations

import json
import re

import pytest

from sts2_env.bridge.live_eval import LiveEvalRecorder


@pytest.fixture
def recorder(tmp_path):
    return LiveEvalRecorder(tmp_path / "runs.jsonl", "model.zip")


def _norm(text):
    """Compare on content, not column padding."""
    return re.sub(r"\s+", " ", text)


def _run(n, floor, result="died", **extra):
    d = {"run": n, "floor": floor, "result": result, "act": 1, "seconds": 60.0}
    d.update(extra)
    return d


def test_records_each_run(recorder):
    recorder(_run(1, 8))
    recorder(_run(2, 19))
    assert recorder.floors() == [8, 19]


def test_reaching_the_boss_is_not_beating_it(recorder):
    """These two tests used to assert that floor 17 meant act 1 was cleared.

    That was wrong, and it was wrong in the direction that flatters the agent.
    The live game's act 1 boss room IS floor 17 -- the simulator's floor 16 is a
    different count -- so a run that dies to the boss ends on 17 with everything
    a victory would have. On 2026-08-05 that reported "CLEARED act 1: 6/30,
    20.0%" for a session in which the boss was never once beaten: all six were
    floor 17, room Boss, 0 HP, act 1.

    A clear is now reaching act 2, which no death can fake.
    """
    recorder(_run(1, 17, room_type="Boss", act=1))
    report = _norm(recorder.report())
    assert "reached the act 1 boss 1/1 100.0%" in report
    assert "died to it 1/1 100.0%" in report
    assert "CLEARED act 1 (reached act 2) 0/1 0.0%" in report


def test_act1_is_cleared_by_reaching_act_2(recorder):
    recorder(_run(1, 18, room_type="Monster", act=2))
    report = _norm(recorder.report())
    assert "CLEARED act 1 (reached act 2) 1/1 100.0%" in report


def test_log_is_written_and_flushed_per_run(tmp_path):
    """A kill -9 must not lose finished runs."""
    path = tmp_path / "runs.jsonl"
    rec = LiveEvalRecorder(path, "model.zip")
    rec(_run(1, 8))
    rec(_run(2, 19))
    # Read without closing: this is the crash case.
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert [x["floor"] for x in lines] == [8, 19]
    assert all(x["model"] == "model.zip" for x in lines)
    rec.close()


def test_log_appends_rather_than_replacing(tmp_path):
    """Restarting the script must not discard yesterday's runs."""
    path = tmp_path / "runs.jsonl"
    first = LiveEvalRecorder(path, "m.zip")
    first(_run(1, 8))
    first.close()

    second = LiveEvalRecorder(path, "m.zip")
    second(_run(1, 20))
    second.close()

    assert len(path.read_text().splitlines()) == 2, "the earlier run must survive"


def test_report_survives_zero_runs(recorder):
    """Ctrl-C before the first run finishes must not raise."""
    assert "nothing to report" in recorder.report()


def test_report_states_its_own_uncertainty(recorder):
    """A 3-run sample must not read like a measurement."""
    for i, fl in enumerate([20, 3, 5], start=1):
        recorder(_run(i, fl))
    assert "1 se over 3 runs" in recorder.report()
