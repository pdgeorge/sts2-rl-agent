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


def test_act1_clear_counts_floor_17_not_16(recorder):
    """Floor 16 is the boss itself; reaching it is not beating it."""
    recorder(_run(1, 16))
    report = _norm(recorder.report())
    assert "reached the act 1 boss (f>=16) 1/1 100.0%" in report
    assert "CLEARED act 1 (f>=17) 0/1 0.0%" in report


def test_act1_clear_counts_a_run_past_the_boss(recorder):
    recorder(_run(1, 17))
    assert "CLEARED act 1 (f>=17) 1/1 100.0%" in _norm(recorder.report())


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
