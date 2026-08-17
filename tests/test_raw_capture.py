"""The raw protocol capture: whole states, verbatim, quota'd per type.

The point of this file is the round trip -- a state written by the capture and
read back by ``load_capture`` must be byte-for-byte the same dict, because the
whole reason the capture exists is to feed the bridge parsers something the mod
really said rather than something we assumed it said. A capture that quietly
normalised a field would reintroduce exactly the class of bug it was built to
catch (PR #8's UPPER_SNAKE potion ids).
"""

from __future__ import annotations

import json

import pytest

from sts2_env.bridge.raw_capture import (
    RawCapture,
    load_capture,
    load_trailer,
)


def _state(msg_type: str = "combat_action", **extra) -> dict:
    state = {"type": msg_type, "floor": 3, "hp": 71}
    state.update(extra)
    return state


def test_a_captured_state_round_trips_unchanged(tmp_path):
    """Verbatim means verbatim: no key dropped, no value coerced."""
    path = tmp_path / "capture.jsonl"
    original = _state(
        potions=["STRENGTH_POTION", None, None],
        deck=[{"id": "STRIKE", "upgraded": False}, {"id": "BASH", "upgraded": True}],
        enemies=[{"id": "SHRINKER_BEETLE", "hp": 40, "intent": "ATTACK"}],
    )

    capture = RawCapture(path)
    capture.observe(original)
    capture.close()

    assert load_capture(path) == [original]


def test_the_quota_keeps_the_first_n_of_each_type_and_counts_the_rest(tmp_path):
    """A long fight must not crowd rare screens out of the sample."""
    path = tmp_path / "capture.jsonl"
    capture = RawCapture(path, per_type=2)

    for i in range(10):
        capture.observe(_state("combat_action", step=i))
    capture.observe(_state("card_reward"))
    capture.close()

    states = load_capture(path)
    combat = [s for s in states if s["type"] == "combat_action"]
    assert len(combat) == 2, "quota not enforced"
    assert [s["step"] for s in combat] == [0, 1], "kept the wrong two"
    # The rare screen survived the flood -- the reason quotas exist at all.
    assert any(s["type"] == "card_reward" for s in states)

    trailer = load_trailer(path)
    # Combats bucket by room, so a state with no room_type lands in ":?".
    assert trailer["seen"]["combat_action:?"] == 10
    assert trailer["kept"]["combat_action:?"] == 2


def test_a_boss_fight_has_its_own_quota(tmp_path):
    """Bosses must not lose the quota to floor-1 monsters, which they always did.

    Sharing one combat bucket meant the first N monster turns filled it and the
    rare, late fight was never sampled: 590 combat states across eleven live
    sessions and zero from a boss. That made replaying live boss fights offline
    -- the one comparison that would explain a 67% offline win rate against 36%
    live -- impossible for want of a single fight on disk.
    """
    path = tmp_path / "capture.jsonl"
    capture = RawCapture(path, per_type=2, boss_quota=2)

    for i in range(10):
        capture.observe(_state("combat_action", step=i, room_type="Monster"))
    for i in range(3):
        capture.observe(_state("combat_action", step=100 + i, room_type="Boss"))
    capture.close()

    states = load_capture(path)
    boss = [s for s in states if s.get("room_type") == "Boss"]
    monster = [s for s in states if s.get("room_type") == "Monster"]
    assert len(monster) == 2, "monster quota not enforced"
    assert len(boss) == 2, "the boss fight was crowded out again"


def test_boss_fights_get_a_larger_quota_than_the_shared_default(tmp_path) -> None:
    """A boss fight is sampled per state, and fights outlast the old quota.

    The 2026-08-16 session replayed two Waterfall Giant fights that ran eleven
    turns at about five states a round; the shared quota of 25 cut both
    captures at round five -- exactly when the fights started being lost. The
    boss bucket therefore keeps its own budget, large enough for a full fight.
    """
    path = tmp_path / "capture.jsonl"
    capture = RawCapture(path, per_type=2)  # boss_quota stays the default

    for i in range(60):
        capture.observe(_state("combat_action", step=i, room_type="Boss"))
    capture.close()

    assert len(load_capture(path)) == 60, (
        "a full boss fight must survive its own capture; the quota cut the "
        "sunday session's Waterfall Giant fights at round five")


def test_observe_reports_whether_it_kept_the_state(tmp_path):
    path = tmp_path / "capture.jsonl"
    capture = RawCapture(path, per_type=1)

    assert capture.observe(_state()) is True
    assert capture.observe(_state()) is False
    capture.close()


def test_states_are_flushed_as_they_arrive_not_at_close(tmp_path):
    """A session killed mid-run must still leave a usable file.

    The runner's loop can exit through Ctrl-C, a lost connection or a crash,
    and a capture that only became readable on the clean path would be empty
    precisely when it is most wanted.
    """
    path = tmp_path / "capture.jsonl"
    capture = RawCapture(path)
    capture.observe(_state("combat_start"))

    # Deliberately not closed -- this is the crash case.
    assert load_capture(path) == [_state("combat_start")]


def test_a_truncated_final_line_does_not_lose_the_states_before_it(tmp_path):
    path = tmp_path / "capture.jsonl"
    capture = RawCapture(path)
    capture.observe(_state("combat_start"))
    capture.observe(_state("combat_action"))
    capture.close()

    # Simulate a kill mid-write.
    text = path.read_text()
    path.write_text(text[: text.rindex("\n") - 5])

    states = load_capture(path)
    assert [s["type"] for s in states] == ["combat_start", "combat_action"]


def test_an_odd_value_is_stringified_rather_than_losing_the_whole_state(tmp_path):
    """A diagnostic must never be able to end a 20-run live session.

    Real payloads arrive as JSON so every value is already JSON-native; this
    only fires on something impossible. When it does, keeping the state with
    one stringified field beats dropping the screen, because the screen is
    what you were trying to look at.
    """
    path = tmp_path / "capture.jsonl"
    capture = RawCapture(path)

    assert capture.observe({"type": "weird", "obj": object()}) is True
    capture.close()

    replayed = load_capture(path)[0]
    assert replayed["type"] == "weird"
    assert isinstance(replayed["obj"], str)


def test_a_circular_state_is_skipped_without_killing_the_run(tmp_path):
    """The case `default=str` cannot rescue still must not raise."""
    path = tmp_path / "capture.jsonl"
    capture = RawCapture(path)

    circular: dict = {"type": "weird"}
    circular["self"] = circular

    assert capture.observe(circular) is False
    assert capture.observe(_state("combat_action")) is True
    capture.close()

    assert [s["type"] for s in load_capture(path)] == ["combat_action"]


def test_an_untyped_state_is_captured_rather_than_dropped(tmp_path):
    """An untyped screen is itself a protocol question, so keep it."""
    path = tmp_path / "capture.jsonl"
    capture = RawCapture(path)
    capture.observe({"floor": 1, "hp": 80})
    capture.close()

    assert load_capture(path) == [{"floor": 1, "hp": 80}]
    assert load_trailer(path)["kept"] == {"<untyped>": 1}


def test_close_is_idempotent_and_writes_one_trailer(tmp_path):
    path = tmp_path / "capture.jsonl"
    capture = RawCapture(path)
    capture.observe(_state())
    capture.close()
    capture.close()

    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert sum(1 for l in lines if l.get("__capture_trailer__")) == 1


def test_a_new_session_truncates_rather_than_appending(tmp_path):
    """Mixing two mod builds in one file is the confusion this file prevents."""
    path = tmp_path / "capture.jsonl"
    first = RawCapture(path)
    first.observe(_state("combat_start"))
    first.close()

    second = RawCapture(path)
    second.observe(_state("card_reward"))
    second.close()

    assert [s["type"] for s in load_capture(path)] == ["card_reward"]


def test_load_trailer_is_none_when_the_session_never_closed(tmp_path):
    path = tmp_path / "capture.jsonl"
    capture = RawCapture(path)
    capture.observe(_state())

    assert load_trailer(path) is None


def test_a_captured_combat_action_feeds_the_bridge_parsers(tmp_path):
    """The end the capture exists for: replay a real payload through the parsers.

    Uses the same synthetic payload shape the bridge tests use. When a real
    session lands a capture, this is the shape of the check to run against it --
    which is the whole point of Phase 0.1.
    """
    situation_mod = pytest.importorskip("sts2_env.search.situation")
    from tests.test_combat_situation_from_bridge import _bridge_state

    path = tmp_path / "capture.jsonl"
    capture = RawCapture(path)
    capture.observe(_bridge_state())
    capture.close()

    replayed = load_capture(path)[0]
    situation = situation_mod.CombatSituation.from_bridge_state(replayed)
    combat = situation.to_combat_mid_fight(replayed)

    assert combat.enemies, "replayed payload built a fight with no enemies"


def test_a_shared_capture_survives_a_restart_and_keeps_its_quota(tmp_path):
    """The bug that cost `postfix` all 68 of its boss fights.

    `live_eval` restarts the game on a crash and calls `run_agent` again. Until
    2026-08-17 `run_agent` built its own `RawCapture` each time, and the class
    opens with "w" -- so every relaunch truncated the file and the session was
    left with whatever the final segment happened to see. `postfix` restarted
    four times, its last segment was one run, and a session that fought 68 act 1
    bosses ended with zero boss states on disk.

    Two things have to hold, and the second is why appending was not the fix.
    The states from earlier segments must survive, AND the quota counters must
    carry over -- a per-segment counter re-fills its floor-1 buckets once per
    crash, which is the original pathology that made the capture useless.
    """
    path = tmp_path / "capture.jsonl"
    capture = RawCapture(path, per_type=3)

    # Segment one: fills the monster bucket, then reaches a boss.
    for _ in range(5):
        capture.observe({"type": "combat_action", "room_type": "Monster",
                         "encounter_seed": 1})
    capture.observe({"type": "combat_action", "room_type": "Boss",
                     "encounter_seed": 99, "floor": 17})

    # The game dies here and `live_eval` calls `run_agent` again with the SAME
    # capture. Nothing reopens the file.
    for _ in range(5):
        capture.observe({"type": "combat_action", "room_type": "Monster",
                         "encounter_seed": 2})
    capture.observe({"type": "combat_action", "room_type": "Boss",
                     "encounter_seed": 100, "floor": 17})
    capture.close()

    states = load_capture(path)
    rooms = [s.get("room_type") for s in states]

    # The first segment's boss is still there. This is the whole point.
    assert rooms.count("Boss") == 2, (
        "a restart truncated the capture and lost the earlier segment's boss")

    # And the quota did not reset: seed 1 and seed 2 are distinct fights, so
    # each gets its own 3, but neither gets a fresh allowance per segment.
    assert rooms.count("Monster") == 6
    seeds = [s.get("encounter_seed") for s in states if s.get("room_type") == "Monster"]
    assert seeds.count(1) == 3 and seeds.count(2) == 3


def test_run_agent_does_not_close_a_capture_it_was_handed(tmp_path):
    """A borrowed capture stays open, or the trailer lands mid-session.

    `run_agent`'s `finally` closes the capture so a Ctrl-C still writes the
    counts. When the capture belongs to the caller that must not fire, because
    `live_eval` will keep feeding it through every remaining restart.
    """
    import inspect
    from sts2_env.bridge import agent_runner

    source = inspect.getsource(agent_runner.run_agent)
    assert "owns_capture" in source, "run_agent must track whether it owns the capture"
    assert "if owns_capture:" in source, "the close must be guarded by ownership"

    # And the seam exists at all: a caller can hand one in.
    assert "capture_raw" in inspect.signature(agent_runner.run_agent).parameters


def test_the_console_summary_is_one_readable_line(tmp_path):
    """The full per-bucket dict is 30 KB on one line and nobody reads it.

    A 54-run session produced 625 buckets -- one per distinct fight, because
    combat is keyed per encounter_seed -- and dumped every one of them to the
    console at shutdown. The counts still matter, so `close` keeps writing them
    to the file's trailer; they just do not belong in a log line.
    """
    path = tmp_path / "capture.jsonl"
    capture = RawCapture(path, per_type=2, boss_quota=3)
    for seed in range(40):
        for _ in range(5):
            capture.observe({"type": "combat_action", "room_type": "Monster",
                             "encounter_seed": seed})
    for _ in range(5):
        capture.observe({"type": "combat_action", "room_type": "Boss",
                         "encounter_seed": 999})
    capture.close()

    line = capture.summary_line
    assert "\n" not in line
    assert len(line) < 300, f"{len(line)} chars is not a log line"
    assert "1 boss fights" in line
    assert "trailer" in line, "must point at where the detail actually lives"

    # The detail is still on disk, which is the whole justification for cutting it.
    trailer = load_trailer(path)
    assert len(trailer["kept"]) == 41
    assert trailer["kept"]["combat_action:Boss:999"] == 3


def test_the_full_summary_still_carries_every_bucket():
    """Programmatic callers keep the detail; only the console line is trimmed."""
    import pathlib
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        capture = RawCapture(pathlib.Path(d) / "c.jsonl", per_type=2)
        capture.observe({"type": "map_select"})
        assert capture.summary["per_type"] == {"map_select": 1}
        assert capture.summary["kept"] == 1
