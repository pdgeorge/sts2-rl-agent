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
    assert trailer["seen"]["combat_action"] == 10
    assert trailer["kept"]["combat_action"] == 2


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
