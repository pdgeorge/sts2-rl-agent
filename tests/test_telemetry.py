"""Milestone telemetry: the journal-to-event mapping, and the never-block rules.

The mapping is pure and tested without a broker. The publisher is tested for
its degradation contract: no broker, no aio_pika, or a dead connection must
cost a log line at most, never the run.
"""

from __future__ import annotations

import json


def _record(event, **fields):
    return {"t": 1.0, "session": "20260816T202852", "run": 7, "event": event,
            "floor": fields.pop("floor", 17), "policy_version": "v001",
            "git_sha": "abc1234", **fields}


def test_run_start_maps_with_envelope():
    from sts2_env.bridge.telemetry import map_journal_record

    events = map_journal_record(_record("run_start", character="Ironclad", ascension=0))
    assert len(events) == 1
    e = events[0]
    assert e["event_type"] == "run_start"
    assert e["run_id"] == "20260816T202852_7"
    assert e["policy_version"] == "v001"
    assert e["git_sha"] == "abc1234"
    assert e["timestamp"]


def test_act_clear_is_boss_beaten_then_act_entered():
    from sts2_env.bridge.telemetry import map_journal_record

    events = map_journal_record(_record("act_clear", act_from=1, act_to=2))
    assert [e["event_type"] for e in events] == ["boss_beaten", "act_entered"]
    assert events[0]["act"] == 1
    assert events[1]["act"] == 2


def test_elite_beaten_only_on_surviving_elite_fights():
    from sts2_env.bridge.telemetry import map_journal_record

    survived = _record("combat_end", room_type="Elite", hp_after=31,
                       enemies=[{"id": "BYRDONIS"}, {"id": "X"}])
    events = map_journal_record(survived)
    assert [e["event_type"] for e in events] == ["elite_beaten"]
    assert events[0]["enemies"] == ["BYRDONIS", "X"]

    died = _record("combat_end", room_type="Elite", hp_after=0)
    assert map_journal_record(died) == []
    monster = _record("combat_end", room_type="Monster", hp_after=60)
    assert map_journal_record(monster) == []


def test_run_end_death_emits_died():
    from sts2_env.bridge.telemetry import map_journal_record

    dead = _record("run_end", run_hp=0, result="terminated", act=1,
                   room_type="Boss", death_enemy_id="WATERFALL_GIANT")
    events = map_journal_record(dead)
    assert [e["event_type"] for e in events] == ["run_end", "died"]
    assert events[0]["outcome"] == "died"
    assert events[1]["death_cause"] == "WATERFALL_GIANT"

    cleared = _record("run_end", run_hp=41, result="done", act=2, act_cleared=True)
    events = map_journal_record(cleared)
    assert [e["event_type"] for e in events] == ["run_end"]
    assert events[0]["outcome"] == "cleared"


def test_potion_used_maps():
    from sts2_env.bridge.telemetry import map_journal_record

    events = map_journal_record(_record("potion_used", potion="FYSH_OIL", slot=1))
    assert events[0]["event_type"] == "potion_used"
    assert events[0]["potion"] == "FYSH_OIL"


def test_policy_version_loaded_maps():
    from sts2_env.bridge.telemetry import map_journal_record

    events = map_journal_record(_record("policy_version_loaded", source="policies/v001.json"))
    assert events[0]["event_type"] == "policy_version_loaded"


def test_publisher_disabled_is_fully_noop():
    from sts2_env.bridge.telemetry import TelemetryPublisher

    pub = TelemetryPublisher(enabled=False)
    assert not pub.enabled
    pub.observe(_record("run_start"))
    pub.emit("update_detected")
    pub.close()


def test_dead_broker_costs_nothing_observable():
    """Port 1 refuses instantly: observe/emit/close must not raise or block."""
    from sts2_env.bridge.telemetry import TelemetryPublisher

    pub = TelemetryPublisher(url="amqp://guest:guest@127.0.0.1:1/", enabled=True)
    try:
        assert pub.enabled
        pub.observe(_record("run_start"))
        pub.observe(_record("run_end", run_hp=0, result="terminated"))
        pub.emit("update_detected")
    finally:
        pub.close()


def test_every_event_serialises_and_routes():
    from sts2_env.bridge.telemetry import TelemetryPublisher, map_journal_record

    pub = TelemetryPublisher(enabled=False)
    samples = [
        _record("run_start"), _record("act_clear", act_from=1, act_to=2),
        _record("combat_end", room_type="Elite", hp_after=10,
                enemies=[{"id": "A"}]),
        _record("run_end", run_hp=0, result="terminated"),
        _record("potion_used", potion="X"),
        _record("policy_version_loaded"),
    ]
    for rec in samples:
        for event in map_journal_record(rec):
            json.dumps(event, default=str)
            assert event["event_type"].replace("_", "").isalnum()
        pub.observe(rec)  # disabled: must not raise
