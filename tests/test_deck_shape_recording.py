"""Every finished run records the shape of the deck that played it.

60 recorded live runs contained ZERO upgraded cards, which read as "the agent
never upgrades". It was really "the mod sent bare card ids, so upgrade state
never left the game" -- and the same entries are what `from_bridge` rebuilds a
deck from to score card rewards, so the battery was evaluating a fully
unupgraded copy of every deck it was asked about.
"""

from __future__ import annotations

from sts2_env.bridge.live_eval import deck_shape

STARTER = ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"]


def test_shape_is_recorded_for_the_flat_string_form():
    """The wire format older mod builds send, and every log already on disk."""
    shape = deck_shape(STARTER)
    assert shape["shape_cards_resolved"] == 10
    assert shape["block_density"] == 0.4
    assert shape["upgrade_density"] == 0.0
    assert shape["cycle_time"] == 2.0


def test_upgrades_are_seen_in_both_wire_forms():
    """A dict with `upgraded`, and the `NAME+` suffix `_read_deck_list` writes."""
    as_dicts = [{"id": "STRIKE_IRONCLAD", "upgraded": True}] * 2 + STARTER[2:]
    as_suffix = ["STRIKE_IRONCLAD+"] * 2 + STARTER[2:]

    assert deck_shape(as_dicts)["upgrade_density"] == 0.2
    assert deck_shape(as_suffix)["upgrade_density"] == 0.2
    assert deck_shape(STARTER)["upgrade_density"] == 0.0


def test_a_bad_entry_costs_a_card_and_not_the_run():
    """A recorder that drops a finished run because one card would not build is
    worse than one with a missing field. Runs are expensive; fields are not."""
    shape = deck_shape(STARTER + ["NOT_A_REAL_CARD", None, 17])
    assert shape["shape_cards_resolved"] == 10
    assert shape["block_density"] == 0.4


def test_no_deck_is_not_an_error():
    assert deck_shape(None) == {}
    assert deck_shape([]) == {}
    assert deck_shape(["NOT_A_REAL_CARD"]) == {}


def test_the_recorder_attaches_the_shape():
    from sts2_env.bridge.live_eval import LiveEvalRecorder

    recorder = LiveEvalRecorder(None, "test-model")
    recorder({"floor": 12, "deck": STARTER})
    assert recorder.runs[0]["block_density"] == 0.4
    assert recorder.runs[0]["cycle_time"] == 2.0
