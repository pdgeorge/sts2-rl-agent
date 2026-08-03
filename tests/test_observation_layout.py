"""Pin the observation layout, so a column cannot move quietly.

**This test is meant to fail when you change the observation.** That is its job.
When it does: read the diff, confirm the change was intended, bump
``OBS_LAYOUT_VERSION``, and update the literals below.

The offsets are written out as literal numbers rather than computed from the
constants. Computing them would make the test agree with whatever the code
currently does, which is precisely the failure mode ``derived_values.py`` records
-- "4,609 passing tests meant 'the repo agrees with itself'". A literal is an
independent second opinion.
"""

from __future__ import annotations

import json

import pytest

from sts2_env.gym_env.layout import (
    COMBAT_OBS_LAYOUT,
    OBS_LAYOUT_VERSION,
    RUN_OBS_LAYOUT,
    Block,
    ObservationLayoutMismatch,
    describe_layout,
    layout_fingerprint,
    layout_manifest,
    layout_size,
    stamp_checkpoint,
    verify_checkpoint,
)

# --- the pinned layout -------------------------------------------------------
# name, start, size -- as literals. Do not compute these.

EXPECTED_RUN_LAYOUT = [
    ("combat",          0,    131),
    ("entity_identity", 131,  1893),
    ("deck_features",   2024, 32),
    ("run_level",       2056, 20),
    ("choices",         2076, 126),
    ("relics_potions",  2202, 384),
]
EXPECTED_RUN_SIZE = 2586  # v3: same shape as v2, intent_dmg changed meaning

EXPECTED_COMBAT_LAYOUT = [
    ("player_state",    0,  4),
    ("player_powers",   4,  6),
    ("hand_cards",      10, 50),
    ("pile_summaries",  60, 6),
    ("enemies",         66, 65),
]
EXPECTED_COMBAT_SIZE = 131


def _as_tuples(layout):
    return [(b.name, b.start, b.size) for b in layout]


def test_run_observation_layout_is_unchanged():
    assert _as_tuples(RUN_OBS_LAYOUT) == EXPECTED_RUN_LAYOUT, (
        "The full-run observation layout changed.\n\n"
        f"{describe_layout(RUN_OBS_LAYOUT)}\n\n"
        "If that was intended: bump OBS_LAYOUT_VERSION and update the literals "
        "in this test. Every existing checkpoint is now invalid."
    )


def test_combat_observation_layout_is_unchanged():
    assert _as_tuples(COMBAT_OBS_LAYOUT) == EXPECTED_COMBAT_LAYOUT, (
        "The combat observation layout changed.\n\n"
        f"{describe_layout(COMBAT_OBS_LAYOUT)}"
    )


def test_layout_sizes_match_the_encoders():
    """The pinned total has to equal what the environments actually produce.

    Guards the case where the layout table is updated but an encoder is not, or
    the reverse -- the table would agree with itself while the vector coming out
    of the env is a different length.
    """
    from sts2_env.gym_env.observation import OBS_SIZE
    from sts2_env.gym_env.run_env import RUN_OBS_SIZE

    assert layout_size(RUN_OBS_LAYOUT) == EXPECTED_RUN_SIZE
    assert layout_size(COMBAT_OBS_LAYOUT) == EXPECTED_COMBAT_SIZE
    assert RUN_OBS_SIZE == EXPECTED_RUN_SIZE
    assert OBS_SIZE == EXPECTED_COMBAT_SIZE


def test_blocks_are_contiguous_and_gapless():
    for layout in (RUN_OBS_LAYOUT, COMBAT_OBS_LAYOUT):
        offset = 0
        for block in layout:
            assert block.start == offset, f"{block.name} starts at a gap"
            assert block.size > 0, f"{block.name} is empty"
            offset = block.stop


def test_a_real_observation_matches_the_pinned_size():
    """End to end: the env's actual vector, not just the constants."""
    from sts2_env.gym_env.combat_env import STS2CombatEnv
    from sts2_env.gym_env.run_env import STS2RunEnv

    env = STS2CombatEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (EXPECTED_COMBAT_SIZE,)

    run_env = STS2RunEnv(max_steps=100)
    run_obs, _ = run_env.reset(seed=0)
    assert run_obs.shape == (EXPECTED_RUN_SIZE,)


# --- the fingerprint ---------------------------------------------------------


def test_fingerprint_is_stable_across_calls():
    assert layout_fingerprint(RUN_OBS_LAYOUT) == layout_fingerprint(RUN_OBS_LAYOUT)


def test_fingerprint_notices_a_reorder_that_keeps_the_total():
    """The dangerous edit: two equal-sized blocks swap and the size is identical.

    A size check passes, every downstream reader keeps its offsets, and the
    policy silently reads one quantity as another.
    """
    original = (Block("a", 0, 10), Block("b", 10, 10))
    swapped = (Block("b", 0, 10), Block("a", 10, 10))
    assert layout_size(original) == layout_size(swapped)
    assert layout_fingerprint(original) != layout_fingerprint(swapped)


def test_fingerprint_notices_a_rename():
    assert layout_fingerprint((Block("a", 0, 4),)) != layout_fingerprint((Block("z", 0, 4),))


def test_fingerprint_notices_a_resize():
    assert layout_fingerprint((Block("a", 0, 4),)) != layout_fingerprint((Block("a", 0, 5),))


# --- checkpoint stamping -----------------------------------------------------


def test_stamp_then_verify_round_trips(tmp_path):
    model = tmp_path / "model.zip"
    model.write_bytes(b"not really a model")

    sidecar = stamp_checkpoint(model)
    assert sidecar.is_file()
    recorded = json.loads(sidecar.read_text())
    assert recorded["obs_layout_version"] == OBS_LAYOUT_VERSION
    assert recorded["size"] == EXPECTED_RUN_SIZE

    verify_checkpoint(model)  # must not raise


def test_verify_rejects_a_checkpoint_from_another_layout(tmp_path):
    model = tmp_path / "model.zip"
    model.write_bytes(b"not really a model")
    stamp_checkpoint(model)

    stale = json.loads((tmp_path / "model.zip.layout.json").read_text())
    stale["fingerprint"] = "0000000000000000"
    stale["size"] = 1234
    (tmp_path / "model.zip.layout.json").write_text(json.dumps(stale))

    with pytest.raises(ObservationLayoutMismatch) as excinfo:
        verify_checkpoint(model)
    # The message has to say what to do, not merely that something differs.
    assert "1234" in str(excinfo.value)
    assert str(EXPECTED_RUN_SIZE) in str(excinfo.value)


def test_unstamped_checkpoint_passes_by_default_and_fails_when_strict(tmp_path):
    """Checkpoints predating this module still load, unless asked otherwise."""
    model = tmp_path / "old_model.zip"
    model.write_bytes(b"pre-existing checkpoint")

    verify_checkpoint(model)  # tolerated
    with pytest.raises(ObservationLayoutMismatch):
        verify_checkpoint(model, allow_unstamped=False)


def test_manifest_is_json_serialisable():
    json.dumps(layout_manifest(RUN_OBS_LAYOUT))
