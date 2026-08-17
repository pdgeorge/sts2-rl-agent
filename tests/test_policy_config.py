"""A policy is a versioned file; the decision path runs what the file says.

These pin the three things PHASE_TWO section 3.1-3.2 exist to guarantee:
the shipped v001 file reproduces the legacy module constants exactly (so
switching to config-driven play changes no behaviour), a malformed config fails
LOUDLY at load rather than silently tuning nothing, and every value actually
reaches the readers that use it.
"""

from __future__ import annotations

import json

import pytest


def _load_module_constants():
    """The legacy shipped defaults, read from the modules themselves."""
    from sts2_env.bridge import agent_runner, card_quality
    from sts2_env.search.evaluate import EvalWeights
    return agent_runner, card_quality, EvalWeights


def test_v001_reproduces_the_shipped_constants_exactly():
    """Loading the default must be a no-op in behaviour.

    If this fails, the file and the code have drifted and one of them is
    tuning a number the other does not know about.
    """
    from sts2_env.policy_config import PolicyConfig
    from sts2_env.search.evaluate import EvalWeights

    agent_runner, card_quality, _ = _load_module_constants()
    policy = PolicyConfig.load()  # v001

    assert policy.policy_version == "v001"
    assert policy.eval_weights == EvalWeights()
    assert policy.room_min_hp_fraction == agent_runner.ROOM_MIN_HP_FRACTION
    assert policy.quality_bar_scale == agent_runner.QUALITY_BAR_SCALE
    assert policy.skip_threshold == card_quality.SKIP_THRESHOLD
    assert policy.large_deck_size == agent_runner.CARD_REWARD_LARGE_DECK_SIZE
    assert policy.scaling_bonus == card_quality.SCALING_BONUS
    assert policy.block_need_bonus == card_quality.BLOCK_NEED_BONUS
    assert policy.git_sha, "the stamp must carry a sha, even 'unknown'"


def test_a_malformed_config_fails_loudly(tmp_path):
    from sts2_env.policy_config import PolicyConfig

    base = json.loads((PolicyConfig.load().source_path and
                       __import__("pathlib").Path(PolicyConfig.load().source_path)).read_text())

    missing = dict(base)
    del missing["eval_weights"]
    p = tmp_path / "missing.json"
    p.write_text(json.dumps(missing))
    with pytest.raises(ValueError, match="missing"):
        PolicyConfig.load(p)

    unknown = dict(base)
    unknown["a_dial_that_does_not_exist"] = 1.0
    p2 = tmp_path / "unknown.json"
    p2.write_text(json.dumps(unknown))
    with pytest.raises(ValueError, match="unknown"):
        PolicyConfig.load(p2)

    unknown_weight = dict(base)
    unknown_weight["eval_weights"] = dict(base["eval_weights"], bogus=0.1)
    p3 = tmp_path / "unknown_weight.json"
    p3.write_text(json.dumps(unknown_weight))
    with pytest.raises(ValueError, match="unknown"):
        PolicyConfig.load(p3)


def test_set_active_policy_writes_every_legacy_reader():
    """Applying a policy must reach the constants the decision path reads."""
    from sts2_env.bridge import agent_runner, card_quality
    from sts2_env.policy_config import PolicyConfig, active_policy, set_active_policy

    original = active_policy()
    try:
        tuned = original.with_weights(enemy_hp=0.5)
        tuned = PolicyConfig(
            policy_version="v_test",
            source_path="<test>",
            git_sha=original.git_sha,
            eval_weights=tuned.eval_weights,
            room_min_hp_fraction={**original.room_min_hp_fraction, "elite": 0.55},
            quality_bar_scale=12.0,
            skip_threshold=0.25,
            large_deck_size=28,
            scaling_bonus=2.0,
            block_need_bonus=0.5,
        )
        set_active_policy(tuned)

        assert agent_runner.ROOM_MIN_HP_FRACTION["elite"] == 0.55
        assert agent_runner.QUALITY_BAR_SCALE == 12.0
        assert agent_runner.CARD_REWARD_LARGE_DECK_SIZE == 28
        assert card_quality.SKIP_THRESHOLD == 0.25
        assert card_quality.SCALING_BONUS == 2.0
        assert card_quality.BLOCK_NEED_BONUS == 0.5
        assert active_policy().policy_version == "v_test"
    finally:
        set_active_policy(original)
        assert active_policy().policy_version == "v001"


def test_with_weights_rejects_unknown_names():
    from sts2_env.policy_config import PolicyConfig

    with pytest.raises(ValueError, match="unknown eval weight"):
        PolicyConfig.load().with_weights(not_a_weight=1.0)
