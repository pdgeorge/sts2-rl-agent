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


class TestPotionHoldIsAPolicyFieldNotAGlobal:
    """The hold ships OFF and is turned on by a policy, never by a patch.

    It was live for one session as a hard-coded frozenset. The measurement said
    the mechanism worked and the hypothesis did not -- trash use of the five
    fell 85% to 12%, and potions held entering the act 1 boss did not move at
    all (0.99 -> 0.97), because they were spent one room earlier on elites.
    Reverted here, and made configurable so the A/B can be run the sanctioned
    way: PHASE_TWO section 3.1 records a sweep that ran 400 runs with its
    baseline arm doing the opposite of its name, because it patched a global.
    """

    def test_the_shipped_policy_holds_nothing(self):
        from sts2_env.policy_config import PolicyConfig
        assert PolicyConfig.load("v001").hold_potions_for_big_fights == ()

    def test_the_experiment_policy_holds_the_five(self):
        from sts2_env.policy_config import PolicyConfig
        held = set(PolicyConfig.load("v002_hold_potions").hold_potions_for_big_fights)
        assert held == {"PowderedDemise", "DistilledChaos", "GigantificationPotion",
                        "OrobicAcid", "Duplicator"}

    def test_applying_a_policy_drives_the_module_constant(self):
        from sts2_env.policy_config import PolicyConfig, apply_active_policy
        from sts2_env.search import potion_policy

        apply_active_policy(PolicyConfig.load("v002_hold_potions"))
        assert len(potion_policy.HOLD_FOR_BIG_FIGHTS) == 5
        apply_active_policy(PolicyConfig.load("v001"))
        assert potion_policy.HOLD_FOR_BIG_FIGHTS == frozenset()

    def test_a_policy_written_before_the_key_existed_still_loads(self):
        """Optional, so old configs keep meaning what they meant."""
        import json
        from sts2_env.policy_config import PolicyConfig

        data = json.loads(open("policies/v001.json").read())
        data.pop("hold_potions_for_big_fights", None)
        assert PolicyConfig.from_dict(data).hold_potions_for_big_fights == ()

    def test_a_misspelled_key_is_still_rejected(self):
        """Optional must not mean unvalidated -- that is the whole check."""
        import json

        import pytest

        from sts2_env.policy_config import PolicyConfig

        data = json.loads(open("policies/v001.json").read())
        data["hold_potions_for_big_fite"] = []
        with pytest.raises(ValueError, match="unknown keys"):
            PolicyConfig.from_dict(data)
