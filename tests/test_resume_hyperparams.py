"""Resuming must honour --lr, and the override must reach the optimizer.

MaskablePPO.load() restores the checkpoint's own hyperparameters, so --lr and
--ent-coef were accepted, printed in the config banner, and then silently ignored.
Fine-tuning at a lower rate was impossible while appearing configured.

That is what a converged resume needs. Resuming alpha at its original 3e-4 ran
20.5M steps for a reward change of -0.03 (-0.1 sem) with approx_kl at 0.037 and
clip_fraction at 0.20 -- large updates every step, rebuilding the same policy
rather than refining it.

The second test is the one that matters mechanically: assigning model.learning_rate
changes nothing on its own, because the optimizer already exists and PPO reads
lr_schedule. Without _setup_lr_schedule() the flag would still be a no-op, and the
banner would still say it had been applied.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure


class _MaskedStub(gym.Env):
    """Smallest env MaskablePPO will accept: Box obs, Discrete actions, a mask.

    Deliberately trivial -- these tests are about hyperparameter plumbing, not
    about learning anything, so the env only has to be constructible.
    """

    def __init__(self):
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._n = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._n = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._n += 1
        return np.zeros(4, dtype=np.float32), 0.0, self._n >= 8, False, {}

    def action_masks(self):
        return np.ones(self.action_space.n, dtype=bool)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A tiny model saved at a known learning rate, standing in for alpha."""
    env = _MaskedStub()
    model = MaskablePPO("MlpPolicy", env, learning_rate=3e-4, ent_coef=0.02,
                        n_steps=64, batch_size=32, verbose=0)
    path = tmp_path_factory.mktemp("ckpt") / "alpha_stub.zip"
    model.save(path)
    return path, env


def test_loading_restores_the_checkpoints_rate(checkpoint):
    """Baseline: this is why --lr was a no-op."""
    path, env = checkpoint
    model = MaskablePPO.load(path, env=env)
    assert model.lr_schedule(1.0) == pytest.approx(3e-4)


def test_the_override_reaches_the_lr_schedule(checkpoint):
    path, env = checkpoint
    model = MaskablePPO.load(path, env=env)

    model.learning_rate = 5e-5
    model._setup_lr_schedule()

    assert model.lr_schedule(1.0) == pytest.approx(5e-5), (
        "the schedule PPO actually reads must reflect the override")


def test_assigning_learning_rate_alone_is_not_enough(checkpoint):
    """Pins the trap: without _setup_lr_schedule the override is invisible.

    If SB3 ever makes the assignment sufficient, this test fails and the extra
    call can go -- which is the right way to find that out.
    """
    path, env = checkpoint
    model = MaskablePPO.load(path, env=env)
    model.learning_rate = 5e-5
    assert model.lr_schedule(1.0) == pytest.approx(3e-4), (
        "assignment alone still reports the old rate")


def test_the_override_survives_into_the_optimizer(checkpoint):
    """The schedule is only useful if training pushes it into the param groups."""
    path, env = checkpoint
    model = MaskablePPO.load(path, env=env)
    model.learning_rate = 5e-5
    model._setup_lr_schedule()
    # _update_learning_rate records the rate, so it needs a logger; training
    # normally attaches one via learn().
    model.set_logger(configure(None, format_strings=[]))
    model._update_learning_rate(model.policy.optimizer)

    for group in model.policy.optimizer.param_groups:
        assert group["lr"] == pytest.approx(5e-5)


def test_ent_coef_override_is_a_plain_assignment(checkpoint):
    """ent_coef is read directly in train(), so no schedule rebuild is needed."""
    path, env = checkpoint
    model = MaskablePPO.load(path, env=env)
    assert model.ent_coef == pytest.approx(0.02)
    model.ent_coef = 0.005
    assert model.ent_coef == pytest.approx(0.005)
