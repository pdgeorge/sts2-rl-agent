"""Hierarchical environment: meta-policy for non-combat, frozen policy for combat.

Wraps STS2RunEnv so that combat phases are fast-forwarded by a combat solver
(frozen RL policy or heuristic) rather than exposed to the meta-policy step-by-step.
The meta-policy only makes non-combat decisions: map, card rewards, shop, rest,
events, and treasure.

This cuts the meta-policy's learning problem in half. The combat policy (which
already achieves 92% win rate on act 1 encounters) handles fights; the
meta-policy only has to learn deckbuilding, pathing, and resource management.

Usage:
    env = HierarchicalRunEnv(combat_policy_path="output/combat_ppo_v3/final_model.zip")
    # or with a heuristic:
    env = HierarchicalRunEnv(combat_solver=HeuristicCombatSolver())
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from sts2_env.gym_env.run_env import STS2RunEnv, DEFAULT_MAX_STEPS, DEFAULT_MAX_COMBAT_TURNS
from sts2_env.run.run_manager import RunManager

logger = logging.getLogger(__name__)


class CombatSolver:
    """Abstract combat solver interface."""

    def solve(self, run_env: STS2RunEnv) -> tuple[float, bool, bool]:
        """Run combat to completion inside the given env.

        Returns (total_reward, terminated, truncated).
        """
        raise NotImplementedError


class HeuristicCombatSolver(CombatSolver):
    """Simple deterministic combat solver.

    Plays the highest-damage playable card each turn, targeting the first
    alive enemy. Uses no potions. Ends turn when out of playable attacks or
    when no damage can be dealt.

    Not meant to be optimal -- just a baseline that finishes combat faster
    than random without requiring a trained model.
    """

    def solve(self, run_env: STS2RunEnv) -> tuple[float, bool, bool]:
        mgr = run_env._mgr
        assert mgr is not None
        total_reward = 0.0
        steps = 0
        max_steps = run_env.max_combat_turns * 10  # generous upper bound
        truncated_seen = False

        while mgr.phase == RunManager.PHASE_COMBAT and steps < max_steps:
            combat = mgr.get_combat_state()
            if combat is None or combat.is_over:
                break

            # Get the action mask
            from sts2_env.gym_env.action_space import get_action_mask
            from sts2_env.gym_env.action_space import action_to_card_and_target

            owner = combat.primary_player
            mask = get_action_mask(combat, owner=owner)
            valid = np.where(mask == 1)[0]
            if len(valid) == 0:
                break

            # Pick highest-damage playable card targeting first alive enemy
            best_action = self._pick_action(combat, valid)
            # Route through step() rather than _step_combat() so the rewards
            # that step() accrues -- COMBAT_WON, ELITE_WON, BOSS_WON,
            # FLOOR_REACHED -- reach the meta-policy. _step_combat bypassed
            # every reward block in step, so the meta-policy saw only the
            # terminal +/- and the card-reward shaping; combat outcome, the
            # main signal, was invisible. The work step() does that _step_combat
            # doesn't (observation encoding) is cheap and only happens while a
            # combat is being fast-forwarded, never on the hot training path.
            _, step_reward, terminated, truncated, _ = run_env.step(best_action)
            total_reward += step_reward
            if truncated:
                truncated_seen = True
            steps += 1

            # Check if run ended during combat (e.g. player died)
            if terminated or truncated or mgr.is_over:
                break

        return total_reward, mgr.is_over, truncated_seen

    def _pick_action(self, combat, valid_actions):
        from sts2_env.gym_env.action_space import action_to_card_and_target

        best_action = None
        best_damage = -1

        for action in valid_actions:
            if action == 0:
                # End turn -- low priority
                continue

            hand_idx, target_idx = action_to_card_and_target(action)
            if hand_idx is None:
                continue

            if hand_idx >= len(combat.hand):
                continue

            card = combat.hand[hand_idx]
            dmg = card.base_damage or 0
            if dmg > best_damage:
                best_damage = dmg
                best_action = action

        # If no damage card found, end turn
        if best_action is None and 0 in valid_actions:
            return 0

        # Fallback: first valid action
        if best_action is None:
            return int(valid_actions[0])

        return int(best_action)


class FrozenRLCombatSolver(CombatSolver):
    """Frozen MaskablePPO policy loaded from a .zip checkpoint.

    Defaults to CPU inference. A 131-dim MLP is microseconds on CPU,
    and CPU avoids CUDA deadlocks when multiple SubprocVecEnv workers
    each try to create a CUDA context on the same GPU.
    """

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        try:
            from sb3_contrib import MaskablePPO
        except ImportError:
            raise ImportError(
                "FrozenRLCombatSolver requires sb3-contrib. "
                "Install with: pip install 'sts2-rl-agent[train]'"
            )
        self.model = MaskablePPO.load(str(model_path), device=device)
        self.model_path = str(model_path)

    def solve(self, run_env: STS2RunEnv) -> tuple[float, bool, bool]:
        mgr = run_env._mgr
        assert mgr is not None
        total_reward = 0.0
        steps = 0
        max_steps = run_env.max_combat_turns * 10

        while mgr.phase == RunManager.PHASE_COMBAT and steps < max_steps:
            combat = mgr.get_combat_state()
            if combat is None or combat.is_over:
                break

            # Build combat observation from the combat state
            from sts2_env.gym_env.observation import encode_observation
            obs = encode_observation(combat)

            # Build action mask
            from sts2_env.gym_env.action_space import get_action_mask
            owner = combat.primary_player
            mask = get_action_mask(combat, owner=owner)

            # Predict action
            action, _ = self.model.predict(
                obs, action_masks=mask, deterministic=True
            )
            # Same change as HeuristicCombatSolver: route through step() so
            # COMBAT_WON / ELITE_WON / BOSS_WON / FLOOR_REACHED reach the
            # meta-policy instead of being silently discarded by
            # _step_combat's bypass of step's reward block.
            _, step_reward, terminated, truncated, _ = run_env.step(int(action))
            total_reward += step_reward
            steps += 1

            if terminated or truncated or mgr.is_over:
                break

        return total_reward, mgr.is_over, False


class HierarchicalRunEnv(gymnasium.Env):
    """Meta-policy environment that delegates combat to a solver.

    Wraps STS2RunEnv. The meta-policy only steps through non-combat phases.
    When combat starts, the solver fast-forwards it internally. The meta-policy
    sees the post-combat state as its next observation.

    Observation space: same as STS2RunEnv (so existing models can load if
    combat_policy_path is None), but combat observations are mostly zeroed
    outside combat.

    Action space: same as STS2RunEnv. During non-combat phases, combat actions
    are masked out. During combat (which is fast-forwarded), no meta-policy step
    occurs.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        character_id: str = "Ironclad",
        ascension_level: int = 0,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_combat_turns: int = DEFAULT_MAX_COMBAT_TURNS,
        combat_solver: CombatSolver | None = None,
        render_mode: str | None = None,
        gamma: float = 0.99,
        enable_reward_shaping: bool = False,
    ):
        super().__init__()

        self._inner = STS2RunEnv(
            character_id=character_id,
            ascension_level=ascension_level,
            max_steps=max_steps,
            max_combat_turns=max_combat_turns,
            render_mode=render_mode,
            gamma=gamma,
            enable_reward_shaping=enable_reward_shaping,
        )
        self.combat_solver = combat_solver or HeuristicCombatSolver()
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return self._inner.reset(seed=seed, options=options)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Meta-policy step.

        If currently in combat, fast-forward with the solver instead of
        taking a meta-policy action.
        """
        # If we're somehow in combat (shouldn't happen if caller is well-behaved,
        # but a previous step's non-combat action could have triggered one),
        # fast-forward first.
        if self._inner._mgr is not None and self._inner._mgr.phase == RunManager.PHASE_COMBAT:
            logger.debug("Fast-forwarding combat at start of meta-policy step")
            combat_reward, terminated, truncated = self.combat_solver.solve(self._inner)
            if terminated or truncated:
                obs = self._inner._encode_obs()
                info = self._inner._build_info()
                return obs, combat_reward, terminated, truncated, info

        # Normal non-combat step
        obs, reward, terminated, truncated, info = self._inner.step(action)

        # If the non-combat action put us into combat, fast-forward it
        if (
            not terminated
            and not truncated
            and self._inner._mgr is not None
            and self._inner._mgr.phase == RunManager.PHASE_COMBAT
        ):
            logger.debug("Fast-forwarding combat after non-combat action")
            combat_reward, terminated, truncated = self.combat_solver.solve(self._inner)
            reward += combat_reward

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._inner.action_masks()

    def render(self) -> str | None:
        return self._inner.render()

    def close(self):
        self._inner.close()
