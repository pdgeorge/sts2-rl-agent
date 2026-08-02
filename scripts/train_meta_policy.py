"""Train a MaskablePPO meta-policy on STS2 full run with frozen combat.

Usage:
    pip install "sts2-rl-agent[train]"
    python scripts/train_meta_policy.py

The meta-policy makes non-combat decisions (map, card rewards, shop, rest,
events, treasure). Combat is fast-forwarded by a frozen combat solver -- either
a pre-trained combat policy or a simple heuristic.

Requires: stable-baselines3, sb3-contrib, torch
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np


def make_masked_env(seed: int, combat_solver=None, enable_reward_shaping: bool = True, max_combat_turns: int | None = None):
    """Create a masked env factory for vectorised envs."""
    try:
        from sb3_contrib.common.wrappers import ActionMasker
    except ImportError:
        print("Training requires sb3-contrib and stable-baselines3.")
        print("Install with: pip install 'sts2-rl-agent[train]'")
        sys.exit(1)

    from sts2_env.gym_env.hierarchical_env import HierarchicalRunEnv

    def mask_fn(env):
        return env.action_masks()

    def _init():
        kwargs = dict(
            max_steps=2000,
            combat_solver=combat_solver,
            enable_reward_shaping=enable_reward_shaping,
        )
        if max_combat_turns is not None:
            kwargs["max_combat_turns"] = max_combat_turns
        env = HierarchicalRunEnv(**kwargs)
        env.reset(seed=seed)
        env = ActionMasker(env, mask_fn)
        return env

    return _init


def train(args):
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            StopTrainingOnNoModelImprovement,
        )
    except ImportError:
        print("Training requires sb3-contrib and stable-baselines3.")
        print("Install with: pip install 'sts2-rl-agent[train]'")
        sys.exit(1)

    print(f"Training MaskablePPO meta-policy on STS2 Full Run")
    from sts2_env.core.game_build import check_decompile_matches_installed, write_fingerprint

    matches, reason = check_decompile_matches_installed()
    if not matches:
        print("Refusing to train: the decompile in use is not the installed game build.\n")
        print(reason)
        if not args.allow_stale_decompile:
            print("\n  --allow-stale-decompile overrides this, if the mismatch is deliberate.")
            sys.exit(2)
        print("\n  --allow-stale-decompile given; continuing anyway.")
    else:
        print(f"Game build: {reason}")

    # Load frozen combat solver if requested
    combat_solver = None
    if args.combat_policy:
        from sts2_env.gym_env.hierarchical_env import FrozenRLCombatSolver
        print(f"  combat solver:   FrozenRLCombatSolver({args.combat_policy})")
        combat_solver = FrozenRLCombatSolver(args.combat_policy)
    else:
        from sts2_env.gym_env.hierarchical_env import HeuristicCombatSolver
        print("  combat solver:   HeuristicCombatSolver (baseline)")
        combat_solver = HeuristicCombatSolver()

    print(f"  n_envs:          {args.n_envs}")
    print(f"  total_timesteps: {args.total_timesteps}")
    if args.lr is None:
        print("  learning_rate:   "
              + ("from the checkpoint" if args.resume_from else f"{_DEFAULT_LR}"))
    else:
        print(f"  learning_rate:   {args.lr}")
    print(f"  batch_size:      {args.batch_size}")
    print(f"  output_dir:      {args.output_dir}")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  build stamp:     {write_fingerprint(output_dir)}")

    # Create vectorised environments
    if args.n_envs > 1:
        train_env = SubprocVecEnv([
            make_masked_env(args.seed + i, combat_solver, args.enable_reward_shaping)
            for i in range(args.n_envs)
        ])
    else:
        train_env = DummyVecEnv([
            make_masked_env(args.seed, combat_solver, args.enable_reward_shaping)
        ])

    eval_env = DummyVecEnv([
        make_masked_env(args.seed + 9999, combat_solver, args.enable_reward_shaping,
                        max_combat_turns=30)
    ])

    # Create model
    if args.resume_from:
        print(f"  resuming from:   {args.resume_from}")
        model = MaskablePPO.load(args.resume_from, env=train_env, device="auto")
        model.set_env(train_env)
        model.tensorboard_log = str(output_dir / "tb_logs")

        if args.lr is not None:
            model.learning_rate = args.lr
            model._setup_lr_schedule()
            print(f"  learning_rate:   overridden to {args.lr}")
        if args.ent_coef is not None:
            model.ent_coef = args.ent_coef
            print(f"  ent_coef:        overridden to {args.ent_coef}")
    else:
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=_DEFAULT_LR if args.lr is None else args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=_DEFAULT_ENT_COEF if args.ent_coef is None else args.ent_coef,
            verbose=1,
            tensorboard_log=str(output_dir / "tb_logs"),
            seed=args.seed,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
        )

    class FloorLogger(BaseCallback):
        """Log floors reached."""

        def _on_step(self) -> bool:
            dones = self.locals.get("dones", [])
            infos = self.locals.get("infos", [])
            floors = [
                info["floor"]
                for info, done in zip(infos, dones)
                if done and isinstance(info, dict) and "floor" in info
            ]
            if floors:
                self.logger.record("run/mean_floor", float(np.mean(floors)))
                self.logger.record("run/max_floor", float(np.max(floors)))
            return True

    stop_callback = None
    if args.stop_after_no_improvement:
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=args.stop_after_no_improvement,
            min_evals=args.stop_min_evals,
            verbose=1,
        )

    class TimeoutMaskableEvalCallback(MaskableEvalCallback):
        """MaskableEvalCallback with a hard wall-clock timeout per eval cycle.

        The synchronous eval can hang the main thread when a combat episode
        enters a non-terminating stall. This raises TimeoutError after
        ``timeout_seconds`` and logs the skip so training continues.
        """

        def __init__(self, *args, timeout_seconds: int = 180, **kwargs):
            super().__init__(*args, **kwargs)
            self.timeout_seconds = timeout_seconds

        def _on_step(self) -> bool:
            def _alarm_handler(signum, frame):
                raise TimeoutError(
                    f"Eval exceeded {self.timeout_seconds}s wall-clock timeout"
                )

            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(self.timeout_seconds)
            try:
                result = super()._on_step()
            except TimeoutError:
                self.logger.record("eval/timeout", 1)
                print(
                    f"\n[EVAL TIMEOUT] Evaluation cycle exceeded "
                    f"{self.timeout_seconds}s. Skipping this eval.\n"
                )
                result = True
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result

    eval_callback = TimeoutMaskableEvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        callback_after_eval=stop_callback,
        timeout_seconds=args.eval_timeout_seconds,
    )

    from stable_baselines3.common.callbacks import CheckpointCallback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.eval_freq // args.n_envs, 1),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ckpt",
    )

    def _save_and_stop(signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _save_and_stop)

    start = time.perf_counter()
    interrupted = False
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_callback, checkpoint_callback, FloorLogger()],
            progress_bar=True,
            reset_num_timesteps=args.resume_from is None,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n\nInterrupted. Saving the model as it stands before exiting.")
    elapsed = time.perf_counter() - start

    final_path = str(output_dir / "final_model")
    model.save(final_path)
    print(f"\nTraining {'interrupted after' if interrupted else 'complete in'} {elapsed:.1f}s")
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {output_dir / 'best_model'}")

    print("\n--- Final Evaluation ---")
    evaluate(model, eval_env, n_episodes=100)

    train_env.close()
    eval_env.close()


def evaluate(model, env, n_episodes: int = 100):
    """Evaluate a trained meta-policy on a vectorized env."""
    from sb3_contrib.common.wrappers import ActionMasker
    from sts2_env.gym_env.hierarchical_env import HierarchicalRunEnv

    def mask_fn(e):
        return e.action_masks()

    wins = 0
    total_rewards = []
    floors_reached = []
    episode_lengths = []

    for ep in range(n_episodes):
        e = ActionMasker(HierarchicalRunEnv(max_steps=2000), mask_fn)
        obs, info = e.reset(seed=ep + 10000)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            masks = e.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = e.step(int(action))
            ep_reward += reward
            steps += 1
            done = terminated or truncated
            if terminated and reward > 0:
                wins += 1
        total_rewards.append(ep_reward)
        episode_lengths.append(steps)
        floors_reached.append(info.get("floor", 0))
        e.close()

    print(f"Episodes:         {n_episodes}")
    print(f"Win rate:         {wins / n_episodes:.1%}")
    print(f"Avg reward:       {np.mean(total_rewards):.3f}")
    print(f"Avg ep length:    {np.mean(episode_lengths):.1f}")
    print(f"Avg floors:       {np.mean(floors_reached):.1f}")
    print(f"Max floors:       {max(floors_reached)}")


_DEFAULT_LR = 3e-4
_DEFAULT_ENT_COEF = 0.02


def main():
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO meta-policy on STS2 full run"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=1_000_000,
        help="Total training timesteps (default: 1000000)",
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default: 3e-4 fresh; on --resume-from, the checkpoint's own rate unless given here)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Minibatch size (default: 256)",
    )
    parser.add_argument(
        "--n-steps", type=int, default=2048,
        help="Steps per rollout per env (default: 2048)",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=10,
        help="PPO epochs per update (default: 10)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.995,
        help="Discount factor (default: 0.995, higher for long episodes)",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=None,
        help="Entropy coefficient (default: 0.02 fresh; on --resume-from, the checkpoint's own value unless given here)",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=20_000,
        help="Evaluate every N steps (default: 20000)",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10,
        help="Episodes per evaluation (default: 10)",
    )
    parser.add_argument(
        "--eval-timeout-seconds", type=int, default=180,
        help="Hard wall-clock timeout per evaluation cycle (default: 180). "
             "If eval hangs, it is aborted and training continues.",
    )
    parser.add_argument(
        "--stop-after-no-improvement", type=int, default=0,
        help="Stop when this many consecutive evals bring no new best model (0 = never)",
    )
    parser.add_argument(
        "--stop-min-evals", type=int, default=20,
        help="Never stop before this many evaluations (default: 20)",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Fine-tune from an existing model .zip instead of starting over",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for envs, network init and sampling (default: 0)",
    )
    parser.add_argument(
        "--allow-stale-decompile", action="store_true",
        help="Train even if the decompile is not the installed game build",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/meta_ppo",
        help="Output directory (default: output/meta_ppo)",
    )
    parser.add_argument(
        "--combat-policy", type=str, default=None,
        help="Path to a frozen combat model .zip (default: use heuristic solver)",
    )
    parser.add_argument(
        "--enable-reward-shaping", action="store_true", default=True,
        help="Give small bonus/penalty for card reward deckbuilding (default: on)",
    )
    parser.add_argument(
        "--no-reward-shaping", action="store_true", default=False,
        help="Disable card reward shaping",
    )
    args = parser.parse_args()
    if args.no_reward_shaping:
        args.enable_reward_shaping = False
    train(args)


if __name__ == "__main__":
    main()
