"""Train a MaskablePPO agent on the STS2 full-run environment.

Usage:
    pip install "sts2-rl-agent[train]"
    python scripts/train_full_run.py

Requires: stable-baselines3, sb3-contrib, torch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def make_env(seed: int = 0):
    """Create a single STS2RunEnv wrapped with ActionMasker."""
    from sts2_env.gym_env.run_env import STS2RunEnv

    def _init():
        env = STS2RunEnv(max_steps=2000)
        env.reset(seed=seed)
        return env

    return _init


def make_masked_env(seed: int):
    """Create a masked env factory for vectorised envs."""
    try:
        from sb3_contrib.common.wrappers import ActionMasker
    except ImportError:
        print("Training requires sb3-contrib and stable-baselines3.")
        print("Install with: pip install 'sts2-rl-agent[train]'")
        sys.exit(1)

    from sts2_env.gym_env.run_env import STS2RunEnv

    def mask_fn(env):
        return env.action_masks()

    def _init():
        env = STS2RunEnv(max_steps=2000)
        # The seed was accepted and dropped here, so every env came up unseeded
        # and no run could be reproduced or compared against another.
        env.reset(seed=seed)
        env = ActionMasker(env, mask_fn)
        return env

    return _init


def train(args):
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        # MaskableEvalCallback, not stable_baselines3's EvalCallback: SB3's
        # evaluate_policy takes no action_masks, so evaluation runs unmasked. In
        # the combat trainer that produced a reward curve pinned flat for 500k
        # steps while the masked final evaluation of the same weights reported an
        # 83% win rate, and a hard hang once evaluation went deterministic. Here
        # max_steps=2000 would bound it rather than hang, but every eval episode
        # would still run to the cap and measure nothing.
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    except ImportError:
        print("Training requires sb3-contrib and stable-baselines3.")
        print("Install with: pip install 'sts2-rl-agent[train]'")
        sys.exit(1)

    print(f"Training MaskablePPO on STS2 Full Run")
    from sts2_env.core.game_build import check_decompile_matches_installed, write_fingerprint

    # Card values are read from a decompile on disk. If it is not the installed
    # build this trains on the previous patch and says nothing about it: logs,
    # reward curve and saved model all look right. Worth refusing over, and doubly
    # so for a run left going unattended.
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

    print(f"  n_envs:          {args.n_envs}")
    print(f"  total_timesteps: {args.total_timesteps}")
    print(f"  learning_rate:   {args.lr}")
    print(f"  batch_size:      {args.batch_size}")
    print(f"  output_dir:      {args.output_dir}")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stamped before training, so an interrupted run still records what it learned.
    print(f"  build stamp:     {write_fingerprint(output_dir)}")

    # Create vectorised environments
    if args.n_envs > 1:
        train_env = SubprocVecEnv([
            make_masked_env(args.seed + i)
            for i in range(args.n_envs)
        ])
    else:
        train_env = DummyVecEnv([
            make_masked_env(args.seed)
        ])

    # Eval env (always single)
    eval_env = DummyVecEnv([
        make_masked_env(args.seed + 9999)
    ])

    # Create model
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=str(output_dir / "tb_logs"),
        # Without this, network init and action sampling are unseeded and two runs
        # of the same command differ by more than most changes worth testing.
        seed=args.seed,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
    )

    # Eval callback
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    # Train
    start = time.perf_counter()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )
    elapsed = time.perf_counter() - start

    # Save final model
    final_path = str(output_dir / "final_model")
    model.save(final_path)
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {output_dir / 'best_model'}")

    # Quick evaluation
    print("\n--- Final Evaluation ---")
    evaluate(model, n_episodes=100)

    train_env.close()
    eval_env.close()


def evaluate(model, n_episodes: int = 100):
    """Evaluate a trained model on the full-run environment."""
    from sb3_contrib.common.wrappers import ActionMasker
    from sts2_env.gym_env.run_env import STS2RunEnv

    def mask_fn(env):
        return env.action_masks()

    env = ActionMasker(
        STS2RunEnv(max_steps=2000),
        mask_fn,
    )

    wins = 0
    total_rewards = []
    floors_reached = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 10000)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            steps += 1
            done = terminated or truncated
            if terminated and reward > 0:
                wins += 1
        total_rewards.append(ep_reward)
        episode_lengths.append(steps)
        floors_reached.append(info.get("floor", 0))

    print(f"Episodes:         {n_episodes}")
    print(f"Win rate:         {wins / n_episodes:.1%}")
    print(f"Avg reward:       {np.mean(total_rewards):.3f}")
    print(f"Avg ep length:    {np.mean(episode_lengths):.1f}")
    print(f"Avg floors:       {np.mean(floors_reached):.1f}")
    print(f"Max floors:       {max(floors_reached)}")


def random_baseline(n_episodes: int = 100):
    """Run a random-action baseline for comparison."""
    from sts2_env.gym_env.run_env import STS2RunEnv

    env = STS2RunEnv(max_steps=2000)
    rng = np.random.RandomState(42)

    wins = 0
    total_rewards = []
    floors_reached = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        while not done:
            mask = info["action_mask"]
            valid = np.where(mask == 1)[0]
            action = int(rng.choice(valid))
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            if terminated and reward > 0:
                wins += 1
        total_rewards.append(ep_reward)
        floors_reached.append(info.get("floor", 0))

    print(f"=== Random Baseline ===")
    print(f"Episodes:         {n_episodes}")
    print(f"Win rate:         {wins / n_episodes:.1%}")
    print(f"Avg reward:       {np.mean(total_rewards):.3f}")
    print(f"Avg floors:       {np.mean(floors_reached):.1f}")
    print(f"Max floors:       {max(floors_reached)}")


def main():
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO on STS2 full run"
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
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)",
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
        "--ent-coef", type=float, default=0.02,
        help="Entropy coefficient (default: 0.02)",
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
        "--seed", type=int, default=0,
        help="Seed for envs, network init and sampling (default: 0)",
    )
    parser.add_argument(
        "--allow-stale-decompile", action="store_true",
        help="Train even if the decompile is not the installed game build",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/run_ppo",
        help="Output directory (default: output/run_ppo)",
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Only run random baseline evaluation (no training)",
    )
    args = parser.parse_args()

    if args.baseline_only:
        random_baseline()
    else:
        # Print random baseline first for reference
        print("Running random baseline for reference...")
        random_baseline(n_episodes=50)
        print()
        train(args)


if __name__ == "__main__":
    main()
