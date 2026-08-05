"""Train a MaskablePPO agent on STS2 combat.

Usage:
    pip install "sts2-rl-agent[train]"
    python scripts/train_combat.py

Requires: stable-baselines3, sb3-contrib, torch
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def make_env(seed: int = 0):
    """Create a single STS2CombatEnv."""
    from sts2_env.gym_env.combat_env import STS2CombatEnv

    def _init():
        env = STS2CombatEnv()
        env.reset(seed=seed)
        return env

    return _init


def train(args):
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from sb3_contrib.common.maskable.utils import get_action_masks
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        # MaskableEvalCallback, not stable_baselines3's EvalCallback. SB3's
        # evaluate_policy has no action_masks parameter, so evaluation ran with no
        # mask at all: the policy picked illegal actions, the env rejected them,
        # and the episode went nowhere. Every strange number this project has
        # produced came from that -- eval episode lengths of 1,814 rising to 8,351
        # (read as the policy learning to stall), a hard hang once deterministic
        # evaluation removed the resampling that had been escaping the loop, and a
        # mean reward pinned at -1.09 for 500k steps while the masked final
        # evaluation of the same weights reported an 83% win rate.
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    except ImportError:
        print("Training requires sb3-contrib and stable-baselines3.")
        print("Install with: pip install 'sts2-rl-agent[train]'")
        sys.exit(1)

    from sts2_env.core.game_build import check_decompile_matches_installed, write_fingerprint
    from sts2_env.gym_env.combat_env import STS2CombatEnv
    from sts2_env.search.situation import load_situations

    # Card values are read from a decompile on disk. If that is not the installed
    # build, this run trains on the previous patch and says nothing about it --
    # the logs, the reward curve and the saved model all look exactly right.
    # Worth refusing over, because the cost of a wrong run is the whole run.
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

    # A situation set, if asked for, replaces the starter-deck-vs-random-
    # encounter defaults with fights harvested from real runs: a 16-card
    # deck at 40 HP holding three relics, against the encounter the map
    # actually rolled. The previous-model plateau (combat_v3_overnight, 40M
    # steps with a flat eval curve) was a starter-deck model asked to play
    # fights it had never seen.
    situation_pool = None
    if args.situation_set:
        situation_pool = load_situations(args.situation_set)
        print(f"Training on {len(situation_pool)} real situations from "
              f"{args.situation_set}")
        if args.resume_from:
            print("  --resume-from: fine-tuning from a model that learned on "
                  "the starter-deck distribution; this re-fits it to the real one.")

    print(f"Training MaskablePPO on STS2 combat")
    print(f"  n_envs:          {args.n_envs}")
    print(f"  total_timesteps: {args.total_timesteps}")
    print(f"  learning_rate:   {args.lr}")
    print(f"  batch_size:      {args.batch_size}")
    print(f"  output_dir:      {args.output_dir}")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stamp the build before training rather than after saving, so a run that is
    # interrupted still leaves a directory that can say what it was learning.
    stamp_path = write_fingerprint(output_dir)
    print(f"  build stamp:     {stamp_path}")

    # Wrap env with action masker
    def mask_fn(env):
        return env.action_masks()

    def make_masked_env(seed: int):
        def _init():
            env = STS2CombatEnv(situation_pool=situation_pool)
            # The seed used to be accepted and then dropped on the floor, so every
            # env -- including the evaluation one, nominally seeded 9999 -- came up
            # unseeded and no run could be reproduced or compared to another.
            # Seeding once here is enough: later reset(seed=None) calls keep the
            # generator rather than replacing it.
            env.reset(seed=seed)
            env = ActionMasker(env, mask_fn)
            return env
        return _init

    # Create vectorized envs
    if args.n_envs > 1:
        train_env = SubprocVecEnv([make_masked_env(args.seed + i) for i in range(args.n_envs)])
    else:
        train_env = DummyVecEnv([make_masked_env(args.seed)])

    # Eval env (always single)
    eval_env = DummyVecEnv([make_masked_env(args.seed + 9999)])

    # Create model
    if args.resume_from:
        # Fine-tune rather than start over. A patch changes card values, not the
        # shape of the game, so the previous model's learned structure is still
        # broadly right and only needs re-fitting to the new numbers.
        #
        # load() checks the observation and action spaces against this env and
        # raises if they disagree, which is the failure we want: it means the
        # env changed shape and a warm start would be silently meaningless.
        print(f"  resuming from:   {args.resume_from}")
        model = MaskablePPO.load(args.resume_from, env=train_env, device="auto")
        model.set_env(train_env)
        model.tensorboard_log = str(output_dir / "tb_logs")
    else:
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
            # Without this the network init and action sampling are unseeded, so two
            # runs of the same command differ (measured: 83% and 79% win rate) and
            # neither can be compared to the other.
            seed=args.seed,
        )

    # Eval callback
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        # Deterministic evaluation, for two reasons. It is what the bridge will do
        # when Cyra actually plays, so it measures the thing that ships. And
        # sampled evaluation here was dominated by a heavy tail: a couple of
        # runaway episodes out of twenty pushed the reported mean episode length
        # into the thousands while the policy's real behaviour was 26 turns, and
        # burned 2.17M env steps against 250k of training.
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
    """Evaluate trained model."""
    from sb3_contrib.common.wrappers import ActionMasker
    from sts2_env.gym_env.combat_env import STS2CombatEnv

    def mask_fn(env):
        return env.action_masks()

    env = ActionMasker(STS2CombatEnv(), mask_fn)
    wins = 0
    total_rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 10000)
        done = False
        ep_reward = 0.0
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated
            if terminated and reward > 0:
                wins += 1
        total_rewards.append(ep_reward)

    print(f"Episodes:    {n_episodes}")
    print(f"Win rate:    {wins / n_episodes:.1%}")
    print(f"Avg reward:  {np.mean(total_rewards):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO on STS2 combat")
    parser.add_argument("--total-timesteps", type=int, default=500_000,
                        help="Total training timesteps (default: 500000)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Minibatch size (default: 256)")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Steps per rollout per env (default: 2048)")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="PPO epochs per update (default: 10)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient (default: 0.01)")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="Evaluate every N steps (default: 10000)")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Episodes per evaluation (default: 20)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Fine-tune from an existing model .zip instead of starting over")
    parser.add_argument("--situation-set", type=str, default=None,
                        help="Path to a fixture of real combat situations (JSON, "
                             "as produced by scripts/harvest_combat_benchmark.py). "
                             "When set, training samples fights from the file "
                             "instead of the Ironclad starter deck against random "
                             "act 1 encounters -- a 16-card deck at 40 HP, against "
                             "the encounter the map actually rolled. Breaks the "
                             "starter-deck plateau; see docs/GLM_ROADMAP_50P_ACT1.md.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for envs, network init and sampling (default: 0)")
    parser.add_argument("--allow-stale-decompile", action="store_true",
                        help="Train even if the decompile is not the installed game build")
    parser.add_argument("--output-dir", type=str, default="output/combat_ppo",
                        help="Output directory (default: output/combat_ppo)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
