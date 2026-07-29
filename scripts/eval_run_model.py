"""Evaluate a full-run model in the simulator and show where it dies.

Reads a checkpoint and plays N runs, reporting the distribution of floors rather
than just the mean -- "average 8.9" hides whether that is every run reaching 9 or
half reaching 3 and half reaching 15, and those want different fixes.

Safe to run against best_model while training is still going; it only reads.

    python scripts/eval_run_model.py output/run_ppo_v2/best_model/best_model.zip
    python scripts/eval_run_model.py <model.zip> --episodes 200 --stochastic
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Path to a full-run model .zip")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--stochastic", action="store_true",
                    help="Sample actions instead of taking the argmax")
    ap.add_argument("--seed-base", type=int, default=90000)
    args = ap.parse_args()

    from sb3_contrib import MaskablePPO
    from sts2_env.gym_env.run_env import STS2RunEnv

    model = MaskablePPO.load(args.model)
    env = STS2RunEnv(max_steps=2000)

    floors, rewards, lengths = [], [], []
    outcomes: Counter = Counter()

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed_base + ep)
        total, steps = 0.0, 0
        while True:
            action, _ = model.predict(
                obs, action_masks=env.action_masks(), deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            steps += 1
            if terminated or truncated:
                outcomes["won" if (terminated and env._mgr.player_won)
                         else ("timeout" if truncated else "died")] += 1
                break
        floors.append(info.get("floor", 0))
        rewards.append(total)
        lengths.append(steps)

    f = np.array(floors)
    print(f"model:    {args.model}")
    print(f"episodes: {args.episodes}  ({'sampled' if args.stochastic else 'deterministic'})")
    print(f"outcomes: {dict(outcomes)}")
    print(f"reward:   {np.mean(rewards):+.2f} +/- {np.std(rewards):.2f}")
    print(f"steps:    {np.mean(lengths):.0f}")
    print(f"\nfloors    mean {f.mean():.1f}   median {np.median(f):.0f}   "
          f"min {f.min()}   max {f.max()}")
    print(f"          p25 {np.percentile(f, 25):.0f}   p75 {np.percentile(f, 75):.0f}   "
          f"p90 {np.percentile(f, 90):.0f}")

    # Where runs actually end. A mean is a bad summary of a bimodal distribution,
    # and this is the shape that says whether the ceiling is one wall or a slope.
    print("\nfloor reached:")
    buckets = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 99)]
    for lo, hi in buckets:
        n = int(((f >= lo) & (f <= hi)).sum())
        if n:
            bar = "#" * max(1, round(40 * n / len(f)))
            print(f"  {lo:>2}-{hi:<2}  {n:>4}  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
