#!/usr/bin/env python3
"""Watch a meta-policy model play STS2 in the simulator with full logging.

Shows every non-combat decision, deck features, and rewards so you can see
what the policy is actually learning. Combat is fast-forwarded by the frozen
combat policy.

Usage:
    source .venv/bin/activate
    python scripts/watch_model.py output/meta_ppo_v4/final_model.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="Path to a meta-policy .zip")
    ap.add_argument("--episodes", type=int, default=3,
                    help="How many runs to watch")
    ap.add_argument("--combat-policy", type=str, default=None,
                    help="Frozen combat policy .zip (default: heuristic)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
    except ImportError:
        print("Requires sb3-contrib. pip install 'sts2-rl-agent[train]'")
        return 1

    from sts2_env.gym_env.hierarchical_env import (
        HierarchicalRunEnv, HeuristicCombatSolver, FrozenRLCombatSolver,
    )
    from sts2_env.gym_env.deck_features import encode_deck_features

    # Load combat solver
    if args.combat_policy:
        combat_solver = FrozenRLCombatSolver(args.combat_policy)
        print(f"Combat solver: FrozenRL {args.combat_policy}")
    else:
        combat_solver = HeuristicCombatSolver()
        print("Combat solver: Heuristic")

    # Load meta-policy
    model = MaskablePPO.load(args.model)
    print(f"Meta-policy:   {args.model}")
    print()

    def mask_fn(env):
        return env.action_masks()

    for ep in range(args.episodes):
        env = ActionMasker(
            HierarchicalRunEnv(max_steps=2000, combat_solver=combat_solver),
            mask_fn,
        )
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        step = 0
        total_reward = 0.0

        print(f"=== Episode {ep + 1} (seed {args.seed + ep}) ===")

        while not done:
            phase = info["phase"]
            floor = info.get("floor", 0)
            hp = info.get("hp", 0)
            max_hp = info.get("max_hp", 0)
            gold = info.get("gold", 0)
            deck_size = info.get("deck_size", 0)

            masks = env.action_masks()
            action, _ = model.predict(
                obs, action_masks=masks, deterministic=True
            )

            # Log interesting phases
            if phase in ("MAP_CHOICE", "CARD_REWARD", "SHOP", "REST_SITE",
                         "EVENT", "BOSS_RELIC"):
                valid = np.where(masks == 1)[0]
                print(f"\n  Step {step:>3} | {phase:12s} | floor {floor:>2} | "
                      f"HP {hp:>2}/{max_hp} | gold {gold} | deck {deck_size}")
                print(f"    Action chosen: {action}")
                print(f"    Valid actions: {valid[:10].tolist()}...")

                # Show deck features
                inner = env.env if hasattr(env, "env") else env
                mgr = getattr(inner, "_mgr", None)
                if mgr is not None:
                    deck = mgr.run_state.player.deck
                    feats = encode_deck_features(deck)
                    top = np.argsort(feats)[-5:][::-1]
                    print(f"    Top deck features:", end="")
                    for idx in top:
                        if feats[idx] > 0.01:
                            print(f" [{idx}]={feats[idx]:.2f}", end="")
                    print()

            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
            step += 1

            # Print combat resolution
            inner = env.env if hasattr(env, "env") else env
            mgr = getattr(inner, "_mgr", None)
            if phase == "COMBAT" and mgr is not None and mgr.phase != "COMBAT":
                print(f"  Combat resolved → floor {info.get('floor', 0)} "
                      f"HP {info.get('hp', 0)}/{info.get('max_hp', 0)}")

        print(f"\n  >>> Result: floor {info.get('floor', 0)} | "
              f"reward {total_reward:.2f} | "
              f"{'WON' if terminated and info.get('phase') == 'VICTORY' else 'DIED'}\n")
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
