"""Train a policy to imitate the measured agent, then play runs with it.

    python scripts/collect_teacher_data.py --runs 200 --out output/teacher.npz
    python scripts/train_distill.py output/teacher.npz --epochs 30
    python scripts/train_distill.py output/teacher.npz --eval-only output/student.pt

WHY IMITATION AND NOT MORE PPO

PPO was tried three times -- 6M, 20M and 40M steps -- and returned a null result
every time; the 40M model plays level with a greedy heuristic. The cause is
measured and it is the REWARD, not the learning:

    one curse added to a ten-card deck moves the deck's score by 0.005 over
    240 fights, and that signal arrives hundreds of steps later mixed with
    every other decision in the run

Imitation never asks a gradient to recover that. Every decision carries its own
target, supplied by a teacher whose choices are measured rather than guessed.
Same network, same observation, a completely different learning problem.

WHAT WOULD MAKE THIS A FAILURE

Validation accuracy that is high while runs are no better. That would mean the
student learned the teacher's habits rather than its judgement, and the honest
response is to record it in MODELS.md as a null result with the accuracy
attached -- exactly as run_ppo_v4 and run_ppo_v2_6m are recorded.

The gate is FLOORS on held-out seeds, not accuracy. Accuracy is a training
diagnostic and nothing more.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


class Student:
    """A small masked-softmax policy over the run action space."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 512):
        import torch.nn as nn

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.obs_dim = obs_dim
        self.n_actions = n_actions

    def logits(self, obs, mask):
        import torch

        out = self.net(obs)
        # Illegal actions are removed, not merely discouraged. A policy that can
        # emit an illegal action live does not degrade gracefully -- the game
        # ignores it, the state comes back unchanged, and a deterministic agent
        # sends it again forever.
        return out.masked_fill(mask == 0, float("-inf"))

    def act(self, obs_row, mask_row) -> int:
        import torch

        with torch.no_grad():
            obs = torch.as_tensor(obs_row, dtype=torch.float32).unsqueeze(0)
            mask = torch.as_tensor(mask_row, dtype=torch.int8).unsqueeze(0)
            return int(self.logits(obs, mask).argmax(dim=1).item())


def train(path: str, epochs: int, batch: int, lr: float, val_fraction: float):
    import torch
    import torch.nn as nn

    data = np.load(path)
    obs, masks, actions = data["observations"], data["masks"], data["actions"]
    print(f"{len(obs)} decisions, {obs.shape[1]} obs dims, {masks.shape[1]} actions")

    # Split by INDEX ORDER, which keeps whole runs together -- decisions inside
    # one run are heavily correlated, and a random split leaks the answer.
    cut = int(len(obs) * (1 - val_fraction))
    tr = slice(0, cut)
    va = slice(cut, len(obs))
    print(f"train {cut}, validate {len(obs) - cut} (split by run order, not shuffled)")

    student = Student(obs.shape[1], masks.shape[1])
    optimiser = torch.optim.Adam(student.net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    to_t = lambda a, d: torch.as_tensor(a, dtype=d)
    obs_tr, mask_tr, act_tr = (to_t(obs[tr], torch.float32),
                               to_t(masks[tr], torch.int8), to_t(actions[tr], torch.long))
    obs_va, mask_va, act_va = (to_t(obs[va], torch.float32),
                               to_t(masks[va], torch.int8), to_t(actions[va], torch.long))

    best = 0.0
    for epoch in range(epochs):
        student.net.train()
        order = torch.randperm(len(obs_tr))
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            optimiser.zero_grad()
            loss = loss_fn(student.logits(obs_tr[idx], mask_tr[idx]), act_tr[idx])
            loss.backward()
            optimiser.step()
            total += loss.detach().item() * len(idx)

        student.net.eval()
        with torch.no_grad():
            pred = student.logits(obs_va, mask_va).argmax(dim=1)
            accuracy = float((pred == act_va).float().mean()) if len(act_va) else 0.0
        best = max(best, accuracy)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:>3}  loss {total / max(1, len(order)):.4f}  "
                  f"val agreement {accuracy:.1%}")
    print(f"\nbest validation agreement with the teacher: {best:.1%}")
    return student


def play(student, runs: int, seed_base: int, max_steps: int = 3000):
    """Play runs with the STUDENT alone. No simulation, no battery."""
    import numpy as np

    from sts2_env.gym_env.run_env import STS2RunEnv

    env = STS2RunEnv(max_steps=max_steps)
    floors = []
    for run in range(runs):
        obs, _info = env.reset(seed=seed_base + run)
        for _ in range(max_steps):
            mask = env.action_masks()
            if not mask.any():
                break
            action = student.act(obs, mask)
            obs, _r, terminated, truncated, _i = env.step(action)
            if terminated or truncated:
                break
        floors.append(env._mgr.run_state.total_floor if env._mgr else 0)
    return floors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--save", default="output/student.pt")
    ap.add_argument("--play", type=int, default=30,
                    help="runs to play with the student afterwards (held-out seeds)")
    ap.add_argument("--play-seed-base", type=int, default=900000)
    args = ap.parse_args()

    import torch

    student = train(args.data, args.epochs, args.batch, args.lr, args.val_fraction)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.net.state_dict(), args.save)
    print(f"saved {args.save}")

    if args.play:
        import statistics

        floors = play(student, args.play, args.play_seed_base)
        sem = statistics.stdev(floors) / len(floors) ** 0.5 if len(floors) > 1 else 0.0
        print(f"\nSTUDENT ALONE, {args.play} held-out seeds:")
        print(f"  mean floor {statistics.mean(floors):.1f} +/- {sem:.1f} sem   "
              f"median {statistics.median(floors)}   max {max(floors)}")
        print("\nThe gate is this number against the teacher's, not the agreement "
              "percentage. A student that matches the teacher's habits and not its "
              "floors is a null result and should be recorded as one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
