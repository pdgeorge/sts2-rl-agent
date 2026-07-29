"""Reward calculation.

KNOWN BUG, not yet fixed: this reward is exploitable, and MaskablePPO finds the
exploit within 10,000 timesteps.

Combat truncates at `max_turns = 200` (gym_env/combat_env.py). Truncation is not
`combat.is_over`, so it scores 0 -- while losing scores -1. A policy that cannot
reliably win therefore does better by never ending combat than by fighting and
losing it. Measured: random play truncated 6 times in 1,000 episodes at a mean
episode length of 25 steps; a policy 10k steps into training evaluated at a mean
episode length of 1,813, standard deviation 4,942. It had learned to stall.

Two fixes, both small:

  1. Truncation must score clearly negative. Stalling has to be worse than losing,
     not better than it.
  2. Shape on HP delta -- credit for damage dealt and HP kept. This matters beyond
     play quality. The point of training is a value function for Cyra's gap gate,
     and `gap = value(model_pick) - value(heart_pick)` needs that function to
     separate a good line from a mediocre one. A reward that only moves at
     terminal states gives a value estimate that is nearly flat through the middle
     of a combat, which is where almost every decision happens. `prev_hp` is
     already threaded in for this and is currently unused.
"""

from __future__ import annotations

from sts2_env.core.combat import CombatState


def compute_reward(combat: CombatState, prev_hp: int) -> float:
    """Compute step reward.

    Sparse reward: +1 for win, -1 for loss, 0 otherwise.
    """
    if combat.is_over:
        return 1.0 if combat.player_won else -1.0
    return 0.0
