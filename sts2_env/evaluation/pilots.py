"""Who plays the test fights. The pilot defines what the battery measures.

A greedy-damage pilot scores a block deck as worthless -- not because it is, but
because a pilot that never blocks never converts block into survival. That bias
is the ceiling on any deck ranking built from it, so the pilot is an explicit
argument everywhere rather than a default buried in a function.

greedy is here because it is fast and, measured on this repo, not bad: 73.3% +/-
8.1% on act 1 encounters, statistically level with a trained PPO model. It is the
right choice for bulk work and the wrong choice for the final word.
"""

from __future__ import annotations

import numpy as np

from sts2_env.core.constants import ACTION_END_TURN
from sts2_env.gym_env.action_space import action_to_card_and_target, get_action_mask


def greedy_pilot(combat) -> int:
    """Highest base damage first, end turn when nothing damaging is playable."""
    valid = np.flatnonzero(get_action_mask(combat))
    if valid.size == 0:
        return ACTION_END_TURN

    best_action, best_damage = None, -1
    for action in valid:
        if action == ACTION_END_TURN:
            continue
        hand_index, _ = action_to_card_and_target(int(action))
        if hand_index is None or hand_index >= len(combat.hand):
            continue
        damage = combat.hand[hand_index].base_damage or 0
        if damage > best_damage:
            best_damage, best_action = damage, int(action)

    if best_action is not None:
        return best_action
    return ACTION_END_TURN if ACTION_END_TURN in valid else int(valid[0])


def search_pilot(rollouts: int = 6, max_rollout_turns: int = 8, seed: int = 0):
    """Flat Monte Carlo. Slower, and not archetype-biased the way greedy is.

    Not established as stronger than greedy -- measured at 63-67% against greedy's
    73.3%, a difference well inside noise at 30 seeds. Offered for spot-checking
    decks where the two disagree, which is where greedy's bias would show.
    """
    from sts2_env.search.flat_mc import FlatMonteCarloPolicy

    policy = FlatMonteCarloPolicy(
        rollouts=rollouts, max_rollout_turns=max_rollout_turns, seed=seed
    )
    return policy.act
