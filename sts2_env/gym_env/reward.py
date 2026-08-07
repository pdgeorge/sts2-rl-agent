"""Reward calculation.

The previous reward was +1 win, -1 loss, 0 otherwise. Two things were wrong with
it, one certain and one that turned out to be weaker than it first looked.

Certain: combat truncates at 200 turns, truncation is not `is_over`, so it scored
0 -- better than the -1 for losing. A policy that could not reliably win was
therefore paid more for never ending combat than for fighting and losing. That is
readable from the code and did not need measuring.

Weaker: the training run's evaluation episode lengths climbed from 1,814 to 8,351,
which looked like the exploit being learned. It was not. Loading that model and
running it on 20 fresh seeds gives a mean of 89 steps over 26 turns, deterministic
or sampled. The eval mean was dominated by a heavy tail -- a couple of runaway
episodes out of twenty -- and 26 turns is a slow fight, not a stall. The tail was
real enough to burn 2.17M env steps in evaluation against 250k of training, but it
was not the policy's typical behaviour.

What was genuinely wrong with the old reward is the flat learning curve: 25
evaluations across 250k steps with no trend at all, oscillating around 0.1. A
signal that only appears at terminal states gives almost nothing to learn from.
That is the reason for the shaping below, and it was always the stronger argument.

WHAT THIS REWARD IS FOR

Not play quality. Its job is to produce a *value function with resolution*, for
Cyra's gap gate: `gap = value(model_pick) - value(heart_pick)` decides whether a
choice is close enough to be worth handing to her slow verbal side. A reward that
only moves at terminal states gives a value estimate that is nearly flat through
the middle of a combat, which is where almost every decision happens -- the gap
would be noise, and the gate would be built on a sensor that cannot see the
difference it exists to detect.

DESIGN

Three parts.

1. Terminal. Win +1, loss -1, truncation -1. Truncation equals a loss rather than
   exceeding it: making it worse would create an incentive to die quickly rather
   than play on, which is a different pathology, not a fix.

2. Potential-based shaping, F = gamma * phi(s') - phi(s), with
   phi = HP_WEIGHT * player_hp_frac - ENEMY_WEIGHT * enemy_hp_frac.

   Potential-based specifically (Ng, Harada & Russell 1999): this form provably
   does not change which policy is optimal. It adds gradient without inventing a
   new game to win, so the agent cannot learn to farm the shaping instead of
   winning. It also telescopes -- total shaping over an episode collapses to
   gamma^T * phi(s_T) - phi(s_0) -- so a long episode cannot accumulate shaping
   reward, which matters given the failure mode being fixed.

   "Damage capped at killing" falls out of this rather than needing a rule.
   Creature.lose_hp clamps at `max(0, hp - amount)`, so a 50-damage hit on a
   5 HP enemy moves phi by 5 HP worth. Overkill is free of charge and free of
   reward. Likewise there is no reward for hitting an already-dead enemy.

3. A small per-turn cost. This is what actually closes the truncation hole, and
   the reason the terminal penalty alone cannot: at gamma = 0.99 with a few steps
   per turn, a -1 arriving at turn 200 is discounted into invisibility. A cost
   felt now is what makes indefinite stalling unprofitable now.

   The turn cost is progressive, and flat pricing could not serve both goals at
   once. Stalling to draw the card you need, to build block before a big hit, or
   to stack a power is real Slay the Spire and has to stay affordable -- but at a
   rate cheap enough for that, a 200-turn fight cost exactly 1.0, the same as a
   win, so stalling to the cap and winning netted zero.

   So the first COMBAT_FREE_TURNS are cheap and everything after is ten times
   dearer. A 10-turn fight pays 0.05; 20 turns pays 0.46, which is worse than
   losing 40% of your HP; 200 turns pays 9.5. The intent is unchanged -- not
   "stalling is bad" but "stalling has to be going somewhere" -- only the pricing
   now says so at the length where it stops being true.

   An earlier version of this note said observed combats run ~26 turns. The
   benchmark reports 10-11, and 26 looks like a mean inflated by exactly the
   stalls this is meant to price -- the same trap as a mean floor of 9.6 across a
   bimodal distribution.
"""

from __future__ import annotations

from sts2_env.core.combat import CombatState

from sts2_env.gym_env.reward_config import (
    COMBAT_ENEMY_WEIGHT,
    COMBAT_HP_WEIGHT,
    COMBAT_LOSS_REWARD,
    COMBAT_SHAPING_SCALE,
    COMBAT_TRUNCATION_REWARD,
    COMBAT_FREE_TURNS,
    COMBAT_STALL_TURN_COST,
    COMBAT_TURN_COST,
    COMBAT_WIN_REWARD,
)

# Module-level aliases, because tests and callers already import these names.
WIN_REWARD = COMBAT_WIN_REWARD
LOSS_REWARD = COMBAT_LOSS_REWARD
TRUNCATION_REWARD = COMBAT_TRUNCATION_REWARD
HP_WEIGHT = COMBAT_HP_WEIGHT
ENEMY_WEIGHT = COMBAT_ENEMY_WEIGHT
TURN_COST = COMBAT_TURN_COST
FREE_TURNS = COMBAT_FREE_TURNS
STALL_TURN_COST = COMBAT_STALL_TURN_COST
SHAPING_SCALE = COMBAT_SHAPING_SCALE


def turn_penalty(turns_elapsed: int) -> float:
    """What a fight of this length costs. Cheap, then steep.

    Flat pricing could not serve both goals at once: low enough that setup turns
    stay affordable meant a 200-turn stall cost the same as one win, and high
    enough to punish stalling made building block before a boss hit unaffordable.
    Charging the two regimes differently gets both -- see COMBAT_STALL_TURN_COST.
    """
    turns = max(0, int(turns_elapsed))
    cheap = min(turns, FREE_TURNS) * TURN_COST
    return cheap + max(0, turns - FREE_TURNS) * STALL_TURN_COST


def potential(combat: CombatState) -> float:
    """phi(s): how well the combat is going, in [-ENEMY_WEIGHT, HP_WEIGHT]."""
    player = combat.player
    player_frac = player.current_hp / player.max_hp if player.max_hp else 0.0

    enemy_current = sum(e.current_hp for e in combat.enemies)
    enemy_max = sum(e.max_hp for e in combat.enemies)
    enemy_frac = enemy_current / enemy_max if enemy_max else 0.0

    return HP_WEIGHT * player_frac - ENEMY_WEIGHT * enemy_frac


def compute_reward(
    combat: CombatState,
    prev_potential: float,
    *,
    truncated: bool = False,
    turns_elapsed: int = 0,
    gamma: float = 0.99,
) -> float:
    """Step reward.

    `prev_potential` is phi(s) from before the action -- the caller holds it,
    because recomputing it from a mutated CombatState is not possible.
    `turns_elapsed` is how many turns passed during this step, normally 0 or 1.
    """
    if combat.is_over:
        terminal = WIN_REWARD if combat.player_won else LOSS_REWARD
        # No shaping term on a terminal transition: phi of a finished combat is
        # not a prediction of anything, and adding it would let a win that ended
        # at low HP score below a loss.
        return terminal - turn_penalty(turns_elapsed)

    if truncated:
        return TRUNCATION_REWARD - turn_penalty(turns_elapsed)

    shaping = gamma * potential(combat) - prev_potential
    return SHAPING_SCALE * shaping - turn_penalty(turns_elapsed)
