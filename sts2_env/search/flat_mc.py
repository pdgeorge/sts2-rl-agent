"""Flat Monte Carlo combat policy.

For each legal action: clone the combat, play the action, then finish the fight
with a cheap rollout policy several times over. Score each action by the mean
outcome and take the best. No tree, no learning, no weights -- which is the
point. A trained policy encodes the rules it was trained on and silently
misplays them after a patch; this one re-reads the rules from the simulator on
every decision, so a card whose damage changed is simulated correctly the same
day.

It also removes the archetype bias that makes a hand-written heuristic unusable
as a measuring instrument. A greedy-damage heuristic scores a block deck badly
because it cannot pilot one, not because the deck is bad. Search plays a block
deck like a block deck, because it simulates forward and sees the block pay off.
That property is what makes this usable as the pilot for deck evaluation, which
is the thing it is really for.

THE THREE THINGS THAT ARE EASY TO GET WRONG

1. Reseeding. ``deepcopy`` copies the RNG along with everything else, so every
   rollout of an action replays the *identical* shuffle and the identical enemy
   choices. You get N copies of one sample while believing you have N samples,
   and the variance you are averaging over never appears. Every clone is
   reseeded before it is used.

2. The rollout policy. A uniform-random rollout ends the turn about as often as
   it plays a card, so every action leads to roughly the same slow death and the
   scores are indistinguishable. ``END_TURN_PROB`` keeps rollouts playing cards
   while leaving the option to stop, which is what separates a good action from a
   bad one. It stays low rather than zero because ending a turn early is
   sometimes correct and the search should be able to see that.

3. Rejected actions. The mask is allowed to be wrong -- that is a documented bug
   class here. An action the engine refuses changes nothing at all, including
   ``turn_count``, so a rollout that keeps picking one runs forever. Rejected
   actions are scored ``REJECTED_SCORE`` (below any real outcome) so search
   routes around a mask bug instead of hanging on it, and the rollout has its own
   idle guard.

CLONING IS NOT ``deepcopy``

Every clone goes through :func:`~sts2_env.search.cloning.clone_combat`. Plain
``deepcopy`` produces a combat whose monsters still act on the *original*
combat's creatures, silently, because monster effects close over their creature
and ``deepcopy`` copies functions by reference. Search on top of that measures a
fight that is partly happening somewhere else.

COST

Measured on this repo: a clone is ~0.6 ms standalone and ~2.6 ms inside a run,
where the combat holds a reference to ``RunState`` and drags it along. That
reference is load-bearing -- relics reach through it -- so it is not detached.
Budget accordingly: cost is ``actions x rollouts`` clones per decision, so
``rollouts`` is the dial. This is fast enough to evaluate with and to teach from,
and too slow to sit inside a PPO training loop; distil it if you need that.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Protocol

import numpy as np

from sts2_env.core.constants import ACTION_END_TURN
from sts2_env.core.rng import Rng
from sts2_env.gym_env.action_space import (
    action_to_card_and_target,
    apply_action,
    get_action_mask,
)
from sts2_env.gym_env.reward import ENEMY_WEIGHT, HP_WEIGHT, potential
from sts2_env.search.cloning import clone_combat

if TYPE_CHECKING:
    # Importing CombatState eagerly makes `import sts2_env.search` circular
    # through the card registry. Every use here is an annotation.
    from sts2_env.core.combat import CombatState

logger = logging.getLogger(__name__)


DEFAULT_ROLLOUTS = 8
"""Rollouts per candidate action. The accuracy dial, and the cost dial."""

DEFAULT_MAX_ROLLOUT_TURNS = 30
"""Turns simulated past the decision before giving up and scoring the position.
Real combats run ~26 turns, so this usually reaches a real terminal state. When
it does not, the position is scored by potential rather than thrown away."""

DEFAULT_END_TURN_PROB = 0.15
"""How often a rollout ends the turn while it still has cards it could play."""

MAX_ROLLOUT_STEPS = 400
"""Absolute step ceiling per rollout. A backstop against an engine state that
neither terminates nor advances the turn counter, not a tuning parameter."""

MAX_IDLE_STEPS = 10
"""Consecutive rejected actions inside a rollout before abandoning it. Every
rejection is a mask bug; this stops one from becoming an infinite loop."""

REJECTED_SCORE = -1.0
"""Score for an action the engine refuses. Below a loss, so search never picks
one, and so a mask bug shows up as an avoided action rather than a hang."""

WIN_BASE = 1.0
"""Any win outranks any non-win. Remaining HP breaks ties between wins, which is
what makes the policy fight efficiently rather than merely survive."""


def _hp_fraction(combat: CombatState) -> float:
    player = combat.player
    if player.max_hp <= 0:
        return 0.0
    return max(0.0, min(1.0, player.current_hp / player.max_hp))


def score_state(combat: CombatState) -> float:
    """Score a combat position in ``[0, 2]``.

    Won: ``1 + hp_fraction`` -- every win beats every non-win, and a win at high
    HP beats a win at low HP. Lost: ``0``. Neither: the position mapped from
    :func:`~sts2_env.gym_env.reward.potential` onto ``[0, 1]``, so it can never
    outrank a real win or undercut a real loss. Reusing ``potential`` keeps the
    player-HP-weighted-double judgement in one place instead of inventing a
    second opinion about what a good position looks like.
    """
    if combat.is_over:
        return WIN_BASE + _hp_fraction(combat) if combat.player_won else 0.0

    span = HP_WEIGHT + ENEMY_WEIGHT
    if span <= 0:
        return 0.5
    normalised = (potential(combat) + ENEMY_WEIGHT) / span
    return float(max(0.0, min(1.0, normalised)))


def _reseed_rng(rng: Rng, seed: int) -> None:
    """Re-key an ``Rng`` in place.

    Re-runs the constructor rather than assigning the private fields it sets.
    Reaching in to set ``_seed``/``_rng``/``_counter`` by hand would silently
    stop covering any field ``Rng`` gains later, which is exactly the kind of
    quiet rot this project is trying to avoid.
    """
    rng.__init__(seed)  # noqa: PLC2801 -- deliberate: callers hold the reference


def reseed_combat(combat: CombatState, seed: int) -> None:
    """Give a cloned combat a fresh random future.

    Without this every rollout of the same action draws the same cards and the
    enemies make the same choices, so averaging over rollouts averages over one
    sample repeated N times. Reseeds the combat's own RNG and, when the combat
    is embedded in a run, the run-level streams it delegates to (shuffle, enemy
    targeting, card selection) -- ``CombatState._run_rng`` reads those from
    ``RunState``, so seeding ``combat.rng`` alone would leave the shuffle
    identical across rollouts.
    """
    _reseed_rng(combat.rng, seed)

    state = getattr(combat, "_primary_player_state", None)
    player_state = getattr(state, "player_state", None)
    run_state = getattr(player_state, "run_state", None)
    rng_set = getattr(run_state, "rng", None)
    if rng_set is None:
        return

    for offset, name in enumerate(sorted(vars(rng_set))):
        stream = getattr(rng_set, name, None)
        if isinstance(stream, Rng):
            _reseed_rng(stream, seed + 1 + offset)


class RolloutPolicy(Protocol):
    """Chooses an action during a rollout. Cheap by construction."""

    def __call__(
        self, combat: CombatState, valid_actions: np.ndarray, rng: random.Random
    ) -> int: ...


class RandomRolloutPolicy:
    """Random among legal actions, biased against ending the turn.

    Uniform-random would end the turn roughly as often as it plays a card, which
    makes every candidate action lead to the same slow death and flattens the
    scores the search depends on. Ending the turn stays reachable, at
    ``end_turn_prob``, because doing so is sometimes right.
    """

    def __init__(self, end_turn_prob: float = DEFAULT_END_TURN_PROB):
        self.end_turn_prob = end_turn_prob

    def __call__(
        self, combat: CombatState, valid_actions: np.ndarray, rng: random.Random
    ) -> int:
        others = [int(a) for a in valid_actions if a != ACTION_END_TURN]
        if not others:
            return ACTION_END_TURN
        can_end = ACTION_END_TURN in valid_actions
        if can_end and rng.random() < self.end_turn_prob:
            return ACTION_END_TURN
        return rng.choice(others)


class GreedyRolloutPolicy:
    """Highest base damage first, with a small chance of a random action.

    In flat Monte Carlo the rollout policy sets the quality of every value
    estimate: playouts that lose at random make every candidate action look
    equally doomed, and the search degenerates into picking among noise. A
    stronger rollout returns a sharper estimate for the same number of samples.

    ``explore_prob`` keeps some randomness so the estimate reflects a
    distribution of continuations rather than one deterministic line -- with no
    exploration every rollout of an action would be identical and the extra
    samples would buy nothing.
    """

    def __init__(self, explore_prob: float = 0.25, end_turn_prob: float = DEFAULT_END_TURN_PROB):
        self.explore_prob = explore_prob
        self.random_fallback = RandomRolloutPolicy(end_turn_prob=end_turn_prob)

    def __call__(
        self, combat: CombatState, valid_actions: np.ndarray, rng: random.Random
    ) -> int:
        if rng.random() < self.explore_prob:
            return self.random_fallback(combat, valid_actions, rng)

        best_action, best_damage = None, -1
        for action in valid_actions:
            if action == ACTION_END_TURN:
                continue
            hand_index, _ = action_to_card_and_target(int(action))
            if hand_index is None or hand_index >= len(combat.hand):
                continue
            damage = combat.hand[hand_index].base_damage or 0
            if damage > best_damage:
                best_damage, best_action = damage, int(action)

        if best_action is None:
            return self.random_fallback(combat, valid_actions, rng)
        return best_action


def rollout(
    combat: CombatState,
    rng: random.Random,
    *,
    policy: RolloutPolicy | None = None,
    max_turns: int = DEFAULT_MAX_ROLLOUT_TURNS,
) -> float:
    """Play ``combat`` forward ``max_turns`` rounds and score it. Mutates ``combat``.

    ``max_turns`` counts **complete rounds** from where the rollout started, not
    from the start of the fight, so the lookahead does not shrink as the fight
    goes on. One round is a player turn plus the enemy turn that answers it.

    The stopping point is deliberate. ``turn_count`` advances in
    ``_start_player_turn``, which runs only after the enemy turn has fully
    resolved, so a rollout that stops on this condition is always scored *after*
    the enemy has attacked. That is what makes block worth anything: the player's
    block is cleared by then, having already absorbed the hit, and the HP figure
    it protected is what gets scored. Stopping at the end of the player's own
    turn instead would score block as spent energy that bought nothing, and a
    defensive line would always look worse than an aggressive one.
    """
    policy = policy or RandomRolloutPolicy()
    turn_limit = combat.turn_count + max_turns
    idle = 0

    for _ in range(MAX_ROLLOUT_STEPS):
        # `>=`, so max_turns=1 simulates one round. With `>` it silently ran
        # max_turns+1, which matters when depth is the variable under test.
        if combat.is_over or combat.turn_count >= turn_limit:
            break

        mask = get_action_mask(combat)
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            break

        if apply_action(combat, policy(combat, valid, rng)):
            idle = 0
        else:
            idle += 1
            if idle >= MAX_IDLE_STEPS:
                logger.debug("Abandoning rollout after %d rejected actions", idle)
                break

    return score_state(combat)


class FlatMonteCarloPolicy:
    """Pick the action with the best mean rollout outcome.

    Deterministic given ``seed``: the same position always yields the same
    action, so an evaluation is reproducible and a regression is attributable.
    """

    def __init__(
        self,
        rollouts: int = DEFAULT_ROLLOUTS,
        *,
        max_rollout_turns: int = DEFAULT_MAX_ROLLOUT_TURNS,
        rollout_policy: RolloutPolicy | None = None,
        seed: int = 0,
    ):
        self.rollouts = max(1, rollouts)
        self.max_rollout_turns = max_rollout_turns
        self.rollout_policy = rollout_policy or RandomRolloutPolicy()
        self.seed = seed
        self._decisions = 0

    def reset(self) -> None:
        """Forget decision history, so a fresh fight replays identically."""
        self._decisions = 0

    def action_scores(self, combat: CombatState) -> dict[int, float]:
        """Mean rollout score per legal action. The policy's reasoning, exposed
        for debugging and for generating training targets to distil from."""
        mask = get_action_mask(combat)
        valid = np.flatnonzero(mask)
        decision = self._decisions
        return {
            int(action): self._evaluate(combat, int(action), decision)
            for action in valid
        }

    def act(self, combat: CombatState) -> int:
        """Choose an action for the current position."""
        mask = get_action_mask(combat)
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return ACTION_END_TURN
        if valid.size == 1:
            self._decisions += 1
            return int(valid[0])

        decision = self._decisions
        self._decisions += 1

        best_action = int(valid[0])
        best_score = float("-inf")
        for action in valid:
            score = self._evaluate(combat, int(action), decision)
            # Strict >: ties go to the lower action index, so the choice is
            # stable rather than dependent on iteration order.
            if score > best_score:
                best_score = score
                best_action = int(action)
        return best_action

    def _evaluate(self, combat: CombatState, action: int, decision: int) -> float:
        total = 0.0
        for k in range(self.rollouts):
            # Distinct per (decision, action, rollout) so no two rollouts share
            # a random future, and reproducible so a replay matches.
            rollout_seed = (
                self.seed * 1_000_003 + decision * 10_007 + action * 101 + k
            ) % 2_147_483_647

            # clone_combat, never plain deepcopy: a deep-copied monster still
            # acts on the original combat's creatures. See search/cloning.py.
            clone = clone_combat(combat)
            reseed_combat(clone, rollout_seed)

            if not apply_action(clone, action):
                # The mask advertised something the engine refuses. Nothing was
                # rolled, so the remaining rollouts would say the same thing.
                return REJECTED_SCORE

            if clone.is_over:
                # Settled by the action itself; N rollouts would agree N times.
                return score_state(clone)

            total += rollout(
                clone,
                random.Random(rollout_seed),
                policy=self.rollout_policy,
                max_turns=self.max_rollout_turns,
            )
        return total / self.rollouts
