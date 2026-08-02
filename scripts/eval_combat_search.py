"""Compare combat policies on a fixed seed set.

    python scripts/eval_combat_search.py --episodes 40 --rollouts 8

Every policy fights the *same* encounters from the *same* seeds, so the
comparison is paired and the differences are not seed luck. Win rates are
reported with a standard error, because selecting on an unqualified point
estimate is how `run_ppo_v4` came to be recorded as an improvement when it was
the maximum of 41 noisy draws (see docs/MODELS.md).

Read the HP column as well as the win rate. A policy that wins at 12 HP has not
won a run -- it has moved the loss to the next fight.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

# Importing sts2_env.core.combat first triggers a circular import through the
# card registry, so CombatState is annotation-only here.
from sts2_env.core.constants import ACTION_END_TURN
from sts2_env.gym_env.action_space import (
    action_to_card_and_target,
    apply_action,
    get_action_mask,
)
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.search.flat_mc import (
    FlatMonteCarloPolicy,
    GreedyRolloutPolicy,
    RandomRolloutPolicy,
)

if TYPE_CHECKING:
    from sts2_env.core.combat import CombatState

MAX_STEPS_PER_COMBAT = 600


@dataclass
class Result:
    name: str
    wins: int
    episodes: int
    hp_fraction_on_win: float
    mean_turns: float
    seconds_per_decision: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.episodes if self.episodes else 0.0

    @property
    def sem(self) -> float:
        """Standard error of the win rate, so 'better' can be qualified."""
        p, n = self.win_rate, self.episodes
        return math.sqrt(p * (1.0 - p) / n) if n else 0.0


def random_policy(seed: int):
    rng = random.Random(seed)

    def choose(combat: CombatState) -> int:
        valid = np.flatnonzero(get_action_mask(combat))
        return int(rng.choice(list(valid))) if valid.size else ACTION_END_TURN

    return choose


def greedy_policy(_seed: int):
    """Highest base damage first. The archetype-biased baseline.

    Worth keeping in the table precisely because it is biased: it is what a
    deck evaluator would be using if it used a hand-written heuristic, and the
    gap between it and search is the size of the measurement error that choice
    would introduce.
    """

    def choose(combat: CombatState) -> int:
        valid = np.flatnonzero(get_action_mask(combat))
        if valid.size == 0:
            return ACTION_END_TURN

        best_action, best_damage = None, -1
        for action in valid:
            if action == ACTION_END_TURN:
                continue
            hand_idx, _ = action_to_card_and_target(int(action))
            if hand_idx is None or hand_idx >= len(combat.hand):
                continue
            damage = combat.hand[hand_idx].base_damage or 0
            if damage > best_damage:
                best_damage, best_action = damage, int(action)

        if best_action is not None:
            return best_action
        return ACTION_END_TURN if ACTION_END_TURN in valid else int(valid[0])

    return choose


def search_policy_factory(rollouts: int, max_rollout_turns: int, rollout_policy: str):
    def build(seed: int):
        policy = FlatMonteCarloPolicy(
            rollouts=rollouts,
            max_rollout_turns=max_rollout_turns,
            rollout_policy=(
                GreedyRolloutPolicy() if rollout_policy == "greedy" else RandomRolloutPolicy()
            ),
            seed=seed,
        )
        return policy.act

    return build


def evaluate(name: str, build_policy, seeds: list[int]) -> Result:
    wins = 0
    hp_fractions: list[float] = []
    turns: list[int] = []
    decisions = 0
    elapsed = 0.0

    for seed in seeds:
        env = STS2CombatEnv()
        env.reset(seed=seed)
        combat = env.combat
        assert combat is not None
        choose = build_policy(seed)

        for _ in range(MAX_STEPS_PER_COMBAT):
            if combat.is_over:
                break
            if np.flatnonzero(get_action_mask(combat)).size == 0:
                break

            started = time.perf_counter()
            action = choose(combat)
            elapsed += time.perf_counter() - started
            decisions += 1

            if not apply_action(combat, action):
                # The mask advertised something the engine refuses. Abandoning
                # is correct: a rejected action does not advance the turn
                # counter, so continuing would spin without terminating.
                break

        turns.append(combat.turn_count)
        if combat.player_won:
            wins += 1
            hp_fractions.append(
                combat.player.current_hp / combat.player.max_hp
                if combat.player.max_hp
                else 0.0
            )

    return Result(
        name=name,
        wins=wins,
        episodes=len(seeds),
        hp_fraction_on_win=float(np.mean(hp_fractions)) if hp_fractions else 0.0,
        mean_turns=float(np.mean(turns)) if turns else 0.0,
        seconds_per_decision=elapsed / decisions if decisions else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--max-rollout-turns", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--rollout-policy",
        choices=["random", "greedy"],
        default="random",
        help="How rollouts are played out. In flat Monte Carlo this sets the "
             "quality of every value estimate.",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Baselines only. Search is by far the slowest entry.",
    )
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))

    entries = [("random", random_policy), ("greedy-damage", greedy_policy)]
    if not args.skip_search:
        entries.append(
            (
                f"flat-mc r={args.rollouts} {args.rollout_policy}",
                search_policy_factory(
                    args.rollouts, args.max_rollout_turns, args.rollout_policy
                ),
            )
        )

    print(f"Act 1 encounters, Ironclad starter deck, {len(seeds)} fixed seeds\n")
    header = f"{'policy':<20} {'win rate':>18} {'HP% on win':>11} {'turns':>7} {'s/decision':>11}"
    print(header)
    print("-" * len(header))

    results = []
    for name, build in entries:
        result = evaluate(name, build, seeds)
        results.append(result)
        rate = f"{result.win_rate:6.1%} +/- {result.sem:.1%}"
        print(
            f"{result.name:<20} {rate:>18} {result.hp_fraction_on_win:>10.1%} "
            f"{result.mean_turns:>7.1f} {result.seconds_per_decision:>11.4f}"
        )

    if len(results) > 1:
        best, baseline = results[-1], results[0]
        gap = best.win_rate - baseline.win_rate
        combined = math.sqrt(best.sem**2 + baseline.sem**2)
        sigma = gap / combined if combined else float("inf")
        print(
            f"\n{best.name} vs {baseline.name}: {gap:+.1%} "
            f"({sigma:.1f} sem){'' if sigma >= 2 else '  -- under 2 sem, not yet a result'}"
        )


if __name__ == "__main__":
    main()
