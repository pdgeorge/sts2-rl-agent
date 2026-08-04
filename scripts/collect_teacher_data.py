"""Play runs with the measured agent and record every decision as training data.

    python scripts/collect_teacher_data.py --runs 200 --out output/teacher.npz

WHY THIS IS THE ML PATH, AND WHY PPO WAS NOT

Three PPO runs -- 6M, 20M and 40M steps -- returned null results, and the 40M
model plays level with a greedy heuristic. That is not evidence that learning
cannot work here. It is evidence about the REWARD:

    adding one curse to a ten-card deck moves the deck's measured score by
    0.005 over 240 fights, and the signal arrives hundreds of steps later
    mixed with every other decision in the run

No policy gradient recovers a signal that small from a terminal reward. But
imitation learning never asks it to. It needs a teacher whose decisions are
good and a target for every single decision, and both now exist:

    card rewards   measured by playing the deck with each candidate
    rest / smith   measured against the act 1 boss and the survival curves
    combat         value_pilot, which beats the trained model 96% to 3% on the
                   act 1 boss with a scaling deck
    map routing    rest-site-aware, worth ~1 upgrade = 30% -> 66% boss

So the student learns from a dense, per-decision target instead of a terminal
one, which is a completely different learning problem from the one that failed.

AND THE STUDENT SHOULD BEAT THE TEACHER ON THE ONE AXIS THAT MATTERS LIVE

The teacher is slow. A measured rest decision takes 12-24 seconds and has
already overrun the mod's 30-second agent timeout, which ends the run outright.
A network answers in microseconds. Distillation is the only route to an agent
that is both good and fast enough to play the real game.

WHAT IS RECORDED

One row per decision: the run-model observation, the legal action mask, and the
action the teacher chose. Observations come from the env itself rather than
being rebuilt here, so the student trains on exactly the encoding it will see.
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np


def _teacher_action(env, manager, layout, pilot, seeds, rng):
    """The measured agent's action, as an index into the run action space.

    Returns None when nothing measured applies, and the caller falls back to a
    legal action -- shops, events and treasure are still heuristic, and a
    student that learns those from random data would learn noise.
    """
    from sts2_env.evaluation.card_choice import best_index, rank_candidates
    from sts2_env.evaluation.rest_choice import rank_rest_options
    from sts2_env.gym_env.action_space import action_to_card_and_target, get_action_mask
    from sts2_env.run.run_manager import RunManager

    phase = manager.phase
    actions = manager.get_available_actions()

    if phase == RunManager.PHASE_COMBAT:
        combat = manager.get_combat_state()
        if combat is None or combat.is_over or not get_action_mask(combat).any():
            return None
        sim_action = int(pilot(combat))
        hand_index, target_index = action_to_card_and_target(sim_action)
        if hand_index is None:
            return layout.combat_start          # end turn
        # The run action space lays combat out the same way the combat space
        # does, offset by combat_start.
        return layout.combat_start + sim_action

    if phase == RunManager.PHASE_MAP_CHOICE:
        from sts2_env.bridge.agent_runner import (
            REST_HP_RATIO_THRESHOLD,
            ROOM_PRIORITY_HEALTHY,
            ROOM_PRIORITY_LOW_HP,
        )

        moves = [a for a in actions if a.get("action") == "move"]
        if not moves:
            return None
        player = manager.run_state.player
        ratio = float(getattr(player, "current_hp", 0)) / max(
            1.0, float(getattr(player, "max_hp", 1))
        )
        priority = (ROOM_PRIORITY_LOW_HP if ratio < REST_HP_RATIO_THRESHOLD
                    else ROOM_PRIORITY_HEALTHY)
        for room in priority:
            for i, move in enumerate(moves):
                kind = str(move.get("point_type", "")).lower().replace("_", "")
                if kind == room:
                    return layout.map_start + min(i, layout.map_size - 1)
        return layout.map_start

    if phase == RunManager.PHASE_CARD_REWARD:
        picks = [a for a in actions if a.get("action") == "pick_card"]
        if not picks:
            return None
        from sts2_env.cards.factory import create_card
        from sts2_env.core.enums import CardId

        offered = []
        for action in picks[:3]:
            card = action.get("card")
            if card is None:
                name = action.get("card_id")
                try:
                    card = create_card(CardId[name]) if name else None
                except Exception:  # noqa: BLE001
                    card = None
            if card is None:
                return None
            offered.append(card)

        deck = list(manager.run_state.player.deck)
        ranked = rank_candidates(
            deck, offered, pilot, floor=manager.run_state.total_floor, seeds=seeds,
        )
        choice = best_index(ranked, offered)
        if choice is None:
            return layout.card_reward_start + 3          # skip
        return layout.card_reward_start + choice

    if phase == RunManager.PHASE_REST_SITE:
        rest = next((a for a in actions if a.get("option_id") == "HEAL"), None)
        smith = next((a for a in actions if a.get("option_id") == "SMITH"), None)
        if rest is None or smith is None:
            return None
        deck = list(manager.run_state.player.deck)
        player = manager.run_state.player
        ranked = rank_rest_options(
            deck, [c for c in deck if not getattr(c, "upgraded", False)], pilot,
            current_hp=int(getattr(player, "current_hp", 1)),
            max_hp=int(getattr(player, "max_hp", 80)),
            floor=manager.run_state.total_floor, seeds=seeds,
        )
        wants_smith = bool(ranked) and ranked[0].kind == "upgrade"
        index = actions.index(smith if wants_smith else rest)
        return layout.rest_start + min(index, layout.rest_size - 1)

    return None


def collect(runs: int, seed_base: int, decision_seeds: tuple[int, ...],
            max_steps: int = 3000):
    import random

    from sts2_env.evaluation.pilots import value_pilot
    from sts2_env.gym_env.run_env import _LAYOUT, STS2RunEnv

    observations, masks, choices, phases = [], [], [], []
    floors = []
    env = STS2RunEnv(max_steps=max_steps)

    for run in range(runs):
        obs, _info = env.reset(seed=seed_base + run)
        rng = random.Random(seed_base + run)
        for _ in range(max_steps):
            manager = env._mgr
            if manager is None or manager.is_over:
                break
            mask = env.action_masks()
            if not mask.any():
                break

            action = _teacher_action(env, manager, _LAYOUT, value_pilot,
                                     decision_seeds, rng)
            teachable = action is not None and 0 <= action < len(mask) and mask[action]
            if not teachable:
                # Not a decision the teacher owns -- step it legally but do NOT
                # record it. A student trained on random shop and event choices
                # learns to imitate noise, which is worse than not learning them.
                action = int(rng.choice(np.flatnonzero(mask)))
            else:
                observations.append(np.asarray(obs, dtype=np.float32))
                masks.append(np.asarray(mask, dtype=np.int8))
                choices.append(int(action))
                phases.append(manager.phase)

            obs, _reward, terminated, truncated, _info = env.step(int(action))
            if terminated or truncated:
                break
        floors.append(env._mgr.run_state.total_floor if env._mgr else 0)
        if (run + 1) % 10 == 0:
            print(f"  {run + 1}/{runs} runs, {len(observations)} decisions recorded",
                  flush=True)

    return (np.array(observations), np.array(masks), np.array(choices),
            phases, floors)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--out", default="output/teacher_data.npz")
    ap.add_argument("--seed-base", type=int, default=500000)
    ap.add_argument("--decision-seeds", type=int, default=3)
    args = ap.parse_args()

    seeds = tuple(range(args.decision_seeds))
    obs, masks, choices, phases, floors = collect(args.runs, args.seed_base, seeds)

    if len(obs) == 0:
        print("no decisions recorded")
        return 1

    np.savez_compressed(args.out, observations=obs, masks=masks, actions=choices)
    print(f"\nwrote {args.out}")
    print(f"  {len(obs)} decisions from {args.runs} runs "
          f"({len(obs) / args.runs:.1f} per run)")
    print(f"  observation {obs.shape[1]} dims, {masks.shape[1]} actions")
    print(f"  by phase: {dict(Counter(phases).most_common())}")
    print(f"  teacher mean floor {np.mean(floors):.1f} "
          f"(this is the bar the student has to match)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
