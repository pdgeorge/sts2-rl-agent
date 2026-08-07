"""Does archetype-aware card picking build better decks? Simulated A/B.

    python scripts/ab_archetype_picking.py --runs 200

Step 10 of the Phase 5 build plan, and the question the previous nine exist to
answer. Both arms walk full runs in the simulator and differ in exactly one
place: which card they take when offered a reward.

    control     rank_cards(offered, deck)              quality only
    archetype   rank_cards(offered, deck, direction)   quality x fit

Everything else is held identical -- same seeds, same combat policy, same map,
rest and shop heuristics -- so a difference in floors reached is attributable to
the card picking and nothing else.

WHAT THIS CANNOT SHOW
---------------------
Live performance. The simulator's combat is the frozen policy, not the turn
search the live agent uses, and a deck built for an agent that loses fights is
not necessarily the deck you want when the fights are played well. What it can
show is whether the archetype signal moves deck quality at all, cheaply, before
an hour of live runs is spent finding out.

Paired on seed: run N of both arms starts from the same seed and sees the same
map and the same card offers, so the comparison is within-pair rather than
between two independent samples. That is worth roughly a halving of the runs
needed for the same resolution.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys

import numpy as np


def _pick_card_reward(mgr, run_mask, rng, use_archetype: bool):
    """Take the best card on offer, by whichever scorer this arm uses.

    Falls back to the walker's random behaviour when the screen is not a plain
    card choice -- relic and potion sub-screens share the phase, and neither is
    what this experiment varies.
    """
    from sts2_env.bridge.card_quality import rank_cards
    from sts2_env.gym_env.run_env import _CARD_RWD_START
    from sts2_env.search.archetypes import DeckDirection

    offered = [
        {"id": a.get("card_id")}
        for a in (mgr.get_available_actions() or [])
        if a.get("action") == "pick_card" and a.get("card_id")
    ]
    if not offered:
        return None

    deck = [{"id": c.card_id.name} for c in mgr.run_state.player.deck]
    direction = None
    if use_archetype:
        direction = DeckDirection()
        direction.observe_deck(c["id"] for c in deck)

    ranked = rank_cards(offered, deck, direction)
    best_score, best_offer_index, _ = ranked[0]
    if best_score <= 0:
        # Nothing worth taking. The skip slot keeps deck sizes honest rather
        # than uniformly maximal, and matches what the live agent does.
        skip = _CARD_RWD_START + 3
        if skip < len(run_mask) and run_mask[skip]:
            return int(skip)
    slot = _CARD_RWD_START + best_offer_index
    return int(slot) if slot < len(run_mask) and run_mask[slot] else None


def walk(seed: int, use_archetype: bool, combat_policy) -> tuple[int, str | None]:
    """One full run. Returns (floor reached, committed archetype or None)."""
    from sts2_env.gym_env.run_env import STS2RunEnv
    from sts2_env.run.run_manager import RunManager
    from sts2_env.search.archetypes import DeckDirection

    sys.path.insert(0, "scripts")
    from harvest_combat_benchmark import _combat_action, _noncombat_action

    rng = np.random.default_rng(seed)
    env = STS2RunEnv()
    env.reset(seed=seed)
    for _ in range(3000):
        mgr = env._mgr
        if mgr is None:
            break
        mask = env.action_masks()
        valid = np.where(mask == 1)[0]
        if not len(valid):
            break

        action = None
        if mgr.phase == RunManager.PHASE_COMBAT:
            if combat_policy is not None:
                action = _combat_action(combat_policy, mgr, mask)
        elif mgr.phase == RunManager.PHASE_CARD_REWARD:
            action = _pick_card_reward(mgr, mask, rng, use_archetype)
            if action is None:
                action = _noncombat_action(mgr, mgr.phase, mask, rng)
        else:
            action = _noncombat_action(mgr, mgr.phase, mask, rng)
        if action is None:
            action = int(rng.choice(valid))

        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    mgr = env._mgr
    floor = int(getattr(getattr(mgr, "run_state", None), "total_floor", 0) or 0)
    direction = DeckDirection()
    if mgr is not None:
        direction.observe_deck(c.card_id.name for c in mgr.run_state.player.deck)
    env.close()
    return floor, direction.committed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--combat-model",
                        default="output/combat_v3_overnight/final_model.zip")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import sts2_env.cards  # noqa: F401
    sys.path.insert(0, "scripts")
    from harvest_combat_benchmark import _load_combat_policy

    policy = _load_combat_policy(args.combat_model)
    print(f"{args.runs} paired runs per arm, combat by {args.combat_model}\n")

    control, archetype, committed = [], [], []
    for i in range(args.runs):
        seed = args.seed + i
        control.append(walk(seed, False, policy)[0])
        floor, plan = walk(seed, True, policy)
        archetype.append(floor)
        committed.append(plan)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{args.runs}  control {statistics.mean(control):.2f}  "
                  f"archetype {statistics.mean(archetype):.2f}")

    c = np.array(control, dtype=float)
    a = np.array(archetype, dtype=float)
    diff = a - c
    se = diff.std(ddof=1) / math.sqrt(len(diff)) if len(diff) > 1 else float("nan")

    print("\n" + "=" * 62)
    print(f"  control    mean floor {c.mean():5.2f}   median {np.median(c):.0f}")
    print(f"  archetype  mean floor {a.mean():5.2f}   median {np.median(a):.0f}")
    print(f"\n  difference {diff.mean():+5.2f} +/- {se:.2f} (1 se, paired)")
    if se == se and se > 0:
        print(f"             {diff.mean() / se:+.1f} standard errors")
        if abs(diff.mean()) < se:
            print("             INSIDE THE NOISE -- no evidence either way")
    print(f"\n  archetype arm won {int((diff > 0).sum())}, "
          f"lost {int((diff < 0).sum())}, tied {int((diff == 0).sum())}")

    import collections
    plans = collections.Counter(p or "(none)" for p in committed)
    print("\n  decks the archetype arm committed to:")
    for name, n in plans.most_common():
        print(f"    {name:<16} {n:>4}  {100 * n / len(committed):.0f}%")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
