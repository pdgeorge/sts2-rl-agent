"""Play full simulated runs with the measured drafter, and report where they die.

    python scripts/eval_drafting.py --runs 40
    python scripts/eval_drafting.py --runs 40 --no-priors --no-density
    python scripts/eval_drafting.py --runs 40 --compare

WHY THIS EXISTS

On 2026-08-04 a dozen drafting changes shipped in one day, each verified in
isolation and none end to end. Live floors went from a 13.0-13.6 baseline to 10.1
over 45 runs, degrading as the day went on -- 11.4 in the first half, 8.8 in the
second. The regression was found by pdgeorge playing twenty runs, which is the
most expensive possible test harness and the slowest possible feedback loop.

Nothing here is new capability. `RunManager` could always be driven directly; no
script did it with the measured drafter, so "does this change make runs better"
had no cheap answer and every change went out on the strength of a unit test that
proved only that it did what it said.

WHAT IT MEASURES, AND WHAT IT DOES NOT

Floors reached, distribution not just mean -- "average 10" hides whether that is
every run reaching 10 or half reaching 3 and half reaching 17, and those want
different fixes. Deck size and block density at death are reported because deck
bloat is the failure mode this was built to catch.

It is the SIMULATOR, so it inherits every parity gap the simulator has, and
combat is flown by a pilot rather than the trained model. Use it for A/B between
two drafting configurations, where both arms share those biases, and not as a
prediction of live floor counts.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter


def play_one(seed: int, *, use_priors: bool, use_density: bool,
             seeds_per_decision: tuple[int, ...], use_rest: bool = True,
             max_steps: int = 4000) -> dict:
    """Drive one full run, using the measured drafter at every card reward."""
    from sts2_env.cards.factory import create_card
    from sts2_env.core.enums import CardId
    from sts2_env.evaluation import card_choice
    from sts2_env.evaluation.card_choice import best_index, rank_candidates
    from sts2_env.evaluation.deck_metrics import block_density
    from sts2_env.evaluation.pilots import greedy_pilot
    from sts2_env.evaluation.rest_choice import rank_rest_options
    from sts2_env.gym_env.action_space import action_to_card_and_target, get_action_mask
    from sts2_env.bridge.agent_runner import (
        REST_HP_RATIO_THRESHOLD,
        ROOM_PRIORITY_HEALTHY,
        ROOM_PRIORITY_LOW_HP,
    )
    from sts2_env.run.run_manager import RunManager

    manager = RunManager(seed=seed)
    rng = __import__("random").Random(seed)

    # Density is applied inside apply_composition_rules; the cheapest honest way
    # to switch it off for an A/B arm is to zero its scale rather than to fork
    # the code path, so both arms run identical code.
    saved_density = card_choice.DENSITY_POINTS
    if not use_density:
        card_choice.DENSITY_POINTS = 0.0

    drafted = skipped = rests = smiths = 0
    try:
        for _ in range(max_steps):
            if manager.is_over:
                break
            actions = manager.get_available_actions()
            if not actions:
                break

            if manager.phase == RunManager.PHASE_COMBAT:
                # Combat MUST go through the manager's own action dicts. Calling
                # apply_action() on the CombatState directly desyncs the two --
                # the fight advances and the manager never notices, so the run
                # sits on floor 1 forever. That was this script's first bug.
                combat = manager.get_combat_state()
                choice = None
                if combat is not None and not combat.is_over and get_action_mask(combat).any():
                    hand_index, target_index = action_to_card_and_target(
                        int(greedy_pilot(combat))
                    )
                    if hand_index is None:
                        choice = next(
                            (a for a in actions if a.get("action") == "end_turn"), None
                        )
                    else:
                        choice = next(
                            (a for a in actions
                             if a.get("action") == "play_card"
                             and a.get("hand_index") == hand_index
                             and (target_index is None
                                  or a.get("target_index") in (None, target_index))),
                            None,
                        )
                manager.take_action(choice or actions[0])
                continue

            picks = [a for a in actions if a.get("action") == "pick_card"]
            if picks:
                deck = list(manager.run_state.player.deck)
                offered, indexes = [], []
                for action in picks:
                    card = action.get("card")
                    if card is None:
                        card_id = action.get("card_id")
                        try:
                            card = create_card(CardId[card_id]) if card_id else None
                        except Exception:  # noqa: BLE001
                            card = None
                    if card is not None:
                        offered.append(card)
                        indexes.append(action["index"])
                can_skip = any(a.get("action") == "skip" for a in actions)
                if offered:
                    ranked = rank_candidates(
                        deck, offered, greedy_pilot,
                        floor=manager.run_state.total_floor,
                        seeds=seeds_per_decision,
                        include_skip=can_skip,
                        use_priors=use_priors,
                    )
                    choice = best_index(ranked, offered)
                    if choice is None and can_skip:
                        skipped += 1
                        manager.take_action(
                            next(a for a in actions if a.get("action") == "skip")
                        )
                    else:
                        drafted += 1
                        manager.take_action(
                            {"action": "pick_card", "index": indexes[choice or 0]}
                        )
                    continue

            # Rest sites, decided by rank_rest_options -- the same call the live
            # agent makes. Random choice here made the whole rest/upgrade model
            # invisible to this harness, which is how a change that drove live
            # upgrade density to 7% could be measured as "no difference".
            if use_rest and manager.phase == RunManager.PHASE_REST_SITE:
                rest = next((a for a in actions if a.get("option_id") == "HEAL"), None)
                smith = next((a for a in actions if a.get("option_id") == "SMITH"), None)
                if rest is not None and smith is not None:
                    deck = list(manager.run_state.player.deck)
                    player = manager.run_state.player
                    ranked = rank_rest_options(
                        deck, [c for c in deck if not getattr(c, "upgraded", False)],
                        greedy_pilot,
                        current_hp=int(getattr(player, "current_hp", 1)),
                        max_hp=int(getattr(player, "max_hp", 80)),
                        floor=manager.run_state.total_floor,
                        seeds=seeds_per_decision,
                    )
                    if ranked and ranked[0].kind == "upgrade":
                        smiths += 1
                        manager.take_action(smith)
                    else:
                        rests += 1
                        manager.take_action(rest)
                    continue

            # Map routing uses the LIVE agent's priority order, not chance.
            #
            # Random routing made this harness unrepresentative in exactly the
            # way that hid three separate changes. It died around floor 7.6 with
            # 12-card decks against the live agent's 11.6 and ~17, so it never
            # reached the deck sizes the bloat term acts on, the densities the
            # block band acts on, or more than 16 rest sites across 25 runs. All
            # three measured "no difference" because the states they act on were
            # off the end of the run.
            moves = [a for a in actions if a.get("action") == "move"]
            if moves:
                player = manager.run_state.player
                hp_ratio = (float(getattr(player, "current_hp", 0))
                            / max(1.0, float(getattr(player, "max_hp", 1))))
                priority = (ROOM_PRIORITY_LOW_HP if hp_ratio < REST_HP_RATIO_THRESHOLD
                            else ROOM_PRIORITY_HEALTHY)
                chosen = None
                for room in priority:
                    for move in moves:
                        if str(move.get("point_type", "")).lower().replace("_", "") == room:
                            chosen = move
                            break
                    if chosen is not None:
                        break
                manager.take_action(chosen or moves[0])
                continue

            # Shops, events and treasure stay random: this is an A/B on drafting
            # and rest, and a heuristic there would be one more untested thing
            # shared by both arms.
            manager.take_action(rng.choice(actions))
    except Exception as error:  # noqa: BLE001
        return {"floor": manager.run_state.total_floor, "error": repr(error)[:120],
                "deck_size": len(manager.run_state.player.deck), "drafted": drafted,
                "skipped": skipped, "block_density": 0.0, "upgrade_density": 0.0,
                "rests": rests, "smiths": smiths}
    finally:
        card_choice.DENSITY_POINTS = saved_density

    deck = list(manager.run_state.player.deck)
    from sts2_env.evaluation.deck_metrics import upgrade_density
    return {
        "floor": manager.run_state.total_floor,
        "won": manager.player_won,
        "deck_size": len(deck),
        "block_density": block_density(deck),
        "upgrade_density": upgrade_density(deck),
        "drafted": drafted,
        "skipped": skipped,
        "rests": rests,
        "smiths": smiths,
    }


def summarise(label: str, rows: list[dict]) -> dict:
    floors = [r["floor"] for r in rows]
    errors = [r for r in rows if r.get("error")]
    decks = [r["deck_size"] for r in rows]
    density = [r["block_density"] for r in rows if r["block_density"]]
    picks = sum(r["drafted"] for r in rows)
    skips = sum(r["skipped"] for r in rows)
    sem = (statistics.stdev(floors) / len(floors) ** 0.5) if len(floors) > 1 else 0.0

    print(f"\n{label}")
    print(f"  floor    mean {statistics.mean(floors):5.1f} +/- {sem:.1f} sem   "
          f"median {statistics.median(floors):4.0f}   max {max(floors):3}")
    print(f"  deck     mean {statistics.mean(decks):5.1f} cards   "
          f"block density {statistics.mean(density) if density else 0:.0%}")
    print(f"  drafting {picks} taken, {skips} skipped "
          f"({skips / max(1, picks + skips):.0%} skip rate)")
    r = sum(x.get("rests", 0) for x in rows); sm = sum(x.get("smiths", 0) for x in rows)
    up = [x.get("upgrade_density", 0.0) for x in rows]
    print(f"  rest     {r} rested, {sm} smithed "
          f"({sm / max(1, r + sm):.0%} smith rate)   "
          f"upgrade density {statistics.mean(up) if up else 0:.0%}")
    if errors:
        print(f"  ERRORS   {len(errors)}/{len(rows)}: {errors[0]['error']}")
    buckets = Counter(min(f // 5 * 5, 30) for f in floors)
    print("  " + "  ".join(f"{b}-{b+4}:{buckets[b]}" for b in sorted(buckets)))
    return {"mean": statistics.mean(floors), "sem": sem}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed-base", type=int, default=70000)
    ap.add_argument("--decision-seeds", type=int, default=4,
                    help="battery seeds per draft decision; live uses 4")
    ap.add_argument("--no-priors", action="store_true")
    ap.add_argument("--no-density", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="run all four on/off combinations on the SAME run seeds")
    args = ap.parse_args()

    decision_seeds = tuple(range(args.decision_seeds))
    seeds = [args.seed_base + i for i in range(args.runs)]

    arms = [("priors+density (live today)", True, True)]
    if args.compare:
        arms = [
            ("priors ON  density ON   (live today)", True, True),
            ("priors ON  density OFF", True, False),
            ("priors OFF density ON", False, True),
            ("priors OFF density OFF (before today)", False, False),
        ]
    elif args.no_priors or args.no_density:
        arms = [(f"priors {'OFF' if args.no_priors else 'ON'} "
                 f"density {'OFF' if args.no_density else 'ON'}",
                 not args.no_priors, not args.no_density)]

    results = {}
    for label, priors, density in arms:
        # SAME run seeds across arms -- paired, so the comparison is not fighting
        # between-run variance on top of the effect.
        rows = [play_one(s, use_priors=priors, use_density=density,
                         seeds_per_decision=decision_seeds) for s in seeds]
        results[label] = summarise(label, rows)

    if len(results) > 1:
        print("\n" + "=" * 60)
        best = max(results.items(), key=lambda kv: kv[1]["mean"])
        print(f"best: {best[0]}  ({best[1]['mean']:.1f} floors)")
        print("Differences under ~2 sem are not differences. See PLAN.md rule 6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
