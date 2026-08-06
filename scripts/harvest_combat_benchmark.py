"""Collect the fights a real run actually presents, and freeze them as a benchmark.

Combat models here have always been measured on `STS2CombatEnv`, which deals the
Ironclad starter deck at full HP against a random act 1 encounter. No run is ever
in that state after floor 1. The model that scores 92% there dies on floor 8 of a
live run holding a 16-card deck at 40 HP, and both numbers are real -- they are
just not about the same problem.

So this walks actual runs and snapshots each fight as it begins: the deck as it
has grown, the HP as it has been spent, the relics collected, against the
encounter the map actually rolled. The result is a fixed set of situations that
every candidate agent can be pointed at, which is what makes "is the searcher
better than the policy" a question with an answer.

    python scripts/harvest_combat_benchmark.py --count 200

The walking policy only decides which situations get harvested, not how good they
are. Random play dies early and would sample floors 1-6 almost exclusively, so
--model spreads the harvest deeper. Coverage is reported per floor band and per
room type, because a benchmark that is 90% floor-1 hallway fights would flatter
anything measured on it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from sts2_env.gym_env.run_env import STS2RunEnv
from sts2_env.run.run_manager import RunManager
from sts2_env.search.situation import CombatSituation, save_situations

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "tests/fixtures/act1_combat_benchmark.json"

FLOOR_BANDS = ((1, 4), (5, 8), (9, 12), (13, 16), (17, 99))


def _room_name(mgr) -> str:
    """The room type as CombatSituation records it, for the --room-types filter."""
    from sts2_env.search.situation import _room_type_name

    return _room_type_name(mgr._current_room_type)


def _load_combat_policy(model_path: str | None):
    """The policy used to fight while walking runs. None means random.

    A combat model rather than a full-run one on purpose: the run observation has
    been relaid out since the last full-run model was trained (that is what
    `output/crash_log.json` records), while the 131-dim combat observation has
    not moved, so `combat_v3_overnight` still loads and still fights. Fighting is
    all that is needed here -- random non-combat choices die around floor 5, and
    the point of the walker is only to reach deep floors, not to play well.
    """
    if model_path is None:
        return None
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        print("Walking runs with a model needs sb3-contrib: pip install 'sts2-rl-agent[train]'")
        sys.exit(1)
    return MaskablePPO.load(model_path, device="cpu")


def _combat_action(policy, mgr, run_mask: np.ndarray) -> int | None:
    """The combat model's choice, as a run-level action index, or None.

    Combat occupies indices 0..114 of the run action space, so a combat action
    index is already a run action index. It is still checked against the run mask
    rather than trusted: the run env masks pending choices and player selection
    differently, and a rejected action would stall the walk rather than fail.
    """
    from sts2_env.gym_env.action_space import get_action_mask
    from sts2_env.gym_env.observation import encode_observation

    combat = mgr.get_combat_state()
    if combat is None or combat.is_over:
        return None
    try:
        obs = encode_observation(combat)
        mask = get_action_mask(combat, owner=combat.primary_player)
    except Exception:
        return None
    if not mask.any():
        return None

    action, _ = policy.predict(obs, action_masks=mask, deterministic=False)
    action = int(action)
    return action if action < len(run_mask) and run_mask[action] else None


def _noncombat_action(mgr, phase: str, run_mask: np.ndarray, rng) -> int | None:
    """A plausible amateur's non-combat choice, or None to fall back to random.

    Not an attempt at good play -- it exists because random non-combat play dies
    around floor 8 with an 11-card deck and one relic, so a randomly-walked
    fixture would contain none of the fights that actually kill runs. Taking the
    reward and healing when low is most of what keeps a run alive, and it is
    enough to reach the boss.

    This does bias the fixture toward decks built the way this heuristic builds
    them. Every agent is measured on the same fixture so the comparison stays
    fair, but a searcher that later builds different decks is a reason to
    re-harvest rather than to reuse this one for ever.
    """
    from sts2_env.gym_env.run_env import (
        _CARD_RWD_START, _REST_START, _SHOP_START, _TREASURE_START,
        _BOSS_RELIC_START,
    )

    def offer(index: int) -> int | None:
        return int(index) if index < len(run_mask) and run_mask[index] else None

    if phase == RunManager.PHASE_CARD_REWARD:
        # Slot 0 takes the first card, the relic, or the potion depending on
        # which sub-screen is up (see run_env._step_card_reward). Skipping
        # sometimes keeps deck sizes varied rather than uniformly maximal.
        if rng.random() < 0.85:
            return offer(_CARD_RWD_START)
        return offer(_CARD_RWD_START + 3)

    if phase == RunManager.PHASE_REST_SITE:
        options = [a for a in mgr.get_available_actions() if a.get("action") == "rest_option"]
        player = mgr.run_state.player
        hurt = player.max_hp > 0 and player.current_hp / player.max_hp < 0.6
        want = "HEAL" if hurt else "SMITH"
        for i, opt in enumerate(options):
            if opt.get("option_id") == want and opt.get("enabled", True):
                return offer(_REST_START + i)
        return offer(_REST_START)

    if phase == RunManager.PHASE_TREASURE:
        return offer(_TREASURE_START)

    if phase == RunManager.PHASE_BOSS_RELIC:
        return offer(_BOSS_RELIC_START)

    if phase == RunManager.PHASE_SHOP:
        # Leave. Shop buying needs gold reasoning this walker does not have, and
        # a random buy mostly wastes the gold that a potion could have been.
        return offer(_SHOP_START)

    return None


def harvest(
    count: int,
    *,
    model_path: str | None = None,
    max_floor: int = 16,
    seed: int = 0,
    max_steps_per_run: int = 3000,
    max_runs: int = 2000,
    room_types: set[str] | None = None,
) -> list[CombatSituation]:
    policy = _load_combat_policy(model_path)
    rng = np.random.default_rng(seed)

    # Quotas per floor band, because every run walks floors 1-4 and few reach 13.
    # Harvesting without them gives a fixture that is half opening hallway fights,
    # which is the easy half -- an agent could look good on it while being no
    # better where runs actually end.
    # A room-type filter turns the floor bands off: the point of harvesting
    # BOSS-only is that every act 1 boss sits on floor 16, so banding by floor
    # would fill one band and starve the rest forever. The quota then applies
    # to the whole harvest rather than per band.
    bands = [b for b in FLOOR_BANDS if b[0] <= max_floor]
    per_band = max(1, count // len(bands))
    band_counts: Counter = Counter()

    def band_of(floor: int) -> tuple[int, int] | None:
        for lo, hi in bands:
            if lo <= floor <= hi:
                return (lo, hi)
        return None

    situations: list[CombatSituation] = []
    run_index = 0
    seen_this_run: set[int] = set()

    while len(situations) < count:
        env = STS2RunEnv()
        obs, _ = env.reset(seed=seed + run_index)
        run_index += 1
        seen_this_run = set()

        for _ in range(max_steps_per_run):
            mgr = env._mgr
            if mgr is None:
                break

            # Snapshot at the moment a fight begins: phase is COMBAT, the manager
            # recorded which encounter it rolled, and this fight has not been
            # taken yet. Keyed on floor because one floor is one fight.
            floor = mgr.run_state.total_floor
            band = band_of(floor)
            if (
                mgr.phase == RunManager.PHASE_COMBAT
                and mgr._last_encounter is not None
                and floor not in seen_this_run
                and floor <= max_floor
                and band is not None
                and (room_types is None or _room_name(mgr) in room_types)
                and (room_types is not None or band_counts[band] < per_band)
            ):
                seen_this_run.add(floor)
                band_counts[band] += 1
                situations.append(
                    CombatSituation.from_run_manager(
                        mgr, f"f{floor:02d}-r{run_index:03d}"
                    )
                )
                if len(situations) >= count:
                    break

            mask = env.action_masks()
            valid = np.where(mask == 1)[0]
            if len(valid) == 0:
                break

            action = None
            if mgr.phase == RunManager.PHASE_COMBAT:
                if policy is not None:
                    action = _combat_action(policy, mgr, mask)
            else:
                action = _noncombat_action(mgr, mgr.phase, mask, rng)
            if action is None:
                action = int(rng.choice(valid))

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        env.close()

        if run_index % 50 == 0:
            logger.info(
                "%d runs, %d/%d situations  %s",
                run_index, len(situations), count,
                " ".join(f"{lo}-{hi}:{band_counts[(lo, hi)]}/{per_band}"
                         for lo, hi in bands),
            )

        if run_index >= max_runs:
            # Said out loud rather than returned quietly. A fixture short on deep
            # floors measures a different thing than one that is not, and finding
            # that out from the file later is worse than reading it here.
            unmet = [f"{lo}-{hi}: {band_counts[(lo, hi)]}/{per_band}"
                     for lo, hi in bands if band_counts[(lo, hi)] < per_band]
            logger.warning(
                "Stopped at the %d-run budget with %d/%d situations. Bands short "
                "of quota: %s. The walker dies before those floors; raise "
                "--max-runs or improve the walking policy.",
                max_runs, len(situations), count, "; ".join(unmet) or "none",
            )
            break

    return situations


def describe(situations: list[CombatSituation]) -> str:
    """Coverage, stated plainly. A benchmark's shape decides what it can measure."""
    if not situations:
        return "No situations harvested."

    lines = ["", f"{len(situations)} situations", ""]

    band_counts = Counter()
    for s in situations:
        for lo, hi in FLOOR_BANDS:
            if lo <= s.total_floor <= hi:
                band_counts[(lo, hi)] += 1
                break
    lines.append("by floor:")
    for lo, hi in FLOOR_BANDS:
        c = band_counts.get((lo, hi), 0)
        if c:
            lines.append(f"  {lo:>2}-{hi:<2} {c:>4}  {'#' * min(40, c)}")

    lines.append("")
    lines.append("by room:")
    for room, c in Counter(s.room_type for s in situations).most_common():
        lines.append(f"  {room:<10} {c:>4}")

    decks = [len(s.deck) for s in situations]
    hp = [s.current_hp / s.max_hp for s in situations if s.max_hp]
    relics = [len(s.relics) for s in situations]
    lines += [
        "",
        f"deck size   mean {np.mean(decks):.1f}   min {min(decks)}   max {max(decks)}",
        f"hp fraction mean {np.mean(hp):.2f}   min {min(hp):.2f}   max {max(hp):.2f}",
        f"relics      mean {np.mean(relics):.1f}   max {max(relics)}",
        f"encounters  {len(set(s.encounter for s in situations))} distinct",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the fights real runs present, as a benchmark fixture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count", type=int, default=200, help="How many situations")
    parser.add_argument("--combat-model", default=None,
                        help="Combat model (131-dim) used to fight while walking runs. "
                             "Without it, runs are walked randomly and die around floor "
                             "5, sampling almost only low floors.")
    parser.add_argument("--max-floor", type=int, default=16,
                        help="Ignore fights above this floor (16 = the act 1 boss)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=2000,
                        help="Run budget. Deep floors are rare, so a balanced "
                             "fixture needs many runs to fill its top band.")
    parser.add_argument("--room-types", default=None,
                        help="Comma-separated room types to harvest (e.g. "
                             "BOSS or BOSS,ELITE). Without it the harvest is "
                             "quota'd by floor band. With it, floor-band "
                             "quotas are off, because every act 1 boss is on "
                             "floor 16 and would fill one band and starve the "
                             "rest. Use for a boss-weighted held-out set: the "
                             "default fixture holds 15 boss fights, which puts "
                             "a ~12 point standard error on any boss win rate "
                             "near 30% and cannot resolve the Phase 3.3 gate.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    situations = harvest(
        args.count, model_path=args.combat_model, max_floor=args.max_floor,
        seed=args.seed, max_runs=args.max_runs,
        room_types=(
            {t.strip().upper() for t in args.room_types.split(",")}
            if args.room_types else None
        ),
    )
    print(describe(situations))
    path = save_situations(situations, args.output)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
