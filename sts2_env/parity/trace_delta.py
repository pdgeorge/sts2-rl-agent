"""Per-step behavioural parity: does the simulator resolve an action like the game?

The existing replay harness (bridge_replay.py) compares a hand-authored scenario:
you write a factory that builds one specific combat, drive that same combat in the
real game, and check the two stay in lockstep. That is exact, and it only works
for combats someone scripted in advance.

This answers a different question about traces nobody scripted. Recording a live
run gives you the game's actual behaviour across whatever encounters it happened
to meet -- but there is no factory for "floor 4, Fuzzy Wurm Crawler, whatever deck
the heuristic built by then", so the whole-combat replay cannot start.

So instead of replaying a combat from the beginning and drifting, this resyncs
from the game at every step:

    for each recorded (state, action, next_state) inside a combat:
        rebuild a simulator combat from `state`
        apply `action`
        compare the *deltas* -- damage dealt, HP lost, block gained --
        against what the game actually did

Deltas rather than absolute state, because a bridge state cannot reconstruct a
combat exactly: draw pile order, relics and some powers are not in it. Absolute
comparison would report a mismatch on every step and say nothing. Deltas ask the
question that matters -- "the game played Bash into this enemy and dealt 14; does
the simulator also deal 14?" -- which is exactly the drift the card-value
derivation cannot catch, because it is behaviour rather than numbers.

Steps that cannot be rebuilt are counted and reported, never silently skipped: a
tool that quietly compares 3 of 63 steps and says "ok" is worse than no tool.

Usage:
    python -m sts2_env.parity.trace_delta output/trace.json [--verbose]
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from sts2_env.cards.factory import create_card
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import CardId, PowerId
from sts2_env.core.rng import Rng

MONSTER_MODULES = (
    "sts2_env.monsters.act1",
    "sts2_env.monsters.act1_weak",
    "sts2_env.monsters.act2",
    "sts2_env.monsters.act3",
    "sts2_env.monsters.act4",
    "sts2_env.monsters.shared",
)
COMBAT_STATE = "combat_action"
REBUILD_SEED = 0


@dataclass
class StepResult:
    index: int
    action: dict[str, Any]
    card_id: str | None
    game: dict[str, int]
    sim: dict[str, int]

    @property
    def agrees(self) -> bool:
        return self.game == self.sim


@dataclass
class Report:
    compared: list[StepResult] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)

    @property
    def mismatches(self) -> list[StepResult]:
        return [s for s in self.compared if not s.agrees]


@lru_cache(maxsize=1)
def monster_factories() -> dict[str, Callable[..., Any]]:
    """monster_id -> factory, discovered by construction.

    There is no registry keyed by monster_id, and the ids only exist on built
    creatures, so every create_* is called once with a fixed seed and asked what
    it made.
    """
    factories: dict[str, Callable[..., Any]] = {}
    for module_name in MONSTER_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in dir(module):
            if not name.startswith("create_"):
                continue
            fn = getattr(module, name)
            if not callable(fn):
                continue
            try:
                creature, _ai = fn(Rng(REBUILD_SEED))
            except Exception:  # noqa: BLE001 - many need specific arguments; skip them
                continue
            monster_id = getattr(creature, "monster_id", None)
            if monster_id:
                factories.setdefault(monster_id, fn)
    return factories


def _power_id(name: str) -> PowerId:
    """Resolve a bridge power name to a simulator PowerId.

    The game names these with a trailing _POWER (VULNERABLE_POWER) where the
    simulator's enum does not (VULNERABLE). The same convention gap is already
    handled for dynamic vars in reference_static_metadata, and it accounted for
    every rebuild failure on the first real trace -- 26 of 63 steps uncomparable
    because of a suffix.
    """
    try:
        return PowerId[name]
    except KeyError:
        return PowerId[name.removesuffix("_POWER")]


def rebuild_combat(state: dict[str, Any], reasons: Counter | None = None) -> CombatState | None:
    """Best-effort simulator combat matching a recorded bridge state.

    Returns None when the state names something the simulator does not have --
    a monster deleted upstream, a card with no factory -- which is itself a
    finding rather than an error.
    """
    player = state.get("player") or {}
    combat = CombatState(
        player_hp=int(player.get("hp", 80)),
        player_max_hp=int(player.get("max_hp", 80)),
        deck=[],
        rng_seed=REBUILD_SEED,
        character_id="Ironclad",
    )

    factories = monster_factories()
    for enemy_state in state.get("enemies", []):
        monster_id = enemy_state.get("id")
        factory = factories.get(monster_id)
        if factory is None:
            if reasons is not None:
                reasons[f"no simulator monster for {monster_id}"] += 1
            return None
        creature, ai = factory(Rng(REBUILD_SEED))
        combat.add_enemy(creature, ai)
        creature.max_hp = int(enemy_state.get("max_hp", creature.max_hp))
        creature.current_hp = int(enemy_state.get("hp", creature.current_hp))
        creature.block = int(enemy_state.get("block", 0))
        # The recorded state is authoritative about powers, so whatever the
        # factory granted innately has to go first. Inklets are created with
        # Slippery; by the time the game reached this state one of them had spent
        # it, and layering the recorded powers on top of the factory's left the
        # simulator dodging a hit the game took -- 1 damage where the game dealt
        # 6, which read as simulator drift and was this.
        creature.powers.clear()
        for power in enemy_state.get("powers", []):
            try:
                combat.apply_power_to(creature, _power_id(power["id"]), int(power["amount"]))
            except (KeyError, ValueError):
                if reasons is not None:
                    reasons[f"unknown enemy power {power.get('id')}"] += 1
                return None

    combat.start_combat()

    # start_combat deals an opening hand and can change HP via relics; overwrite
    # with what the game actually had, so only the action under test differs.
    combat.player.current_hp = int(player.get("hp", combat.player.current_hp))
    combat.player.block = int(player.get("block", 0))
    combat.energy = int(player.get("energy", combat.energy))
    for power in player.get("powers", []):
        try:
            combat.apply_power_to(combat.player, _power_id(power["id"]), int(power["amount"]))
        except (KeyError, ValueError):
            if reasons is not None:
                reasons[f"unknown player power {power.get('id')}"] += 1
            return None

    hand = []
    for card_state in state.get("hand", []):
        try:
            card = create_card(CardId[card_state["id"]], upgraded=bool(card_state.get("upgraded")))
        except (KeyError, ValueError):
            if reasons is not None:
                reasons[f"no simulator card for {card_state.get('id')}"] += 1
            return None
        hand.append(card)
    combat.hand = hand
    return combat


def _keyed(entries: list[tuple[str, int, int]]) -> dict[str, int]:
    """Key enemies by identity, not list position.

    The game drops a dead enemy out of the list, so position N after an action is
    not position N before it. Comparing by index made a kill look like the next
    enemy healing -- one step reported enemy0_hp +34 -- and every disagreement
    this tool first produced was that artifact rather than a real difference.
    """
    seen: Counter = Counter()
    snap: dict[str, int] = {}
    for monster_id, max_hp, hp, block in entries:
        # max_hp is part of the key because an encounter can hold several of the
        # same monster -- three INKLETs at 14, 12 and 16 -- and an ordinal alone
        # still shifts when one of them dies, which read as the survivors taking
        # 1 damage from a 6-damage Strike.
        base = f"{monster_id}@{max_hp}"
        key = f"{base}#{seen[base]}"
        seen[base] += 1
        snap[f"{key}_hp"] = hp
        snap[f"{key}_block"] = block
    return snap


def _snapshot(state: dict[str, Any]) -> dict[str, int]:
    player = state.get("player") or {}
    snap = {
        "player_hp": int(player.get("hp", 0)),
        "player_block": int(player.get("block", 0)),
    }
    snap.update(_keyed([
        (e.get("id", "?"), int(e.get("max_hp", 0)), int(e.get("hp", 0)), int(e.get("block", 0)))
        for e in state.get("enemies", [])
        if e.get("is_alive", True)
    ]))
    return snap


def _sim_snapshot(combat: CombatState) -> dict[str, int]:
    snap = {
        "player_hp": combat.player.current_hp,
        "player_block": combat.player.block,
    }
    snap.update(_keyed([
        (e.monster_id or "?", e.max_hp, e.current_hp, e.block)
        for e in combat.enemies
        if e.is_alive
    ]))
    return snap


def _delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Changes to things present on both sides, plus who stopped being present.

    An enemy that vanishes between the two states was killed. That is a real and
    comparable outcome -- did the simulator kill it too? -- so it is reported
    rather than dropped for having no 'after' value.
    """
    out = {k: after[k] - before[k] for k in before if k in after and after[k] != before[k]}
    for key in before:
        if key.endswith("_hp") and key not in after:
            out[key.removesuffix("_hp") + "_KILLED"] = 1
    return out


def compare_trace(path: str | Path) -> Report:
    data = json.loads(Path(path).read_text())
    report = Report()

    prior = data["initial_state"]
    for index, step in enumerate(data["steps"]):
        result_state = step["resulting_state"]
        action = step["action"]

        # Only card plays are compared. end_turn resolves the whole enemy turn,
        # whose intents depend on AI state this rebuild cannot restore, so a
        # mismatch there would say nothing about the simulator's card logic.
        if prior.get("type") != COMBAT_STATE or result_state.get("type") != COMBAT_STATE:
            report.skipped["not a combat-to-combat step"] += 1
            prior = result_state
            continue
        if action.get("action") != "play":
            report.skipped[f"action {action.get('action')!r} not compared"] += 1
            prior = result_state
            continue

        combat = rebuild_combat(prior, report.skipped)
        if combat is None:
            prior = result_state
            continue

        card_index = int(action["card_index"])
        card_id = None
        if 0 <= card_index < len(combat.hand):
            card_id = combat.hand[card_index].card_id.name

        before = _sim_snapshot(combat)
        if not combat.play_card(card_index, action.get("target_index")):
            report.skipped["simulator rejected the card the game played"] += 1
            prior = result_state
            continue

        report.compared.append(StepResult(
            index=index,
            action=action,
            card_id=card_id,
            game=_delta(_snapshot(prior), _snapshot(result_state)),
            sim=_delta(before, _sim_snapshot(combat)),
        ))
        prior = result_state

    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="Path to a bridge replay trace")
    ap.add_argument("--verbose", action="store_true", help="Show every compared step")
    args = ap.parse_args()

    report = compare_trace(args.trace)
    total = len(report.compared)
    bad = report.mismatches

    print(f"compared:  {total} card plays")
    print(f"agreeing:  {total - len(bad)}")
    print(f"differing: {len(bad)}")
    if report.skipped:
        print("\nnot compared:")
        for reason, count in report.skipped.most_common():
            print(f"  {count:>4}  {reason}")

    if bad:
        by_card = defaultdict(list)
        for step in bad:
            by_card[step.card_id or "?"].append(step)
        print(f"\ndisagreements by card ({len(by_card)} cards):")
        for card, steps in sorted(by_card.items(), key=lambda kv: -len(kv[1])):
            print(f"\n  {card}  x{len(steps)}")
            example = steps[0]
            print(f"    game: {example.game}")
            print(f"    sim:  {example.sim}")

    if args.verbose:
        print("\nall compared steps:")
        for step in report.compared:
            flag = "ok " if step.agrees else "DIFF"
            print(f"  [{step.index:>3}] {flag} {step.card_id}: game={step.game} sim={step.sim}")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
