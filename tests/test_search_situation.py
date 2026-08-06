"""A situation rebuilds the same fight, every time, from JSON alone.

The benchmark rests entirely on this. If two rebuilds of one situation differ,
then two agents measured on it did not face the same test and the comparison
between them means nothing.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sts2_env.core.enums import RoomType
from sts2_env.gym_env.run_env import STS2RunEnv
from sts2_env.run.run_manager import RunManager
from sts2_env.search.situation import (
    CardRef,
    CombatSituation,
    encounter_registry,
    load_situations,
    resolve_encounter,
    save_situations,
)


def _situation(**overrides) -> CombatSituation:
    base = dict(
        situation_id="test",
        character_id="Ironclad",
        current_hp=54,
        max_hp=80,
        deck=tuple(
            [CardRef("STRIKE_IRONCLAD")] * 5
            + [CardRef("DEFEND_IRONCLAD")] * 4
            + [CardRef("BASH", upgraded=True)]
        ),
        encounter="setup_shrinker_beetle_weak",
        encounter_seed=1234,
        combat_seed=5678,
        relics=("BURNING_BLOOD", "BRONZE_SCALES"),
        room_type="MONSTER",
        act_floor=6,
        total_floor=6,
    )
    base.update(overrides)
    return CombatSituation(**base)


def _fingerprint(combat) -> dict:
    return {
        "hp": (combat.player.current_hp, combat.player.max_hp),
        "energy": combat.energy,
        "hand": [c.card_id.name + ("+" if c.upgraded else "") for c in combat.hand],
        "draw": [c.card_id.name for c in combat.draw_pile],
        "enemies": [
            (e.monster_id, e.current_hp, e.max_hp, e.block) for e in combat.enemies
        ],
        "intents": [
            combat.enemy_ais[e.combat_id].current_move.state_id
            for e in combat.enemies
            if e.combat_id in combat.enemy_ais
        ],
    }


# -- reproducibility --------------------------------------------------------

def test_two_rebuilds_are_identical() -> None:
    situation = _situation()
    assert _fingerprint(situation.to_combat()) == _fingerprint(situation.to_combat())


def test_json_round_trip_preserves_the_situation() -> None:
    situation = _situation()
    assert CombatSituation.from_dict(situation.to_dict()) == situation


def test_a_round_tripped_situation_rebuilds_the_same_fight() -> None:
    situation = _situation()
    restored = CombatSituation.from_dict(situation.to_dict())
    assert _fingerprint(restored.to_combat()) == _fingerprint(situation.to_combat())


def test_saving_and_loading_a_fixture(tmp_path) -> None:
    situations = [_situation(situation_id=f"s{i}", combat_seed=i) for i in range(3)]
    path = save_situations(situations, tmp_path / "fixture.json")
    assert load_situations(path) == situations


# -- the state actually described -------------------------------------------

def test_the_player_is_as_described() -> None:
    combat = _situation(current_hp=41, max_hp=77).to_combat()
    assert combat.player.current_hp == 41
    assert combat.player.max_hp == 77


def test_relics_are_present_and_active() -> None:
    combat = _situation(relics=("BURNING_BLOOD", "BRONZE_SCALES")).to_combat()
    active = {type(r).__name__ for r in combat.current_player_state.relics}
    assert {"BurningBlood", "BronzeScales"} <= active


def test_the_deck_is_rebuilt_with_upgrades() -> None:
    combat = _situation().to_combat()
    everything = list(combat.hand) + list(combat.draw_pile) + list(combat.discard_pile)
    assert len(everything) == 10
    bash = [c for c in everything if c.card_id.name == "BASH"]
    assert len(bash) == 1 and bash[0].upgraded


def test_potions_are_rebuilt_into_their_slots() -> None:
    combat = _situation(potions=("FirePotion", None, "BlockPotion")).to_combat()
    slots = [p.potion_id if p is not None else None for p in combat.potions]
    assert slots[0] == "FirePotion"
    assert slots[1] is None
    assert slots[2] == "BlockPotion"


def test_an_unknown_potion_is_dropped_with_a_warning_not_a_crash() -> None:
    """The previous behaviour was a KeyError that crashed `to_combat` -- and
    when the bridge sent a potion id the simulator did not know, every step
    of the live fight raised in LiveSearch, the runner fell back to END_TURN
    for every combat action, and the player died on the first encounter
    without playing a card. Tests for potions the simulator does not yet
    know now drop the slot to None with a log line rather than raising,
    because the searcher running against a fight missing one potion is
    useful; the searcher crashing is not.
    """
    combat = _situation(potions=("APotionThisBuildRemoved",)).to_combat()
    assert combat.potions[0] is None


def test_the_run_level_rng_streams_are_wired() -> None:
    """Without a run_state, `shuffle_rng` and `monster_ai_rng` silently fall back
    to the combat's own RNG -- reproducible, but drawing from a different stream
    than any real run, which makes the fixture unrepresentative rather than
    merely different."""
    combat = _situation().to_combat()
    run_state = combat.current_player_state.player_state.run_state
    assert run_state is not None
    assert combat.shuffle_rng is run_state.rng.shuffle
    assert combat.monster_ai_rng is run_state.rng.monster_ai


def test_elite_and_boss_rooms_are_built_as_such() -> None:
    elite = _situation(room_type="ELITE", encounter="setup_shrinker_beetle_weak").to_combat()
    assert elite.room.is_elite


# -- failing loudly ---------------------------------------------------------

def test_an_unknown_encounter_is_a_clear_error() -> None:
    with pytest.raises(KeyError, match="No encounter setup"):
        _situation(encounter="setup_a_monster_this_build_removed").to_combat()


def test_an_unknown_card_is_a_clear_error() -> None:
    with pytest.raises(KeyError, match="No card named"):
        CardRef("A_CARD_THIS_BUILD_REMOVED").instantiate()


def test_the_encounter_registry_finds_every_act() -> None:
    registry = encounter_registry()
    assert len(registry) > 50
    assert resolve_encounter("setup_shrinker_beetle_weak") is registry["setup_shrinker_beetle_weak"]


# -- capture ----------------------------------------------------------------

def test_capturing_from_a_live_run_reproduces_its_enemies() -> None:
    """The enemies, their HP rolls, powers and intents come back exactly.

    The opening shuffle deliberately does not: a run's shuffle stream has been
    advanced by every earlier fight and the rebuild starts it at zero. See the
    module docstring in situation.py.
    """
    env = STS2RunEnv()
    env.reset(seed=3)
    rng = np.random.default_rng(0)

    captured = None
    for _ in range(600):
        mgr = env._mgr
        if mgr.phase == RunManager.PHASE_COMBAT and mgr._last_encounter is not None:
            captured = CombatSituation.from_run_manager(mgr, "captured")
            break
        mask = env.action_masks()
        valid = np.where(mask == 1)[0]
        if not len(valid):
            break
        _, _, terminated, truncated, _ = env.step(int(rng.choice(valid)))
        if terminated or truncated:
            break

    assert captured is not None, "the walk never reached a combat"
    original = _fingerprint(env._mgr.get_combat_state())
    rebuilt = _fingerprint(captured.to_combat())

    assert rebuilt["enemies"] == original["enemies"]
    assert rebuilt["intents"] == original["intents"]
    assert rebuilt["hp"] == original["hp"]
    assert sorted(rebuilt["hand"] + rebuilt["draw"]) == sorted(
        original["hand"] + original["draw"]
    ), "the same cards should be in play, however they were shuffled"


def test_capture_refuses_when_no_encounter_was_rolled() -> None:
    mgr = RunManager(seed=1, character_id="Ironclad")
    with pytest.raises(ValueError, match="has not entered a combat"):
        CombatSituation.from_run_manager(mgr, "nope")


# -- the shipped fixture ----------------------------------------------------

BENCHMARK = Path(__file__).parent / "fixtures" / "act1_combat_benchmark.json"


@pytest.mark.skipif(not BENCHMARK.exists(), reason="benchmark fixture not generated")
def test_the_shipped_benchmark_still_rebuilds() -> None:
    """The canary for a game update.

    Every card, potion, relic and encounter the fixture names has to still exist
    and still build a fight. When Mega Crit renames one, this fails with the name
    in the message -- which is the whole point of storing names rather than a
    pickle. Regenerate the fixture; do not edit it by hand.
    """
    situations = load_situations(BENCHMARK)
    assert len(situations) >= 100

    for situation in situations:
        combat = situation.to_combat()
        assert combat.enemies, f"{situation.situation_id} rebuilt with no enemies"
        assert combat.player.current_hp > 0


@pytest.mark.skipif(not BENCHMARK.exists(), reason="benchmark fixture not generated")
def test_the_benchmark_covers_the_fights_that_end_runs() -> None:
    """A fixture of only floor-1 hallway fights would flatter anything measured
    on it. Elites and bosses are where act 1 runs actually stop."""
    situations = load_situations(BENCHMARK)
    rooms = Counter(s.room_type for s in situations)
    assert rooms["ELITE"] >= 10
    assert rooms["BOSS"] >= 5

    deep = [s for s in situations if s.total_floor >= 13]
    assert len(deep) >= 25, "no coverage of the floors where runs end"

    hurt = [s for s in situations if s.current_hp < 0.6 * s.max_hp]
    assert len(hurt) >= 25, "every fight starting near full HP is not a real run"


# --- sibling HP uniqueness (SetUniqueMonsterHpValue) ------------------------


def test_two_siblings_never_share_an_hp_total():
    """`Creature.cs:371` picks from the range MINUS what siblings already hold.

    This simulator rolled each monster independently and produced colliding
    totals in 11.4% of multi-enemy fights, which the game cannot do.
    """
    import sts2_env.cards  # resolve package import order
    from sts2_env.search.situation import load_situations

    situations = load_situations("tests/fixtures/act1_combat_train_2000.json")
    checked = collisions = 0
    for situation in situations[:400]:
        try:
            combat = situation.to_combat()
        except Exception:
            continue
        totals = [e.max_hp for e in combat.enemies]
        if len(totals) < 2:
            continue
        checked += 1
        if len(set(totals)) != len(totals):
            collisions += 1

    assert checked > 50, "fixture produced too few multi-enemy fights to test"
    assert collisions == 0, f"{collisions}/{checked} multi-enemy fights collided"


def test_uniqueness_is_enforced_across_species_not_just_within_one():
    """`CombatState.cs:496` passes the whole `_enemies` list, not same-species.

    So a Zapbot at 24 stops a Stabbot being 24, odd as that reads.
    """
    import sts2_env.cards
    from sts2_env.core.combat import CombatState
    from sts2_env.core.creature import Creature
    from sts2_env.monsters.state_machine import MonsterAI, MoveState

    combat = CombatState(player_hp=80, player_max_hp=80, deck=[], rng_seed=3)

    def bot(monster_id, hp):
        creature = Creature(max_hp=hp, monster_id=monster_id,
                            min_initial_hp=hp, max_initial_hp=hp + 6)
        ai = MonsterAI(
            states={"M": MoveState(state_id="M", intents=[],
                                   effect_fn=lambda combat: None)},
            initial_state_id="M",
        )
        return creature, ai

    first, first_ai = bot("ALPHA", 24)
    combat.add_enemy(first, first_ai)
    second, second_ai = bot("BETA", 24)
    combat.add_enemy(second, second_ai)

    assert first.max_hp != second.max_hp
    assert 24 <= second.max_hp <= 30


def test_a_monster_with_a_unique_total_keeps_its_roll_untouched():
    """No collision means no re-draw -- a deliberate HP must survive."""
    import sts2_env.cards
    from sts2_env.core.combat import CombatState
    from sts2_env.core.creature import Creature
    from sts2_env.monsters.state_machine import MonsterAI, MoveState

    combat = CombatState(player_hp=80, player_max_hp=80, deck=[], rng_seed=3)
    creature = Creature(max_hp=37, monster_id="ALPHA",
                        min_initial_hp=30, max_initial_hp=40)
    ai = MonsterAI(
        states={"M": MoveState(state_id="M", intents=[],
                               effect_fn=lambda combat: None)},
        initial_state_id="M",
    )
    combat.add_enemy(creature, ai)

    assert creature.max_hp == 37
    assert creature.current_hp == 37
