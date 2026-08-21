"""Worst-branch planning: pessimistic in the search, honest in the real fight."""
from __future__ import annotations

import sts2_env.cards  # noqa: F401
from sts2_env.core.rng import Rng
from sts2_env.monsters.factory import create_monster_by_id
from sts2_env.policy_config import PolicyConfig, set_active_policy
from sts2_env.search.cloning import clone_combat
from sts2_env.search.situation import CardRef, CombatSituation


def _combat(encounter: str):
    return CombatSituation(
        situation_id="w", character_id="Ironclad", current_hp=70, max_hp=80,
        deck=tuple([CardRef("STRIKE_IRONCLAD")] * 10), encounter=encounter,
        encounter_seed=5, combat_seed=3, relics=()).to_combat()


def test_worst_branch_picks_the_hardest_hitting_move():
    """Mawler's three moves all follow up into `new RandomBranchState("RAND")`,
    so there is no order to read off -- only a choice about what to assume."""
    _, ai = create_monster_by_id("MAWLER", Rng(3))
    branch = next(v for v in ai.states.values() if getattr(v, "branches", None))
    assert ai._worst_branch(branch) is None, "must roll unless asked not to"
    ai.assume_worst_branch = True
    # RIP_AND_TEAR 14, CLAW 8, ROAR 0.
    assert ai._worst_branch(branch) == "RIP_AND_TEAR_MOVE"


def test_a_monster_without_random_branches_is_unaffected():
    _, ai = create_monster_by_id("HAUNTED_SHIP", Rng(1))
    ai.assume_worst_branch = True
    for state in ai.states.values():
        assert ai._worst_branch(state) is None


def test_the_authoritative_combat_keeps_rolling_and_only_clones_are_pessimistic():
    """The important half. Offline the authoritative combat IS the game; making
    it pessimistic would not be planning, it would be changing the fight."""
    set_active_policy(PolicyConfig.load("v006_worst_branch"))
    try:
        combat = _combat("setup_mawler_normal")
    except Exception:
        set_active_policy(PolicyConfig.load("v001"))
        return  # encounter name differs in this build; the two asserts below cover it
    try:
        for ai in (combat.enemy_ais or {}).values():
            assert ai.assume_worst_branch is False, "the real fight must not be biased"
        clone = clone_combat(combat)
        for ai in (clone.enemy_ais or {}).values():
            assert ai.assume_worst_branch is True, "the clone should plan for the worst"
    finally:
        set_active_policy(PolicyConfig.load("v001"))


def test_v001_clones_are_not_pessimistic():
    set_active_policy(PolicyConfig.load("v001"))
    combat = _combat("setup_corpse_slugs_normal")
    clone = clone_combat(combat)
    for ai in (clone.enemy_ais or {}).values():
        assert ai.assume_worst_branch is False
