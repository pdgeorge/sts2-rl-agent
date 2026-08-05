"""A bridge state must rebuild the same fight from its JSON that the live game started.

`CombatSituation.from_run_manager` builds a reproducible fight from a
RunManager mid-simulator-run. `from_bridge_state` is its live-game
counterpart -- it reads the JSON the C# mod sends at combat_start and
produces a `CombatSituation` the SearchAgent can clone. The two must agree
on every field, because a searcher that clones a different fight from the
one on screen is worse than useless: it looks correct and is not.

This tests the JSON path against the same `to_combat` that the simulator
uses, asserting that the player, deck and enemies emerge correctly from a
realistic bridge payload.
"""

from __future__ import annotations

import pytest

from sts2_env.search.situation import CombatSituation


# -- a realistic bridge payload ------------------------------------------------
# Real mod fields (RlRunInfo.Attach) plus the Phase 1.1 additions
# (encounter, encounter_seed, combat_seed, character_id). The deck is sent
# as dicts with the upgraded flag -- the target spec -- so an upgraded Bash
# is not silently read as a base Bash on the live path.

def _bridge_state(*, encounter="setup_nibbits_weak", deck_extra=None,
                  upgraded_cards=None, no_encounter=False, **overrides):
    """Build a bridge combat_action state, field by field.

    The defaults match what RlRunInfo.Attach writes, plus the three
    encounter-identification fields the Phase 1.1 patch adds. The deck
    defaults to the actual Ironclad starter (5 Strike, 4 Defend, 1 Bash) so
    the test situation mirrors a real floor-2 hallway fight.
    """
    deck = list(deck_extra or ["STRIKE_IRONCLAD", "STRIKE_IRONCLAD",
                               "STRIKE_IRONCLAD", "STRIKE_IRONCLAD",
                               "STRIKE_IRONCLAD", "DEFEND_IRONCLAD",
                               "DEFEND_IRONCLAD", "DEFEND_IRONCLAD",
                               "DEFEND_IRONCLAD", "BASH"])
    if upgraded_cards is not None:
        assert len(upgraded_cards) == len(deck)
        deck = [
            {"id": cid, "upgraded": up}
            for cid, up in zip(deck, upgraded_cards)
        ]

    state = {
        "type": "combat_action",
        "floor": 2,
        "act": 1,
        "act_floor": 2,
        "room_type": "Monster",
        "run_hp": 72,
        "run_max_hp": 80,
        "gold": 99,
        "deck_size": len(deck),
        "relic_count": 1,
        "potion_count": 0,
        "max_potion_slots": 3,
        "ascension": 0,
        "relics": ["BURNING_BLOOD"],
        "potion_slots": [None, None, None],
        "deck": deck,
        "character_id": "Ironclad",
        "encounter": encounter,
        "encounter_seed": 4242,
        "combat_seed": 4242,
        "combat_state": {
            "player": {"hp": 72, "max_hp": 80, "block": 0,
                       "energy": 3, "max_energy": 3},
            "hand": [],
            "enemies": [{"id": "NIBBIT", "hp": 16, "max_hp": 16,
                         "is_alive": True},
                        {"id": "NIBBIT", "hp": 16, "max_hp": 16,
                         "is_alive": True}],
        },
    }
    if no_encounter:
        del state["encounter"]
    state.update(overrides)
    return state


# -- the round trip: JSON -> situation -> a real fight ------------------------

def test_to_combat_rebuilds_the_enemies_from_the_encounter_and_seed() -> None:
    situation = CombatSituation.from_bridge_state(_bridge_state())
    combat = situation.to_combat()

    # setup_nibbits_weak spawns at least one Nibbit. The seed fixes the HP
    # roll, so two rebuilds from the same situation agree exactly (pinned in
    # the next test). Here we just assert the encounter ran.
    alive = [e for e in combat.enemies if e.is_alive]
    assert len(alive) >= 1
    assert all(e.monster_id == "NIBBIT" for e in alive)


def test_the_player_starts_with_the_hp_the_bridge_reported() -> None:
    situation = CombatSituation.from_bridge_state(_bridge_state(run_hp=55))
    combat = situation.to_combat()
    assert combat.player.current_hp == 55
    assert combat.player.max_hp == 80


def test_the_deck_is_built_from_the_bridge_deck_list() -> None:
    deck_list = ["STRIKE_IRONCLAD", "BASH", "ANGER", "CLASH", "DEFEND_IRONCLAD"]
    situation = CombatSituation.from_bridge_state(
        _bridge_state(deck_extra=deck_list, upgraded_cards=None)
    )
    assert tuple(d.card_id for d in situation.deck) == tuple(deck_list)
    assert all(d.upgraded is False for d in situation.deck)

    combat = situation.to_combat()
    # After start_combat the deck is split between hand and draw_pile; both
    # together, plus discard/exhaust, must sum to the deck the bridge sent.
    total = (len(combat.hand) + len(combat.draw_pile)
             + len(combat.discard_pile) + len(combat.exhaust_pile))
    assert total == len(deck_list)


def test_upgraded_cards_are_preserved_when_the_mod_sends_dicts() -> None:
    deck_list = ["STRIKE_IRONCLAD", "BASH", "DEFEND_IRONCLAD"]
    situation = CombatSituation.from_bridge_state(
        _bridge_state(
            deck_extra=deck_list,
            upgraded_cards=[False, True, False],
        )
    )
    assert situation.deck[0].upgraded is False
    assert situation.deck[1].upgraded is True
    assert situation.deck[2].upgraded is False


def test_the_relics_arrive_intact() -> None:
    situation = CombatSituation.from_bridge_state(
        _bridge_state(relics=["BURNING_BLOOD", "FISHING_ROD"])
    )
    assert tuple(situation.relics) == ("BURNING_BLOOD", "FISHING_ROD")


def test_the_potion_slots_keep_their_positions() -> None:
    situation = CombatSituation.from_bridge_state(
        _bridge_state(potion_slots=["FIRE_POTION", None, None])
    )
    assert tuple(situation.potions) == ("FIRE_POTION", None, None)


def test_room_type_and_floor_fields_round_trip() -> None:
    situation = CombatSituation.from_bridge_state(
        _bridge_state(floor=17, act_floor=17, room_type="Boss",
                      encounter="setup_vantom_boss")
    )
    assert situation.total_floor == 17
    assert situation.act_floor == 17
    assert situation.room_type == "BOSS"


def test_character_id_defaults_to_ironclad_when_absent() -> None:
    """The current mod hardcodes Ironclad and does not send character_id."""
    state = _bridge_state()
    del state["character_id"]
    situation = CombatSituation.from_bridge_state(state)
    assert situation.character_id == "Ironclad"


# -- the encounter-identification gap is loud, not quiet -----------------------

def test_missing_encounter_raises_rather_than_falling_back() -> None:
    with pytest.raises(ValueError, match="encounter"):
        CombatSituation.from_bridge_state(_bridge_state(no_encounter=True))


def test_missing_encounter_seed_defaults_to_zero() -> None:
    state = _bridge_state()
    del state["encounter_seed"]
    situation = CombatSituation.from_bridge_state(state)
    assert situation.encounter_seed == 0


def test_missing_combat_seed_falls_back_to_encounter_seed() -> None:
    """The mod could send one seed rather than two; combat_seed then matches."""
    state = _bridge_state(encounter_seed=999)
    del state["combat_seed"]
    situation = CombatSituation.from_bridge_state(state)
    assert situation.combat_seed == 999


# -- the same seed produces the same fight, twice -----------------------------

def test_two_rebuilds_of_one_situation_agree_on_enemy_hp() -> None:
    """Reproducibility is the property a benchmark rests on.

    Two calls to `to_combat` from the same `CombatSituation` must produce
    enemies with the same HP, which is also what the search path relies on
    when it clones the fight every turn.
    """
    situation = CombatSituation.from_bridge_state(_bridge_state(encounter_seed=7))
    combat_a = situation.to_combat()
    combat_b = situation.to_combat()
    hp_a = sorted(e.current_hp for e in combat_a.enemies)
    hp_b = sorted(e.current_hp for e in combat_b.enemies)
    assert hp_a == hp_b


# -- a known-card deck round-trips through from_bridge_state -------------------

def test_running_the_simulator_from_a_bridge_state_does_not_raise() -> None:
    """The end-to-end smoke: a situation built from JSON must play a turn."""
    situation = CombatSituation.from_bridge_state(_bridge_state())
    combat = situation.to_combat()

    from sts2_env.gym_env.action_space import get_action_mask
    mask = get_action_mask(combat)
    # The starter deck always has at least one playable card on turn 1.
    assert mask.any(), "no legal actions on the first turn of a fresh fight"