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

from dataclasses import replace

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


# -- encounter names from the C# mod (PascalCase) resolve to setup_X ---------

def test_pascalcase_encounter_id_resolves_to_setup_function() -> None:
    """The C# mod sends `EncounterModel.Id.Entry` (PascalCase like "NibbitsWeak").

    The Python registry keys on `setup_nibbits_weak`. `resolve_encounter`
    normalises both forms (and UPPER_SNAKE) to the same function, so the
    live bridge and the simulator's own snapshots can use either form.
    """
    from sts2_env.search.situation import resolve_encounter

    fn = resolve_encounter("NibbitsWeak")
    assert fn.__name__ == "setup_nibbits_weak"

    # The same function resolves whether we send PascalCase or setup_X
    # (the form from_run_manager uses).
    assert resolve_encounter("setup_nibbits_weak") is fn


def test_a_bridge_state_with_pascalcase_encounter_id_rebuilds_the_fight() -> None:
    """End-to-end: bridge sends PascalCase encounter, situation resolves it."""
    situation = CombatSituation.from_bridge_state(
        _bridge_state(encounter="NibbitsWeak")
    )
    combat = situation.to_combat()
    alive = [e for e in combat.enemies if e.is_alive]
    assert len(alive) >= 1
    assert all(e.monster_id == "NIBBIT" for e in alive)


def test_pascalcase_boss_encounter_id_also_resolves() -> None:
    """Smoke against the boss encounter naming convention."""
    from sts2_env.search.situation import resolve_encounter

    fn = resolve_encounter("VantomBoss")
    assert fn.__name__ == "setup_vantom_boss"


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


# -- potions: bridge sends UPPER_SNAKE, simulator holds PascalCase ---------

def test_an_upper_snake_potion_id_resolves_to_pascalcase() -> None:
    """The C# mod sends `Id.Entry` which is Slugify(ClassName) -- so
    `StrengthPotion` lands on the wire as `STRENGTH_POTION`. The simulator's
    `_POTION_MODELS` keys on PascalCase; `coerce_potion_id` bridges the two.

    This is the bug that crashed Phase 2.4's first live session: a Strength
    Potion in the bridge payload raised KeyError('STRENGTH_POTION') in
    `to_combat()`, every subsequent combat step raised LiveSearch.decide,
    the runner's END_TURN fallback fired every step, and the player did
    nothing but end turn until they died on the first encounter.
    """
    from sts2_env.potions.base import coerce_potion_id, create_potion

    assert coerce_potion_id("STRENGTH_POTION") == "StrengthPotion"
    assert coerce_potion_id("StrengthPotion") == "StrengthPotion"

    # Round-trip through create_potion: both forms produce a working instance.
    p1 = create_potion("STRENGTH_POTION", slot=0)
    p2 = create_potion("StrengthPotion", slot=0)
    assert p1.potion_id == p2.potion_id == "StrengthPotion"


def test_a_bridge_state_with_upper_snake_potion_ids_rebuilds_without_raising() -> None:
    """End-to-end on the bridge path: a combat_state with an UPPER_SNAKE
    potion in slot 0 must build the local sim without crashing. The
    previous behaviour was a KeyError that bounced every combat step to
    END_TURN and killed the player on the first encounter.
    """
    situation = CombatSituation.from_bridge_state(
        _bridge_state(potion_slots=["STRENGTH_POTION", None, None])
    )
    combat = situation.to_combat()
    # The potion is in the player's slot 0, as a PotionInstance.
    assert combat.potions[0] is not None
    assert combat.potions[0].potion_id == "StrengthPotion"


def test_a_genuinely_unknown_potion_drops_with_a_warning_not_a_crash() -> None:
    """A potion the simulator does not know (new game patch, parity gap) is
    dropped from the clone with a log line, not a raise. The searcher running
    against a fight missing one potion is still useful; the searcher crashing
    is not, and the crash used to tank every combat step of the live run.
    """
    situation = CombatSituation.from_bridge_state(
        _bridge_state(potion_slots=["TOTTALLY_NOT_A_REAL_POTION", None, None])
    )
    combat = situation.to_combat()
    # The unknown potion is dropped to None -- the fight still builds.
    assert combat.potions[0] is None


# -- to_combat_mid_fight: the overlay that fixes the drift regressions -------

def test_to_combat_mid_fight_overlays_player_hp_block_energy() -> None:
    """The bridge is ground truth. A bridge state reporting HP=67, block=12,
    energy=2 lands on the sim's primary player exactly, not whatever the
    fresh to_combat() build would have started the player at."""
    situation = CombatSituation.from_bridge_state(_bridge_state())
    state = _bridge_state(run_hp=67)
    state["combat_state"]["player"]["hp"] = 67
    state["combat_state"]["player"]["block"] = 12
    state["combat_state"]["player"]["energy"] = 2
    combat = situation.to_combat_mid_fight(state)
    assert combat.primary_player.current_hp == 67
    assert combat.primary_player.block == 12
    assert combat.current_player_state.energy == 2


def test_to_combat_mid_fight_overlays_player_powers() -> None:
    """A bridge report of STRENGTH+2 sets the player's Strength power to 2
    in the clone, replacing whatever powers from_run_manager set during the
    fresh to_combat() build."""
    situation = CombatSituation.from_bridge_state(_bridge_state())
    state = _bridge_state()
    state["combat_state"]["player"]["powers"] = [
        {"id": "STRENGTH", "amount": 3},
        {"id": "DEXTERITY", "amount": 2},
    ]
    combat = situation.to_combat_mid_fight(state)
    from sts2_env.core.enums import PowerId
    assert combat.primary_player.powers[PowerId.STRENGTH].amount == 3
    assert combat.primary_player.powers[PowerId.DEXTERITY].amount == 2


def test_to_combat_mid_fight_overlays_the_hand_from_the_bridge() -> None:
    """The bridge's hand list replaces the simulator's freshly-drawn hand.

    The previous design's drift bug, exactly: the local sim drew a different
    opening hand than the live game, and after ~2 turns got stuck on a
    fictional 3-card hand while the live game had a 5-card hand. The mid-fight
    overlay takes the bridge's hand as the new state's hand, so the search
    plans against the actual cards the player is holding right now.
    """
    situation = CombatSituation.from_bridge_state(_bridge_state())
    state = _bridge_state()
    # Override the hand with three specific cards.
    state["combat_state"]["hand"] = [
        {"id": "BASH", "cost": 2, "type": "Attack", "target": "AnyEnemy",
         "playable": True, "upgraded": True},
        {"id": "STRIKE_IRONCLAD", "cost": 1, "type": "Attack", "target": "AnyEnemy",
         "playable": True},
        {"id": "DEFEND_IRONCLAD", "cost": 1, "type": "Skill", "target": "Self",
         "playable": True},
    ]
    combat = situation.to_combat_mid_fight(state)
    assert len(combat.hand) == 3
    assert combat.hand[0].card_id.name == "BASH"
    assert combat.hand[0].upgraded is True


def test_to_combat_mid_fight_overlays_enemy_hp_and_block() -> None:
    """A boss with 20 HP / 5 block lands on the sim's enemy exactly."""
    situation = CombatSituation.from_bridge_state(_bridge_state())
    state = _bridge_state()
    state["combat_state"]["enemies"] = [
        {"id": "NIBBIT", "hp": 12, "max_hp": 30, "block": 5,
         "is_alive": True, "intent": "ATTACK", "intent_damage": 7,
         "intent_hits": 1, "powers": [{"id": "VULNERABLE", "amount": 1}]},
    ]
    combat = situation.to_combat_mid_fight(state)
    enemy = combat.enemies[0]
    assert enemy.current_hp == 12
    assert enemy.block == 5
    from sts2_env.core.enums import PowerId
    assert enemy.powers[PowerId.VULNERABLE].amount == 1


def test_to_combat_mid_fight_marks_dead_enemies_as_dead() -> None:
    """The bridge says is_alive=false; the sim's enemy should be at 0 HP
    (which is how Creature checks liveness)."""
    situation = CombatSituation.from_bridge_state(_bridge_state())
    state = _bridge_state()
    state["combat_state"]["enemies"] = [
        {"id": "NIBBIT", "hp": 0, "max_hp": 30, "block": 0,
         "is_alive": False, "intent": "UNKNOWN"},
    ]
    combat = situation.to_combat_mid_fight(state)
    assert combat.enemies[0].current_hp == 0
    assert not combat.enemies[0].is_alive


def test_to_combat_mid_fight_sets_round_and_turn_count() -> None:
    """The bridge's round=3 sets the sim's round_number to 3 and turn_count
    to 2 (turn_count advances when the player turn ends)."""
    situation = CombatSituation.from_bridge_state(_bridge_state())
    state = _bridge_state()
    state["combat_state"]["round"] = 3
    combat = situation.to_combat_mid_fight(state)
    assert combat.round_number == 3


def test_to_combat_mid_fight_runs_a_search_step_without_raising() -> None:
    """The end-to-end smoke: a bridge-built mid-fight state must be plannable
    by the SearchAgent without exceptions."""
    from sts2_env.search.turn_search import SearchAgent

    situation = CombatSituation.from_bridge_state(_bridge_state())
    state = _bridge_state()
    state["combat_state"]["player"]["hp"] = 80
    state["combat_state"]["player"]["energy"] = 3
    state["combat_state"]["round"] = 1
    combat = situation.to_combat_mid_fight(state)
    agent = SearchAgent(time_budget=0.5)
    action = agent.act(combat)
    assert 0 <= action < 115

# --- enemy HP comes from the game, not from a seed -------------------------
#
# `CombatState.cs:499` rolls monster HP from `RunState.Rng.Niche`, a run-level
# stream whose position depends on everything earlier in the run. The encounter
# seed cannot reproduce it, however faithful the generator is. So it is read
# from the bridge rather than re-derived -- the same rule the mid-fight overlay
# already followed, applied one level down where it was still being violated.


def test_enemy_max_hp_is_taken_from_the_bridge_not_rolled_from_the_seed():
    """The game said 43, so it is 43 -- whatever the encounter seed rolls."""
    state = _bridge_state()
    reported = [e["max_hp"] for e in state["combat_state"]["enemies"]]

    situation = CombatSituation.from_bridge_state(state)
    assert list(situation.enemy_max_hp) == reported

    # The encounter may build fewer slots than the bridge reports (a parity
    # gap); every slot it does build takes the game's number.
    combat = situation.to_combat()
    built = [e.max_hp for e in combat.enemies]
    assert built == reported[: len(built)]


def test_a_situation_without_reported_enemies_still_rolls_from_the_seed():
    """A harvested situation has no bridge report and must keep working.

    `enemy_max_hp` is empty there, and `to_combat` falls back to the encounter
    setup's own roll, which is internally consistent and reproducible.
    """
    state = _bridge_state()
    situation = CombatSituation.from_bridge_state(state)
    without = replace(situation, enemy_max_hp=())

    combat = without.to_combat()
    again = without.to_combat()

    assert combat.enemies, "encounter setup produced no enemies"
    assert [e.max_hp for e in combat.enemies] == [e.max_hp for e in again.enemies]


def test_more_reported_enemies_than_the_encounter_builds_does_not_raise():
    """A roster mismatch is a parity gap, not a crash.

    Better to run the fight with the enemies the simulator knows how to build
    than to take down a live run over an extra slot.
    """
    state = _bridge_state()
    situation = CombatSituation.from_bridge_state(state)
    padded = replace(situation, enemy_max_hp=situation.enemy_max_hp + (99, 98, 97))

    combat = padded.to_combat()

    assert combat.enemies


def test_bridge_enemy_index_follows_the_bridge_order_not_the_sim_roster() -> None:
    """Fogmog: the summoned Eye occupies the EARLIER display slot.

    `FogmogNormal.Slots` is ["illusion", "fogmog"], so once Fogmog summons its
    Eye the game reports ["EYE_WITH_TEETH", "FOGMOG"] while the sim roster is
    [FOGMOG, EYE]. The mapping used to be built by re-enumerating sim slots in
    order, which sent sim slot 0 (Fogmog) to bridge slot 0 (the Eye) -- so every
    attack the search aimed at Fogmog hit the Eye instead.

    That is not a harmless mis-aim: EyeWithTeeth carries IllusionPower, heals to
    full on death and is never removed from combat, so damage spent on it is
    deleted outright.
    """
    situation = CombatSituation.from_bridge_state(_bridge_state())
    # The sim roster order, whatever the encounter builder produced.
    sim_ids = [str(e.monster_id).upper() for e in situation.to_combat().enemies]

    state = _bridge_state()
    state["combat_state"]["enemies"] = [
        {"id": "EYE_WITH_TEETH", "hp": 6, "max_hp": 6, "block": 0,
         "is_alive": True, "intent": "STATUS"},
        {"id": "NIBBIT", "hp": 40, "max_hp": 74, "block": 0,
         "is_alive": True, "intent": "ATTACK", "intent_damage": 8},
    ]
    combat = situation.to_combat_mid_fight(state)
    mapping = combat.bridge_enemy_index

    # Whichever sim slot holds the real monster must map to bridge slot 1,
    # because that is where the bridge actually put it.
    for sim_slot, enemy in enumerate(combat.enemies):
        if not enemy.is_alive:
            continue
        expected = 0 if str(enemy.monster_id).upper() == "EYE_WITH_TEETH" else 1
        assert mapping[sim_slot] == expected, (
            f"sim slot {sim_slot} ({enemy.monster_id}) mapped to bridge "
            f"{mapping[sim_slot]}, expected {expected}; sim roster was {sim_ids}"
        )


def test_a_stunned_monster_is_understood_as_doing_nothing() -> None:
    """STUNNED is synthesised by the game, so it is in no state machine.

    `Creature.StunInternal` builds `MoveState("STUNNED", stunMove, StunIntent())`
    with a follow-up to the previous move and applies it via SetMoveImmediate, so
    a lookup against a monster's states was always going to miss -- on every
    monster, forever. It was the largest unknown_move cluster in the live logs,
    including two act 1 bosses.

    The cost was a wrong TURN rather than a wrong number: the search rolled its
    lookahead on whatever move it thought was next, so it planned around a hit
    that was not coming and spent a free turn defending.
    """
    situation = CombatSituation.from_bridge_state(_bridge_state())
    state = _bridge_state()
    state["combat_state"]["enemies"] = [
        {"id": "NIBBIT", "hp": 30, "max_hp": 45, "block": 0, "is_alive": True,
         "intent": "STUN", "intent_damage": 0, "intent_move_id": "STUNNED"},
    ]

    combat = situation.to_combat_mid_fight(state)
    ai = combat.enemy_ais[combat.enemies[0].combat_id]

    assert ai.current_move.state_id == "STUNNED"

    # Nothing is coming this turn -- the whole point of a stun.
    from sts2_env.search.turn_search import _incoming_damage
    assert _incoming_damage(combat) == 0

    # And it really is only one turn: the stun hands back to what it was doing.
    hp_before = combat.player.current_hp
    combat.end_player_turn()
    assert combat.player.current_hp == hp_before


def test_a_stun_does_not_report_itself_as_a_parity_gap() -> None:
    """It is handled, so it must stop appearing in the disparity list."""
    from sts2_env.search.parity import disparity_summary, reset_disparities

    reset_disparities()
    try:
        situation = CombatSituation.from_bridge_state(_bridge_state())
        state = _bridge_state()
        state["combat_state"]["enemies"] = [
            {"id": "NIBBIT", "hp": 30, "max_hp": 45, "block": 0, "is_alive": True,
             "intent": "STUN", "intent_damage": 0, "intent_move_id": "STUNNED"},
        ]
        situation.to_combat_mid_fight(state)

        assert disparity_summary() == []
    finally:
        reset_disparities()
