"""Ironclad card-model parity tests for combat edge cases and owner behavior."""

import sts2_env.powers  # noqa: F401

from sts2_env.cards.ironclad import (
    create_ironclad_starter_deck,
    make_aggression,
    make_anger,
    make_barricade,
    make_bloodletting,
    make_blood_wall,
    make_body_slam,
    make_brand,
    make_break,
    make_breakthrough,
    make_bludgeon,
    make_cinder,
    make_colossus,
    make_conflagration,
    make_corruption,
    make_crimson_mantle,
    make_demon_form,
    make_demonic_shield,
    make_dismantle,
    make_dominate,
    make_drum_of_battle,
    make_evil_eye,
    make_expect_a_fight,
    make_fiend_fire,
    make_fight_me,
    make_flame_barrier,
    make_forgotten_ritual,
    make_hemokinesis,
    make_howl_from_beyond,
    make_impervious,
    make_infernal_blade,
    make_inferno,
    make_iron_wave,
    make_mangle,
    make_molten_fist,
    make_pacts_end,
    make_pillage,
    make_pyre,
    make_setup_strike,
    make_spite,
    make_stomp,
    make_stampede,
    make_stone_armor,
    make_sword_boomerang,
    make_taunt,
    make_thunderclap,
    make_thrash,
    make_tremble,
    make_unmovable,
    make_unrelenting,
    make_uppercut,
    make_whirlwind,
)
from sts2_env.cards.ironclad_basic import make_defend_ironclad, make_strike_ironclad
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import CardId, CombatSide, PowerId, ValueProp
from sts2_env.core.hooks import fire_after_block_gained, fire_after_turn_end, fire_before_turn_end
from sts2_env.core.rng import Rng
from sts2_env.monsters.act1_weak import create_shrinker_beetle
from sts2_env.powers.base import PowerInstance
from sts2_env.relics.registry import create_relic_by_name
from sts2_env.run.run_state import PlayerState


COLOSSUS_BLOCK = 5
COLOSSUS_UPGRADED_BLOCK = 8
COLOSSUS_POWER_AMOUNT = 1
DOMINATE_TARGET_VULNERABLE = 3
DOMINATE_STRENGTH_GAIN = DOMINATE_TARGET_VULNERABLE
FIGHT_ME_DAMAGE_PER_HIT = 5
FIGHT_ME_HITS = 2
FIGHT_ME_SELF_STRENGTH = 2
FIGHT_ME_ENEMY_STRENGTH = 1
FIGHT_ME_UPGRADED_DAMAGE_PER_HIT = 6
FIGHT_ME_UPGRADED_SELF_STRENGTH = 3
TEST_ENEMY_HP = 100
EXHAUST_KEYWORD = "exhaust"


class _StrengthOnExhaustPower(PowerInstance):
    def __init__(self):
        super().__init__(PowerId.FEEL_NO_PAIN, 1)

    def after_card_exhausted(self, owner, card, combat):
        owner.apply_power(PowerId.STRENGTH, 1)


class _CannotHitPower(PowerInstance):
    def __init__(self):
        super().__init__(PowerId.COVERED, 1)

    def should_allow_hitting(self, owner, combat):
        return False


def _make_combat(*, extra_enemies: int = 0, seed: int = 1234) -> CombatState:
    combat = CombatState(
        player_hp=80,
        player_max_hp=80,
        deck=create_ironclad_starter_deck(),
        rng_seed=seed,
        character_id="Ironclad",
    )
    creature, ai = create_shrinker_beetle(Rng(seed))
    combat.add_enemy(creature, ai)
    for i in range(extra_enemies):
        extra_creature, extra_ai = create_shrinker_beetle(Rng(seed + 100 + i))
        combat.add_enemy(extra_creature, extra_ai)
    combat.start_combat()
    return combat


class TestIroncladCombatEdgeCardModelParity:
    def test_colossus_gains_block_and_applies_reference_power(self):
        combat = _make_combat()
        combat.hand = [make_colossus()]
        combat.energy = 1

        assert combat.play_card(0)
        assert combat.player.block == COLOSSUS_BLOCK
        assert combat.player.get_power_amount(PowerId.COLOSSUS) == COLOSSUS_POWER_AMOUNT

        upgraded_combat = _make_combat()
        upgraded_combat.hand = [make_colossus(upgraded=True)]
        upgraded_combat.energy = 1

        assert upgraded_combat.play_card(0)
        assert upgraded_combat.player.block == COLOSSUS_UPGRADED_BLOCK
        assert upgraded_combat.player.get_power_amount(PowerId.COLOSSUS) == COLOSSUS_POWER_AMOUNT

    def test_dominate_gains_strength_equal_to_target_vulnerable_and_upgrade_removes_exhaust(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.apply_power(PowerId.VULNERABLE, DOMINATE_TARGET_VULNERABLE, applier=combat.player)
        card = make_dominate()
        combat.hand = [card]
        combat.energy = 1

        assert combat.play_card(0, 0)
        assert combat.player.get_power_amount(PowerId.STRENGTH) == DOMINATE_STRENGTH_GAIN
        assert card in combat.exhaust_pile
        assert EXHAUST_KEYWORD not in make_dominate(upgraded=True).keywords

    def test_fight_me_deals_two_hits_and_applies_reference_strength(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = TEST_ENEMY_HP
        enemy.current_hp = TEST_ENEMY_HP
        combat.hand = [make_fight_me()]
        combat.energy = 2

        assert combat.play_card(0, 0)
        assert enemy.current_hp == TEST_ENEMY_HP - FIGHT_ME_DAMAGE_PER_HIT * FIGHT_ME_HITS
        assert combat.player.get_power_amount(PowerId.STRENGTH) == FIGHT_ME_SELF_STRENGTH
        assert enemy.get_power_amount(PowerId.STRENGTH) == FIGHT_ME_ENEMY_STRENGTH

        upgraded_combat = _make_combat()
        upgraded_enemy = upgraded_combat.enemies[0]
        upgraded_enemy.max_hp = TEST_ENEMY_HP
        upgraded_enemy.current_hp = TEST_ENEMY_HP
        upgraded_combat.hand = [make_fight_me(upgraded=True)]
        upgraded_combat.energy = 2

        assert upgraded_combat.play_card(0, 0)
        assert upgraded_enemy.current_hp == TEST_ENEMY_HP - FIGHT_ME_UPGRADED_DAMAGE_PER_HIT * FIGHT_ME_HITS
        assert upgraded_combat.player.get_power_amount(PowerId.STRENGTH) == FIGHT_ME_UPGRADED_SELF_STRENGTH
        assert upgraded_enemy.get_power_amount(PowerId.STRENGTH) == FIGHT_ME_ENEMY_STRENGTH

    def test_anger_adds_a_matching_copy_to_discard(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        combat.hand = [make_anger(upgraded=True)]
        combat.energy = 0

        assert combat.play_card(0, 0)
        copies = [card for card in combat.discard_pile if card.card_id == CardId.ANGER]
        assert len(copies) == 2
        assert all(card.upgraded is True for card in copies)
        assert all(card.cost == 0 for card in copies)

    def test_breakthrough_loses_hp_before_hitting_all_enemies(self):
        combat = _make_combat(extra_enemies=1)
        for enemy in combat.enemies:
            enemy.max_hp = 100
            enemy.current_hp = 100
        start_hp = combat.player.current_hp
        combat.hand = [make_breakthrough()]
        combat.energy = 1

        assert combat.play_card(0)
        assert combat.player.current_hp == start_hp - 1
        assert [enemy.current_hp for enemy in combat.enemies] == [91, 91]
        assert [event[1] for event in combat._damage_events_combat[-3:]] == [combat.player, *combat.enemies]  # noqa: SLF001

    def test_breakthrough_hits_only_hittable_enemies(self):
        combat = _make_combat(extra_enemies=1)
        blocked, hittable = combat.enemies
        blocked.max_hp = blocked.current_hp = 100
        hittable.max_hp = hittable.current_hp = 100
        blocked.powers[PowerId.COVERED] = _CannotHitPower()
        combat.hand = [make_breakthrough()]
        combat.energy = 1

        assert combat.play_card(0)
        assert blocked.current_hp == 100
        assert hittable.current_hp == 91

    def test_breakthrough_does_not_attack_after_self_damage_ends_combat(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy_hp = enemy.current_hp
        combat.player.current_hp = 1
        combat.hand = [make_breakthrough()]
        combat.energy = 1

        assert combat.play_card(0)
        assert combat.is_over
        assert enemy.current_hp == enemy_hp

    def test_blood_wall_does_not_gain_block_after_self_damage_ends_combat(self):
        combat = _make_combat()
        combat.player.current_hp = 2
        combat.hand = [make_blood_wall()]
        combat.energy = 2

        assert combat.play_card(0)
        assert combat.is_over
        assert combat.player.block == 0

    def test_setup_strike_grants_temporary_strength_until_turn_end(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        combat.hand = [make_setup_strike(upgraded=True)]
        combat.energy = 1

        assert combat.play_card(0, 0)
        assert enemy.current_hp == 91
        assert combat.player.get_power_amount(PowerId.SETUP_STRIKE) == 3
        assert combat.player.get_power_amount(PowerId.STRENGTH) == 3

        combat.player.powers[PowerId.SETUP_STRIKE].after_turn_end(combat.player, CombatSide.PLAYER, combat)

        assert combat.player.get_power_amount(PowerId.SETUP_STRIKE) == 0
        assert combat.player.get_power_amount(PowerId.STRENGTH) == 0

    def test_demonic_shield_does_not_gain_block_after_self_damage_ends_combat(self):
        combat = _make_combat()
        ally = combat.add_ally_player(
            PlayerState(player_id=2, character_id="Ironclad", max_hp=60, current_hp=60)
        )
        ally.apply_power(PowerId.DEXTERITY, 3)
        combat.player.current_hp = 1
        combat.hand = [make_demonic_shield()]
        combat.energy = 0

        assert combat.play_card(0, 0)
        assert combat.is_over
        assert ally.block == 0

    def test_unmovable_doubles_owners_card_block(self):
        combat = _make_combat()
        combat.hand = [make_unmovable(), make_defend_ironclad()]
        combat.energy = 3

        assert combat.play_card(0)
        assert combat.play_card(0)
        assert combat.player.block == 10

    def test_unmovable_does_not_double_other_players_card_block(self):
        combat = _make_combat()
        ally = combat.add_ally_player(
            PlayerState(player_id=2, character_id="Ironclad", max_hp=60, current_hp=60)
        )
        ally_state = combat.combat_player_state_for(ally)
        assert ally_state is not None
        defend = make_defend_ironclad()
        defend.owner = ally
        ally_state.hand = [defend]
        combat.player.apply_power(PowerId.UNMOVABLE, 1)
        ally_state.energy = 1

        assert combat.play_card_from_creature(ally, 0)
        assert ally.block == 5

    def test_pyre_and_stone_armor_apply_reference_power_amounts(self):
        combat = _make_combat()
        combat.hand = [make_pyre(upgraded=True), make_stone_armor(upgraded=True)]
        combat.energy = 3

        assert combat.play_card(0)
        assert combat.player.get_power_amount(PowerId.PYRE) == 2
        assert combat.max_energy == 5
        assert combat.play_card(0)
        assert combat.player.get_power_amount(PowerId.PLATING) == 6

    def test_tremble_and_taunt_apply_reference_vulnerable_and_block(self):
        tremble_combat = _make_combat()
        tremble_enemy = tremble_combat.enemies[0]
        tremble_combat.hand = [make_tremble(upgraded=True)]
        tremble_combat.energy = 1

        assert tremble_combat.play_card(0, 0)
        assert tremble_enemy.get_power_amount(PowerId.VULNERABLE) == 3

        taunt_combat = _make_combat()
        taunt_enemy = taunt_combat.enemies[0]
        taunt_combat.hand = [make_taunt(upgraded=True)]
        taunt_combat.energy = 1

        assert taunt_combat.play_card(0, 0)
        assert taunt_combat.player.block == 8
        assert taunt_enemy.get_power_amount(PowerId.VULNERABLE) == 2

    def test_unrelenting_makes_next_owner_attack_free(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        combat.hand = [make_unrelenting(), make_strike_ironclad()]
        combat.energy = 2

        assert combat.play_card(0, 0)
        assert combat.energy == 0
        assert combat.player.get_power_amount(PowerId.FREE_ATTACK) == 1
        assert combat.play_card(0, 0)
        assert combat.energy == 0
        assert combat.player.get_power_amount(PowerId.FREE_ATTACK) == 0
        assert enemy.current_hp == 82

    def test_stampede_autoplays_random_attack_before_turn_end(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        strike = make_strike_ironclad()
        defend = make_defend_ironclad()
        combat.hand = [make_stampede()]
        combat.energy = 2

        assert combat.play_card(0)
        assert combat.player.get_power_amount(PowerId.STAMPEDE) == 1

        combat.hand = [strike, defend]
        fire_before_turn_end(CombatSide.PLAYER, combat)

        assert strike not in combat.hand
        assert defend in combat.hand
        assert enemy.current_hp == 94

    def test_cinder_exhausts_top_draw_card_after_shuffle_if_needed(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        kept = make_strike_ironclad()
        to_exhaust = make_defend_ironclad()
        combat.hand = [make_cinder(), kept]
        combat.draw_pile = []
        combat.discard_pile = [to_exhaust]
        combat.energy = 2

        assert combat.play_card(0, 0)
        assert enemy.current_hp == 83
        assert combat.hand == [kept]
        assert to_exhaust in combat.exhaust_pile

    def test_thunderclap_applies_vulnerable_after_all_damage_to_survivors(self):
        combat = _make_combat(extra_enemies=1)
        first, second = combat.enemies
        first.max_hp = 4
        first.current_hp = 4
        second.max_hp = 100
        second.current_hp = 100
        combat.hand = [make_thunderclap()]
        combat.energy = 1

        assert combat.play_card(0)
        assert first.is_dead
        assert first.get_power_amount(PowerId.VULNERABLE) == 0
        assert second.current_hp == 96
        assert second.get_power_amount(PowerId.VULNERABLE) == 1

    def test_evil_eye_gains_block_twice_after_owner_exhausted_card_this_turn(self):
        combat = _make_combat()
        combat.hand = [make_evil_eye()]
        combat.energy = 1

        assert combat.play_card(0)
        assert combat.player.block == 8

        exhausted_combat = _make_combat()
        fodder = make_defend_ironclad()
        exhausted_combat.hand = [fodder]
        exhausted_combat.exhaust_card(fodder)
        exhausted_combat.hand = [make_evil_eye(upgraded=True)]
        exhausted_combat.energy = 1

        assert exhausted_combat.play_card(0)
        assert exhausted_combat.player.block == 22

    def test_evil_eye_second_block_is_skipped_if_first_block_ends_combat(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.current_hp = 5
        exhausted = make_defend_ironclad()
        exhausted.owner = combat.player
        combat.record_card_exhausted(exhausted)
        combat.player.apply_power(PowerId.JUGGERNAUT, 5)
        combat.hand = [make_evil_eye()]
        combat.energy = 1

        assert combat.play_card(0)
        assert combat.is_over
        assert combat.player.block == 8

    def test_drum_of_battle_draws_then_applies_power(self):
        combat = _make_combat()
        drawn = [make_strike_ironclad(), make_defend_ironclad(), make_strike_ironclad()]
        combat.hand = [make_drum_of_battle(upgraded=True)]
        combat.draw_pile = list(drawn)
        combat.energy = 0

        assert combat.play_card(0)
        assert combat.hand == drawn
        assert combat.player.get_power_amount(PowerId.DRUM_OF_BATTLE) == 1

    def test_conflagration_is_four_hits_of_two_and_does_not_scale(self):
        """It never scaled with attacks played -- that mechanic is not in the game.

        This test used to assert the opposite, and pinned an invented card in
        place for as long as it stood. `Conflagration.cs` is a flat
        `DamageVar(2m)` with `RepeatVar(4)` against all opponents; the simulator
        dealt `8 + 2 * attacks_played_this_turn` as a single hit. Playing two
        Strikes first must now change nothing about what Conflagration does.
        """
        combat = _make_combat(extra_enemies=1)
        for enemy in combat.enemies:
            enemy.max_hp = 100
            enemy.current_hp = 100
        combat.hand = [make_strike_ironclad(), make_strike_ironclad(), make_conflagration()]
        combat.energy = 3

        assert combat.play_card(0, 0)
        assert combat.play_card(0, 0)
        before = [enemy.current_hp for enemy in combat.enemies]
        assert combat.play_card(0)
        dealt = [b - e.current_hp for b, e in zip(before, combat.enemies)]
        assert dealt == [8, 8], "2 damage x 4 hits to every enemy, whatever came before"

    def test_conflagration_multiplies_strength_once_per_hit(self):
        """Four hits take Strength four times -- the reason the shape matters.

        At +2 Strength the real card deals 16 where a single 8-damage hit would
        deal 10, so the old model under-valued it in exactly the deck built to
        use it.
        """
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = enemy.current_hp = 100
        combat.player.apply_power(PowerId.STRENGTH, 2, applier=combat.player)
        combat.hand = [make_conflagration()]
        combat.energy = 1

        assert combat.play_card(0)
        assert enemy.current_hp == 84, "4 x (2 + 2 Strength)"

    def test_conflagration_hits_only_hittable_enemies(self):
        combat = _make_combat(extra_enemies=1)
        blocked, hittable = combat.enemies
        blocked.max_hp = blocked.current_hp = 100
        hittable.max_hp = hittable.current_hp = 100
        blocked.powers[PowerId.COVERED] = _CannotHitPower()
        combat.hand = [make_strike_ironclad(), make_conflagration()]
        combat.energy = 2

        assert combat.play_card(0, 1)
        assert combat.play_card(0)

        assert blocked.current_hp == 100
        # Strike for 6, then Conflagration for 2 x 4.
        assert hittable.current_hp == 100 - 6 - 8

    def test_fiend_fire_exhausts_all_hand_cards_before_damage_hits(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        combat.player.powers[PowerId.FEEL_NO_PAIN] = _StrengthOnExhaustPower()
        combat.hand = [make_fiend_fire(), make_defend_ironclad(), make_defend_ironclad()]
        combat.energy = 2

        assert combat.play_card(0, 0)
        assert enemy.current_hp == 82
        assert combat.player.get_power_amount(PowerId.STRENGTH) == 3

    def test_crimson_mantle_increments_self_damage_when_played(self):
        combat = _make_combat()
        start_hp = combat.player.current_hp
        combat.hand = [make_crimson_mantle()]
        combat.energy = 1

        assert combat.play_card(0)
        combat.player.block = 0
        combat._start_player_turn()  # noqa: SLF001
        assert combat.player.current_hp == start_hp - 1
        assert combat.player.block == 8

    def test_thrash_exhausts_random_attack_and_adds_its_damage(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        thrash = make_thrash()
        defend = make_defend_ironclad()
        strike = make_strike_ironclad()
        combat.hand = [thrash, defend, strike]
        combat.energy = 1

        assert combat.play_card(0, 0)
        assert enemy.current_hp == 92
        assert combat.hand == [defend]
        assert strike in combat.exhaust_pile
        assert thrash.base_damage == 10

    def test_thrash_absorbed_damage_ignores_target_vulnerable_and_paper_phrog(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        enemy.apply_power(PowerId.VULNERABLE, 1, applier=combat.player)
        combat.current_player_state.relics.append(create_relic_by_name("PaperPhrog"))
        thrash = make_thrash()
        strike = make_strike_ironclad()
        combat.hand = [thrash, strike]
        combat.energy = 1

        assert combat.play_card(0, 0)
        assert enemy.current_hp == 86
        assert thrash.base_damage == 10

    def test_thrash_absorbed_damage_keeps_owner_strength(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        combat.player.apply_power(PowerId.STRENGTH, 2)
        thrash = make_thrash()
        strike = make_strike_ironclad()
        combat.hand = [thrash, strike]
        combat.energy = 1

        assert combat.play_card(0, 0)
        assert enemy.current_hp == 88
        assert thrash.base_damage == 12

    def test_body_slam_uses_current_block_as_damage(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        combat.player.gain_block(17)
        combat.hand = [make_body_slam()]
        combat.energy = 1

        assert combat.play_card(0, 0)
        assert enemy.current_hp == 83

    def test_bludgeon_uses_original_damage_values(self):
        assert make_bludgeon().base_damage == 32
        assert make_bludgeon(upgraded=True).base_damage == 42

        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        combat.hand = [make_bludgeon(upgraded=True)]
        combat.energy = 3

        assert combat.play_card(0, 0)
        assert enemy.current_hp == 58

    def test_break_deals_damage_then_applies_vulnerable(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        combat.hand = [make_break(upgraded=True)]
        combat.energy = 2

        assert combat.play_card(0, 0)
        assert enemy.current_hp == 75
        assert enemy.get_power_amount(PowerId.VULNERABLE) == 7

    def test_break_does_not_apply_vulnerable_after_killing_target(self):
        combat = _make_combat(extra_enemies=1)
        first, second = combat.enemies
        first.max_hp = 20
        first.current_hp = 20
        second.max_hp = 100
        second.current_hp = 100
        combat.hand = [make_break()]
        combat.energy = 2

        assert combat.play_card(0, 0)
        assert not combat.is_over
        assert first.escaped
        assert first.get_power_amount(PowerId.VULNERABLE) == 0
        assert second.get_power_amount(PowerId.VULNERABLE) == 0

    def test_flame_barrier_gains_block_and_retaliation_power(self):
        combat = _make_combat()
        combat.hand = [make_flame_barrier(upgraded=True)]
        combat.energy = 2

        assert combat.play_card(0)
        assert combat.player.block == 16
        assert combat.player.get_power_amount(PowerId.FLAME_BARRIER) == 6

    def test_barricade_applies_single_block_retention_power(self):
        combat = _make_combat()
        combat.player.block = 12
        combat.hand = [make_barricade(upgraded=True)]
        combat.energy = 2

        assert combat.play_card(0)
        assert combat.player.get_power_amount(PowerId.BARRICADE) == 1
        combat.player.clear_block(combat)
        assert combat.player.block == 12

    def test_demon_form_applies_original_strength_gain_power(self):
        combat = _make_combat()
        combat.hand = [make_demon_form(upgraded=True)]
        combat.energy = 3

        assert combat.play_card(0)
        assert combat.player.get_power_amount(PowerId.DEMON_FORM) == 3
        combat.player.powers[PowerId.DEMON_FORM].after_side_turn_start(
            combat.player,
            CombatSide.PLAYER,
            combat,
        )
        assert combat.player.get_power_amount(PowerId.STRENGTH) == 3

    def test_corruption_applies_skill_cost_and_exhaust_rules(self):
        combat = _make_combat()
        combat.hand = [make_corruption(upgraded=True)]
        combat.energy = 2

        assert combat.play_card(0)
        assert combat.player.get_power_amount(PowerId.CORRUPTION) == 1

        skill = make_defend_ironclad()
        combat.hand = [skill]
        combat.energy = 0

        assert combat.modified_card_cost(combat.player, skill) == 0
        assert combat.play_card(0)
        assert skill in combat.exhaust_pile

    def test_infernal_blade_adds_random_ironclad_attack_and_makes_it_free_this_turn(self):
        combat = _make_combat()
        combat.hand = [make_infernal_blade()]
        combat.energy = 1

        assert combat.play_card(0)
        assert len(combat.hand) == 1
        generated = combat.hand[0]
        assert generated.card_type.name == "ATTACK"
        assert generated.card_id != CardId.INFERNAL_BLADE
        assert generated.cost == 0

    def test_iron_wave_gains_block_then_deals_damage(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        combat.hand = [make_iron_wave(upgraded=True)]
        combat.energy = 1

        assert combat.play_card(0, 0)
        assert combat.player.block == 7
        assert enemy.current_hp == 93

    def test_sword_boomerang_hits_random_enemies_expected_number_of_times(self):
        combat = _make_combat(extra_enemies=1)
        card = make_sword_boomerang(upgraded=True)
        assert card.effect_vars["repeat"] == 4
        combat.hand = [card]
        combat.energy = 1
        starting = [enemy.current_hp for enemy in combat.enemies]

        assert combat.play_card(0)
        total_damage = sum(before - enemy.current_hp for before, enemy in zip(starting, combat.enemies, strict=True))
        assert total_damage == 12

    def test_sword_boomerang_random_hits_use_only_hittable_enemies(self):
        combat = _make_combat(extra_enemies=1)
        blocked, hittable = combat.enemies
        blocked.max_hp = blocked.current_hp = 100
        hittable.max_hp = hittable.current_hp = 100
        blocked.powers[PowerId.COVERED] = _CannotHitPower()
        combat.hand = [make_sword_boomerang()]
        combat.energy = 1

        assert combat.play_card(0)

        assert blocked.current_hp == 100
        assert hittable.current_hp == 91

    def test_whirlwind_hits_only_hittable_enemies(self):
        combat = _make_combat(extra_enemies=1)
        blocked, hittable = combat.enemies
        blocked.max_hp = blocked.current_hp = 100
        hittable.max_hp = hittable.current_hp = 100
        blocked.powers[PowerId.COVERED] = _CannotHitPower()
        combat.hand = [make_whirlwind()]
        combat.energy = 2

        assert combat.play_card(0)

        assert blocked.current_hp == 100
        assert hittable.current_hp == 90

    def test_uppercut_deals_damage_and_applies_both_debuffs(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        combat.hand = [make_uppercut(upgraded=True)]
        combat.energy = 2

        assert combat.play_card(0, 0)
        assert enemy.current_hp == 87
        assert enemy.get_power_amount(PowerId.WEAK) == 2
        assert enemy.get_power_amount(PowerId.VULNERABLE) == 2

    def test_hemokinesis_self_damage_and_attack_values_match_reference(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        hp_before = combat.player.current_hp
        combat.hand = [make_hemokinesis()]
        combat.energy = 1

        assert combat.play_card(0, 0)
        assert combat.player.current_hp == hp_before - 2
        assert enemy.current_hp == 86

    def test_inferno_increments_start_turn_self_damage_when_played(self):
        combat = _make_combat()
        start_hp = combat.player.current_hp
        combat.hand = [make_inferno()]
        combat.energy = 1

        assert combat.play_card(0)
        power = combat.player.powers[PowerId.INFERNO]
        assert power.amount == 6
        assert power.self_damage == 1

        combat.player.block = 0
        combat._start_player_turn()  # noqa: SLF001
        assert combat.player.current_hp == start_hp - 1

    def test_inferno_damage_hits_only_hittable_enemies(self):
        combat = _make_combat(extra_enemies=1)
        blocked, hittable = combat.enemies
        blocked.current_hp = blocked.max_hp = 100
        hittable.current_hp = hittable.max_hp = 100
        blocked.powers[PowerId.COVERED] = _CannotHitPower()
        combat.apply_power_to(combat.player, PowerId.INFERNO, 6)
        combat.player.block = 0

        combat.deal_damage(
            dealer=blocked,
            target=combat.player,
            amount=1,
            props=ValueProp.UNBLOCKABLE | ValueProp.UNPOWERED,
        )

        assert blocked.current_hp == 100
        assert hittable.current_hp == 94

    def test_molten_fist_duplicates_existing_vulnerable_only(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        combat.hand = [make_molten_fist()]
        combat.energy = 1

        assert combat.play_card(0, 0)
        assert enemy.get_power_amount(PowerId.VULNERABLE) == 0

        doubled_combat = _make_combat()
        enemy = doubled_combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        enemy.apply_power(PowerId.VULNERABLE, 2)
        doubled_combat.hand = [make_molten_fist()]
        doubled_combat.energy = 1

        assert doubled_combat.play_card(0, 0)
        assert enemy.get_power_amount(PowerId.VULNERABLE) == 4

    def test_mangle_applies_temporary_strength_loss_then_restores(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        enemy.apply_power(PowerId.STRENGTH, 4)
        combat.hand = [make_mangle(upgraded=True)]
        combat.energy = 3

        assert combat.play_card(0, 0)
        assert enemy.current_hp == 80
        assert enemy.get_power_amount(PowerId.MANGLE) == 15
        assert enemy.get_power_amount(PowerId.STRENGTH) == -11

        fire_after_turn_end(CombatSide.ENEMY, combat)

        assert enemy.get_power_amount(PowerId.MANGLE) == 0
        assert enemy.get_power_amount(PowerId.STRENGTH) == 4

    def test_impervious_grants_thirty_block_and_exhausts(self):
        combat = _make_combat()
        combat.hand = [make_impervious()]
        combat.energy = 2

        assert combat.play_card(0)
        assert combat.player.block == 30
        assert any(card.card_id == CardId.IMPERVIOUS for card in combat.exhaust_pile)

    def test_howl_from_beyond_autoplays_from_exhaust_before_hand_draw(self):
        combat = _make_combat(extra_enemies=1)
        for enemy in combat.enemies:
            enemy.max_hp = 100
            enemy.current_hp = 100
        card = make_howl_from_beyond()
        card.owner = combat.player
        combat.hand = []
        combat.draw_pile = []
        combat.exhaust_pile = [card]

        combat._apply_card_before_hand_draw(combat.player)  # noqa: SLF001

        assert [enemy.current_hp for enemy in combat.enemies] == [84, 84]
        assert card in combat.discard_pile

    def test_stomp_cost_drops_for_owner_attacks_played_this_turn(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        stomp = make_stomp()
        combat.hand = [make_strike_ironclad(), make_strike_ironclad()]
        combat.energy = 2

        assert combat.play_card(0, 0)
        combat.move_card_to_creature_hand(combat.player, stomp)
        assert stomp.cost == 2

        assert combat.play_card(0, 0)
        assert stomp.cost == 1

    def test_stomp_entering_on_enemy_turn_ignores_player_turn_attacks(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        stomp = make_stomp()
        combat.hand = [make_strike_ironclad()]
        combat.energy = 1

        assert combat.play_card(0, 0)

        combat.current_side = CombatSide.ENEMY
        combat._reset_side_turn_history()  # noqa: SLF001
        assert combat.current_side == CombatSide.ENEMY

        combat.move_card_to_creature_hand(combat.player, stomp)

        assert stomp.cost == stomp.original_cost
        assert enemy.is_alive

    def test_pillage_draws_until_non_attack(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        first = make_strike_ironclad()
        second = make_strike_ironclad()
        stop = make_defend_ironclad()
        remaining = make_strike_ironclad()
        combat.hand = [make_pillage()]
        combat.draw_pile = [first, second, stop, remaining]
        combat.energy = 1

        assert combat.play_card(0, 0)
        assert enemy.current_hp == 94
        assert combat.hand == [first, second, stop]
        assert combat.draw_pile == [remaining]

    def test_pillage_continues_when_drawn_attack_leaves_hand(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        enemy.max_hp = 100
        enemy.current_hp = 100
        strike = make_strike_ironclad()
        stop = make_defend_ironclad()
        remaining = make_strike_ironclad()
        combat.hand = [make_pillage()]
        combat.draw_pile = [strike, stop, remaining]
        combat.energy = 1
        combat.player.apply_power(PowerId.HELLRAISER, 1)

        assert combat.play_card(0, 0)
        assert strike not in combat.hand
        assert stop in combat.hand
        assert combat.draw_pile == [remaining]

    def test_spite_draws_only_after_owner_took_unblocked_damage_this_turn(self):
        combat = _make_combat()
        enemy = combat.enemies[0]
        missed_draw = make_defend_ironclad()
        combat.hand = [make_spite()]
        combat.draw_pile = [missed_draw]
        combat.energy = 0

        assert combat.play_card(0, 0)
        assert combat.hand == []
        assert combat.draw_pile == [missed_draw]

        damaged_combat = _make_combat()
        enemy = damaged_combat.enemies[0]
        drawn = make_defend_ironclad()
        damaged_combat.hand = [make_bloodletting(), make_spite()]
        damaged_combat.draw_pile = [drawn]
        damaged_combat.energy = 1

        assert damaged_combat.play_card(0)
        assert damaged_combat.play_card(0, 0)
        assert damaged_combat.hand == [drawn]

    def test_expect_a_fight_gains_energy_for_attacks_in_hand_not_skills(self):
        combat = _make_combat()
        combat.hand = [
            make_expect_a_fight(),
            make_strike_ironclad(),
            make_strike_ironclad(),
            make_defend_ironclad(),
        ]
        combat.energy = 2

        assert combat.play_card(0)
        assert combat.energy == 2

    def test_brand_still_gains_strength_when_selection_returns_none(self):
        combat = _make_combat()
        strike = make_strike_ironclad()
        defend = make_defend_ironclad()
        combat.hand = [make_brand(), strike, defend]
        combat.energy = 1
        starting_hp = combat.player.current_hp

        assert combat.play_card(0)
        assert combat.pending_choice is not None

        resolver = combat.pending_choice.resolver
        combat.pending_choice = None
        resolver([])
        combat._resume_pending_play()  # noqa: SLF001

        assert combat.player.current_hp == starting_hp - 1
        assert strike in combat.hand
        assert defend in combat.hand
        assert combat.player.powers[PowerId.STRENGTH].amount == 1

    def test_brand_does_not_open_selection_after_self_damage_ends_combat(self):
        combat = _make_combat()
        strike = make_strike_ironclad()
        combat.player.current_hp = 1
        combat.hand = [make_brand(), strike]
        combat.energy = 1

        assert combat.play_card(0)
        assert combat.is_over
        assert combat.pending_choice is None
        assert strike in combat.hand
        assert strike not in combat.exhaust_pile
        assert combat.player.get_power_amount(PowerId.STRENGTH) == 0

    def test_pacts_end_requires_three_exhausted_cards(self):
        combat = _make_combat()
        card = make_pacts_end()
        combat.hand = [card]
        combat.energy = 0
        combat.exhaust_pile = [make_strike_ironclad(), make_defend_ironclad()]

        assert combat.can_play_card(card) is False
        combat.exhaust_pile.append(make_anger())
        assert combat.can_play_card(card) is True

    def test_forgotten_ritual_only_gains_energy_after_owner_exhausted_card_this_turn(self):
        combat = _make_combat()
        combat.hand = [make_forgotten_ritual()]
        combat.energy = 1

        assert combat.play_card(0)
        assert combat.energy == 0

        exhausted_combat = _make_combat()
        fodder = make_defend_ironclad()
        exhausted_combat.hand = [fodder]
        exhausted_combat.exhaust_card(fodder)
        exhausted_combat.hand = [make_forgotten_ritual()]
        exhausted_combat.energy = 1

        assert exhausted_combat.play_card(0)
        assert exhausted_combat.energy == 3
