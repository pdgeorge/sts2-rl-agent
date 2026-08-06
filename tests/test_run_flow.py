"""Integration tests for run flow.

Tests RunState initialization, room transitions, act transitions,
rest site healing, and full run lifecycle.
"""

import math
import pytest

from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.cards.factory import create_card
from sts2_env.core.enums import (
    CardId, CardRarity, CardType, MapPointType, RoomType, TargetType,
)
from sts2_env.core.rng import Rng
from sts2_env.cards.base import CardInstance
from sts2_env.map.map_point import MapCoord, MapPoint
from sts2_env.run.run_state import RunState, PlayerState, UNLOCK_STATE_NUMBER_OF_RUNS_KEY
from sts2_env.run.rooms import create_room, CombatRoom, ShopRoom, RestSiteRoom, EventRoom, TreasureRoom
from sts2_env.run.rest_site import generate_rest_site_options, HealOption, SmithOption
from sts2_env.run.odds import UnknownMapPointOdds, PotionRewardOdds
from sts2_env.run.run_manager import RunManager
from sts2_env.run.modifiers import FlightModifier


class TestRunStateInitialization:
    def test_default_ironclad_stats(self):
        rs = RunState(seed=42, character_id="Ironclad")
        assert rs.player.max_hp == 80
        assert rs.player.current_hp == 80
        assert rs.player.gold == 99
        assert rs.player.max_potion_slots == 3

    def test_ascension_3_reduces_gold(self):
        rs = RunState(seed=42, ascension_level=3)
        rs.initialize_run()
        assert rs.player.gold == round(99 * 0.75)  # 74

    def test_ascension_4_reduces_potion_slots(self):
        rs = RunState(seed=42, ascension_level=4)
        rs.initialize_run()
        assert rs.player.max_potion_slots == 2

    def test_ascension_5_adds_curse(self):
        rs = RunState(seed=42, ascension_level=5)
        rs.initialize_run()
        curses = [c for c in rs.player.deck if c.card_id == CardId.ASCENDERS_BANE]
        assert len(curses) == 1
        assert curses[0].card_type == CardType.CURSE

    def test_initialize_generates_map(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        assert rs.map is not None
        assert len(rs.map.room_points()) > 0

    def test_initial_act_is_zero(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        assert rs.current_act_index == 0

    def test_visited_coords_empty_initially(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        assert len(rs.visited_map_coords) == 0


class TestPlayerState:
    def test_heal_caps_at_max(self):
        p = PlayerState(max_hp=80, current_hp=60)
        healed = p.heal(30)
        assert p.current_hp == 80
        assert healed == 20

    def test_heal_exact(self):
        p = PlayerState(max_hp=80, current_hp=70)
        healed = p.heal(5)
        assert p.current_hp == 75
        assert healed == 5

    def test_lose_hp(self):
        p = PlayerState(max_hp=80, current_hp=50)
        lost = p.lose_hp(20)
        assert p.current_hp == 30
        assert lost == 20

    def test_lose_hp_floors_at_zero(self):
        p = PlayerState(max_hp=80, current_hp=10)
        lost = p.lose_hp(30)
        assert p.current_hp == 0
        assert p.is_dead

    def test_gain_gold(self):
        p = PlayerState(gold=50)
        p.gain_gold(30)
        assert p.gold == 80

    def test_lose_gold(self):
        p = PlayerState(gold=50)
        lost = p.lose_gold(20)
        assert p.gold == 30
        assert lost == 20

    def test_lose_gold_caps_at_zero(self):
        p = PlayerState(gold=10)
        lost = p.lose_gold(20)
        assert p.gold == 0
        assert lost == 10

    def test_gain_max_hp(self):
        p = PlayerState(max_hp=80, current_hp=70)
        p.gain_max_hp(10)
        assert p.max_hp == 90
        assert p.current_hp == 80

    def test_lose_max_hp(self):
        p = PlayerState(max_hp=80, current_hp=80)
        p.lose_max_hp(10)
        assert p.max_hp == 70
        assert p.current_hp == 70

    def test_potion_slots(self):
        from sts2_env.potions.base import create_potion
        import sts2_env.potions.all  # noqa: F401
        p = PlayerState(max_potion_slots=3)
        assert p.add_potion(create_potion("FirePotion"))
        assert p.add_potion(create_potion("BlockPotion"))
        assert p.add_potion(create_potion("StrengthPotion"))
        # 4th should fail
        assert not p.add_potion(create_potion("EnergyPotion"))
        assert len(p.held_potions()) == 3

    def test_remove_potion(self):
        from sts2_env.potions.base import create_potion
        import sts2_env.potions.all  # noqa: F401
        p = PlayerState(max_potion_slots=3)
        p.add_potion(create_potion("FirePotion"))
        removed = p.remove_potion(0)
        assert removed is not None
        assert removed.potion_id == "FirePotion"
        assert len(p.held_potions()) == 0

    def test_obtain_relic_applies_after_obtained_hook(self):
        rs = RunState(seed=42, character_id="Ironclad")
        rs.initialize_run()
        starting_gold = rs.player.gold

        assert rs.player.obtain_relic("OLD_COIN")
        assert rs.player.gold == starting_gold + 300

    def test_card_added_to_deck_notifies_relic_hooks(self):
        rs = RunState(seed=42, character_id="Ironclad")
        rs.initialize_run()
        rs.player.obtain_relic("LUCKY_FYSH")
        starting_gold = rs.player.gold

        card = CardInstance(
            card_id=CardId.STRIKE_IRONCLAD,
            cost=1,
            card_type=CardType.ATTACK,
            target_type=TargetType.ANY_ENEMY,
            rarity=CardRarity.BASIC,
            base_damage=6,
        )
        rs.player.add_card_instance_to_deck(card)

        assert rs.player.gold == starting_gold + 15

    def test_gold_gain_notifies_relic_hooks(self):
        rs = RunState(seed=42, character_id="Ironclad")
        rs.initialize_run()
        rs.player.obtain_relic("DRAGON_FRUIT")
        starting_gold = rs.player.gold
        starting_max_hp = rs.player.max_hp

        rs.player.gain_gold(20)

        assert rs.player.gold == starting_gold + 20
        assert rs.player.max_hp == starting_max_hp + 1

    def test_gold_gain_can_be_blocked_by_relic_hook(self):
        rs = RunState(seed=42, character_id="Ironclad")
        rs.initialize_run()
        rs.player.obtain_relic("ECTOPLASM")
        starting_gold = rs.player.gold

        rs.player.gain_gold(20)

        assert rs.player.gold == starting_gold

    def test_card_added_to_deck_can_duplicate_via_relic_hook(self):
        rs = RunState(seed=42, character_id="Ironclad")
        rs.initialize_run()
        rs.player.obtain_relic("BING_BONG")
        starting_deck = len(rs.player.deck)

        card = CardInstance(
            card_id=CardId.STRIKE_IRONCLAD,
            cost=1,
            card_type=CardType.ATTACK,
            target_type=TargetType.ANY_ENEMY,
            rarity=CardRarity.BASIC,
            base_damage=6,
        )
        rs.player.add_card_instance_to_deck(card)

        assert len(rs.player.deck) == starting_deck + 2

    def test_bing_bong_duplicate_still_notifies_other_card_added_relics(self):
        rs = RunState(seed=42, character_id="Ironclad")
        rs.initialize_run()
        rs.player.obtain_relic("BING_BONG")
        rs.player.obtain_relic("DARKSTONE_PERIAPT")
        starting_deck = len(rs.player.deck)
        starting_max_hp = rs.player.max_hp

        rs.player.add_card_instance_to_deck(create_card(CardId.INJURY))

        assert len(rs.player.deck) == starting_deck + 2
        assert len({card.instance_id for card in rs.player.deck}) == len(rs.player.deck)
        assert rs.player.max_hp == starting_max_hp + 12


class TestMapNavigation:
    def test_available_coords_at_start(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        coords = rs.get_available_next_coords()
        assert len(coords) >= 2  # at least 2 starting paths

    def test_available_coords_all_row_1(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        coords = rs.get_available_next_coords()
        for c in coords:
            assert c.row == 1

    def test_visit_coord_updates_state(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        coords = rs.get_available_next_coords()
        assert rs.add_visited_coord(coords[0]) is True
        assert len(rs.visited_map_coords) == 1
        assert rs.act_floor == coords[0].row + 1
        assert rs.total_floor == 1

    def test_visit_coord_ignores_duplicate_coords(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        coord = rs.get_available_next_coords()[0]

        assert rs.add_visited_coord(coord, room_type=RoomType.MONSTER) is True
        assert rs.add_visited_coord(coord, room_type=RoomType.MONSTER) is False

        assert rs.visited_map_coords == [coord]
        assert rs.total_floor == 1
        assert len(rs.map_point_history) == 1

    def test_visit_coord_records_map_point_history(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        coord = rs.get_available_next_coords()[0]
        point = rs.map.get_point(coord)
        room_type = rs.resolve_room_type(point.point_type)

        rs.add_visited_coord(coord, room_type=room_type)

        assert len(rs.map_point_history) == 1
        assert rs.map_point_history[0].map_point_type == point.point_type
        assert rs.map_point_history[0].room_type == room_type

    def test_second_move_goes_to_row_2(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        coords = rs.get_available_next_coords()
        rs.add_visited_coord(coords[0])
        next_coords = rs.get_available_next_coords()
        for c in next_coords:
            assert c.row == 2

    def test_flight_modifier_allows_any_next_row_room(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        rs.modifiers = [FlightModifier()]
        first_coord = rs.get_available_next_coords()[0]
        rs.add_visited_coord(first_coord)

        assert set(rs.get_available_next_coords()) == {point.coord for point in rs.map.get_row(2)}

    def test_full_path_to_boss(self):
        """Walk a full path from start to boss."""
        rs = RunState(seed=42)
        rs.initialize_run()

        steps = 0
        while True:
            coords = rs.get_available_next_coords()
            if not coords:
                break
            rs.add_visited_coord(coords[0])
            steps += 1
            point = rs.map.get_point(coords[0])
            if point and point.point_type == MapPointType.BOSS:
                break

        assert steps > 0
        assert rs.total_floor == steps

    def test_run_manager_ignores_duplicate_map_move(self):
        mgr = RunManager(seed=42)
        coord = mgr.get_available_actions()[0]["coord"]

        first = mgr.take_action({"action": "move", "coord": coord})
        mgr._phase = RunManager.PHASE_MAP_CHOICE
        mgr._available_coords = [MapCoord(*coord)]
        second = mgr.take_action({"action": "move", "coord": coord})

        assert first["phase"] != RunManager.PHASE_MAP_CHOICE
        assert second["phase"] == RunManager.PHASE_MAP_CHOICE
        assert "already visited" in second["description"]
        assert len(mgr.run_state.visited_map_coords) == 1
        assert mgr.run_state.total_floor == 1


class TestRoomTypeResolution:
    def test_monster_resolves_to_monster(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        assert rs.resolve_room_type(MapPointType.MONSTER) == RoomType.MONSTER

    def test_elite_resolves_to_elite(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        assert rs.resolve_room_type(MapPointType.ELITE) == RoomType.ELITE

    def test_boss_resolves_to_boss(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        assert rs.resolve_room_type(MapPointType.BOSS) == RoomType.BOSS

    def test_shop_resolves_to_shop(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        assert rs.resolve_room_type(MapPointType.SHOP) == RoomType.SHOP

    def test_rest_site_resolves_to_rest_site(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        assert rs.resolve_room_type(MapPointType.REST_SITE) == RoomType.REST_SITE

    def test_treasure_resolves_to_treasure(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        assert rs.resolve_room_type(MapPointType.TREASURE) == RoomType.TREASURE

    def test_unknown_resolves_to_valid_type(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        rs.player.unlock_state[UNLOCK_STATE_NUMBER_OF_RUNS_KEY] = 1
        result = rs.resolve_room_type(MapPointType.UNKNOWN)
        valid = {RoomType.MONSTER, RoomType.ELITE, RoomType.SHOP,
                 RoomType.TREASURE, RoomType.EVENT}
        assert result in valid

    def test_first_run_unknown_rooms_follow_tutorial_sequence(self):
        rs = RunState(seed=42)
        rs.player.unlock_state[UNLOCK_STATE_NUMBER_OF_RUNS_KEY] = 0

        assert rs.resolve_room_type(MapPointType.UNKNOWN) == RoomType.EVENT

        rs.append_to_map_point_history(MapPointType.UNKNOWN, RoomType.EVENT)
        assert rs.resolve_room_type(MapPointType.UNKNOWN) == RoomType.EVENT

        rs.append_to_map_point_history(MapPointType.UNKNOWN, RoomType.EVENT)
        assert rs.resolve_room_type(MapPointType.UNKNOWN) == RoomType.MONSTER

    def test_ancient_resolves_to_event(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        assert rs.resolve_room_type(MapPointType.ANCIENT) == RoomType.EVENT

    def test_first_run_unassigned_resolves_to_tutorial_event_once(self):
        rs = RunState(seed=42)
        rs.player.unlock_state[UNLOCK_STATE_NUMBER_OF_RUNS_KEY] = 0

        assert rs.resolve_room_type(MapPointType.UNASSIGNED) == RoomType.EVENT

        rs.append_to_map_point_history(MapPointType.UNASSIGNED, RoomType.EVENT)
        assert rs.resolve_room_type(MapPointType.UNASSIGNED) == RoomType.MONSTER

    def test_unassigned_tutorial_event_requires_no_completed_runs(self):
        rs = RunState(seed=42)
        rs.player.unlock_state[UNLOCK_STATE_NUMBER_OF_RUNS_KEY] = 1

        assert rs.resolve_room_type(MapPointType.UNASSIGNED) == RoomType.MONSTER

    def test_unknown_room_blacklists_shop_after_shop(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        rs.player.unlock_state[UNLOCK_STATE_NUMBER_OF_RUNS_KEY] = 1
        rs.append_to_map_point_history(MapPointType.SHOP, RoomType.SHOP)
        rs.unknown_odds._current[RoomType.SHOP] = 1.0

        blacklist = rs.build_room_type_blacklist()
        assert RoomType.SHOP in blacklist
        assert rs.resolve_room_type(MapPointType.UNKNOWN, blacklist=blacklist) is not RoomType.SHOP

    def test_unknown_room_blacklists_shop_when_all_next_nodes_are_shops(self):
        rs = RunState(seed=43)
        rs.initialize_run()
        rs.player.unlock_state[UNLOCK_STATE_NUMBER_OF_RUNS_KEY] = 1
        next_points = [
            MapPoint(MapCoord(0, 2), MapPointType.SHOP),
            MapPoint(MapCoord(1, 2), MapPointType.SHOP),
        ]
        rs.unknown_odds._current[RoomType.SHOP] = 1.0

        blacklist = rs.build_room_type_blacklist(next_points)

        assert blacklist == {RoomType.SHOP}
        assert rs.resolve_room_type(MapPointType.UNKNOWN, blacklist=blacklist) is not RoomType.SHOP


class TestRoomCreation:
    def test_create_monster_room(self):
        room = create_room(RoomType.MONSTER)
        assert isinstance(room, CombatRoom)
        assert room.room_type == RoomType.MONSTER

    def test_create_elite_room(self):
        room = create_room(RoomType.ELITE)
        assert isinstance(room, CombatRoom)
        assert room.is_elite

    def test_create_boss_room(self):
        room = create_room(RoomType.BOSS)
        assert isinstance(room, CombatRoom)
        assert room.is_boss

    def test_create_shop_room(self):
        room = create_room(RoomType.SHOP)
        assert isinstance(room, ShopRoom)


class TestConcreteRunObjects:
    def test_shop_purchase_adds_real_card_and_potion(self):
        mgr = RunManager(seed=64, character_id="Ironclad")
        mgr._enter_shop()
        inv = mgr._shop_inventory
        assert inv is not None
        # Whether the starting gold happens to cover the rolled prices is
        # incidental to "a purchase adds a real card and potion" -- and when
        # it did not, this failed as a bare StopIteration out of the `next`
        # below, which reads as a broken shop rather than an unaffordable one.
        mgr.run_state.player.gold = 9_999

        card_entry = next(entry for entry in inv.cards if entry.card_id and mgr.run_state.player.gold >= entry.price)
        gold_before = mgr.run_state.player.gold
        deck_before = len(mgr.run_state.player.deck)
        result = mgr._do_shop_action({"action": "buy_card", "index": inv.cards.index(card_entry)})
        assert result["phase"] == RunManager.PHASE_SHOP
        assert mgr.run_state.player.gold < gold_before
        assert len(mgr.run_state.player.deck) == deck_before + 1

        potion_entry = next(entry for entry in inv.potions if entry.potion_id and mgr.run_state.player.gold >= entry.price)
        potions_before = len(mgr.run_state.player.held_potions())
        mgr._do_shop_action({"action": "buy_potion", "index": inv.potions.index(potion_entry)})
        assert len(mgr.run_state.player.held_potions()) == potions_before + 1

    def test_shop_foul_potion_use_gains_gold_and_consumes_potion(self):
        from sts2_env.potions.base import create_potion

        mgr = RunManager(seed=65, character_id="Ironclad")
        assert mgr.run_state.player.add_potion(create_potion("FoulPotion"))
        mgr.run_state.player.gold = 12
        mgr._enter_shop()

        actions = mgr.get_available_actions()
        use_action = next(action for action in actions if action["action"] == "use_potion")
        result = mgr.take_action(use_action)

        assert result["phase"] == RunManager.PHASE_SHOP
        assert mgr.run_state.player.gold == 112
        assert mgr.run_state.player.held_potions() == []

    def test_noncombat_blood_potion_heals_target_player(self):
        from sts2_env.potions.base import create_potion

        mgr = RunManager(seed=66, character_id="Ironclad")
        mgr.run_state.player.current_hp = 40
        assert mgr.run_state.player.add_potion(create_potion("BloodPotion"))

        result = mgr.take_action({"action": "use_potion", "slot_index": 0, "target_player_id": 1})

        assert result["description"] == "Healed 16 HP."
        assert mgr.run_state.player.current_hp == 56
        assert mgr.run_state.player.held_potions() == []

    def test_noncombat_fruit_juice_gains_target_player_max_hp(self):
        from sts2_env.potions.base import create_potion

        mgr = RunManager(seed=67, character_id="Ironclad")
        assert mgr.run_state.player.add_potion(create_potion("FruitJuice"))

        result = mgr.take_action({"action": "use_potion", "slot_index": 0, "target_player_id": 1})

        assert result["description"] == "Gained 5 Max HP."
        assert mgr.run_state.player.max_hp == 85
        assert mgr.run_state.player.current_hp == 85
        assert mgr.run_state.player.held_potions() == []

    def test_noncombat_entropic_brew_fills_empty_potion_slots(self):
        from sts2_env.potions.base import create_potion

        mgr = RunManager(seed=68, character_id="Ironclad")
        assert mgr.run_state.player.add_potion(create_potion("EntropicBrew"))

        result = mgr.take_action({"action": "use_potion", "slot_index": 0})

        assert result["description"] == "Filled 3 potion slot(s)."
        assert len(mgr.run_state.player.held_potions()) == 3
        assert all(potion.potion_id != "EntropicBrew" for potion in mgr.run_state.player.held_potions())

    def test_foul_potion_cannot_be_used_outside_shop_or_fake_merchant_event(self):
        from sts2_env.potions.base import create_potion

        mgr = RunManager(seed=69, character_id="Ironclad")
        assert mgr.run_state.player.add_potion(create_potion("FoulPotion"))

        result = mgr.take_action({"action": "use_potion", "slot_index": 0})

        assert result["description"] == "Cannot use potion."
        assert mgr.run_state.player.gold == 99
        assert [potion.potion_id for potion in mgr.run_state.player.held_potions()] == ["FoulPotion"]

    def test_boss_relic_options_are_real_ids(self):
        mgr = RunManager(seed=73, character_id="Ironclad")
        mgr._enter_boss_relic()
        assert len(mgr._boss_relics) == 3
        assert all(relic_id.isupper() for relic_id in mgr._boss_relics)

    def test_rest_site_smith_uses_pending_choice_flow(self):
        mgr = RunManager(seed=75, character_id="Ironclad")
        mgr.run_state.player.deck = create_ironclad_starter_deck()
        mgr._enter_rest_site()

        result = mgr._do_rest_site({"option_id": "SMITH"})
        assert result["phase"] == RunManager.PHASE_REST_SITE

        actions = mgr.get_available_actions()
        assert any(action["action"] == "choose" for action in actions)

        final = mgr.take_action({"action": "choose", "index": 0})
        assert final["phase"] == RunManager.PHASE_MAP_CHOICE
        assert any(card.upgraded for card in mgr.run_state.player.deck)

    def test_create_rest_site_room(self):
        room = create_room(RoomType.REST_SITE)
        assert isinstance(room, RestSiteRoom)

    def test_create_treasure_room(self):
        room = create_room(RoomType.TREASURE)
        assert isinstance(room, TreasureRoom)

    def test_create_event_room(self):
        room = create_room(RoomType.EVENT)
        assert isinstance(room, EventRoom)


class TestActTransition:
    def test_enter_next_act(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        assert rs.current_act_index == 0
        result = rs.enter_next_act()
        assert result is True
        assert rs.current_act_index == 1
        assert rs.map is not None
        assert len(rs.visited_map_coords) == 0

    def test_final_act_wins_run(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        rs.enter_act(len(rs.acts) - 1)
        result = rs.enter_next_act()
        assert result is False
        assert rs.is_over
        assert rs.player_won

    def test_act_transition_resets_unknown_odds(self):
        rs = RunState(seed=42)
        rs.initialize_run()
        # Mutate unknown odds
        rs.unknown_odds._current[RoomType.MONSTER] = 0.99
        rs.enter_next_act()
        # Should be reset to base
        assert rs.unknown_odds._current[RoomType.MONSTER] == pytest.approx(
            UnknownMapPointOdds.BASE_MONSTER
        )


class TestRestSiteHealing:
    def test_heal_30_percent(self):
        p = PlayerState(max_hp=80, current_hp=50)
        options = generate_rest_site_options(p)
        heal_opt = next(o for o in options if o.option_id == "HEAL")
        result = heal_opt.execute(p)
        expected_heal = math.floor(80 * 0.3)  # 24
        assert p.current_hp == 50 + expected_heal
        assert "24" in result

    def test_heal_does_not_exceed_max(self):
        p = PlayerState(max_hp=80, current_hp=78)
        options = generate_rest_site_options(p)
        heal_opt = next(o for o in options if o.option_id == "HEAL")
        heal_opt.execute(p)
        assert p.current_hp == 80  # capped

    def test_smith_available(self):
        p = PlayerState()
        p.deck.append(CardInstance(
            card_id=CardId.STRIKE_IRONCLAD, cost=1, card_type=CardType.ATTACK,
            target_type=TargetType.ANY_ENEMY,
        ))
        options = generate_rest_site_options(p)
        smith = next(o for o in options if o.option_id == "SMITH")
        assert smith.enabled

    def test_smith_disabled_when_all_upgraded(self):
        p = PlayerState()
        card = CardInstance(
            card_id=CardId.STRIKE_IRONCLAD, cost=1, card_type=CardType.ATTACK,
            target_type=TargetType.ANY_ENEMY, upgraded=True,
        )
        p.deck.append(card)
        options = generate_rest_site_options(p)
        smith = next(o for o in options if o.option_id == "SMITH")
        assert not smith.enabled

    def test_smith_upgrades_card(self):
        p = PlayerState()
        card = CardInstance(
            card_id=CardId.STRIKE_IRONCLAD, cost=1, card_type=CardType.ATTACK,
            target_type=TargetType.ANY_ENEMY,
        )
        p.deck.append(card)
        options = generate_rest_site_options(p)
        smith = next(o for o in options if o.option_id == "SMITH")
        smith.execute(p, card_index=0)
        assert p.deck[0].upgraded

    def test_dig_available_with_shovel(self):
        p = PlayerState()
        options = generate_rest_site_options(p, relic_ids=["Shovel"])
        dig = [o for o in options if o.option_id == "DIG"]
        assert len(dig) == 1

    def test_lift_available_with_girya(self):
        p = PlayerState()
        options = generate_rest_site_options(p, relic_ids=["Girya"])
        lift = [o for o in options if o.option_id == "LIFT"]
        assert len(lift) == 1


class TestUnknownMapPointOdds:
    def test_initial_odds(self):
        odds = UnknownMapPointOdds()
        assert odds._current[RoomType.MONSTER] == pytest.approx(0.10)
        assert odds._current[RoomType.ELITE] == pytest.approx(-1.00)
        assert odds._current[RoomType.TREASURE] == pytest.approx(0.02)
        assert odds._current[RoomType.SHOP] == pytest.approx(0.03)

    def test_reset_to_base(self):
        odds = UnknownMapPointOdds()
        odds._current[RoomType.MONSTER] = 0.99
        odds.reset_to_base()
        assert odds._current[RoomType.MONSTER] == pytest.approx(0.10)

    def test_odds_shift_over_time(self):
        """Non-rolled types should increase, rolled type should reset."""
        odds = UnknownMapPointOdds()
        rs = RunState(42)
        rs.initialize_run()
        rs.player.unlock_state[UNLOCK_STATE_NUMBER_OF_RUNS_KEY] = 1
        rng = Rng(42)

        initial_monster = odds._current[RoomType.MONSTER]
        result = odds.roll(rng, rs)

        if result == RoomType.MONSTER:
            assert odds._current[RoomType.MONSTER] == pytest.approx(initial_monster)
        else:
            assert odds._current[RoomType.MONSTER] > initial_monster

    def test_blacklisted_room_type_does_not_increase_odds(self):
        odds = UnknownMapPointOdds()
        rs = RunState(42)
        rs.initialize_run()
        rs.player.unlock_state[UNLOCK_STATE_NUMBER_OF_RUNS_KEY] = 1
        rng = Rng(42)

        initial_shop = odds._current[RoomType.SHOP]

        odds.roll(rng, rs, blacklist={RoomType.SHOP})

        assert odds._current[RoomType.SHOP] == pytest.approx(initial_shop)


class TestPotionRewardOdds:
    def test_initial_value(self):
        odds = PotionRewardOdds()
        assert odds.current_value == pytest.approx(0.40)

    def test_oscillation(self):
        """Odds should swing by 0.10 each roll."""
        odds = PotionRewardOdds()
        rng = Rng(42)
        initial = odds.current_value
        got = odds.roll(rng)
        if got:
            assert odds.current_value == pytest.approx(initial - 0.10)
        else:
            assert odds.current_value == pytest.approx(initial + 0.10)

    def test_elite_bonus(self):
        """Elite rolls should have higher drop rate."""
        rng_reg = Rng(42)
        rng_elite = Rng(42)
        n = 5000
        odds_reg = PotionRewardOdds()
        odds_elite = PotionRewardOdds()

        reg_drops = sum(1 for _ in range(n) if odds_reg.roll(rng_reg, is_elite=False))
        elite_drops = sum(1 for _ in range(n) if odds_elite.roll(rng_elite, is_elite=True))

        assert elite_drops > reg_drops

    def test_forced_drop_reduces_future_odds(self):
        odds = PotionRewardOdds()
        rng = Rng(42)
        initial = odds.current_value

        assert odds.roll(rng, force=True)
        assert odds.current_value == pytest.approx(initial - 0.10)


class TestRunDeterminism:
    def test_same_seed_same_state(self):
        rs1 = RunState(seed=42)
        rs1.initialize_run()
        rs2 = RunState(seed=42)
        rs2.initialize_run()

        assert rs1.player.gold == rs2.player.gold
        assert rs1.player.max_hp == rs2.player.max_hp

        coords1 = rs1.get_available_next_coords()
        coords2 = rs2.get_available_next_coords()
        assert coords1 == coords2


class TestWinLose:
    def test_win_run(self):
        rs = RunState(seed=42)
        rs.win_run()
        assert rs.is_over
        assert rs.player_won

    def test_lose_run(self):
        rs = RunState(seed=42)
        rs.lose_run()
        assert rs.is_over
        assert not rs.player_won
