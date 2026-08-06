"""Tests for post-combat reward sequencing in RunManager."""

from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.cards.status import make_guilty
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import CardId, RoomType
from sts2_env.core.rng import Rng
from sts2_env.monsters.act1_weak import create_shrinker_beetle
from sts2_env.potions.base import create_potion
from sts2_env.run.run_manager import RunManager


def _won_combat(extra_card_rewards: int = 0) -> CombatState:
    combat = CombatState(
        player_hp=80,
        player_max_hp=80,
        deck=create_ironclad_starter_deck(),
        rng_seed=7,
        character_id="Ironclad",
    )
    creature, ai = create_shrinker_beetle(Rng(7))
    combat.add_enemy(creature, ai)
    combat.player_won = True
    combat.extra_card_rewards = extra_card_rewards
    return combat


def test_the_hunt_extra_reward_creates_second_card_reward_screen():
    mgr = RunManager(seed=1, character_id="Ironclad")
    mgr._combat = _won_combat(extra_card_rewards=1)
    mgr._current_room_type = RoomType.MONSTER
    mgr._run_state.potion_reward_odds.current_value = -1.0

    result = mgr._resolve_combat_end()

    assert result["player_won"] is True
    assert mgr.phase == RunManager.PHASE_CARD_REWARD
    assert len(mgr._offered_cards) == 3

    skip_one = mgr._do_card_reward_skip()
    assert skip_one["phase"] == RunManager.PHASE_CARD_REWARD
    assert len(mgr._offered_cards) == 3

    skip_two = mgr._do_card_reward_skip()
    assert skip_two["phase"] == RunManager.PHASE_MAP_CHOICE


def test_normal_victory_still_uses_single_card_reward_screen():
    mgr = RunManager(seed=2, character_id="Ironclad")
    mgr._combat = _won_combat(extra_card_rewards=0)
    mgr._current_room_type = RoomType.MONSTER
    mgr._run_state.potion_reward_odds.current_value = -1.0

    mgr._resolve_combat_end()

    assert mgr.phase == RunManager.PHASE_CARD_REWARD
    skip = mgr._do_card_reward_skip()
    assert skip["phase"] == RunManager.PHASE_MAP_CHOICE


def test_elite_victory_offers_relic_reward_object():
    """An elite victory offers a relic, whatever else it offers first.

    Setting base potion odds to 0.0 does NOT suppress the potion on an elite:
    `PotionRewardOdds.roll` adds `ELITE_BONUS * 0.50` to the return check, so
    an elite keeps a 12.5% potion chance at zero base odds. The test used to
    assume the relic came immediately after the card skip, which held only
    because seed 3 happened to miss that 12.5% window. It is the ordering that
    is incidental; the relic is the invariant.
    """
    mgr = RunManager(seed=3, character_id="Ironclad")
    mgr.run_state.potion_reward_odds.current_value = 0.0
    mgr._combat = _won_combat(extra_card_rewards=0)
    mgr._current_room_type = RoomType.ELITE

    mgr._resolve_combat_end()

    assert mgr.phase == RunManager.PHASE_CARD_REWARD
    assert mgr._offered_relic is None

    # Skip through whatever precedes the relic (a potion, on some seeds).
    for _ in range(4):
        mgr._do_card_reward_skip()
        if mgr._offered_relic is not None:
            break

    assert mgr.phase == RunManager.PHASE_CARD_REWARD
    assert mgr._offered_relic is not None


def test_combat_end_syncs_player_max_hp_back_to_run_state():
    mgr = RunManager(seed=4, character_id="Ironclad")
    mgr._combat = _won_combat(extra_card_rewards=0)
    mgr._current_room_type = RoomType.MONSTER
    mgr._run_state.potion_reward_odds.current_value = -1.0
    mgr._heal_after_combat = 0
    mgr._combat.player.max_hp = 86
    mgr._combat.player.current_hp = 70

    mgr._resolve_combat_end()

    assert mgr.run_state.player.max_hp == 86
    assert mgr.run_state.player.current_hp == 70


def test_guilty_counts_combat_victories_and_removes_itself_after_five():
    mgr = RunManager(seed=5, character_id="Ironclad")
    guilty = make_guilty()
    mgr.run_state.player.deck = [guilty]
    mgr.run_state.potion_reward_odds.current_value = -1.0
    mgr._heal_after_combat = 0

    for expected_seen in range(1, 5):
        mgr._combat = _won_combat(extra_card_rewards=0)
        mgr._current_room_type = RoomType.MONSTER
        mgr._current_room = None

        mgr._resolve_combat_end()

        assert mgr.run_state.player.deck == [guilty]
        assert guilty.effect_vars["combats_seen"] == expected_seen
        assert guilty.effect_vars["combats"] == 5 - expected_seen

    mgr._combat = _won_combat(extra_card_rewards=0)
    mgr._current_room_type = RoomType.MONSTER
    mgr._current_room = None

    mgr._resolve_combat_end()

    assert all(card.card_id != CardId.GUILTY for card in mgr.run_state.player.deck)


def test_guilty_deck_combat_end_behavior_lives_on_card_hook():
    mgr = RunManager(seed=55, character_id="Ironclad")
    guilty = make_guilty()

    for expected_seen in range(1, 5):
        assert guilty.after_combat_end_in_deck(mgr.run_state.player)
        assert guilty.effect_vars["combats_seen"] == expected_seen
        assert guilty.effect_vars["combats"] == 5 - expected_seen

    assert not guilty.after_combat_end_in_deck(mgr.run_state.player)
    assert guilty.effect_vars["combats_seen"] == 5
    assert guilty.effect_vars["combats"] == 0


def test_fruit_juice_updates_persistent_player_state_inside_combat():
    combat = _won_combat()
    combat.player_won = False
    starting_max_hp = combat.current_player_state.player_state.max_hp

    potion = create_potion("FruitJuice")
    potion.use(combat, combat.player)

    assert combat.player.max_hp == starting_max_hp + 5
    assert combat.current_player_state.player_state.max_hp == starting_max_hp + 5
