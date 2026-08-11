"""Potion use-effect implementations for all 63 potions.

Each effect function has the signature:
    (combat: CombatState, user: Creature, target: Creature | None) -> None

Matches the OnUse methods from decompiled MegaCrit.Sts2.Core.Models.Potions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sts2_env.cards.factory import create_cards_from_ids, create_character_cards, eligible_registered_cards
from sts2_env.core.card_pools import CardPoolId
from sts2_env.core.enums import (
    CardRarity, CardType, OrbType, PowerId, ValueProp, PotionTargetType, CardId,
)
from sts2_env.core.constants import PERCENT_DENOMINATOR
from sts2_env.core.damage import calculate_damage, apply_damage
from sts2_env.potions.all import (
    AMBERGRIS_HEAL_PERCENT,
    BLOOD_POTION_HEAL_PERCENT,
    BLOOD_POTION_ID,
    ENTROPIC_BREW_ID,
    FOUL_POTION_ID,
    FRUIT_JUICE_ID,
    FRUIT_JUICE_MAX_HP,
)
from sts2_env.potions.base import register_potion_effect

if TYPE_CHECKING:
    from sts2_env.core.creature import Creature
    from sts2_env.core.combat import CombatState


# ─── Helpers ──────────────────────────────────────────────────────────

def _gain_unpowered_block(target: Creature, amount: int, combat: CombatState) -> int:
    before = target.block
    target.gain_block(amount, unpowered=True)
    gained = target.block - before
    if gained > 0:
        from sts2_env.core.hooks import fire_after_block_gained

        fire_after_block_gained(target, gained, combat)
    return gained


def _deal_unpowered_damage(
    combat: CombatState, dealer: Creature, target: Creature, base_damage: int,
) -> None:
    """Deal unpowered damage (no Str/Dex scaling) to a single target."""
    final = calculate_damage(base_damage, dealer, target, ValueProp.UNPOWERED, combat)
    apply_damage(target, final, ValueProp.UNPOWERED, combat=combat, dealer=dealer)


def _deal_unpowered_damage_all(
    combat: CombatState, dealer: Creature, base_damage: int,
) -> None:
    """Deal unpowered damage to all hittable enemies."""
    for enemy in list(combat.hittable_enemies):
        _deal_unpowered_damage(combat, dealer, enemy, base_damage)


_SOURCE_CARD_RARITY_ORDER = {
    CardRarity.BASIC: 1,
    CardRarity.COMMON: 2,
    CardRarity.UNCOMMON: 3,
    CardRarity.RARE: 4,
    CardRarity.ANCIENT: 5,
    CardRarity.STATUS: 6,
    CardRarity.CURSE: 7,
    CardRarity.EVENT: 8,
    CardRarity.QUEST: 9,
}

ENERGY_POTION_ENERGY_GAIN = 2
SWIFT_POTION_DRAW_COUNT = 3
CURE_ALL_ENERGY_GAIN = 1
CURE_ALL_DRAW_COUNT = 2
RADIANT_TINCTURE_ENERGY_GAIN = 1
RADIANT_TINCTURE_RADIANCE = 3
SNECKO_OIL_DRAW_COUNT = 7
SNECKO_OIL_MIN_COST = 0
SNECKO_OIL_MAX_COST = 3
SOLDIERS_STEW_REPLAY_GAIN = 1
BOTTLED_POTENTIAL_DRAW_COUNT = 5
GLOWWATER_DRAW_COUNT = 10


def _source_card_order(card) -> tuple[int, str]:
    return (_SOURCE_CARD_RARITY_ORDER[card.rarity], card.card_id.name)


# =====================================================================
#  COMMON POTIONS (20)
# =====================================================================

def _attack_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Generate 3 random Attack cards; add one with cost 0 this turn to hand.
    """
    state = combat.combat_player_state_for(user)
    if state is None:
        return
    generated = create_character_cards(
        state.character_id,
        combat.combat_card_generation_rng,
        3,
        card_type=CardType.ATTACK,
        distinct=True,
        generation_context="combat",
        is_multiplayer=combat.is_multiplayer,
    )
    for generated_card in generated:
        generated_card.owner = user
        generated_card.set_temporary_free_this_turn()
    if generated:
        combat.request_card_choice(
            prompt="Choose an Attack card",
            cards=generated,
            source_pile="generated",
            resolver=combat.move_card_to_hand,
            allow_skip=True,
            owner=user,
        )


def _block_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 12 block (unpowered)."""
    t = target if target is not None else user
    _gain_unpowered_block(t, 12, combat)


def _blood_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Heal target for 20% of max HP."""
    t = target if target is not None else user
    amount = t.max_hp * BLOOD_POTION_HEAL_PERCENT // PERCENT_DENOMINATOR
    t.heal(amount)


def _colorless_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Generate a random Colorless card (cost 0) in hand.
    """
    state = combat.combat_player_state_for(user)
    if state is None:
        return
    colorless_ids = eligible_registered_cards(
        card_pool=CardPoolId.COLORLESS,
        generation_context="combat",
        is_multiplayer=combat.is_multiplayer,
    )
    generated = create_cards_from_ids(colorless_ids, combat.combat_card_generation_rng, 3, distinct=True)
    for generated_card in generated:
        generated_card.owner = user
        generated_card.set_temporary_free_this_turn()
    if generated:
        combat.request_card_choice(
            prompt="Choose a Colorless card",
            cards=generated,
            source_pile="generated",
            resolver=combat.move_card_to_hand,
            allow_skip=True,
            owner=user,
        )


def _dexterity_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 2 Dexterity."""
    t = target if target is not None else user
    combat.apply_power_to(t, PowerId.DEXTERITY, 2, applier=user)


def _energy_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 2 energy."""
    t = target if target is not None else user
    combat.gain_energy(t, ENERGY_POTION_ENERGY_GAIN)


def _explosive_ampoule(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Deal 10 unpowered damage to ALL enemies."""
    _deal_unpowered_damage_all(combat, user, 10)


def _fire_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Deal 20 unpowered damage to target enemy."""
    if target is not None:
        _deal_unpowered_damage(combat, user, target, 20)


def _flex_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 5 temporary Strength (FlexPotion power removes at end of turn)."""
    t = target if target is not None else user
    combat.apply_power_to(t, PowerId.FLEX_POTION, 5, applier=user)


def _focus_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """User gains 2 Focus."""
    combat.apply_power_to(user, PowerId.FOCUS, 2, applier=user)


def _poison_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Apply 6 Poison to target enemy."""
    if target is not None:
        combat.apply_power_to(target, PowerId.POISON, 6, applier=user)


def _potion_of_doom(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Apply 33 Doom to target enemy."""
    if target is not None:
        combat.apply_power_to(target, PowerId.DOOM, 33, applier=user)


def _power_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Generate a random Power card (cost 0) in hand.
    """
    state = combat.combat_player_state_for(user)
    if state is None:
        return
    generated = create_character_cards(
        state.character_id,
        combat.combat_card_generation_rng,
        3,
        card_type=CardType.POWER,
        distinct=True,
        generation_context="combat",
        is_multiplayer=combat.is_multiplayer,
    )
    for generated_card in generated:
        generated_card.owner = user
        generated_card.set_temporary_free_this_turn()
    if generated:
        combat.request_card_choice(
            prompt="Choose a Power card",
            cards=generated,
            source_pile="generated",
            resolver=combat.move_card_to_hand,
            allow_skip=True,
            owner=user,
        )


def _skill_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Generate a random Skill card (cost 0) in hand.
    """
    state = combat.combat_player_state_for(user)
    if state is None:
        return
    generated = create_character_cards(
        state.character_id,
        combat.combat_card_generation_rng,
        3,
        card_type=CardType.SKILL,
        distinct=True,
        generation_context="combat",
        is_multiplayer=combat.is_multiplayer,
    )
    for generated_card in generated:
        generated_card.owner = user
        generated_card.set_temporary_free_this_turn()
    if generated:
        combat.request_card_choice(
            prompt="Choose a Skill card",
            cards=generated,
            source_pile="generated",
            resolver=combat.move_card_to_hand,
            allow_skip=True,
            owner=user,
        )


def _speed_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 5 temporary Dexterity (SpeedPotion power removes at end of turn)."""
    t = target if target is not None else user
    combat.apply_power_to(t, PowerId.SPEED_POTION, 5, applier=user)


def _star_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """User gains 3 stars.

    Stars are a Regent mechanic; calls gain_stars if available.
    """
    combat.gain_stars(user, 3)


def _strength_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 2 Strength."""
    t = target if target is not None else user
    combat.apply_power_to(t, PowerId.STRENGTH, 2, applier=user)


def _swift_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target draws 3 cards."""
    t = target if target is not None else user
    combat.draw_cards(t, SWIFT_POTION_DRAW_COUNT)


def _vulnerable_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Apply 3 Vulnerable to target enemy."""
    if target is not None:
        combat.apply_power_to(target, PowerId.VULNERABLE, 3, applier=user)


def _weak_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Apply 3 Weak to target enemy."""
    if target is not None:
        combat.apply_power_to(target, PowerId.WEAK, 3, applier=user)


# =====================================================================
#  UNCOMMON POTIONS (22)
# =====================================================================

def _ashwater(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Exhaust any number of cards from hand.
    """
    owner_state = combat.combat_player_state_for(user)
    if owner_state is None or not owner_state.hand:
        return
    combat.request_multi_card_choice(
        prompt="Choose any number of hand cards to exhaust",
        cards=list(owner_state.hand),
        source_pile="hand",
        resolver=lambda selected_cards: [combat.exhaust_card(selected) for selected in selected_cards],
        min_count=0,
        max_count=len(owner_state.hand),
        allow_skip=True,
        owner=user,
    )


def _blessing_of_the_forge(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Upgrade all upgradable cards in hand."""
    state = combat.combat_player_state_for(user)
    if state is None:
        return
    for card in list(state.hand):
        combat.upgrade_card(card)


def _bone_brew(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Summon Osty with 15 HP.

    Necrobinder mechanic; calls summon_osty if available.
    """
    if hasattr(combat, "summon_osty"):
        combat.summon_osty(user, 15)


def _ambergris(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Heal 50% of max HP, and gain Ambergris while a fight is on.

    `CreatureCmd.Heal(target, MaxHp * HealPercent / 100)` with HealPercent 50,
    then `Apply<AmbergrisPower>(target, 1)` guarded by
    `CombatManager.Instance.IsInProgress` -- so the power only lands if it is
    drunk in combat, while the heal happens either way.
    """
    t = target if target is not None else user
    t.heal(t.max_hp * AMBERGRIS_HEAL_PERCENT // PERCENT_DENOMINATOR)
    if not combat.is_over:
        combat.apply_power_to(t, PowerId.AMBERGRIS, 1, applier=user)


def _clarity(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target draws 1 card and gains 3 Clarity (retain cards)."""
    t = target if target is not None else user
    combat.draw_cards(t, 1)
    combat.apply_power_to(t, PowerId.CLARITY, 3, applier=user)


def _cunning_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Add 3 upgraded Shivs to hand.
    """
    from sts2_env.cards.status import make_shiv

    for _ in range(3):
        combat.add_generated_card_to_creature_hand(user, make_shiv(upgraded=True))


def _cure_all(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 1 energy and draws 2 cards."""
    t = target if target is not None else user
    combat.gain_energy(t, CURE_ALL_ENERGY_GAIN)
    combat.draw_cards(t, CURE_ALL_DRAW_COUNT)


def _duplicator(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Gain 1 Duplication (next card is played twice)."""
    combat.apply_power_to(user, PowerId.DUPLICATION, 1, applier=user)


def _fortifier(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Double target's current block."""
    t = target if target is not None else user
    _gain_unpowered_block(t, t.block * 2, combat)


def _fysh_oil(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 1 Strength and 1 Dexterity."""
    t = target if target is not None else user
    combat.apply_power_to(t, PowerId.STRENGTH, 1, applier=user)
    combat.apply_power_to(t, PowerId.DEXTERITY, 1, applier=user)


def _gamblers_brew(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Discard any number of cards and draw that many.
    """
    owner_state = combat.combat_player_state_for(user)
    if owner_state is None or not owner_state.hand:
        return

    def _resolver(selected_cards):
        count = len(selected_cards)
        combat.discard_cards(selected_cards, count)

    combat.request_multi_card_choice(
        prompt="Choose any number of hand cards to discard",
        cards=list(owner_state.hand),
        source_pile="hand",
        resolver=_resolver,
        min_count=0,
        max_count=len(owner_state.hand),
        allow_skip=True,
        owner=user,
    )


def _heart_of_iron(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 7 Plating."""
    t = target if target is not None else user
    combat.apply_power_to(t, PowerId.PLATING, 7, applier=user)


def _kings_courage(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 15 Forge.

    Regent mechanic; calls gain_forge if available.
    """
    t = target if target is not None else user
    if hasattr(t, "gain_forge"):
        t.gain_forge(15, source="KingsCourage")


def _liquid_bronze(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 3 Thorns."""
    t = target if target is not None else user
    combat.apply_power_to(t, PowerId.THORNS, 3, applier=user)


def _potion_of_binding(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Apply 1 Weak and 1 Vulnerable to all enemies."""
    for enemy in list(combat.hittable_enemies):
        combat.apply_power_to(enemy, PowerId.WEAK, 1, applier=user)
        combat.apply_power_to(enemy, PowerId.VULNERABLE, 1, applier=user)


def _potion_of_capacity(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Add 2 orb slots.

    Defect mechanic; calls add_orb_slots if available.
    """
    if hasattr(combat, "add_orb_slots"):
        combat.add_orb_slots(user, 2)


def _powdered_demise(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Apply 9 Demise to target enemy."""
    if target is not None:
        combat.apply_power_to(target, PowerId.DEMISE, 9, applier=user)


def _radiant_tincture(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 1 energy and 3 Radiance."""
    t = target if target is not None else user
    combat.gain_energy(t, RADIANT_TINCTURE_ENERGY_GAIN)
    combat.apply_power_to(t, PowerId.RADIANCE, RADIANT_TINCTURE_RADIANCE, applier=user)


def _regen_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 5 Regen."""
    t = target if target is not None else user
    combat.apply_power_to(t, PowerId.REGEN, 5, applier=user)


def _stable_serum(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 2 RetainHand (retain entire hand for N turns)."""
    t = target if target is not None else user
    combat.apply_power_to(t, PowerId.RETAIN_HAND, 2, applier=user)


def _touch_of_insanity(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Choose a card from hand; it costs 0 for the rest of combat.
    """
    owner_state = combat.combat_player_state_for(user)
    if owner_state is None:
        return
    candidates = [
        card for card in owner_state.hand
        if (
            (
                not card.has_energy_cost_x
                and (
                    card.cost > 0
                    or combat.modified_card_cost(user, card) > 0
                )
            )
            or (
                not card.has_star_cost_x
                and (
                    card.star_cost > 0
                    or combat.modified_star_cost(user, card) > 0
                )
            )
        )
    ]
    if not candidates:
        return

    def _resolver(selected):
        if selected is None:
            return
        selected.set_free_this_combat()

    combat.request_card_choice(
        prompt="Choose a hand card to set to 0 cost",
        cards=candidates,
        source_pile="hand",
        resolver=_resolver,
        owner=user,
    )


# =====================================================================
#  RARE POTIONS (20)
# =====================================================================

def _beetle_juice(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Apply 4 Shrink to target enemy (reduces damage dealt by 30% per stack)."""
    if target is not None:
        combat.apply_power_to(target, PowerId.SHRINK, 4, applier=user)


def _bottled_potential(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Put hand into draw pile, shuffle, then draw 5."""
    t = target if target is not None else user
    user_state = combat.combat_player_state_for(user)
    if user_state is None:
        return
    cards_in_hand = list(user_state.hand)
    user_state.hand.clear()
    user_state.draw.extend(cards_in_hand)
    target_state = combat.combat_player_state_for(t)
    if target_state is None:
        return
    target_state.draw.extend(target_state.discard)
    target_state.discard.clear()
    combat.shuffle_rng.shuffle(target_state.draw)
    combat.draw_cards(t, BOTTLED_POTENTIAL_DRAW_COUNT)


def _cosmic_concoction(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Generate 3 upgraded Colorless cards in hand.
    """
    colorless_ids = eligible_registered_cards(
        card_pool=CardPoolId.COLORLESS,
        generation_context="combat",
        is_multiplayer=combat.is_multiplayer,
    )
    generated = create_cards_from_ids(colorless_ids, combat.combat_card_generation_rng, 3, distinct=True)
    for generated_card in generated:
        combat.upgrade_card(generated_card)
        combat.add_generated_card_to_creature_hand(user, generated_card)


def _distilled_chaos(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Auto-play top 3 cards from draw pile.
    """
    combat.auto_play_from_draw(user, 3)


def _droplet_of_precognition(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Choose a card from draw pile and add to hand.
    """
    state = combat.combat_player_state_for(user)
    if state is None or not state.draw:
        return
    candidates = sorted(state.draw, key=_source_card_order)
    combat.request_card_choice(
        prompt="Choose a draw pile card",
        cards=candidates,
        source_pile="draw",
        resolver=combat.move_card_to_hand,
        owner=user,
    )


def _entropic_brew(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Fill all empty potion slots with random potions.

    Uses the combat's potion slots.
    """
    combat.fill_empty_potion_slots(in_combat=False)


def _essence_of_darkness(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Channel Dark orbs equal to orb slot count.

    Defect mechanic; calls channel_orb if available.
    """
    orb_queue = getattr(combat.combat_player_state_for(user), "orb_queue", None)
    if orb_queue is not None and hasattr(combat, "channel_orb"):
        for _ in range(orb_queue.capacity):
            combat.channel_orb(user, OrbType.DARK)


def _fairy_in_a_bottle(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """On death: heal 30% max HP and prevent death.

    Automatic usage; this effect fires when triggered by should_die check.
    """
    t = target if target is not None else user
    heal_amount = t.max_hp * 30 // 100
    if t.is_dead:
        t.current_hp = min(t.max_hp, heal_amount)
    else:
        t.heal(heal_amount)


def _fruit_juice(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 5 max HP."""
    t = target if target is not None else user
    combat.gain_max_hp(t, FRUIT_JUICE_MAX_HP)


def _ghost_in_a_jar(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 1 Intangible."""
    t = target if target is not None else user
    combat.apply_power_to(t, PowerId.INTANGIBLE, 1, applier=user)


def _gigantification_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 1 Gigantification (double damage for 1 turn)."""
    t = target if target is not None else user
    combat.apply_power_to(t, PowerId.GIGANTIFICATION, 1, applier=user)


def _liquid_memories(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Choose a card from discard pile, set cost to 0, add to hand.
    """
    state = combat.combat_player_state_for(user)
    if state is None or not state.discard:
        return

    def _resolver(selected):
        if selected is None:
            return
        selected.set_temporary_cost_for_turn(0)
        combat.move_card_to_hand(selected)

    combat.request_card_choice(
        prompt="Choose a discard pile card",
        cards=list(state.discard),
        source_pile="discard",
        resolver=_resolver,
        owner=user,
    )


def _lucky_tonic(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 1 Buffer (negate next HP loss)."""
    t = target if target is not None else user
    combat.apply_power_to(t, PowerId.BUFFER, 1, applier=user)


def _mazaleths_gift(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 1 Ritual (gain Strength each turn)."""
    t = target if target is not None else user
    combat.apply_power_to(t, PowerId.RITUAL, 1, applier=user)


def _orobic_acid(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Generate 1 random Attack, 1 Skill, 1 Power, each with cost 0, in hand.
    """
    state = combat.combat_player_state_for(user)
    if state is None:
        return
    generated = []
    generated.extend(
        create_character_cards(
            state.character_id, combat.combat_card_generation_rng, 1,
            card_type=CardType.ATTACK, distinct=True, generation_context="combat",
            is_multiplayer=combat.is_multiplayer,
        )
    )
    generated.extend(
        create_character_cards(
            state.character_id, combat.combat_card_generation_rng, 1,
            card_type=CardType.SKILL, distinct=True, generation_context="combat",
            is_multiplayer=combat.is_multiplayer,
        )
    )
    generated.extend(
        create_character_cards(
            state.character_id, combat.combat_card_generation_rng, 1,
            card_type=CardType.POWER, distinct=True, generation_context="combat",
            is_multiplayer=combat.is_multiplayer,
        )
    )
    for generated_card in generated:
        generated_card.owner = user
        generated_card.set_temporary_free_this_turn()
        combat.add_generated_card_to_creature_hand(user, generated_card)


def _pot_of_ghouls(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Add 2 Soul cards to hand.

    Necrobinder mechanic; calls create_cards_in_hand if available.
    """
    from sts2_env.cards.status import make_soul

    for _ in range(2):
        combat.add_generated_card_to_creature_hand(user, make_soul())


def _shackling_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Apply 7 ShacklingPotion power to all enemies (temporary Strength loss)."""
    for enemy in list(combat.hittable_enemies):
        combat.apply_power_to(enemy, PowerId.SHACKLING_POTION, 7, applier=user)


def _ship_in_a_bottle(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Target gains 10 block now and 10 block next turn."""
    t = target if target is not None else user
    _gain_unpowered_block(t, 10, combat)
    combat.apply_power_to(t, PowerId.BLOCK_NEXT_TURN, 10, applier=user)


def _snecko_oil(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Draw 7 cards and randomize costs of all cards in hand (0-3)."""
    t = target if target is not None else user
    combat.draw_cards(t, SNECKO_OIL_DRAW_COUNT)
    state = combat.combat_player_state_for(t)
    if state is None:
        return
    for card in state.hand:
        if hasattr(card, "cost") and not card.has_energy_cost_x and card.cost >= 0:
            card.set_temporary_cost_for_turn(
                combat.combat_energy_costs_rng.next_int(SNECKO_OIL_MIN_COST, SNECKO_OIL_MAX_COST)
            )


def _soldiers_stew(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """All Strike-tagged cards gain +1 base replay count.

    Adds replay_count attribute if available.
    """
    from sts2_env.core.enums import CardTag
    t = target if target is not None else user
    state = combat.combat_player_state_for(t)
    if state is None:
        return
    for pile in state.all_piles:
        for card in pile:
            if hasattr(card, "tags") and CardTag.STRIKE in card.tags:
                if hasattr(card, "base_replay_count"):
                    card.base_replay_count += SOLDIERS_STEW_REPLAY_GAIN


# =====================================================================
#  EVENT / TOKEN POTIONS (3)
# =====================================================================

def _foul_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """In combat: deal 12 unpowered damage to ALL creatures (enemies + self).
    Out of combat at merchant: gain 100 gold (handled outside combat).
    """
    for creature in list(combat.all_creatures):
        if creature.is_alive:
            _deal_unpowered_damage(combat, user, creature, 12)


def _glowwater_potion(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Exhaust entire hand, then draw 10."""
    state = combat.combat_player_state_for(user)
    if state is None:
        return
    for card in list(state.hand):
        card.owner = user
        combat.exhaust_card(card)
    combat.draw_cards(user, GLOWWATER_DRAW_COUNT)


def _potion_shaped_rock(combat: CombatState, user: Creature, target: Creature | None) -> None:
    """Deal 15 unpowered damage to target enemy."""
    if target is not None:
        _deal_unpowered_damage(combat, user, target, 15)


# =====================================================================
#  Registration
# =====================================================================

_ALL_EFFECTS: dict[str, object] = {
    # Common (20)
    "AttackPotion":        _attack_potion,
    "Ambergris":           _ambergris,
    "BlockPotion":         _block_potion,
    BLOOD_POTION_ID:       _blood_potion,
    "ColorlessPotion":     _colorless_potion,
    "DexterityPotion":     _dexterity_potion,
    "EnergyPotion":        _energy_potion,
    "ExplosiveAmpoule":    _explosive_ampoule,
    "FirePotion":          _fire_potion,
    "FlexPotion":          _flex_potion,
    "FocusPotion":         _focus_potion,
    "PoisonPotion":        _poison_potion,
    "PotionOfDoom":        _potion_of_doom,
    "PowerPotion":         _power_potion,
    "SkillPotion":         _skill_potion,
    "SpeedPotion":         _speed_potion,
    "StarPotion":          _star_potion,
    "StrengthPotion":      _strength_potion,
    "SwiftPotion":         _swift_potion,
    "VulnerablePotion":    _vulnerable_potion,
    "WeakPotion":          _weak_potion,
    # Uncommon (22 -- 20 in pool + BoneBrew & CunningPotion use char-specific subsystems)
    "Ashwater":            _ashwater,
    "BlessingOfTheForge":  _blessing_of_the_forge,
    "BoneBrew":            _bone_brew,
    "Clarity":             _clarity,
    "CunningPotion":       _cunning_potion,
    "CureAll":             _cure_all,
    "Duplicator":          _duplicator,
    "Fortifier":           _fortifier,
    "FyshOil":             _fysh_oil,
    "GamblersBrew":        _gamblers_brew,
    "HeartOfIron":         _heart_of_iron,
    "KingsCourage":        _kings_courage,
    "LiquidBronze":        _liquid_bronze,
    "PotionOfBinding":     _potion_of_binding,
    "PotionOfCapacity":    _potion_of_capacity,
    "PowderedDemise":      _powdered_demise,
    "RadiantTincture":     _radiant_tincture,
    "RegenPotion":         _regen_potion,
    "StableSerum":         _stable_serum,
    "TouchOfInsanity":     _touch_of_insanity,
    # Rare (20)
    "BeetleJuice":         _beetle_juice,
    "BottledPotential":    _bottled_potential,
    "CosmicConcoction":    _cosmic_concoction,
    "DistilledChaos":      _distilled_chaos,
    "DropletOfPrecognition": _droplet_of_precognition,
    ENTROPIC_BREW_ID:      _entropic_brew,
    "EssenceOfDarkness":   _essence_of_darkness,
    "FairyInABottle":      _fairy_in_a_bottle,
    FRUIT_JUICE_ID:        _fruit_juice,
    "GhostInAJar":         _ghost_in_a_jar,
    "GigantificationPotion": _gigantification_potion,
    "LiquidMemories":      _liquid_memories,
    "LuckyTonic":          _lucky_tonic,
    "MazalethsGift":       _mazaleths_gift,
    "OrobicAcid":          _orobic_acid,
    "PotOfGhouls":         _pot_of_ghouls,
    "ShacklingPotion":     _shackling_potion,
    "ShipInABottle":       _ship_in_a_bottle,
    "SneckoOil":           _snecko_oil,
    "SoldiersStew":        _soldiers_stew,
    # Event / Token (3)
    FOUL_POTION_ID:        _foul_potion,
    "GlowwaterPotion":     _glowwater_potion,
    "PotionShapedRock":    _potion_shaped_rock,
}

for _pid, _fn in _ALL_EFFECTS.items():
    register_potion_effect(_pid, _fn)
