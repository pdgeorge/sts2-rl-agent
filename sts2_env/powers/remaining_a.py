"""Remaining powers batch A (46 powers).

Each power is verified against the decompiled C# source in
decompiled/MegaCrit.Sts2.Core.Models.Powers/.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from sts2_env.core.enums import (
    CardPilePosition,
    CardId,
    CardKeyword,
    CardTag,
    CardType,
    CombatSide,
    PileType,
    PowerId,
    PowerType,
    PowerStackType,
    ValueProp,
)
from sts2_env.powers.base import PowerInstance

if TYPE_CHECKING:
    from sts2_env.core.creature import Creature
    from sts2_env.core.combat import CombatState


def _gain_unpowered_block(owner: Creature, amount: int, combat: CombatState) -> int:
    before = owner.block
    owner.gain_block(amount, unpowered=True)
    gained = owner.block - before
    if gained > 0:
        from sts2_env.core.hooks import fire_after_block_gained

        fire_after_block_gained(owner, gained, combat)
    return gained


# =====================================================================
#  1. AccelerantPower
# =====================================================================
class AccelerantPower(PowerInstance):
    """Buff/counter. Increases the number of times Poison ticks each turn.

    C# ref: AccelerantPower.cs
    StackType.Counter. The actual repeat-tick logic lives in PoisonPower;
    this power is a counter that PoisonPower reads.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.ACCELERANT, amount)

    # PoisonPower checks owner.get_power(ACCELERANT) to determine extra ticks.
    # No hook methods needed -- this is a passive counter.


# =====================================================================
#  2. AdaptablePower
# =====================================================================
class AdaptablePower(PowerInstance):
    """Buff/single. Monster revives after death (TestSubject boss mechanic).

    C# ref: AdaptablePower.cs
    - AfterDeath: triggers dead-state / revive on TestSubject.
    - ShouldAllowHitting: false while reviving.
    - ShouldStopCombatFromEnding: true.
    - ShouldCreatureBeRemovedFromCombatAfterDeath: false for owner.
    StackType.Single.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.SINGLE

    def __init__(self, amount: int = 1):
        super().__init__(PowerId.ADAPTABLE, amount)
        self.is_reviving: bool = False

    def after_death(
        self,
        owner: Creature,
        creature: Creature,
        combat: CombatState,
        was_removal_prevented: bool = False,
    ) -> None:
        if was_removal_prevented or creature is not owner:
            return
        self.is_reviving = True
        combat.set_enemy_state(owner, "RESPAWN_MOVE")

    def do_revive(self) -> None:
        """Called by the encounter system when the revive completes."""
        self.is_reviving = False

    def should_stop_combat_ending(self) -> bool:
        return True

    def should_allow_hitting(self, owner: Creature, combat: CombatState) -> bool:
        return not self.is_reviving

    def should_creature_be_removed_from_combat_after_death(
        self,
        owner: Creature,
        combat: CombatState,
    ) -> bool:
        return False

    def should_power_be_removed_after_owner_death(
        self,
        owner: Creature,
        combat: CombatState,
    ) -> bool:
        return False

    def should_creature_be_removed_from_combat_after_death(
        self,
        owner: Creature,
        combat: CombatState,
    ) -> bool:
        return False


# =====================================================================
#  3. AnticipatePower  (TemporaryDexterity variant)
# =====================================================================
class AnticipatePower(PowerInstance):
    """Temporary Dexterity: grants Dexterity on application, removes it
    at end of owner's turn.

    C# ref: AnticipatePower.cs extends TemporaryDexterityPower (IsPositive=true).
    - BeforeApplied: applies +Amount Dexterity.
    - AfterTurnEnd (owner side): removes self and applies -Amount Dexterity.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER
    is_temporary = True

    def __init__(self, amount: int):
        super().__init__(PowerId.ANTICIPATE, amount)

    def after_power_amount_changed(
        self,
        owner: Creature,
        target: Creature,
        power_id: PowerId,
        amount: int,
        applier: Creature | None,
        source: object | None,
        combat: CombatState,
    ) -> None:
        if owner is target and power_id == PowerId.ANTICIPATE and owner.powers.get(PowerId.ANTICIPATE) is self and not self.consume_ignore_next_instance():
            owner.apply_power(PowerId.DEXTERITY, amount, applier=applier, source=source)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            owner.apply_power(PowerId.DEXTERITY, -self.amount, applier=owner)
            self.amount = 0


# =====================================================================
#  4. ArsenalPower
# =====================================================================
class ArsenalPower(PowerInstance):
    """After playing a Colorless card, gain Amount Strength.

    C# ref: ArsenalPower.cs
    - AfterCardPlayed: if card is colorless and owned by owner's player,
      apply StrengthPower(Amount) to owner.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.ARSENAL, amount)

    def after_card_played(self, owner: Creature, card: object, combat: CombatState) -> None:
        is_visual_colorless = getattr(card, "is_visual_colorless", None)
        is_colorless = bool(
            getattr(card, "is_colorless", False)
            or getattr(card, "visual_card_pool_is_colorless", False)
            or (callable(is_visual_colorless) and is_visual_colorless())
        )
        card_owner = getattr(card, "owner", None)
        if is_colorless and (card_owner is owner or card_owner is None):
            owner.apply_power(PowerId.STRENGTH, self.amount)


# =====================================================================
#  5. AutomationPower
# =====================================================================
class AutomationPower(PowerInstance):
    """After every 10 cards drawn, gain Amount energy.

    C# ref: AutomationPower.cs
    - AfterCardDrawn: decrement internal counter; at 0, gain energy and reset.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    _BASE_CARDS = 10

    def __init__(self, amount: int):
        super().__init__(PowerId.AUTOMATION, amount)
        self.cards_left: int = self._BASE_CARDS
        self._instances: list[tuple[int, int]] = [(amount, self._BASE_CARDS)]

    def _add_instance(self, amount: int) -> None:
        self._instances.append((amount, self._BASE_CARDS))
        self.amount = amount

    def after_power_amount_changed(
        self,
        owner: Creature,
        target: Creature,
        power_id: PowerId,
        amount: int,
        applier: Creature | None,
        source: object | None,
        combat: CombatState,
    ) -> None:
        if owner is target and power_id == self.power_id and amount > 0 and self.amount != amount:
            self._add_instance(amount)

    def on_card_drawn(
        self,
        owner: Creature,
        card: object,
        from_hand_draw: bool,
        combat: CombatState,
    ) -> None:
        """Called by the draw pipeline when a card is drawn."""
        card_owner = getattr(card, "owner", None)
        if card_owner is not owner:
            return
        updated_instances: list[tuple[int, int]] = []
        total_energy = 0
        for amount, cards_left in self._instances:
            cards_left -= 1
            if cards_left <= 0:
                total_energy += amount
                cards_left = self._BASE_CARDS
            updated_instances.append((amount, cards_left))
        self._instances = updated_instances
        if total_energy > 0:
            combat.gain_energy(owner, total_energy)
        if self._instances:
            _, self.cards_left = self._instances[-1]


# =====================================================================
#  6. BattlewornDummyTimeLimitPower
# =====================================================================
class BattlewornDummyTimeLimitPower(PowerInstance):
    """Countdown timer for the Battleworn Dummy event. Decrements at end
    of owner's turn; at 1, the creature escapes.

    C# ref: BattlewornDummyTimeLimitPower.cs
    - AfterTurnEnd (owner side): decrement; at 1 -> creature escapes.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.BATTLEWORN_DUMMY_TIME_LIMIT, amount)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side != owner.side:
            return
        if self.amount > 1:
            self.amount -= 1
            return
        # Time's up -- creature escapes
        escape = getattr(combat, "escape_creature", None)
        if escape is not None:
            escape(owner)
        self.amount = 0


# =====================================================================
#  7. BeaconOfHopePower
# =====================================================================
class BeaconOfHopePower(PowerInstance):
    """When owner gains block during their own turn, grant 50% * Amount of
    that block (unpowered) to all teammates.

    C# ref: BeaconOfHopePower.cs
    - AfterBlockGained: if creature == owner and on owner's side turn,
      give 50% * Amount of gained block to each teammate.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.BEACON_OF_HOPE, amount)

    def after_block_gained(self, owner: Creature, creature: Creature,
                           amount: int, combat: CombatState) -> None:
        if creature is not owner or amount < 1:
            return
        if not combat.is_owner_side_turn(owner):
            return
        share = int(amount * 0.5 * self.amount)
        if share < 1:
            return
        teammates = getattr(combat, "get_teammates_of", None)
        if teammates is not None:
            for ally in teammates(owner):
                if ally is not owner and getattr(ally, "is_alive", True):
                    _gain_unpowered_block(ally, share, combat)


# =====================================================================
#  8. BlackHolePower
# =====================================================================
class BlackHolePower(PowerInstance):
    """Deal Amount damage (unpowered) to all enemies whenever stars are
    spent on a card or whenever stars are gained.

    C# ref: BlackHolePower.cs
    - AfterCardPlayed: if stars were spent, damage all enemies.
    - AfterStarsGained: if stars > 0 gained, damage all enemies.
    StackType.Counter.

    # Simplified: Stars system is Regent-specific. We expose a trigger
    # method the combat system calls when stars are spent or gained.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.BLACK_HOLE, amount)

    def _deal_damage_to_all_enemies(self, owner: Creature, combat: CombatState) -> None:
        for enemy in combat.hittable_enemies:
            combat.deal_damage(
                dealer=owner,
                target=enemy,
                amount=self.amount,
                props=ValueProp.UNPOWERED,
            )

    def after_card_played(self, owner: Creature, card: object, combat: CombatState) -> None:
        if (
            getattr(card, "owner", None) is owner
            and getattr(card, "combat_vars", {}).get("_stars_spent_for_play", 0) > 0
            and combat.active_card_play_is_last_in_series
        ):
            self._deal_damage_to_all_enemies(owner, combat)

    def on_stars_gained(self, owner: Creature, stars: int, combat: CombatState) -> None:
        """Called by the combat system when stars are gained."""
        if stars > 0:
            self._deal_damage_to_all_enemies(owner, combat)


# =====================================================================
#  9. BladeOfInkPower
# =====================================================================
class BladeOfInkPower(PowerInstance):
    """Each Attack card played this turn grants +Amount Strength (temporary).
    At end of turn, remove the accumulated Strength and remove self.

    C# ref: BladeOfInkPower.cs
    - AfterCardPlayed: if Attack, apply StrengthPower(Amount) silently.
    - AfterTurnEnd (owner side): apply -strength_applied, remove self.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.BLADE_OF_INK, amount)
        self._strength_applied: int = 0

    def after_card_played(self, owner: Creature, card: object, combat: CombatState) -> None:
        card_type = getattr(card, "card_type", None) or getattr(card, "type", None)
        card_owner = getattr(card, "owner", None)
        if card_owner is owner and card_type == CardType.ATTACK:
            owner.apply_power(PowerId.STRENGTH, self.amount)
            self._strength_applied += self.amount

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            if self._strength_applied != 0:
                owner.apply_power(PowerId.STRENGTH, -self._strength_applied)
            self._strength_applied = 0
            self.amount = 0  # signal removal


# =====================================================================
#  10. CallOfTheVoidPower
# =====================================================================
class CallOfTheVoidPower(PowerInstance):
    """At start of turn, generate Amount random (non-basic, non-ancient)
    cards with Ethereal into hand.

    C# ref: CallOfTheVoidPower.cs
    - BeforeHandDraw: generate Amount distinct cards from card pool,
      apply Ethereal, add to hand.
    StackType.Counter.

    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.CALL_OF_THE_VOID, amount)

    def before_hand_draw(self, owner: Creature, combat: CombatState) -> None:
        if not getattr(owner, "is_player", False):
            return
        from sts2_env.cards.factory import create_card, create_cards_from_ids, eligible_character_cards
        from sts2_env.core.enums import CardRarity

        state = combat.combat_player_state_for(owner)
        if state is None or self.amount <= 0:
            return
        candidates = [
            card_id
            for card_id in eligible_character_cards(
                state.character_id,
                generation_context="combat",
                is_multiplayer=combat.is_multiplayer,
            )
            if create_card(card_id).rarity not in {CardRarity.BASIC, CardRarity.ANCIENT}
        ]
        if not candidates:
            return
        generated = create_cards_from_ids(
            candidates,
            combat.combat_card_generation_rng,
            self.amount,
            distinct=False,
        )
        for card in generated:
            card.keywords = frozenset(set(card.keywords) | {"ethereal"})
        combat._add_generated_cards_to_hand(generated, owner=owner)


# =====================================================================
#  11. ChainsOfBindingPower
# =====================================================================
class ChainsOfBindingPower(PowerInstance):
    """Debuff. When owner draws a card during their turn, the first Amount
    cards drawn are "Bound" (can only be played once this turn). Bound
    afflictions clear at end of turn.

    C# ref: ChainsOfBindingPower.cs
    - AfterCardDrawn: afflict drawn card with Bound (up to Amount per turn).
    - ShouldPlay: prevent playing a second Bound card.
    - BeforeTurnEnd: clear Bound afflictions.
    StackType.Counter.

    # Simplified: tracks bound state. Card system should check this power.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.CHAINS_OF_BINDING, amount)
        self._bound_cards_this_turn: int = 0
        self._bound_card_played: bool = False

    def on_card_drawn(
        self,
        owner: Creature,
        card: object,
        from_hand_draw: bool,
        combat: CombatState,
    ) -> None:
        """Called by draw pipeline. Afflicts the card if limit not reached."""
        if getattr(card, "owner", None) is not owner:
            return
        if not combat.is_owner_side_turn(owner):
            return
        if self._bound_cards_this_turn < self.amount:
            afflict = getattr(card, "afflict", None)
            if callable(afflict):
                if not afflict("bound"):
                    return
            elif hasattr(card, "bound"):
                if getattr(card, "bound", False):
                    return
                card.bound = True
            self._bound_cards_this_turn += 1

    def before_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        self._bound_card_played = False
        self._bound_cards_this_turn = 0
        state = combat.combat_player_state_for(owner)
        if state is not None:
            for pile in state.all_piles:
                for card in pile:
                    clear_affliction = getattr(card, "clear_affliction", None)
                    if callable(clear_affliction):
                        clear_affliction("bound")
                    elif getattr(card, "bound", False):
                        card.bound = False

    def before_side_turn_start(self, owner: Creature, side: CombatSide,
                               combat: CombatState) -> None:
        if side == owner.side:
            self._bound_cards_this_turn = 0
            self._bound_card_played = False

    def before_card_played(self, owner: Creature, card: object, combat: CombatState) -> None:
        if getattr(card, "combat_vars", {}).get("_is_clone"):
            return
        if getattr(card, "owner", None) is not owner:
            return
        if not getattr(card, "bound", False):
            return
        self._bound_card_played = True

    def should_play(self, owner: Creature, card: object, combat: CombatState) -> bool:
        if getattr(card, "owner", None) is not owner:
            return True
        if not getattr(card, "bound", False):
            return True
        return not self._bound_card_played


# =====================================================================
#  12. ChildOfTheStarsPower
# =====================================================================
class ChildOfTheStarsPower(PowerInstance):
    """Whenever stars are spent, gain Amount * stars_spent block (unpowered).

    C# ref: ChildOfTheStarsPower.cs
    - AfterStarsSpent: gain Amount * spent_amount block.
    StackType.Counter.

    # Simplified: Regent star system. Exposes trigger method.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.CHILD_OF_THE_STARS, amount)

    def on_stars_spent(self, owner: Creature, stars: int, combat: CombatState) -> None:
        """Called by the combat system when stars are spent."""
        if stars > 0:
            _gain_unpowered_block(owner, self.amount * stars, combat)


# =====================================================================
#  13. ClarityPower
# =====================================================================
class ClarityPower(PowerInstance):
    """Draw 1 extra card per turn. Decrements each turn.

    C# ref: ClarityPower.cs
    - ModifyHandDraw: +1 card.
    - AfterSideTurnStart: decrement (removes at 0).
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.CLARITY, amount)

    def modify_hand_draw(self, owner: Creature, draw: int) -> int:
        return draw + 1

    def after_side_turn_start(self, owner: Creature, side: CombatSide,
                              combat: CombatState) -> None:
        if side == owner.side:
            self.amount -= 1


# =====================================================================
#  14. ConfusedPower
# =====================================================================
class ConfusedPower(PowerInstance):
    """Debuff/single. When a card is drawn, randomize its energy cost (0-3).

    C# ref: ConfusedPower.cs
    - AfterCardDrawn: set card energy cost to random 0-3.
    StackType.Single.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.SINGLE

    def __init__(self, amount: int = 1):
        super().__init__(PowerId.CONFUSED, amount)

    def on_card_drawn(
        self,
        owner: Creature,
        card: object,
        from_hand_draw: bool,
        combat: CombatState,
    ) -> None:
        """Called by draw pipeline. Randomizes card cost."""
        card_owner = getattr(card, "owner", None)
        if card_owner is not owner:
            return
        canonical_cost = getattr(card, "energy_cost", -1)
        if canonical_cost is not None and canonical_cost < 0:
            return
        new_cost = combat.combat_energy_costs_rng.next_int(0, 3)
        if hasattr(card, "energy_cost"):
            card.energy_cost = new_cost
        elif hasattr(card, "set_combat_cost"):
            card.set_combat_cost(new_cost)


# =====================================================================
#  15. ConquerorPower
# =====================================================================
class ConquerorPower(PowerInstance):
    """Debuff/duration. Sovereign Blade cards deal 2x damage to the owner.
    Ticks down at end of owner's turn.

    C# ref: ConquerorPower.cs
    - ModifyDamageMultiplicative: 2x for Sovereign Blade against owner.
    - AfterTurnEnd (owner side): tick down duration.
    StackType.Counter (used as duration).
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.CONQUEROR, amount)

    def modify_damage_multiplicative(
        self, owner: Creature, dealer: Creature | None, target: Creature, props: ValueProp
    ) -> float:
        from sts2_env.cards.status import is_sovereign_blade

        card_source = getattr(owner.combat_state, "active_card_source", None)
        if not is_sovereign_blade(card_source):
            return 1.0
        if target is not owner:
            return 1.0
        if not props.is_powered_attack():
            return 1.0
        return 2.0

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            self.amount -= 1


# =====================================================================
#  16. ConsumingShadowPower
# =====================================================================
class ConsumingShadowPower(PowerInstance):
    """At end of owner's turn, evoke the last orb Amount times.

    C# ref: ConsumingShadowPower.cs
    - AfterTurnEnd (owner side): evoke last orb Amount times if orbs exist.
    StackType.Counter.

    # Simplified: delegates orb evocation to combat system.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.CONSUMING_SHADOW, amount)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side != owner.side:
            return
        evoke = getattr(combat, "evoke_last_orb", None)
        if evoke is not None:
            for _ in range(self.amount):
                evoke(owner)


# =====================================================================
#  17. CoolantPower
# =====================================================================
class CoolantPower(PowerInstance):
    """At start of owner's turn, gain block equal to (distinct orb types) * Amount.

    C# ref: CoolantPower.cs
    - AfterSideTurnStart (owner side): count distinct orb types, gain
      block = count * Amount (unpowered).
    StackType.Counter.

    # Simplified: delegates orb counting to combat system.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.COOLANT, amount)

    def after_side_turn_start(self, owner: Creature, side: CombatSide,
                              combat: CombatState) -> None:
        if side != owner.side:
            return
        distinct_orbs = getattr(combat, "count_distinct_orb_types", None)
        if distinct_orbs is not None:
            count = distinct_orbs(owner)
            if count > 0:
                _gain_unpowered_block(owner, count * self.amount, combat)


# =====================================================================
#  18. CorrosiveWavePower
# =====================================================================
class CorrosiveWavePower(PowerInstance):
    """Whenever the owner draws a card, apply Amount Poison to all enemies.
    Removed at end of owner's turn.

    C# ref: CorrosiveWavePower.cs
    - AfterCardDrawn: apply PoisonPower(Amount) to all hittable enemies.
    - AfterTurnEnd (owner side): remove self.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.CORROSIVE_WAVE, amount)

    def on_card_drawn(
        self,
        owner: Creature,
        card: object,
        from_hand_draw: bool,
        combat: CombatState,
    ) -> None:
        """Called by draw pipeline. Applies Poison to all enemies."""
        card_owner = getattr(card, "owner", None)
        if card_owner is not None and card_owner is not owner:
            return
        for enemy in combat.hittable_enemies:
            combat.apply_power_to(enemy, PowerId.POISON, self.amount, applier=owner)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            self.amount = 0  # signal removal


# =====================================================================
#  19. CrimsonMantlePower
# =====================================================================
class CrimsonMantlePower(PowerInstance):
    """At start of owner's player turn, deal self-damage (unblockable,
    unpowered) then gain Amount block (unpowered). Self-damage increments
    over time via IncrementSelfDamage().

    C# ref: CrimsonMantlePower.cs
    - AfterPlayerTurnStart: deal SelfDamage to self, then gain Amount block.
    - IncrementSelfDamage(): increases the self-damage each call.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.CRIMSON_MANTLE, amount)
        self.self_damage: int = 0

    def after_player_turn_start(self, owner: Creature, combat: CombatState) -> None:
        if not getattr(owner, "is_player", False):
            return
        if self.self_damage > 0:
            combat.deal_damage(
                dealer=owner,
                target=owner,
                amount=self.self_damage,
                props=ValueProp.UNBLOCKABLE | ValueProp.UNPOWERED,
            )
        _gain_unpowered_block(owner, self.amount, combat)

    def increment_self_damage(self) -> None:
        """Called by the card system when Crimson Mantle card is played."""
        self.self_damage += 1


# =====================================================================
#  20. CrushUnderPower  (TemporaryStrength variant -- negative)
# =====================================================================
class CrushUnderPower(PowerInstance):
    """Negative temporary Strength (debuff). Applies -Amount Strength on
    application. At end of turn, removes self and restores the Strength.

    C# ref: CrushUnderPower.cs extends TemporaryStrengthPower (IsPositive=false).
    StackType.Counter.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER
    is_temporary = True

    def __init__(self, amount: int):
        super().__init__(PowerId.CRUSH_UNDER, amount)

    def after_power_amount_changed(
        self,
        owner: Creature,
        target: Creature,
        power_id: PowerId,
        amount: int,
        applier: Creature | None,
        source: object | None,
        combat: CombatState,
    ) -> None:
        if owner is target and power_id == self.power_id and amount != 0 and not self.consume_ignore_next_instance():
            owner.apply_power(PowerId.STRENGTH, -amount, applier=applier, source=source)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            # Reverse: the initial application reduced Strength by Amount,
            # so now we restore it.
            owner.apply_power(PowerId.STRENGTH, self.amount)
            self.amount = 0  # signal removal


# =====================================================================
#  21. CuriousPower
# =====================================================================
class CuriousPower(PowerInstance):
    """Reduces the energy cost of Power-type cards by Amount.

    C# ref: CuriousPower.cs
    - TryModifyEnergyCostInCombat: if card is Power type with cost > 0,
      reduce by Amount (min 0).
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.CURIOUS, amount)

    def modify_card_cost(self, owner: Creature, card: object) -> int | None:
        """Return modified cost for Power cards, None otherwise."""
        if getattr(card, "owner", None) is not owner:
            return None
        card_type = getattr(card, "card_type", None) or getattr(card, "type", None)
        if card_type != CardType.POWER:
            return None
        original_cost = getattr(card, "energy_cost", 0)
        if original_cost is None or original_cost <= 0:
            return None
        return max(0, original_cost - self.amount)


# =====================================================================
#  22. DampenPower
# =====================================================================
class DampenPower(PowerInstance):
    """Debuff. On application, downgrades all upgraded cards. When all
    casters die (or power is removed), re-upgrades them.

    C# ref: DampenPower.cs
    - AfterApplied: downgrade all upgraded cards.
    - AfterDeath: if all casters dead, remove self.
    - AfterRemoved: re-upgrade all downgraded cards.
    StackType.None (does not stack).

    # Simplified: tracks downgraded state. Card upgrade/downgrade is
    # delegated to the combat system.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.SINGLE

    def __init__(self, amount: int = 1):
        super().__init__(PowerId.DAMPEN, amount)
        self._active: bool = False
        self.casters: set[Creature] = set()
        self._downgraded_cards: list[object] = []

    def add_caster(self, creature: Creature) -> None:
        self.casters.add(creature)

    def after_power_amount_changed(
        self,
        owner: Creature,
        target: Creature,
        power_id: PowerId,
        amount: int,
        applier: Creature | None,
        source: object | None,
        combat: CombatState,
    ) -> None:
        if owner is target and power_id == PowerId.DAMPEN and amount > 0 and not self._active:
            self._active = True
            self._downgrade_upgraded_cards(owner, combat)

    def _downgrade_upgraded_cards(self, owner: Creature, combat: CombatState) -> None:
        from sts2_env.cards.base import capture_self_mutating_card_progress, restore_self_mutating_card_progress
        from sts2_env.cards.factory import create_card

        state = combat.combat_player_state_for(owner)
        if state is None:
            return
        for pile in state.all_piles:
            for card in pile:
                if not getattr(card, "upgraded", False):
                    continue
                progress = capture_self_mutating_card_progress(card)
                try:
                    base = create_card(card.card_id, upgraded=False)
                except KeyError:
                    continue
                self._downgraded_cards.append(card)
                current_cost = card.cost
                had_turn_override = "_turn_cost_override" in card.combat_vars
                card.card_type = base.card_type
                card.target_type = base.target_type
                card.rarity = base.rarity
                card.base_damage = base.base_damage
                card.base_block = base.base_block
                card.upgraded = False
                card.keywords = base.keywords
                card.tags = base.tags
                card.can_be_generated_in_combat = base.can_be_generated_in_combat
                card.can_be_generated_by_modifiers = base.can_be_generated_by_modifiers
                card.has_turn_end_in_hand_effect = base.has_turn_end_in_hand_effect
                card.effect_vars = dict(base.effect_vars)
                card.has_energy_cost_x = base.has_energy_cost_x
                card.star_cost = base.star_cost
                card.original_cost = base.original_cost
                card.cost = current_cost if had_turn_override else base.cost
                restore_self_mutating_card_progress(card, progress)
                if owner.has_power(PowerId.HEX):
                    card.keywords = frozenset(set(card.keywords) | {"ethereal"})

    def on_removed(self, owner: Creature, combat: CombatState) -> None:
        """Called when the power is removed. Re-upgrades cards."""
        if self._active:
            for card in self._downgraded_cards:
                combat.upgrade_card(card)
                if owner.has_power(PowerId.HEX):
                    card.keywords = frozenset(set(card.keywords) | {"ethereal"})
            self._downgraded_cards.clear()
            self._active = False

    def on_ally_death(
        self,
        owner: Creature,
        creature: Creature,
        combat: CombatState,
        was_removal_prevented: bool = False,
    ) -> None:
        if was_removal_prevented:
            return
        if creature not in self.casters:
            return
        self.casters.remove(creature)
        if self.casters:
            return
        self.on_removed(owner, combat)
        owner.powers.pop(self.power_id, None)


# =====================================================================
#  23. DanseMacabrePower
# =====================================================================
class DanseMacabrePower(PowerInstance):
    """Before playing a card that costs >= 2 energy, gain Amount block (unpowered).

    C# ref: DanseMacabrePower.cs
    - BeforeCardPlayed: if card cost >= Energy DynamicVar (default 2),
      gain Amount block.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    _ENERGY_THRESHOLD = 2

    def __init__(self, amount: int):
        super().__init__(PowerId.DANSE_MACABRE, amount)

    def before_card_played(self, owner: Creature, card: object, combat: CombatState) -> None:
        card_owner = getattr(card, "owner", None)
        if card_owner is not None and card_owner is not owner:
            return
        cost = getattr(card, "energy_cost", 0) or 0
        if cost >= self._ENERGY_THRESHOLD:
            _gain_unpowered_block(owner, self.amount, combat)


# =====================================================================
#  24. DarkShacklesPower  (TemporaryStrength variant -- negative)
# =====================================================================
class DarkShacklesPower(PowerInstance):
    """Negative temporary Strength (debuff). Same as CrushUnder.

    C# ref: DarkShacklesPower.cs extends TemporaryStrengthPower (IsPositive=false).
    StackType.Counter.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER
    is_temporary = True

    def __init__(self, amount: int):
        super().__init__(PowerId.DARK_SHACKLES, amount)

    def after_power_amount_changed(
        self,
        owner: Creature,
        target: Creature,
        power_id: PowerId,
        amount: int,
        applier: Creature | None,
        source: object | None,
        combat: CombatState,
    ) -> None:
        if owner is target and power_id == self.power_id and amount != 0 and not self.consume_ignore_next_instance():
            owner.apply_power(PowerId.STRENGTH, -amount, applier=applier, source=source)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            owner.apply_power(PowerId.STRENGTH, self.amount)
            self.amount = 0  # signal removal


# =====================================================================
#  25. DemesnePower
# =====================================================================
class DemesnePower(PowerInstance):
    """Draw Amount extra cards each turn AND gain Amount extra max energy.

    C# ref: DemesnePower.cs
    - ModifyHandDraw: +Amount.
    - ModifyMaxEnergy: +Amount.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.DEMESNE, amount)

    def modify_hand_draw(self, owner: Creature, draw: int) -> int:
        return draw + self.amount

    def modify_max_energy(self, owner: Creature, energy: int) -> int:
        return energy + self.amount


# =====================================================================
#  26. DemisePower
# =====================================================================
class DemisePower(PowerInstance):
    """Debuff. At end of owner's turn, deal Amount unblockable/unpowered
    damage to self.

    C# ref: DemisePower.cs
    - AfterTurnEnd (owner side): deal Amount damage (unblockable+unpowered)
      to owner with no dealer.
    StackType.Counter.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.DEMISE, amount)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            combat.deal_damage(
                dealer=None,
                target=owner,
                amount=self.amount,
                props=ValueProp.UNBLOCKABLE | ValueProp.UNPOWERED,
            )


# =====================================================================
#  27. DevourLifePower
# =====================================================================
class DevourLifePower(PowerInstance):
    """When the owner plays a Soul card, summon Amount Osty minion(s).

    C# ref: DevourLifePower.cs
    - AfterCardPlayed: if card is Soul and owned by owner, summon Amount osties.
    StackType.Counter.

    # Simplified: Necrobinder-specific. Soul card + Osty summon system.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.DEVOUR_LIFE, amount)

    def after_card_played(self, owner: Creature, card: object, combat: CombatState) -> None:
        from sts2_env.cards.status import is_soul

        card_owner = getattr(card, "owner", None)
        if is_soul(card) and (card_owner is owner or card_owner is None):
            summon = getattr(combat, "summon_osty", None)
            if summon is not None:
                summon(owner, self.amount)


# =====================================================================
#  28. DiamondDiademPower
# =====================================================================
class DiamondDiademPower(PowerInstance):
    """Buff/single. Owner takes 50% damage from powered attacks.
    Removed at end of enemy turn.

    C# ref: DiamondDiademPower.cs
    - ModifyDamageMultiplicative: 0.5 for powered attacks targeting owner.
    - AfterTurnEnd (Enemy side): remove self.
    StackType.Single.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.SINGLE

    def __init__(self, amount: int = 1):
        super().__init__(PowerId.DIAMOND_DIADEM, amount)

    def modify_damage_multiplicative(
        self, owner: Creature, dealer: Creature | None, target: Creature, props: ValueProp
    ) -> float:
        if self.amount > 0 and target is owner and props.is_powered_attack():
            return 0.5
        return 1.0

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == CombatSide.ENEMY:
            combat._remove_power(owner, self.power_id)


# =====================================================================
#  29. DieForYouPower
# =====================================================================
class DieForYouPower(PowerInstance):
    """Buff/single. Redirects powered attack damage targeting the pet owner
    to this creature (the pet) instead.

    C# ref: DieForYouPower.cs
    - ModifyUnblockedDamageTarget: redirect damage from pet owner to self.
    - ShouldCreatureBeRemovedFromCombatAfterDeath: false for owner.
    - ShouldPowerBeRemovedAfterOwnerDeath: false.
    StackType.Single.

    # Simplified: The pet/owner relationship is model-specific. This power
    # exposes a redirect method the damage pipeline can call.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.SINGLE

    def __init__(self, amount: int = 1):
        super().__init__(PowerId.DIE_FOR_YOU, amount)

    def modify_unblocked_damage_target(
        self,
        owner: Creature,
        target: Creature,
        amount: float,
        props: ValueProp,
        dealer: Creature | None,
    ) -> Creature:
        pet_owner = getattr(owner, "pet_owner", None)
        if pet_owner is None:
            return target
        if target is not pet_owner:
            return target
        if not getattr(owner, "is_alive", True):
            return target
        if not props.is_powered_attack():
            return target
        return owner

    def should_redirect_damage(
        self, owner: Creature, target: Creature, props: ValueProp
    ) -> Creature | None:
        redirected = self.modify_unblocked_damage_target(owner, target, 0, props, None)
        return redirected if redirected is not target else None

    def should_allow_hitting(self, owner: Creature, combat: CombatState) -> bool:
        return owner.is_alive

    def should_power_be_removed_after_owner_death(
        self,
        owner: Creature,
        combat: CombatState,
    ) -> bool:
        return False

    def should_creature_be_removed_from_combat_after_death(
        self,
        owner: Creature,
        combat: CombatState,
    ) -> bool:
        return False


# =====================================================================
#  30. DisintegrationPower
# =====================================================================
class DisintegrationPower(PowerInstance):
    """Debuff. At the end of owner's turn (late phase), deal Amount damage
    (unpowered) to self.

    C# ref: DisintegrationPower.cs
    - AfterTurnEndLate (owner side): deal Amount unpowered damage to self.
    StackType.Counter.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.DISINTEGRATION, amount)

    def after_turn_end_late(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            combat.deal_damage(
                dealer=owner,
                target=owner,
                amount=self.amount,
                props=ValueProp.UNPOWERED,
            )


# =====================================================================
#  31. DyingStarPower  (TemporaryStrength variant -- negative)
# =====================================================================
class DyingStarPower(PowerInstance):
    """Negative temporary Strength (debuff). Same as CrushUnder/DarkShackles.

    C# ref: DyingStarPower.cs extends TemporaryStrengthPower (IsPositive=false).
    StackType.Counter.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER
    is_temporary = True

    def __init__(self, amount: int):
        super().__init__(PowerId.DYING_STAR, amount)

    def after_power_amount_changed(
        self,
        owner: Creature,
        target: Creature,
        power_id: PowerId,
        amount: int,
        applier: Creature | None,
        source: object | None,
        combat: CombatState,
    ) -> None:
        if owner is target and power_id == self.power_id and amount != 0 and not self.consume_ignore_next_instance():
            owner.apply_power(PowerId.STRENGTH, -amount, applier=applier, source=source)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            owner.apply_power(PowerId.STRENGTH, self.amount)
            self.amount = 0  # signal removal


# =====================================================================
#  32. EnfeeblingTouchPower  (TemporaryStrength variant -- negative)
# =====================================================================
class EnfeeblingTouchPower(PowerInstance):
    """Negative temporary Strength (debuff). Same as CrushUnder/DarkShackles.

    C# ref: EnfeeblingTouchPower.cs extends TemporaryStrengthPower (IsPositive=false).
    StackType.Counter.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER
    is_temporary = True

    def __init__(self, amount: int):
        super().__init__(PowerId.ENFEEBLING_TOUCH, amount)

    def after_power_amount_changed(
        self,
        owner: Creature,
        target: Creature,
        power_id: PowerId,
        amount: int,
        applier: Creature | None,
        source: object | None,
        combat: CombatState,
    ) -> None:
        if owner is target and power_id == self.power_id and amount != 0 and not self.consume_ignore_next_instance():
            owner.apply_power(PowerId.STRENGTH, -amount, applier=applier, source=source)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            owner.apply_power(PowerId.STRENGTH, self.amount)
            self.amount = 0  # signal removal


# =====================================================================
#  33. EntropyPower
# =====================================================================
class EntropyPower(PowerInstance):
    """At start of owner's turn, transform Amount cards from hand into
    random cards.

    C# ref: EntropyPower.cs
    - AfterPlayerTurnStart: select Amount cards from hand, transform each
      to a random card.
    StackType.Counter.

    # Simplified: Card selection/transform delegated to combat system.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.ENTROPY, amount)

    def after_player_turn_start(self, owner: Creature, combat: CombatState) -> None:
        if not getattr(owner, "is_player", False):
            return
        transform = getattr(combat, "transform_cards_from_hand", None)
        if transform is not None:
            transform(owner, self.amount)


# =====================================================================
#  34. EscapeArtistPower
# =====================================================================
class EscapeArtistPower(PowerInstance):
    """Countdown buff. Decrements at end of owner's turn. Starts pulsing
    at 1 (visual indicator that the creature will escape next turn).

    C# ref: EscapeArtistPower.cs
    - AfterTurnEnd (owner side): decrement if > 1. At 1, start pulsing.
    StackType.Counter.

    The actual escape logic is in the monster's move state machine.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.ESCAPE_ARTIST, amount)

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            if self.amount > 1:
                self.amount -= 1


# =====================================================================
#  35. FastenPower
# =====================================================================
class FastenPower(PowerInstance):
    """Adds Amount block to Defend-tagged cards (powered block from cards
    with the Defend tag).

    C# ref: FastenPower.cs
    - ModifyBlockAdditive: +Amount if target is owner, powered card block,
      and card has Defend tag.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.FASTEN, amount)

    def modify_block_additive(
        self, owner: Creature, target: Creature, props: ValueProp,
        card_source: object | None = None, card_play: object | None = None,
    ) -> int:
        if target is not owner:
            return 0
        if not props.is_powered_card_or_monster_move_block():
            return 0
        if card_source is not None and CardTag.DEFEND not in getattr(card_source, "tags", set()):
            return 0
        return self.amount


# =====================================================================
#  36. FeralPower
# =====================================================================
class FeralPower(PowerInstance):
    """The first Amount 0-cost Attack cards played each turn return to hand
    instead of going to discard.

    C# ref: FeralPower.cs
    - ModifyCardPlayResultPileTypeAndPosition: return to hand if 0-cost
      Attack and limit not reached.
    - AfterSideTurnStart: reset counter.
    StackType.Counter.

    # Simplified: exposes a check the card-play system calls.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER
    FREE_ATTACK_ENERGY_VALUE = 0

    def __init__(self, amount: int):
        super().__init__(PowerId.FERAL, amount)
        self._zero_cost_attacks_played: int = 0

    def after_power_amount_changed(
        self,
        owner: Creature,
        target: Creature,
        power_id: PowerId,
        amount: int,
        applier: Creature | None,
        source: object | None,
        combat: CombatState,
    ) -> None:
        if owner is target and power_id == PowerId.FERAL and amount > 0:
            self._zero_cost_attacks_played = combat.count_card_play_starts_this_turn(
                owner,
                card_type=CardType.ATTACK,
                energy_value=self.FREE_ATTACK_ENERGY_VALUE,
            )

    def modify_card_play_result_pile_type_and_position(
        self,
        owner: Creature,
        card: object,
        is_auto_play: bool,
        energy_value: int,
        pile_type: PileType,
        position: CardPilePosition,
    ) -> tuple[PileType, CardPilePosition]:
        card_type = getattr(card, "card_type", None) or getattr(card, "type", None)
        if getattr(card, "owner", None) is not owner:
            return pile_type, position
        if card_type != CardType.ATTACK:
            return pile_type, position
        if energy_value != self.FREE_ATTACK_ENERGY_VALUE:
            return pile_type, position
        if self._zero_cost_attacks_played >= self.amount:
            return pile_type, position
        return PileType.HAND, CardPilePosition.TOP

    def after_modifying_card_play_result_pile_or_position(
        self,
        card: object,
        pile_type: PileType,
        position: CardPilePosition,
        combat: CombatState,
    ) -> None:
        self._zero_cost_attacks_played += 1

    def before_side_turn_start(self, owner: Creature, side: CombatSide,
                               combat: CombatState) -> None:
        if side == owner.side:
            self._zero_cost_attacks_played = 0


# =====================================================================
#  37. FlankingPower
# =====================================================================
class FlankingPower(PowerInstance):
    """Debuff. Damage from powered attacks by players OTHER than the
    applier is multiplied by Amount. Removed at end of owner's turn.

    C# ref: FlankingPower.cs
    - ModifyDamageMultiplicative: Amount multiplier for attacks on owner
      from dealers that are NOT the applier.
    - AfterTurnEnd (owner side): remove self.
    StackType.Counter. Instanced.
    """

    power_type = PowerType.DEBUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.FLANKING, amount)
        self._instances: list[tuple[int, Creature | None]] = [(amount, None)]

    def after_power_amount_changed(
        self,
        owner: Creature,
        target: Creature,
        power_id: PowerId,
        amount: int,
        applier: Creature | None,
        source: object | None,
        combat: CombatState,
    ) -> None:
        if owner is not target or power_id != self.power_id or amount <= 0:
            return
        if len(self._instances) == 1 and self._instances[0][1] is None:
            self._instances[0] = (self._instances[0][0], applier)
            return
        self._instances.append((amount, applier))

    def modify_damage_multiplicative(
        self, owner: Creature, dealer: Creature | None, target: Creature, props: ValueProp
    ) -> float:
        if target is not owner:
            return 1.0
        if not props.is_powered_attack():
            return 1.0
        multiplier = 1.0
        for amount, applier in self._instances:
            if amount <= 0 or dealer is (applier or self.applier):
                continue
            multiplier *= float(amount)
        return multiplier

    def after_turn_end(self, owner: Creature, side: CombatSide, combat: CombatState) -> None:
        if side == owner.side:
            combat._remove_power(owner, self.power_id)


# =====================================================================
#  38. FlutterPower
# =====================================================================
class FlutterPower(PowerInstance):
    """Owner takes 50% damage from powered attacks. When hit with unblocked
    powered damage, decrement. At 0, the owner is stunned.

    C# ref: FlutterPower.cs
    - ModifyDamageMultiplicative: 0.5 (DamageDecrease/100) for powered
      attacks targeting owner.
    - AfterDamageReceived: decrement on unblocked powered hit; stun at 0.
    StackType.Counter.

    # Simplified: stun is handled by monster AI.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER
    should_scale_in_multiplayer = True

    _DAMAGE_DECREASE_PCT = 50

    def __init__(self, amount: int):
        super().__init__(PowerId.FLUTTER, amount)

    def modify_damage_multiplicative(
        self, owner: Creature, dealer: Creature | None, target: Creature, props: ValueProp
    ) -> float:
        if target is owner and props.is_powered_attack():
            return self._DAMAGE_DECREASE_PCT / 100.0
        return 1.0

    def after_damage_received(
        self, owner: Creature, target: Creature, dealer: Creature | None,
        damage: int, props: ValueProp, combat: CombatState
    ) -> None:
        if target is owner and damage > 0 and props.is_powered_attack():
            self.amount -= 1
            if self.amount <= 0:
                combat._remove_power(owner, PowerId.FLUTTER)
                combat.stun_enemy(owner)


# =====================================================================
#  39. FocusPower
# =====================================================================
class FocusPower(PowerInstance):
    """Modifies orb passive/evoke values by Amount. AllowNegative.

    C# ref: FocusPower.cs
    - ModifyOrbValue: add Amount to orb value (min 0).
    StackType.Counter. AllowNegative = true.

    # The orb system reads this power's amount when calculating orb values.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER
    allow_negative = True

    def __init__(self, amount: int):
        super().__init__(PowerId.FOCUS, amount)


# =====================================================================
#  40. ForbiddenGrimoirePower
# =====================================================================
class ForbiddenGrimoirePower(PowerInstance):
    """After combat ends, offer Amount card removal(s) as a reward.

    C# ref: ForbiddenGrimoirePower.cs
    - AfterCombatEnd: add Amount CardRemovalReward(s).
    StackType.Counter.

    # Simplified: the reward system reads this power post-combat.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.FORBIDDEN_GRIMOIRE, amount)

    def after_combat_end(self, owner: Creature, combat: CombatState) -> None:
        room = getattr(combat, "room", None)
        state = combat.combat_player_state_for(owner)
        if room is None or state is None or not hasattr(room, "add_extra_reward"):
            return
        from sts2_env.run.reward_objects import RemoveCardReward

        player_id = state.player_state.player_id
        for _ in range(self.amount):
            room.add_extra_reward(player_id, RemoveCardReward(player_id))


# =====================================================================
#  41. ForegoneConclusionPower
# =====================================================================
class ForegoneConclusionPower(PowerInstance):
    """Before hand draw, choose Amount cards from the draw pile to put
    into hand. Then remove self.

    C# ref: ForegoneConclusionPower.cs
    - BeforeHandDraw: shuffle if necessary, let player pick Amount cards
      from draw pile -> hand. Remove self.
    StackType.Counter.

    # Simplified: card selection delegated to combat system.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.FOREGONE_CONCLUSION, amount)

    def before_hand_draw(self, owner: Creature, combat: CombatState) -> None:
        if not getattr(owner, "is_player", False):
            return
        combat._shuffle_if_needed(owner)
        select = getattr(combat, "search_draw_pile_to_hand", None)
        if select is not None:
            select(owner, self.amount)
        self.amount = 0  # signal removal


# =====================================================================
#  42. FriendshipPower
# =====================================================================
class FriendshipPower(PowerInstance):
    """Gain Amount extra max energy each turn.

    C# ref: FriendshipPower.cs
    - ModifyMaxEnergy: +Amount.
    StackType.Counter.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.FRIENDSHIP, amount)

    def modify_max_energy(self, owner: Creature, energy: int) -> int:
        return energy + self.amount


# =====================================================================
#  43. FurnacePower
# =====================================================================
class FurnacePower(PowerInstance):
    """At start of owner's turn, forge Amount times (upgrade a card in hand).

    C# ref: FurnacePower.cs
    - AfterSideTurnStart (owner side): forge Amount times.
    StackType.Counter.

    # Simplified: Regent forge mechanic. Delegates to combat system.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.FURNACE, amount)

    def after_side_turn_start(self, owner: Creature, side: CombatSide,
                              combat: CombatState) -> None:
        if side != owner.side:
            return
        forge = getattr(combat, "forge", None)
        if forge is not None:
            forge(owner, self.amount, source=self)


# =====================================================================
#  44. GenesisPower
# =====================================================================
class GenesisPower(PowerInstance):
    """At energy reset (start of turn), gain Amount stars.

    C# ref: GenesisPower.cs
    - AfterEnergyReset: gain Amount stars for owner's player.
    StackType.Counter.

    # Simplified: Regent star mechanic. Delegates to combat system.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.GENESIS, amount)

    def after_energy_reset(self, owner: Creature, combat: CombatState) -> None:
        if not getattr(owner, "is_player", False):
            return
        gain_stars = getattr(combat, "gain_stars", None)
        if gain_stars is not None:
            gain_stars(owner, self.amount)

    def after_side_turn_start(self, owner: Creature, side: CombatSide,
                              combat: CombatState) -> None:
        if (
            side != owner.side
            or not getattr(owner, "is_player", False)
            or combat.has_energy_reset_this_turn(owner)
        ):
            return
        self.after_energy_reset(owner, combat)


# =====================================================================
#  45. GigantificationPower
# =====================================================================
class GigantificationPower(PowerInstance):
    """The first powered attack card each turn deals 3x damage. Decrements
    after the first modified attack resolves.

    C# ref: GigantificationPower.cs
    - BeforeAttack: record first attack command from a card.
    - ModifyDamageMultiplicative: 3x for the recorded attack.
    - AfterAttack: clear recorded command, decrement.
    StackType.Counter.

    # Simplified: tracks whether the first attack this turn has been
    # boosted. The damage pipeline should check this power.
    """

    power_type = PowerType.BUFF
    stack_type = PowerStackType.COUNTER

    def __init__(self, amount: int):
        super().__init__(PowerId.GIGANTIFICATION, amount)
        self._command_to_modify: object | None = None

    def before_attack(self, owner: Creature, attack: object, combat: CombatState) -> None:
        card = getattr(attack, "model_source", None)
        if card is None or getattr(card, "owner", None) is not owner:
            return
        card_type = getattr(card, "card_type", None) or getattr(card, "type", None)
        if card_type != CardType.ATTACK:
            return
        if not getattr(attack, "damage_props", ValueProp.NONE).is_powered_attack():
            return
        if self._command_to_modify is not None:
            return
        self._command_to_modify = attack

    def modify_damage_multiplicative(
        self, owner: Creature, dealer: Creature | None, target: Creature, props: ValueProp
    ) -> float:
        combat = getattr(owner, "combat_state", None)
        attack = getattr(combat, "active_attack", None) if combat is not None else None
        card = getattr(attack, "model_source", None)
        if card is None and combat is not None:
            card = getattr(combat, "active_card_source", None)
        if card is None:
            return 1.0
        if getattr(card, "owner", None) is not owner:
            return 1.0
        if not props.is_powered_attack():
            return 1.0
        if self._command_to_modify is None or card is getattr(self._command_to_modify, "model_source", None):
            return 3.0
        return 1.0

    def after_attack(self, owner: Creature, attack: object, combat: CombatState) -> None:
        if attack is self._command_to_modify:
            self._command_to_modify = None
            self.amount -= 1
            if self.amount <= 0:
                owner.powers.pop(self.power_id, None)


from sts2_env.core.creature import register_power_class  # noqa: E402


_ALL_POWERS: dict[PowerId, type[PowerInstance]] = {
    PowerId.ACCELERANT: AccelerantPower,
    PowerId.ADAPTABLE: AdaptablePower,
    PowerId.ANTICIPATE: AnticipatePower,
    PowerId.ARSENAL: ArsenalPower,
    PowerId.AUTOMATION: AutomationPower,
    PowerId.BATTLEWORN_DUMMY_TIME_LIMIT: BattlewornDummyTimeLimitPower,
    PowerId.BEACON_OF_HOPE: BeaconOfHopePower,
    PowerId.BLACK_HOLE: BlackHolePower,
    PowerId.BLADE_OF_INK: BladeOfInkPower,
    PowerId.CALL_OF_THE_VOID: CallOfTheVoidPower,
    PowerId.CHAINS_OF_BINDING: ChainsOfBindingPower,
    PowerId.CHILD_OF_THE_STARS: ChildOfTheStarsPower,
    PowerId.CLARITY: ClarityPower,
    PowerId.CONFUSED: ConfusedPower,
    PowerId.CONQUEROR: ConquerorPower,
    PowerId.CONSUMING_SHADOW: ConsumingShadowPower,
    PowerId.COOLANT: CoolantPower,
    PowerId.CORROSIVE_WAVE: CorrosiveWavePower,
    PowerId.CRIMSON_MANTLE: CrimsonMantlePower,
    PowerId.CRUSH_UNDER: CrushUnderPower,
    PowerId.CURIOUS: CuriousPower,
    PowerId.DAMPEN: DampenPower,
    PowerId.DANSE_MACABRE: DanseMacabrePower,
    PowerId.DARK_SHACKLES: DarkShacklesPower,
    PowerId.DEMESNE: DemesnePower,
    PowerId.DEMISE: DemisePower,
    PowerId.DEVOUR_LIFE: DevourLifePower,
    PowerId.DIAMOND_DIADEM: DiamondDiademPower,
    PowerId.DIE_FOR_YOU: DieForYouPower,
    PowerId.DISINTEGRATION: DisintegrationPower,
    PowerId.DYING_STAR: DyingStarPower,
    PowerId.ENFEEBLING_TOUCH: EnfeeblingTouchPower,
    PowerId.ENTROPY: EntropyPower,
    PowerId.ESCAPE_ARTIST: EscapeArtistPower,
    PowerId.FASTEN: FastenPower,
    PowerId.FERAL: FeralPower,
    PowerId.FLANKING: FlankingPower,
    PowerId.FLUTTER: FlutterPower,
    PowerId.FOCUS: FocusPower,
    PowerId.FORBIDDEN_GRIMOIRE: ForbiddenGrimoirePower,
    PowerId.FOREGONE_CONCLUSION: ForegoneConclusionPower,
    PowerId.FRIENDSHIP: FriendshipPower,
    PowerId.FURNACE: FurnacePower,
    PowerId.GENESIS: GenesisPower,
    PowerId.GIGANTIFICATION: GigantificationPower,
}

for _pid, _cls in _ALL_POWERS.items():
    register_power_class(_pid, _cls)
