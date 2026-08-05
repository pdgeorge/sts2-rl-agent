"""Action space and masking logic."""

from __future__ import annotations

import numpy as np

from sts2_env.core.combat import CombatState
from sts2_env.core.creature import Creature
from sts2_env.core.enums import PotionTargetType, TargetType
from sts2_env.core.constants import (
    ACTION_END_TURN,
    ACTION_SPACE_SIZE,
    MAX_ENEMIES,
    MAX_HAND_SIZE,
    MAX_POTION_SLOTS,
    POTION_ACTION_SIZE,
    POTION_ACTION_START,
    POTION_TARGET_OPTIONS,
)


def is_potion_action(action: int) -> bool:
    return POTION_ACTION_START <= action < ACTION_SPACE_SIZE


def action_to_card_and_target(action: int) -> tuple[int | None, int | None]:
    """Convert action index to (hand_index, target_index).

    Returns (None, None) for end turn.
    Returns (hand_idx, None) for self/all-target cards.
    Returns (hand_idx, enemy_idx) for targeted cards.
    Returns (None, None) for potion actions.
    """
    if action == ACTION_END_TURN or is_potion_action(action):
        return None, None
    if action <= MAX_HAND_SIZE:
        return action - 1, None
    adjusted = action - 1 - MAX_HAND_SIZE
    hand_idx = adjusted // MAX_ENEMIES
    enemy_idx = adjusted % MAX_ENEMIES
    return hand_idx, enemy_idx


def action_to_potion_and_target(action: int) -> tuple[int | None, int | None]:
    """Convert action index to (slot_index, target_index) for potion actions."""
    if not is_potion_action(action):
        return None, None
    adjusted = action - POTION_ACTION_START
    if adjusted < 0 or adjusted >= POTION_ACTION_SIZE:
        return None, None
    slot_idx = adjusted // POTION_TARGET_OPTIONS
    if slot_idx >= MAX_POTION_SLOTS:
        return None, None
    target_offset = adjusted % POTION_TARGET_OPTIONS
    if target_offset == 0:
        return slot_idx, None
    return slot_idx, target_offset - 1


def apply_combat_action(combat: CombatState, action: int) -> bool:
    """Perform `action` on `combat`. Returns whether anything actually happened.

    The single definition of what an action index *does*, as `action_to_*` above
    are the single definition of what one *means*. Both the training env and the
    searcher call this, because an agent that plays a line in search and then
    sends a different one to the game is the bug class that has cost this project
    the most runs -- and two copies of this dispatch is how that happens.

    A False return means the action was rejected and the state is unchanged.
    That should be impossible when the caller respects `get_action_mask`, so it
    is a mask bug rather than a legal outcome; callers cut the episode off after
    a run of them rather than looping forever.
    """
    if combat.pending_choice is not None:
        if action == ACTION_END_TURN:
            combat.resolve_pending_choice(None)
        else:
            combat.resolve_pending_choice(action - 1)
        return True

    if action == ACTION_END_TURN:
        combat.end_player_turn()
        return True

    if is_potion_action(action):
        slot_idx, target_idx = action_to_potion_and_target(action)
        return bool(
            slot_idx is not None
            and combat.use_potion(slot_idx, target_index=target_idx)
        )

    hand_idx, target_idx = action_to_card_and_target(action)
    return bool(hand_idx is not None and combat.play_card(hand_idx, target_idx))


def get_action_mask(combat: CombatState, owner: Creature | None = None) -> np.ndarray:
    """Return boolean mask of valid actions."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
    acting_owner = owner or combat.primary_player
    owner_state = combat.combat_player_state_for(acting_owner)

    if combat.is_over:
        mask[ACTION_END_TURN] = 1
        return mask

    if combat.pending_choice is not None:
        if combat.pending_choice.can_confirm():
            mask[ACTION_END_TURN] = 1
        for i in range(min(combat.pending_choice.num_options, ACTION_SPACE_SIZE - 1)):
            if combat.pending_choice.is_multi:
                if combat.pending_choice.can_toggle(i):
                    mask[1 + i] = 1
            else:
                mask[1 + i] = 1
        return mask

    # Can always end turn
    mask[ACTION_END_TURN] = 1

    hand = owner_state.hand if owner_state is not None else combat.hand
    for i in range(min(len(hand), MAX_HAND_SIZE)):
        card = hand[i]
        if not combat.can_play_card(card):
            continue

        target_type = card.target_type_for(acting_owner)
        if target_type in (TargetType.SELF, TargetType.NONE, TargetType.ALL_ENEMIES, TargetType.ALL_ALLIES):
            mask[1 + i] = 1
        elif target_type in (TargetType.ANY_ENEMY, TargetType.RANDOM_ENEMY):
            for j in range(min(len(combat.enemies), MAX_ENEMIES)):
                if combat.enemies[j].is_alive:
                    mask[1 + MAX_HAND_SIZE + i * MAX_ENEMIES + j] = 1
        elif target_type == TargetType.ANY_ALLY:
            allies = combat.get_player_allies_of(acting_owner)
            for j in range(min(len(allies), MAX_ENEMIES)):
                mask[1 + MAX_HAND_SIZE + i * MAX_ENEMIES + j] = 1

    if ACTION_SPACE_SIZE > POTION_ACTION_START:
        potions = owner_state.potions if owner_state is not None else combat.potions
        for slot_idx in range(min(len(potions), MAX_POTION_SLOTS)):
            potion = potions[slot_idx]
            if potion is None:
                continue

            action_base = POTION_ACTION_START + slot_idx * POTION_TARGET_OPTIONS
            if potion.target_type in (
                PotionTargetType.SELF,
                PotionTargetType.ANY_PLAYER,
                PotionTargetType.ALL_ENEMIES,
            ):
                if combat.can_use_potion(slot_idx, owner=acting_owner):
                    mask[action_base] = 1
                continue

            if potion.target_type == PotionTargetType.ANY_ENEMY:
                for j in range(min(len(combat.enemies), MAX_ENEMIES)):
                    if combat.can_use_potion(slot_idx, target_index=j, owner=acting_owner):
                        mask[action_base + 1 + j] = 1

    return mask
