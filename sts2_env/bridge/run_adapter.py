"""Full-run observation and action mask, built from a bridge state message.

StateAdapter emits the 131-dim combat observation and the 115-action combat mask,
which is why a full-run model could not be pointed at the live game at all: it
needs 277 and 157, and MaskablePPO.load refuses the mismatch. That refusal is
correct and it is also the entire blocker.

This is the other half. It reuses the combat adapter for the first 131 dims, then
adds the same run-level and choice blocks the simulator writes, through the same
encoders -- so the two agree by construction rather than by two implementations
being kept in step by hand.

WHAT IT DOES NOT DO

It does not invent information. Where the bridge cannot say something the dim is
zero, and the honest fix is to teach the mod to send it rather than to guess here.
Every field the run-level block needs is sent as of RlRunInfo; if that stops being
true, tests/test_run_adapter_parity.py fails rather than the agent quietly
misreading a run.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sts2_env.bridge.state_adapter import StateAdapter
from sts2_env.core.constants import ACTION_SPACE_SIZE as COMBAT_ACTION_SPACE_SIZE
from sts2_env.gym_env.choice_encoding import (
    CHOICE_OBS_SIZE,
    choices_from_bridge_state,
    encode_choices,
)
from sts2_env.gym_env.observation import OBS_SIZE as COMBAT_OBS_SIZE
from sts2_env.gym_env.run_env import (
    _BOSS_RELIC_START,
    _CARD_RWD_START,
    _EVENT_START,
    _MAP_START,
    _REST_START,
    _SHOP_START,
    _TREASURE_START,
    RUN_OBS_SIZE,
    TOTAL_ACTIONS,
)
from sts2_env.gym_env.entity_encoding import (
    ENTITY_OBS_SIZE,
    encode_entities,
    entities_from_bridge_state,
)
from sts2_env.gym_env.deck_features import (
    DECK_FEATURE_SIZE,
    encode_deck_features,
)
from sts2_env.gym_env.relic_potion_encoding import (
    RELIC_POTION_OBS_SIZE,
    encode_relics_and_potions,
    potion_slots_from_bridge_state,
    relics_from_bridge_state,
)
from sts2_env.gym_env.run_level_encoding import (
    RUN_LEVEL_SIZE,
    encode_run_level,
    run_level_from_bridge_state,
)

# Which slice of the run action space each bridge state chooses from, and how many
# options that state can offer. Mirrors run_env's action layout; the layout is
# imported rather than restated so it cannot drift.
_STATE_ACTION_SLICE = {
    "map_select": (_MAP_START, 5),
    "card_reward": (_CARD_RWD_START, 4),
    "card_bundle": (_CARD_RWD_START, 4),
    "reward_screen": (_CARD_RWD_START, 4),
    "boss_relic": (_BOSS_RELIC_START, 3),
    "shop": (_SHOP_START, 10),
    "rest_site": (_REST_START, 5),
    "event": (_EVENT_START, 4),
    "treasure": (_TREASURE_START, 1),
}

# Which key holds the list of options, per state.
_STATE_OPTION_KEY = {
    "map_select": "nodes",
    "card_reward": "cards",
    "card_bundle": "bundles",
    "reward_screen": "options",
    "boss_relic": "relics",
    "shop": "items",
    "rest_site": "options",
    "event": "options",
}


class RunStateAdapter:
    """Bridge state -> full-run observation and action mask."""

    def __init__(self) -> None:
        self._combat = StateAdapter()

    def encode_observation(self, state: dict[str, Any]) -> np.ndarray:
        """The RUN_OBS_SIZE-dim observation, laid out exactly as run_env writes it."""
        obs = np.zeros(RUN_OBS_SIZE, dtype=np.float32)

        # Combat block. Only meaningful in combat; elsewhere it stays zero, which
        # is what the simulator does when there is no combat state.
        if state.get("type") == "combat_action":
            combat_obs = self._combat.encode_observation(state)
            obs[:COMBAT_OBS_SIZE] = combat_obs[:COMBAT_OBS_SIZE]

        idx = COMBAT_OBS_SIZE

        # Identity block, in the same order run_env writes it. Present outside
        # combat too: the deck is what a card reward is deciding about.
        obs[idx:idx + ENTITY_OBS_SIZE] = encode_entities(
            **entities_from_bridge_state(state))
        idx += ENTITY_OBS_SIZE

        # Deck quality features -- same block run_env writes.
        deck = state.get("deck") or []
        obs[idx:idx + DECK_FEATURE_SIZE] = encode_deck_features(deck)
        idx += DECK_FEATURE_SIZE

        obs[idx:idx + RUN_LEVEL_SIZE] = encode_run_level(**run_level_from_bridge_state(state))
        idx += RUN_LEVEL_SIZE

        obs[idx:idx + CHOICE_OBS_SIZE] = encode_choices(**choices_from_bridge_state(state))
        idx += CHOICE_OBS_SIZE

        # Relics and potions by identity, the same block run_env writes. The mod
        # has to send "relics" and "potion_slots"; an older mod sends neither and
        # this block stays zero, which reads as "she owns nothing". That is wrong
        # rather than absent, so agent_runner checks for the fields on connect
        # instead of letting a relic-blind run look normal.
        obs[idx:idx + RELIC_POTION_OBS_SIZE] = encode_relics_and_potions(
            relics_from_bridge_state(state),
            potion_slots_from_bridge_state(state),
        )

        # Same clip the simulator applies so the two sides stay byte-identical.
        from sts2_env.gym_env.run_env import OBS_VALUE_LOW, OBS_VALUE_HIGH
        np.clip(obs, OBS_VALUE_LOW, OBS_VALUE_HIGH, out=obs)
        return obs

    def compute_action_mask(self, state: dict[str, Any]) -> np.ndarray:
        """The 157-action mask.

        In combat this is the combat mask in the first 115 slots. Elsewhere it is
        the slice for that decision, one bit per option actually offered -- so the
        policy cannot pick a card reward slot that is not there.
        """
        mask = np.zeros(TOTAL_ACTIONS, dtype=np.int8)
        state_type = str(state.get("type", ""))

        if state_type == "combat_action":
            mask[:COMBAT_ACTION_SPACE_SIZE] = self._combat.compute_action_mask(state)
            return mask

        entry = _STATE_ACTION_SLICE.get(state_type)
        if entry is None:
            # Nothing known to choose from. Leaving the mask empty would make
            # MaskablePPO raise, so fall back to the first combat action, which
            # the runner treats as "no opinion" and resolves with its heuristic.
            mask[0] = 1
            return mask

        start, width = entry
        options = state.get(_STATE_OPTION_KEY.get(state_type, "options"), []) or []
        count = min(len(options), width) if options else 0

        if count == 0:
            # A decision with no listed options still has to be answerable.
            mask[start] = 1
            return mask

        # The last card_reward slot is the skip, always -- run_env's
        # _step_card_reward treats local==3 as skip unconditionally and puts a
        # fourth card in the separate card_reward_extra region. So the cards here
        # get width-1 slots, not width.
        #
        # Without this cap, a reward offering four cards unmasked the skip slot as
        # "card 4". The model, trained on the simulator's meaning, would select it
        # intending to skip and the bridge would pick a card -- a silent
        # mistranslation of the policy's actual decision, with nothing logged.
        if state_type == "card_reward":
            count = min(count, width - 1)

        for offset in range(count):
            mask[start + offset] = 1

        if state_type == "card_reward" and state.get("can_skip"):
            mask[start + width - 1] = 1

        if not mask[start:start + width].any():
            # Every slot masked out (no options, or skip unavailable) still has to
            # be answerable; an all-zero mask makes MaskablePPO raise mid-run.
            mask[start] = 1
        return mask

    def decode_action(self, action: int, state: dict[str, Any]) -> dict[str, Any]:
        """Run action index -> the bridge action to send.

        Combat delegates to the combat adapter, which already knows the card and
        potion layout. Everything else is a choose-by-index, which is what every
        non-combat bridge screen accepts.
        """
        state_type = str(state.get("type", ""))
        if state_type == "combat_action":
            return self._combat.decode_action(action, state)

        entry = _STATE_ACTION_SLICE.get(state_type)
        if entry is None:
            return {"action": "choose", "index": 0}

        start, width = entry
        offset = max(0, min(action - start, width - 1))

        # Matches the mask above and run_env: the last slot means skip whether or
        # not the game can honour it. Decoding it as "choose index 3" instead would
        # turn an intended skip into a card pick.
        if state_type == "card_reward" and offset == width - 1:
            return {"action": "skip"}
        return {"action": "choose", "index": offset}
