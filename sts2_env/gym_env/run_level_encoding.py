"""The 20 run-level dims, encoded once and used by both sides.

The simulator reads them from RunState; the bridge reads them from a JSON state
message. Before this they were built in two places, and the bridge's version did
not exist at all -- the adapter emitted only the 131-dim combat vector, so a
full-run model could not be pointed at the live game.

Building them twice would be the same mistake in slower motion. A field scaled by
50 in one and 20 in the other, or a phase one-hot in a different order, produces a
policy that reads the live game wrongly and shows it only as worse play. So this
takes primitives and both callers pass what they have.

Two conversions that are easy to get wrong and are therefore done here:

  * `act` on the wire is 1-based, because that is what a human reading a log
    wants. The observation uses RunState.current_act_index, which is 0-based.
  * Room type: the simulator has RoomType.ELITE / BOSS, the game sends
    MapPointType "Elite" / "Boss". Both normalise through the same helper the
    choice encoder uses.
"""

from __future__ import annotations

import numpy as np

from sts2_env.gym_env.choice_encoding import normalize_enum_name

RUN_LEVEL_SIZE = 20

# Scales must match what run_env has always used, or a model trained before this
# module existed would read every one of these dims differently.
ACT_SCALE = 3.0
TOTAL_FLOOR_SCALE = 50.0
ACT_FLOOR_SCALE = 20.0
GOLD_SCALE = 1000.0
DECK_SIZE_SCALE = 40.0
RELIC_COUNT_SCALE = 30.0
MAX_POTION_SLOTS_SCALE = 5.0
ASCENSION_SCALE = 20.0

# Order is the observation layout. Changing it silently reinterprets every dim.
PHASE_ORDER = (
    "MAP_CHOICE", "COMBAT", "CARD_REWARD", "BOSS_RELIC",
    "SHOP", "REST_SITE", "EVENT", "TREASURE",
)
NUM_PHASES = len(PHASE_ORDER)

# Bridge state type -> the phase the observation encodes. Several bridge states
# map to one phase because they are steps within it, which is what the runner's
# own phase mapping already does.
BRIDGE_STATE_TO_PHASE = {
    "map_select": "MAP_CHOICE",
    "combat_action": "COMBAT",
    "card_select": "COMBAT",
    "card_reward": "CARD_REWARD",
    "card_bundle": "CARD_REWARD",
    "reward_screen": "CARD_REWARD",
    "crystal_sphere": "CARD_REWARD",
    "boss_relic": "BOSS_RELIC",
    "shop": "SHOP",
    "rest_site": "REST_SITE",
    "event": "EVENT",
    "treasure": "TREASURE",
}


def encode_run_level(
    *,
    act_index: int = 0,
    total_floor: int = 0,
    act_floor: int = 0,
    hp: int = 0,
    max_hp: int = 0,
    gold: int = 0,
    deck_size: int = 0,
    relic_count: int = 0,
    potion_count: int = 0,
    max_potion_slots: int = 0,
    phase: str = "",
    ascension: int = 0,
    is_elite: bool = False,
    is_boss: bool = False,
) -> np.ndarray:
    """The 20 dims, in the layout run_env has always written."""
    out = np.zeros(RUN_LEVEL_SIZE, dtype=np.float32)

    out[0] = act_index / ACT_SCALE
    out[1] = total_floor / TOTAL_FLOOR_SCALE
    out[2] = act_floor / ACT_FLOOR_SCALE

    out[3] = hp / max(max_hp, 1)
    out[4] = gold / GOLD_SCALE

    out[5] = deck_size / DECK_SIZE_SCALE
    out[6] = relic_count / RELIC_COUNT_SCALE

    out[7] = potion_count / max(max_potion_slots, 1)
    out[8] = max_potion_slots / MAX_POTION_SLOTS_SCALE

    normalized_phase = normalize_enum_name(phase)
    if normalized_phase in PHASE_ORDER:
        out[9 + PHASE_ORDER.index(normalized_phase)] = 1.0

    out[17] = ascension / ASCENSION_SCALE
    out[18] = 1.0 if is_elite else 0.0
    out[19] = 1.0 if is_boss else 0.0
    return out


def run_level_from_bridge_state(state: dict) -> dict:
    """Bridge JSON to encode_run_level kwargs.

    Absent fields fall back to zero. That is correct for a state the mod has not
    been taught to carry, and wrong-looking for one it has -- so RlRunInfo attaches
    every field to every state rather than only the ones a given screen cares
    about.
    """
    room = normalize_enum_name(str(state.get("room_type", "")))
    phase = BRIDGE_STATE_TO_PHASE.get(str(state.get("type", "")), "")

    return {
        # The wire is 1-based; the observation is 0-based. Off by one here would
        # shift every act reading by a third of the scale.
        "act_index": max(0, int(state.get("act", 1)) - 1),
        "total_floor": int(state.get("floor", 0) or 0),
        "act_floor": int(state.get("act_floor", 0) or 0),
        "hp": int(state.get("run_hp", 0) or 0),
        "max_hp": int(state.get("run_max_hp", 0) or 0),
        "gold": int(state.get("gold", 0) or 0),
        "deck_size": int(state.get("deck_size", 0) or 0),
        "relic_count": int(state.get("relic_count", 0) or 0),
        "potion_count": int(state.get("potion_count", 0) or 0),
        "max_potion_slots": int(state.get("max_potion_slots", 0) or 0),
        "phase": phase,
        "ascension": int(state.get("ascension", 0) or 0),
        "is_elite": room == "ELITE",
        "is_boss": room == "BOSS",
    }
