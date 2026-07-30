"""The simulator and the bridge must put the same relic in the same column.

If they disagree, nothing raises. The policy trains believing column 277 means
Burning Blood and plays live believing something else, and the only symptom is
that it plays worse than it evaluated. That is the failure mode this whole file
exists for, and it has already happened three times on other fields.

Golden values are pinned deliberately. Building the expectation with the same
encoder under test would pass no matter what the encoder did -- an earlier parity
suite here did exactly that, and a scale change from 1000 to 100 passed every
test.
"""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.bridge.run_adapter import RunStateAdapter
from sts2_env.gym_env.relic_potion_encoding import (
    POTION_TYPE_COUNT,
    RELIC_OBS_SIZE,
    RELIC_POTION_OBS_SIZE,
    potion_index,
    relic_index,
)
from sts2_env.gym_env.run_env import RUN_OBS_SIZE, STS2RunEnv

BLOCK_START = RUN_OBS_SIZE - RELIC_POTION_OBS_SIZE


def _bridge_obs(**state):
    base = {"type": "map_select", "act": 1, "floor": 3, "nodes": []}
    base.update(state)
    return RunStateAdapter().encode_observation(base)


def test_the_block_is_where_both_sides_think_it_is():
    """Pinned: 131 combat + 3142 identity + 20 run-level + 126 choices."""
    assert BLOCK_START == 3419
    assert RUN_OBS_SIZE == 4038


def test_the_simulators_starting_relic_is_in_the_column_the_bridge_uses():
    env = STS2RunEnv(max_steps=50)
    sim_obs, _ = env.reset(seed=7)

    bridge_obs = _bridge_obs(relics=["BURNING_BLOOD"])

    col = BLOCK_START + relic_index("BURNING_BLOOD")
    assert sim_obs[col] == 1.0, "simulator sets Burning Blood"
    assert bridge_obs[col] == 1.0, "bridge sets the same column"


def test_a_relic_the_simulator_holds_reads_the_same_from_the_bridge():
    """Give the simulator a second relic and check both encodings agree."""
    env = STS2RunEnv(max_steps=50)
    env.reset(seed=7)
    env._mgr.run_state.player.relics.append("BLACK_BLOOD")
    sim_obs = env._encode_obs()

    bridge_obs = _bridge_obs(relics=["BURNING_BLOOD", "BLACK_BLOOD"])

    sim_block = sim_obs[BLOCK_START:BLOCK_START + RELIC_OBS_SIZE]
    bridge_block = bridge_obs[BLOCK_START:BLOCK_START + RELIC_OBS_SIZE]
    np.testing.assert_array_equal(sim_block, bridge_block)
    assert sim_block.sum() == 2.0


def test_potion_slots_line_up_across_the_two_sides():
    """A potion in slot 1 must be slot 1 on both sides, not compacted to 0."""
    from sts2_env.potions.base import create_potion

    env = STS2RunEnv(max_steps=50)
    env.reset(seed=7)
    player = env._mgr.run_state.player
    player.potions = [None, create_potion("BlockPotion", slot=1)]
    sim_obs = env._encode_obs()

    bridge_obs = _bridge_obs(potion_slots=[None, "BlockPotion"])

    start = BLOCK_START + RELIC_OBS_SIZE
    np.testing.assert_array_equal(
        sim_obs[start:start + 5 * POTION_TYPE_COUNT],
        bridge_obs[start:start + 5 * POTION_TYPE_COUNT],
    )
    occupied = start + POTION_TYPE_COUNT + potion_index("BlockPotion")
    assert sim_obs[occupied] == 1.0, "slot 1, not slot 0"


def test_a_mod_that_sends_nothing_leaves_the_block_empty():
    """Documents the risk rather than hiding it.

    An older mod sends no relics field, and this reads as "owns nothing" -- wrong
    rather than absent. agent_runner warns on connect; this pins the behaviour so
    the warning cannot quietly stop mattering.
    """
    obs = _bridge_obs()
    assert obs[BLOCK_START:].sum() == 0.0


@pytest.mark.parametrize("relic", ["BURNING_BLOOD", "BLACK_BLOOD", "CRACKED_CORE"])
def test_each_relic_gets_its_own_column(relic):
    obs = _bridge_obs(relics=[relic])
    col = BLOCK_START + relic_index(relic)
    assert obs[col] == 1.0
    assert obs[BLOCK_START:BLOCK_START + RELIC_OBS_SIZE].sum() == 1.0
