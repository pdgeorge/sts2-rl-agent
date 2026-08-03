"""The simulator and the bridge must produce the same hashed relic/potion vector.

With feature hashing there are no longer guaranteed-unique columns for every item.
Instead the invariant is: both sides use the SAME hasher with the SAME seed, so
the same input (e.g. Burning Blood) produces the SAME output vector on both
sides. The policy learns "this hash pattern means Burning Blood" rather than
"column 277 means Burning Blood."

If they disagree, nothing raises. The policy trains believing a pattern means
Burning Blood and plays live believing something else, and the only symptom is
that it plays worse than it evaluated. These tests verify byte-identical
agreement between the simulator's encoder and the bridge's encoder.
"""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.bridge.run_adapter import RunStateAdapter
from sts2_env.gym_env.relic_potion_encoding import (
    RELIC_POTION_OBS_SIZE,
    encode_relics_and_potions,
)
from sts2_env.gym_env.run_env import RUN_OBS_SIZE, STS2RunEnv
from sts2_env.gym_env.hashing import RELIC_BUCKETS, POTION_BUCKETS

BLOCK_START = RUN_OBS_SIZE - RELIC_POTION_OBS_SIZE


def _bridge_obs(**state):
    base = {"type": "map_select", "act": 1, "floor": 3, "nodes": []}
    base.update(state)
    return RunStateAdapter().encode_observation(base)


def test_the_block_is_where_both_sides_think_it_is():
    """Pinned: 131 combat + 1893 identity + 32 deck features + 20 run-level + 126 choices.

    Identity grew from 1690 to 1893 at layout v2, when card identity moved from
    feature hashing to frozen text embeddings: the hand block became per-slot
    (10 x 65) rather than a 256-bucket bag, and the deck block became a pooled 65.
    """
    from sts2_env.gym_env.deck_features import DECK_FEATURE_SIZE
    assert BLOCK_START == 2170 + DECK_FEATURE_SIZE
    assert RUN_OBS_SIZE == 2586


def test_the_simulators_starting_relic_matches_the_bridge():
    env = STS2RunEnv(max_steps=50)
    sim_obs, _ = env.reset(seed=7)

    bridge_obs = _bridge_obs(relics=["BURNING_BLOOD"])

    # Compare the full relic/potion block byte-for-byte.
    sim_block = sim_obs[BLOCK_START:BLOCK_START + RELIC_POTION_OBS_SIZE]
    bridge_block = bridge_obs[BLOCK_START:BLOCK_START + RELIC_POTION_OBS_SIZE]
    np.testing.assert_array_equal(sim_block, bridge_block)
    assert np.any(sim_block != 0.0), "Burning Blood should hash to nonzero"


def test_a_relic_the_simulator_holds_reads_the_same_from_the_bridge():
    """Give the simulator a second relic and check both encodings agree."""
    env = STS2RunEnv(max_steps=50)
    env.reset(seed=7)
    env._mgr.run_state.player.relics.append("BLACK_BLOOD")
    sim_obs = env._encode_obs()

    bridge_obs = _bridge_obs(relics=["BURNING_BLOOD", "BLACK_BLOOD"])

    sim_block = sim_obs[BLOCK_START:BLOCK_START + RELIC_POTION_OBS_SIZE]
    bridge_block = bridge_obs[BLOCK_START:BLOCK_START + RELIC_POTION_OBS_SIZE]
    np.testing.assert_array_equal(sim_block, bridge_block)
    assert np.any(sim_block != 0.0)


def test_potion_slots_line_up_across_the_two_sides():
    """A potion in slot 1 must produce a different vector than slot 0 on both
    sides, because the slot index is baked into the hash key."""
    from sts2_env.potions.base import create_potion

    env = STS2RunEnv(max_steps=50)
    env.reset(seed=7)
    player = env._mgr.run_state.player
    player.potions = [None, create_potion("BlockPotion", slot=1)]
    sim_obs = env._encode_obs()

    # Include the simulator's starting relic so the full block matches.
    bridge_obs = _bridge_obs(
        relics=[r for r in player.relics],
        potion_slots=[None, "BlockPotion"],
    )

    sim_block = sim_obs[BLOCK_START:BLOCK_START + RELIC_POTION_OBS_SIZE]
    bridge_block = bridge_obs[BLOCK_START:BLOCK_START + RELIC_POTION_OBS_SIZE]
    np.testing.assert_array_equal(sim_block, bridge_block)

    # Slot 0 empty vs slot 1 occupied should differ.
    slot0 = _bridge_obs(potion_slots=["BlockPotion", None])
    slot1 = _bridge_obs(potion_slots=[None, "BlockPotion"])
    assert not np.array_equal(slot0, slot1)


def test_a_mod_that_sends_nothing_leaves_the_block_empty():
    """Documents the risk rather than hiding it.

    An older mod sends no relics field, and this reads as "owns nothing" -- wrong
    rather than absent. agent_runner warns on connect; this pins the behaviour so
    the warning cannot quietly stop mattering.
    """
    obs = _bridge_obs()
    assert obs[BLOCK_START:].sum() == 0.0


@pytest.mark.parametrize("relic", ["BURNING_BLOOD", "BLACK_BLOOD", "CRACKED_CORE"])
def test_each_relic_gets_a_unique_hash_signature(relic):
    obs = _bridge_obs(relics=[relic])
    block = obs[BLOCK_START:BLOCK_START + RELIC_BUCKETS]
    assert np.any(block != 0.0), f"{relic} should hash to nonzero"
    # Different relics must produce different hash signatures.
    for other in ("BURNING_BLOOD", "BLACK_BLOOD", "CRACKED_CORE"):
        if other == relic:
            continue
        other_obs = _bridge_obs(relics=[other])
        other_block = other_obs[BLOCK_START:BLOCK_START + RELIC_BUCKETS]
        assert not np.array_equal(block, other_block), f"{relic} vs {other} must differ"
