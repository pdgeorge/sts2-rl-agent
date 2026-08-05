"""STS2 Combat Gymnasium Environment."""

from __future__ import annotations

import logging

import gymnasium
import numpy as np
from gymnasium import spaces

from sts2_env.cards.base import reset_instance_counter
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.core.combat import CombatState
from sts2_env.core.constants import ACTION_END_TURN, ACTION_SPACE_SIZE, IRONCLAD_STARTING_HP
from sts2_env.encounters.act1 import ALL_ACT1_ENCOUNTERS, EncounterSetup
from sts2_env.core.rng import INT_MAX_EXCLUSIVE, Rng
from sts2_env.gym_env.action_space import (
    apply_combat_action,
    get_action_mask,
)
from sts2_env.gym_env.observation import OBS_SIZE, encode_observation
from sts2_env.gym_env.reward import compute_reward, potential

logger = logging.getLogger(__name__)


class STS2CombatEnv(gymnasium.Env):
    """Gymnasium environment for a single STS2 combat encounter.

    Observation: flat float32 vector encoding player state, hand, piles, enemies.
    Action: fixed discrete combat action space including cards, end turn, and potions.

    Two ways to seed a fight:

    1. Starter-deck-vs-random-encounter (the default, what every model so far
       has trained on). `reset` builds an Ironclad starter deck at full HP and
       picks an act 1 encounter at random.
    2. Real situations (`situation_pool`). Pass a list of `CombatSituation`
       objects -- typically loaded from a fixture harvested by
       `scripts/harvest_combat_benchmark.py` -- and `reset` picks one at
       random and calls `to_combat()`. The deck, HP, relics, potions, room
       type and encounter are exactly what a real run presented, which is the
       gap that made a 92% starter-deck model die on floor 8 of a live run:
       the model had never seen a 16-card deck at 40 HP holding three relics.

    The two paths are mutually exclusive. When `situation_pool` is set, the
    starter-deck, encounter-pool, player_hp and player_max_hp arguments are
    ignored -- the situation owns every state field. They stay on the
    constructor for backwards compatibility and for the tests that still
    exercise the starter-deck path.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        encounter_pool: list[EncounterSetup] | None = None,
        player_hp: int = IRONCLAD_STARTING_HP,
        player_max_hp: int = IRONCLAD_STARTING_HP,
        max_turns: int = 200,
        render_mode: str | None = None,
        gamma: float = 0.99,
        max_idle_steps: int = 25,
        situation_pool: list | None = None,
    ):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.encounter_pool = encounter_pool or ALL_ACT1_ENCOUNTERS
        self.player_hp = player_hp
        self.player_max_hp = player_max_hp
        self.max_turns = max_turns
        self.render_mode = render_mode
        # Must match the trainer's discount: potential-based shaping is only
        # policy-invariant when the gamma in gamma*phi(s') - phi(s) is the same
        # one the algorithm discounts with.
        self.gamma = gamma
        # Consecutive rejected actions before the episode is cut off. Rejected
        # actions are meant to be impossible -- the action mask should exclude
        # them -- so any run of them is a mask bug, and this is the backstop that
        # keeps a mask bug from becoming an infinite episode.
        self.max_idle_steps = max_idle_steps
        self._idle_steps = 0
        # When set, reset() draws from this list instead of the starter-deck
        # path. CombatSituation.to_combat() builds the whole fight, so the
        # starter-deck, encounter_pool, player_hp and player_max_hp on this
        # env are not consulted.
        self._situation_pool = list(situation_pool) if situation_pool else None

        self.combat: CombatState | None = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        reset_instance_counter()
        self._idle_steps = 0

        if self._situation_pool is not None:
            # The situation owns deck, relics, potions, HP, encounter, seeds.
            # `to_combat()` mirrors RunManager._enter_combat step-for-step and
            # calls start_combat() internally, so the rest of reset is just
            # observation and mask.
            idx = int(self.np_random.integers(0, len(self._situation_pool)))
            self.combat = self._situation_pool[idx].to_combat()
            obs = encode_observation(self.combat)
            info = {"action_mask": get_action_mask(self.combat)}
            return obs, info

        rng_seed = int(self.np_random.integers(0, INT_MAX_EXCLUSIVE))
        rng = Rng(rng_seed)

        # Create deck
        deck = create_ironclad_starter_deck()

        # Create combat
        self.combat = CombatState(
            player_hp=self.player_hp,
            player_max_hp=self.player_max_hp,
            deck=deck,
            rng_seed=rng_seed,
            character_id="Ironclad",
        )

        # Setup encounter
        encounter_idx = int(self.np_random.integers(0, len(self.encounter_pool)))
        encounter_setup = self.encounter_pool[encounter_idx]
        encounter_setup(self.combat, rng)

        # Start combat
        self.combat.start_combat()

        obs = encode_observation(self.combat)
        info = {"action_mask": get_action_mask(self.combat)}
        return obs, info

    def step(self, action: int):
        assert self.combat is not None, "Must call reset() first"

        # Snapshot before the action: phi(s) cannot be recovered from a mutated
        # CombatState, and turn_count is what the per-turn cost is charged on.
        prev_potential = potential(self.combat)
        prev_turn_count = self.combat.turn_count
        acted = apply_combat_action(self.combat, action)
        if not acted:
            logger.debug("Ignored invalid action %d", action)

        # A rejected action changes nothing -- including turn_count, which is what
        # truncation is keyed to. So a policy that keeps choosing one runs forever:
        # never terminating, never truncating, burning steps. Sampling used to hide
        # this by eventually picking something else, so it showed up as absurd
        # evaluation episode lengths rather than as a hang; a deterministic policy
        # has no such escape and simply stops.
        self._idle_steps = 0 if acted else self._idle_steps + 1

        obs = encode_observation(self.combat)
        terminated = self.combat.is_over
        truncated = (
            self.combat.turn_count > self.max_turns
            or self._idle_steps >= self.max_idle_steps
        )
        reward = compute_reward(
            self.combat,
            prev_potential,
            truncated=truncated,
            turns_elapsed=max(0, self.combat.turn_count - prev_turn_count),
            gamma=self.gamma,
        )
        info = {"action_mask": get_action_mask(self.combat)}

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return the current action mask (for sb3-contrib MaskablePPO)."""
        if self.combat is None:
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
            mask[0] = 1
            return mask
        return get_action_mask(self.combat)

    def render(self):
        if self.render_mode == "ansi" and self.combat is not None:
            return str(self.combat)
        return None
