"""SearchAgent-driven combat decisions for the live bridge path.

The trained PPO model's combat is myopic for the runner -- it sees one turn
with no lookahead, scores one turn, and is the bar that ``combat_v3_overnight``
cleared at 6.7% boss win rate on the harvested benchmark
(`docs/MODELS.md:28`). The turn searcher in :mod:`sts2_env.search.turn_search`
lifts that to 20% by planning a turn: enumerate legal orderings of cards in
hand, play each on a copy, end the turn on the copy, let the enemies
actually reply, and keep the line that came out best. Reported in
``MODELS.md:120``.

This module is the seam: it builds a local ``CombatState`` from the bridge's
``combat_action`` JSON at the start of each fight, mirrors every action the
runner takes into that local sim so it stays in lockstep with the live game,
and asks the ``SearchAgent`` what to play next. The agent's answer is an
integer in the same ``Discrete(115)`` combat action layout the runner already
decodes via :class:`sts2_env.bridge.state_adapter.StateAdapter`, so no other
plumbing changes.

WHAT IS NOT DONE HERE

Drift recovery mid-fight. ``CombatSituation.to_combat`` starts a fresh fight
-- it calls ``start_combat`` which draws the opening hand and fires opening
relics. There is no path to build a mid-fight ``CombatState`` from JSON, so
if the local sim falls out of sync with the live game we cannot rebuild it
at the right point. Instead we log loudly and trust the SearchAgent's
plan-divergence detection (``SearchAgent.act`` at ``turn_search.py:566``
replans when its planned action is rejected by the live mask): a drifted
local sim would shift the mask slightly, the agent would replan against
the wrong state, and the run would continue until the next combat_start,
which rebuilds fresh.

The right next step on this file is a mid-fight reconstruction path through
``CombatState.from_bridge_partial`` -- a to_combat variant that does not
start_combat. Until it exists, this module degrades to the default trained
model whenever drift is severe (see ``_DriftException``), so a live session
run under ``--live-search`` is no worse than one run without the flag.
"""

from __future__ import annotations

import logging
from typing import Any

from sts2_env.core.combat import CombatState
from sts2_env.gym_env.action_space import get_action_mask
from sts2_env.search.evaluate import EvalWeights, DEFAULT_WEIGHTS
from sts2_env.search.situation import CombatSituation
from sts2_env.search.turn_search import SearchAgent

logger = logging.getLogger(__name__)


# Tolerance for the four synchronisation fields. If the local sim and the
# bridge disagree on any of these by more than the reported threshold, we
# consider the local sim to have drifted. The thresholds are deliberately
# loose: the simulator's RNG is bit-parity-equivalent with the game's
# (per PARITY_GAPS.md L27-62), but the first call to to_combat rolls the
# opening shuffle independently of the live game -- which already happened
# before the bridge's first combat_action -- so the local draw-pile order is
# almost certainly different from the live one. HP and block, by contrast,
# derive from actions the runner can see and mirror, so they should match;
# if they don't the local sim missed something (a relic we don't model, an
# enemy power we silently no-op per core/hooks.py:6, or a card effect we
# diverge on per cards/derived_values.py:51).
DRIFT_TOLERANCE = {
    "player_hp": 6,
    "player_block": 6,
    "energy": 1,
    "hand_size": 1,
}


class _DriftException(Exception):
    """Raised when the local sim has drifted too far from the live state."""


class LiveSearch:
    """Keeps a local mirror of the live fight and asks the SearchAgent to play it.

    Each ``decide(bridge_state, None)`` is one combat_action: the bridge has
    sent the state, the runner has decided the previous action was A_prev,
    and now we want to know what to do next.

    Lifecycle per fight:

    1. First ``decide`` on a `combat_action` state whose previous state was
       not a `combat_action` (a turn boundary, or combat_start): build the
       local sim fresh via ``CombatSituation.from_bridge_state(state).to_combat()``.
    2. Subsequent ``decide`` calls in the same fight: apply the previously
       returned action to the local sim (mirror), check the resulting
       local state against the new bridge state's HP, block, energy, hand
       size; if any exceed the tolerance, log loudly and rebuild the
       SearchAgent but keep the local sim -- we don't have a mid-fight
       builder, so we ride the drift until the fight ends.
    3. When the bridge sends a state whose `round` field has incremented
       past the local sim's round, that's the start of a new player turn
       (after the enemies' reply); we advance the local sim accordingly.
    """

    def __init__(
        self,
        *,
        weights: EvalWeights = DEFAULT_WEIGHTS,
        time_budget: float = 3.0,
        lookahead_turns: int = 2,
    ):
        self._search = SearchAgent(
            weights=weights,
            time_budget=time_budget,
            lookahead_turns=lookahead_turns,
        )
        self._local: CombatState | None = None
        self._last_action: int | None = None
        self._last_round: int | None = None
        self._drift_count = 0
        self._rebuild_count = 0

    # -- math ---------------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        return {
            "drift_count": self._drift_count,
            "rebuild_count": self._rebuild_count,
            "searches": self._search.searches,
            "budget_exhausted": self._search.budget_exhausted_count,
        }

    # -- the public API -----------------------------------------------------

    def reset_for_new_fight(self) -> None:
        """Clear the mirror. Called by the runner when a combat starts.

        Strictly speaking the next `decide` call would rebuild anyway, but
        the explicit reset keeps the journal independent of caller order and
        stops a stuck plan from a previous fight leaking into the next.
        """
        self._local = None
        self._last_action = None
        self._last_round = None
        self._search = SearchAgent(
            weights=self._search.weights,
            time_budget=self._search.time_budget,
            lookahead_turns=self._search.lookahead_turns,
        )

    def decide(self, bridge_state: dict[str, Any], *, prev_action: int | None = None) -> int:
        """Return the next action index for the runner to send to the bridge."""
        from sts2_env.gym_env.action_space import apply_combat_action

        if self._local is None:
            # First call of a fight. Build the local sim fresh from the bridge
            # state; we cannot have drift on the first call by construction.
            self._local = _build_local(bridge_state)
            self._last_action = None
            self._last_round = self._local.turn_count
            self._rebuild_count += 1
        else:
            # Subsequent call. Catch the local sim up with the live game by
            # applying the action we previously returned.
            if self._last_action is not None:
                apply_combat_action(self._local, self._last_action)
                self._last_action = None
            # Verify drift; if the bridge reports a state meaningfully
            # different from our local mirror, log and continue -- the
            # SearchAgent will replan based on the (drifted) local state, its
            # plan-divergence handler will catch any obvious gap, and the
            # fight ends with a fresh rebuild on the next round.
            self._check_drift(bridge_state)

        # The bridge's report of which round we're on might have incremented
        # past the local sim's turn_count -- meaning the live game moved
        # into the enemy reply and back. SearchAgent's plan handles the turn
        # boundary (its _plan empties at end_turn, replans next call), so no
        # manual fiddling with the local sim's round field is needed here.

        try:
            action = self._search.act(self._local)
        except Exception:
            logger.exception("search agent raised; falling back to END_TURN")
            return 0  # ACTION_END_TURN

        self._last_action = action
        return action

    # -- internals ----------------------------------------------------------

    def _check_drift(self, bridge_state: dict[str, Any]) -> None:
        """Log when the local sim has drifted from the bridge-reported state."""
        from sts2_env.bridge.state_adapter import StateAdapter

        crash = bridge_state.get("combat_state") or bridge_state
        player = crash.get("player", {}) if isinstance(crash, dict) else {}
        if not player or self._local is None:
            return

        local_hp = self._local.player.current_hp
        local_block = self._local.player.block
        local_energy = self._local.primary_player_energy if hasattr(
            self._local, "primary_player_energy"
        ) else 0
        # The player's energy is most reliably read off the primary player's
        # combat state, but it varies between sites; fall back to combat.energy
        # which is the same thing on a one-player run.
        try:
            local_energy = self._local.energy
        except AttributeError:
            pass
        local_hand = len(self._local.hand)

        bridge_hp = int(player.get("hp", 0))
        bridge_block = int(player.get("block", 0))
        bridge_energy = int(player.get("energy", 0))
        bridge_hand = len(crash.get("hand", [])) if isinstance(crash, dict) else 0

        drift_report: dict[str, tuple[int, int]] = {}
        if abs(local_hp - bridge_hp) > DRIFT_TOLERANCE["player_hp"]:
            drift_report["player_hp"] = (local_hp, bridge_hp)
        if abs(local_block - bridge_block) > DRIFT_TOLERANCE["player_block"]:
            drift_report["player_block"] = (local_block, bridge_block)
        if abs(local_energy - bridge_energy) > DRIFT_TOLERANCE["energy"]:
            drift_report["energy"] = (local_energy, bridge_energy)
        if abs(local_hand - bridge_hand) > DRIFT_TOLERANCE["hand_size"]:
            drift_report["hand_size"] = (local_hand, bridge_hand)

        if drift_report:
            self._drift_count += 1
            logger.warning(
                "live_search drift: %s. Continuing with the local sim; "
                "SearchAgent will replan if the mask diverges.",
                {k: f"{a}->{b}" for k, (a, b) in drift_report.items()},
            )


def _build_local(bridge_state: dict[str, Any]) -> CombatState:
    """Build a fresh CombatState from the bridge's combat_action JSON.

    Delegates to :meth:`CombatSituation.from_bridge_state`, which raises
    `ValueError` if the mod has not been patched (Phase 1.1) to send
    `encounter` / `encounter_seed` -- which is the intended loud failure.
    A silent fallback would clone a different fight from the one on screen,
    which is worse than the error.
    """
    situation = CombatSituation.from_bridge_state(bridge_state)
    return situation.to_combat()