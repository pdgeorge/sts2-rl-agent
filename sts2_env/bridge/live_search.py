"""SearchAgent-driven combat decisions for the live bridge path.

The trained PPO model's combat is myopic for the runner -- it sees one turn
with no lookahead, scores one turn, and is the bar that ``combat_v3_overnight``
cleared at 6.7% boss win rate on the harvested benchmark
(`docs/MODELS.md:28`). The turn searcher in :mod:`sts2_env.search.turn_search`
lifts that to 20% by planning a turn: enumerate legal orderings of cards in
hand, play each on a copy, end the turn on the copy, let the enemies
actually reply, and keep the line that came out best. Reported in
``MODELS.md:120``.

This module is the seam: it reconstructs a local ``CombatState`` from the
bridge's ``combat_action`` JSON on every decide call, asks the ``SearchAgent``
what to play, and returns the action index in the same ``Discrete(115)``
layout the runner already decodes via
:class:`sts2_env.bridge.state_adapter.StateAdapter`.

THE BRIDGE IS GROUND TRUTH -- NO DRIFT TOLERANCE

The first attempt kept a local sim across calls and tried to "mirror" the
runner's previous action into it. Within 2-3 turns the local sim's
prediction diverged from the live game's truth (different shuffle,
different enemy intent rolls, relic trigger order), but the code logged
the "drift" and kept using the wrong sim -- which then planned END_TURN
forever because in its frozen fiction energy was 0 and the hand was empty.

The fix is structural: rebuild the local sim from the bridge's report on
every single decide call. The bridge sends HP, block, energy, powers,
hand, enemy HP/block/powers/intent every state; we overwrite the fresh
``to_combat()`` build with that report and the search plans against the
live game's actual position. The draw pile order and the enemy's
next-next move remain approximations (the bridge sends only counts, not
the draw-pile order), which is the same approximation the offline search
tolerated at its 20% boss measurement.

That approximation is also why we do NOT keep a local sim across calls:
the search's lookahead uses ``_playout`` (a cheap policy-driven
continuation), which is itself only an approximation. Adding drift
on top of approximation stacked two errors; rebuilding every call keeps
exactly one.
"""

from __future__ import annotations

import logging
from typing import Any

from sts2_env.gym_env.action_space import ACTION_END_TURN
from sts2_env.search.evaluate import EvalWeights, DEFAULT_WEIGHTS
from sts2_env.search.situation import CombatSituation
from sts2_env.search.turn_search import SearchAgent

logger = logging.getLogger(__name__)


class LiveSearch:
    """Reconstructs a CombatState from the bridge JSON every step and asks
    the SearchAgent to play it.

    Each ``decide(bridge_state, prev_action=...)`` is one combat_action:
    the bridge has sent the live position, and we want to know what to play.

    Lifecycle per fight:

    1. First ``decide`` of a fight: build a fresh CombatState via
       ``CombatSituation.from_bridge_state(state).to_combat()``. The situation
       carries the deck, relics, potions, encounter setup -- static fields
       that do not change mid-fight -- and the bridge's combat_action is
       then overlaid on the freshly-started fight via
       ``to_combat_mid_fight``.

    2. Every subsequent ``decide``: rebuild from the bridge state the same
       way. The runner's previous action is dropped on the floor -- its
       effect is already in the bridge's report. The only state kept across
       calls is the SearchAgent's turn-plan (so a multi-card line plays in
       order), which the SearchAgent itself invalidates when its next
       ``act`` is told the state has changed.
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

    # -- stats --------------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        return {
            "searches": self._search.searches,
            "budget_exhausted": self._search.budget_exhausted_count,
        }

    # -- the public API -----------------------------------------------------

    def reset_for_new_fight(self) -> None:
        """Clear the search plan. Called by the runner when a combat starts.

        The next ``decide`` rebuilds the local sim from the bridge state
        regardless, so this is only an internal reset to stop a stale plan
        from a previous fight being replayed.
        """
        self._search = SearchAgent(
            weights=self._search.weights,
            time_budget=self._search.time_budget,
            lookahead_turns=self._search.lookahead_turns,
        )

    def decide(self, bridge_state: dict[str, Any], *, prev_action: int | None = None) -> int:
        """Return the next action index for the runner to send to the bridge.

        ``prev_action`` is kept in the signature for runner-side clarity but
        is not consulted: the bridge's next state already reflects whatever
        the runner sent, so re-applying it would be double-counting. The
        previous version of this module kept a local sim and applied the
        previous action to it; that was the source of the drift cascade, and
        dropping the kept sim is the fix.
        """
        # A raise here propagates on purpose: the runner's two-strike fallback
        # switches this combat to the trained model. The likely causes are the
        # mod missing the Phase 1.1 fields (encounter, encounter_seed) or a
        # card id the simulator cannot resolve -- both are "this fight cannot
        # be reconstructed", which the model can still play.
        situation = CombatSituation.from_bridge_state(bridge_state)
        combat = situation.to_combat_mid_fight(bridge_state)

        try:
            action = self._search.act(combat)
        except Exception:
            logger.exception("search agent raised; falling back to END_TURN")
            return ACTION_END_TURN

        return _retarget_for_bridge(action, combat)


def _retarget_for_bridge(action: int, combat) -> int:
    """Rewrite a searched action's enemy index into the one the game uses.

    The live game compacts its enemy list as monsters die; `to_combat` always
    rebuilds the full opening roster, so a survivor sits in a different slot on
    each side. `to_combat_mid_fight` records the translation on the combat as
    `bridge_enemy_index` ({sim slot: bridge slot}), and this applies it to the
    action just before the runner turns it into a PLAY.

    Skipping this is what produced the first stuck live session: on a
    3-slime SLIMES_WEAK where two slimes were already dead, the search named an
    enemy index the game had nothing at, the game ignored the play, and the same
    state came back until the stuck-detector ended the run.

    Untargeted actions (end turn, self-target, potions) pass through unchanged.
    """
    from sts2_env.core.constants import MAX_ENEMIES, MAX_HAND_SIZE
    from sts2_env.gym_env.action_space import action_to_card_and_target

    mapping = getattr(combat, "bridge_enemy_index", None)
    if not mapping:
        return action

    hand_index, enemy_index = action_to_card_and_target(action)
    if hand_index is None or enemy_index is None:
        return action

    bridge_index = mapping.get(enemy_index)
    if bridge_index is None:
        # The search targeted a slot the bridge never reported -- a monster the
        # game does not have. Nothing sensible to send, so end the turn rather
        # than issue a play that will be silently ignored and stall the run.
        logger.warning(
            "search targeted sim enemy %d, which the bridge did not report "
            "(mapping %s); ending the turn instead of stalling.",
            enemy_index, mapping,
        )
        return ACTION_END_TURN

    if bridge_index == enemy_index:
        return action
    return 1 + MAX_HAND_SIZE + hand_index * MAX_ENEMIES + bridge_index
