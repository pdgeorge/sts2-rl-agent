"""Every line you could play this turn, tried, and the best one kept.

The idea the whole approach rests on. Slay the Spire tells you what the enemies
are about to do, and a turn has only a few hundred legal orderings of a few
cards, so there is no need to *predict* how a turn will go: play each ordering on
a copy, end the turn, let the enemies actually act, and look at what is left.

Measured on this repo: 49-469 orderings for a starter-deck turn, 20-225 ms
exhaustively. A live decision already waits 2.6-3.6 s on the game's animations.

WHAT THIS FINDS THAT A POLICY NET DOES NOT

Sequencing. Bash before the Strikes because Vulnerable multiplies what follows.
The exact block that survives an 11-damage intent rather than the block that
looks tidy. Killing the second enemy *this* turn because it removes 6 damage a
turn for the rest of the fight. These are combinatorial facts about the position,
recomputed from scratch every turn -- which is also why a rebalance patch does not
break it. It reads the new numbers out of the simulator.

WHAT IT DOES NOT SEE

Anything beyond the end of this turn plus the enemies' reply. It will not spend a
turn setting up a combo that pays off in three. That is a real limit and the
honest place to fix it is a deeper search, not a fudge factor here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from sts2_env.core.constants import ACTION_END_TURN
from sts2_env.gym_env.action_space import (
    apply_combat_action,
    get_action_mask,
    is_potion_action,
)
from sts2_env.search.cloning import clone_combat
from sts2_env.search.evaluate import DEFAULT_WEIGHTS, EvalWeights, evaluate

if TYPE_CHECKING:
    from sts2_env.core.combat import CombatState

logger = logging.getLogger(__name__)

DEFAULT_MAX_NODES = 20_000
DEFAULT_TIME_BUDGET = 3.0
DEFAULT_MAX_DEPTH = 12


@dataclass
class SearchResult:
    """The chosen line, and how sure the search was about it."""

    actions: tuple[int, ...]
    """The card plays, in order, NOT including the final end-turn."""

    score: float
    runner_up: float | None = None

    nodes: int = 0
    leaves: int = 0
    elapsed: float = 0.0
    exhausted: bool = True
    """False when a budget cut the search short -- the answer is then the best of
    what was looked at, not the best there is. Reported rather than hidden."""

    @property
    def gap(self) -> float | None:
        """How far ahead the best line finished.

        Feeds Cyra's `gut_phrase`: unlike a softmax over logits, this is a real
        margin between two lines the agent actually played out. The number still
        never leaves the process -- it becomes a phrase, not a percentage.
        """
        if self.runner_up is None:
            return None
        return self.score - self.runner_up

    def first_action(self) -> int:
        """What to do right now. End the turn if the line is empty."""
        return self.actions[0] if self.actions else ACTION_END_TURN


def _state_key(combat: "CombatState") -> tuple:
    """A cheap fingerprint for spotting positions already looked at.

    Two orderings of Strike, Strike, Defend reach the same place; without this the
    search re-explores every permutation of an interchangeable hand. Deliberately
    coarse and cheap -- it only has to be right about *sameness*, and anything it
    misses costs time rather than correctness.
    """
    player = combat.player
    return (
        player.current_hp,
        player.block,
        combat.energy,
        tuple(sorted((c.card_id.value, c.upgraded) for c in combat.hand)),
        len(combat.draw_pile),
        len(combat.discard_pile),
        tuple(sorted((p.value, inst.amount) for p, inst in player.powers.items())),
        tuple(
            (
                e.current_hp,
                e.block,
                tuple(sorted((p.value, inst.amount) for p, inst in e.powers.items())),
            )
            for e in combat.enemies
        ),
    )


def _branch_actions(combat: "CombatState") -> list[int]:
    """The moves worth branching on here, end-turn excluded.

    During a pending choice the mask means something different -- the options are
    choice indices, and index 0 is confirm rather than end-turn -- so the caller
    must not treat this as a list of card plays.
    """
    mask = get_action_mask(combat)
    actions = [int(a) for a in np.where(mask == 1)[0]]
    if combat.pending_choice is not None:
        return actions
    return [a for a in actions if a != ACTION_END_TURN]


def search_turn(
    combat: "CombatState",
    *,
    weights: EvalWeights = DEFAULT_WEIGHTS,
    max_nodes: int = DEFAULT_MAX_NODES,
    time_budget: float = DEFAULT_TIME_BUDGET,
    max_depth: int = DEFAULT_MAX_DEPTH,
    include_potions: bool = True,
) -> SearchResult:
    """Find the best sequence of plays for the turn `combat` is in.

    Every reachable ordering is tried, and each one is scored by ending the turn
    on a copy and letting the enemies reply. The score is therefore what actually
    happened to a copy of the fight, not a guess about what would.
    """
    started = time.perf_counter()
    counters = {"nodes": 0, "leaves": 0}
    seen: set[tuple] = set()
    exhausted = True

    best_score = -float("inf")
    best_actions: tuple[int, ...] = ()
    second_score = -float("inf")

    def out_of_budget() -> bool:
        return (
            counters["nodes"] >= max_nodes
            or (time.perf_counter() - started) >= time_budget
        )

    def consider_ending(state: "CombatState", path: tuple[int, ...]) -> None:
        """Score 'stop here and end the turn', by doing exactly that."""
        nonlocal best_score, best_actions, second_score

        if state.pending_choice is not None:
            return  # cannot end a turn mid-choice

        counters["leaves"] += 1
        ended = clone_combat(state)
        ended.end_player_turn()
        score = evaluate(ended, weights)

        if score > best_score:
            second_score = best_score
            best_score, best_actions = score, path
        elif score > second_score:
            second_score = score

    def descend(state: "CombatState", path: tuple[int, ...]) -> None:
        nonlocal exhausted

        if out_of_budget():
            exhausted = False
            return

        key = _state_key(state)
        if key in seen:
            return
        seen.add(key)

        consider_ending(state, path)

        if len(path) >= max_depth or state.is_over:
            return

        for action in _branch_actions(state):
            if out_of_budget():
                exhausted = False
                return
            if not include_potions and is_potion_action(action):
                continue

            child = clone_combat(state)
            if not apply_combat_action(child, action):
                # Masked as legal but refused. A mask bug rather than a legal
                # outcome; skip the branch rather than recurse on a state that
                # did not change, which would not terminate.
                continue

            counters["nodes"] += 1
            descend(child, path + (action,))

    descend(combat, ())

    if best_score == -float("inf"):
        # Nothing was playable, or the budget went before the first leaf. Ending
        # the turn is always legal, and is what an empty line means.
        best_score, best_actions = 0.0, ()

    return SearchResult(
        actions=best_actions,
        score=best_score,
        runner_up=second_score if second_score > -float("inf") else None,
        nodes=counters["nodes"],
        leaves=counters["leaves"],
        elapsed=time.perf_counter() - started,
        exhausted=exhausted,
    )


class SearchAgent:
    """A `CombatAgent` that plans a turn and then plays it out.

    Planning once per turn rather than once per card is not only cheaper, it is
    also correct here: the plan was produced by playing these exact actions from
    this exact state on a copy, and the copy carries the same RNG, so replaying
    them reproduces what search saw. The plan is still re-validated against the
    live mask before each action, and abandoned if the position is not the one
    that was planned for -- a plan that has stopped matching reality is the bug
    class this project has paid for most.
    """

    def __init__(
        self,
        weights: EvalWeights = DEFAULT_WEIGHTS,
        *,
        max_nodes: int = DEFAULT_MAX_NODES,
        time_budget: float = DEFAULT_TIME_BUDGET,
        max_depth: int = DEFAULT_MAX_DEPTH,
        include_potions: bool = True,
        name: str | None = None,
    ):
        self.weights = weights
        self.max_nodes = max_nodes
        self.time_budget = time_budget
        self.max_depth = max_depth
        self.include_potions = include_potions
        self.name = name or f"search(nodes<={max_nodes}, t<={time_budget}s)"

        self._plan: list[int] = []
        self._last_gap: float | None = None
        self.searches = 0
        self.total_nodes = 0
        self.total_seconds = 0.0
        self.budget_exhausted_count = 0

    @property
    def last_gap(self) -> float | None:
        """The margin behind the most recent decision, for milestone phrasing."""
        return self._last_gap

    def _replan(self, combat: "CombatState") -> None:
        result = search_turn(
            combat,
            weights=self.weights,
            max_nodes=self.max_nodes,
            time_budget=self.time_budget,
            max_depth=self.max_depth,
            include_potions=self.include_potions,
        )
        self._plan = list(result.actions)
        self._last_gap = result.gap
        self.searches += 1
        self.total_nodes += result.nodes
        self.total_seconds += result.elapsed
        if not result.exhausted:
            self.budget_exhausted_count += 1

    def act(self, combat: "CombatState") -> int:
        mask = get_action_mask(combat)

        # A pending choice was not part of the planned line; decide it on its own
        # terms and then plan afresh.
        if combat.pending_choice is not None:
            self._plan = []
            result = search_turn(
                combat,
                weights=self.weights,
                max_nodes=self.max_nodes,
                time_budget=self.time_budget,
                max_depth=self.max_depth,
                include_potions=self.include_potions,
            )
            self._last_gap = result.gap
            self.searches += 1
            self.total_nodes += result.nodes
            self.total_seconds += result.elapsed
            action = result.first_action()
            return action if mask[action] else int(np.where(mask == 1)[0][0])

        if not self._plan:
            self._replan(combat)

        while self._plan:
            action = self._plan.pop(0)
            if action < len(mask) and mask[action]:
                return action
            # The position is not the one that was planned for.
            logger.debug("Plan diverged at action %d; replanning", action)
            self._plan = []
            self._replan(combat)
            if not self._plan:
                break

        return ACTION_END_TURN

    def stats(self) -> dict[str, float]:
        return {
            "searches": self.searches,
            "nodes_per_search": self.total_nodes / self.searches if self.searches else 0.0,
            "seconds_per_search": self.total_seconds / self.searches if self.searches else 0.0,
            "budget_exhausted": self.budget_exhausted_count,
        }
