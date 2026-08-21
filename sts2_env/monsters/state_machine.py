"""Monster AI state machine.

Direct port of the C# MonsterMoveStateMachine, MoveState,
RandomBranchState, and ConditionalBranchState.
"""

from __future__ import annotations

import copy
import types
from itertools import takewhile
from typing import Any, Callable, TYPE_CHECKING

from sts2_env.core.enums import MoveRepeatType
from sts2_env.monsters.intents import Intent

if TYPE_CHECKING:
    from sts2_env.core.combat import CombatState
    from sts2_env.core.rng import Rng


# ---------------------------------------------------------------------------
# Copying states that hold closures
# ---------------------------------------------------------------------------
#
# Every monster is built as a factory that closes over its own Creature:
#
#     def create_nibbit(rng, ...):
#         creature = Creature(...)
#         def butt(combat):
#             _deal_damage_to_player(combat, creature, 6)   # <-- captured
#
# `copy.deepcopy` treats a function as atomic and returns the same object, and
# it has no way to rewrite a closure cell. So a deep-copied combat used to end up
# with monsters whose moves still acted on the *original* creature: rollouts
# buffed and damaged the real fight instead of the copy (a Nibbit reached 140
# block from search alone), while the copy's own monsters did nothing, so the
# search systematically believed incoming damage was far lower than it was.
#
# Nothing raised. The searcher simply planned against a fiction.
#
# Rebuilding the function with memo-copied cells fixes it at the root: the memo
# already holds the copied Creature, so the rebound closure picks up the copy and
# every monster acts on the fight it belongs to.

def _copy_callable(fn: Any, memo: dict) -> Any:
    """A copy of `fn` whose closure points at the copied objects."""
    if not isinstance(fn, types.FunctionType) or not fn.__closure__:
        return fn
    cells = tuple(
        types.CellType(copy.deepcopy(cell.cell_contents, memo))
        for cell in fn.__closure__
    )
    copied = types.FunctionType(
        fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, cells,
    )
    copied.__kwdefaults__ = fn.__kwdefaults__
    copied.__dict__.update(fn.__dict__)
    return copied


def _copy_value(value: Any, memo: dict) -> Any:
    """Deep-copy, looking inside containers for callables to rebind."""
    if isinstance(value, types.FunctionType):
        return _copy_callable(value, memo)
    if isinstance(value, list):
        return [_copy_value(v, memo) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_value(v, memo) for v in value)
    if isinstance(value, dict):
        return {k: _copy_value(v, memo) for k, v in value.items()}
    return copy.deepcopy(value, memo)


class _ClosureAwareCopy:
    """Mixin: deep-copy this object, rebinding any closures it holds."""

    def __deepcopy__(self, memo: dict):
        cls = self.__class__
        copied = cls.__new__(cls)
        memo[id(self)] = copied
        for key, value in self.__dict__.items():
            copied.__dict__[key] = _copy_value(value, memo)
        return copied


class MonsterState(_ClosureAwareCopy):
    """Base class for all state machine states."""

    def __init__(self, state_id: str):
        self.state_id = state_id
        self.should_appear_in_logs: bool = True

    @property
    def is_move(self) -> bool:
        return False

    @property
    def can_transition_away(self) -> bool:
        return True

    def get_next_state(self, state_log: list[str], rng: Rng) -> str:
        raise NotImplementedError


class MoveState(MonsterState):
    """A concrete move that a monster performs."""

    def __init__(
        self,
        state_id: str,
        effect_fn: Callable[[CombatState], None],
        intents: list[Intent],
        follow_up_id: str | None = None,
        must_perform_once: bool = False,
    ):
        super().__init__(state_id)
        self.effect_fn = effect_fn
        self.intents = intents
        self.follow_up_id = follow_up_id
        self.must_perform_once = must_perform_once
        self._performed_at_least_once: bool = False

    @property
    def is_move(self) -> bool:
        return True

    @property
    def can_transition_away(self) -> bool:
        if self.must_perform_once:
            return self._performed_at_least_once
        return True

    def perform(self, combat: CombatState) -> None:
        self._performed_at_least_once = True
        flush_pending = getattr(combat, "flush_pending_attack_context", None)
        if callable(flush_pending):
            flush_pending()
        try:
            self.effect_fn(combat)
        finally:
            if callable(flush_pending):
                flush_pending()

    def on_exit_state(self) -> None:
        self._performed_at_least_once = False

    def get_next_state(self, state_log: list[str], rng: Rng) -> str:
        if self.follow_up_id is None:
            raise ValueError(f"MoveState '{self.state_id}' has no follow_up_id")
        return self.follow_up_id


class WeightedBranch:
    """A single branch option in a RandomBranchState."""

    __slots__ = ("state_id", "repeat_type", "max_times", "base_weight", "cooldown")

    def __init__(
        self,
        state_id: str,
        repeat_type: MoveRepeatType = MoveRepeatType.CAN_REPEAT_FOREVER,
        max_times: int = 1,
        weight: float | Callable[[], float] = 1.0,
        cooldown: int = 0,
    ):
        self.state_id = state_id
        self.repeat_type = repeat_type
        self.max_times = max_times
        self.base_weight = weight
        self.cooldown = cooldown

    def __deepcopy__(self, memo: dict) -> WeightedBranch:
        # Its own, rather than the mixin's: __slots__ means there is no __dict__
        # to walk. `base_weight` may be a callable closing over the creature, so
        # it needs the same rebinding as a move's effect.
        cls = self.__class__
        copied = cls.__new__(cls)
        memo[id(self)] = copied
        for slot in cls.__slots__:
            setattr(copied, slot, _copy_value(getattr(self, slot), memo))
        return copied

    def get_weight(self, state_log: list[str]) -> float:
        """Calculate effective weight given the state history."""
        if self.repeat_type == MoveRepeatType.USE_ONLY_ONCE:
            if self.state_id in state_log:
                return 0.0

        if self.repeat_type == MoveRepeatType.CANNOT_REPEAT:
            if state_log and state_log[-1] == self.state_id:
                return 0.0

        if self.repeat_type == MoveRepeatType.CAN_REPEAT_X_TIMES:
            # Count consecutive occurrences at end of log
            consecutive = sum(
                1 for _ in takewhile(lambda x: x == self.state_id, reversed(state_log))
            )
            if consecutive >= self.max_times:
                return 0.0

        if self.cooldown > 0:
            # Check last N move entries
            move_entries = state_log[-self.cooldown:]
            if self.state_id in move_entries:
                return 0.0

        if callable(self.base_weight):
            return float(self.base_weight())
        return self.base_weight


class RandomBranchState(MonsterState):
    """Randomly selects among branches with weights and repeat constraints."""

    def __init__(self, state_id: str):
        super().__init__(state_id)
        self.branches: list[WeightedBranch] = []
        self.should_appear_in_logs = False

    def add_branch(
        self,
        state_id: str,
        repeat_type: MoveRepeatType = MoveRepeatType.CAN_REPEAT_FOREVER,
        max_times: int = 1,
        weight: float | Callable[[], float] = 1.0,
        cooldown: int = 0,
    ) -> RandomBranchState:
        self.branches.append(WeightedBranch(state_id, repeat_type, max_times, weight, cooldown))
        return self

    def get_next_state(self, state_log: list[str], rng: Rng) -> str:
        weights = [b.get_weight(state_log) for b in self.branches]
        total_weight = sum(weights)
        if total_weight <= 0:
            # Fallback: all branches exhausted, pick first available
            for b in self.branches:
                return b.state_id
            raise ValueError(f"RandomBranchState '{self.state_id}' has no branches")

        roll = rng.next_float(total_weight)
        cumulative = 0.0
        for branch, w in zip(self.branches, weights):
            if w <= 0:
                continue
            cumulative += w
            if roll < cumulative:
                return branch.state_id

        # Floating point edge case: return last valid branch
        for branch, w in zip(reversed(self.branches), reversed(weights)):
            if w > 0:
                return branch.state_id
        raise ValueError("unreachable")


class ConditionalBranchState(MonsterState):
    """Selects the first branch whose condition is true."""

    def __init__(self, state_id: str):
        super().__init__(state_id)
        self.branches: list[tuple[Callable[[], bool], str]] = []
        self.should_appear_in_logs = False

    def add_branch(self, condition: Callable[[], bool], state_id: str) -> ConditionalBranchState:
        self.branches.append((condition, state_id))
        return self

    def get_next_state(self, state_log: list[str], rng: Rng) -> str:
        for condition, state_id in self.branches:
            if condition():
                return state_id
        raise ValueError(f"ConditionalBranchState '{self.state_id}': no condition matched")


class MonsterAI:
    """Container for a monster's state machine."""

    def __init__(self, states: dict[str, MonsterState], initial_state_id: str, rng: Rng | None = None):
        self.states = states
        self.state_log: list[str] = []
        self._current_state_id = initial_state_id
        self._performed_first_move: bool = False
        self.assume_worst_branch: bool = False
        """Resolve `RandomBranchState` to its hardest-hitting branch instead of
        rolling for it. Set ONLY on the search's own clones -- see
        `search/cloning.py`. The authoritative combat must keep rolling, because
        offline that combat IS the game and biasing it would be cheating rather
        than planning."""

        # Resolve initial state to a MoveState (walk through branches)
        self._resolve_to_move(rng)

    @property
    def current_move(self) -> MoveState:
        state = self.states[self._current_state_id]
        assert state.is_move, f"Current state {self._current_state_id} is not a MoveState"
        return state

    def _resolve_to_move(self, rng: Rng | None) -> None:
        """Walk through branch states until we reach a MoveState."""
        safety = 100
        while not self.states[self._current_state_id].is_move:
            if safety <= 0:
                raise RuntimeError("Infinite loop in state machine resolution")
            safety -= 1
            state = self.states[self._current_state_id]
            # Use a temporary rng for initial resolution if none provided
            if rng is None:
                from sts2_env.core.rng import Rng as RngClass
                rng = RngClass(0)
            worst = self._worst_branch(state)
            self._current_state_id = (
                worst if worst is not None
                else state.get_next_state(self.state_log, rng))

    def roll_move(self, rng: Rng) -> MoveState:
        """Advance the state machine to the next move.

        Per C#: first move is held until performed. After that,
        each call advances to the next MoveState.
        """
        current = self.states[self._current_state_id]

        # Don't advance if first move hasn't been performed yet
        if not self._performed_first_move and current.is_move:
            return current

        # Don't advance if current state must be performed first
        if current.is_move and not current.can_transition_away:
            return current

        # Get next state ID from current state
        worst = self._worst_branch(current)
        next_id = (worst if worst is not None
                   else current.get_next_state(self.state_log, rng))
        if current.is_move:
            current.on_exit_state()
        self._current_state_id = next_id

        # Resolve through branch states until we reach a MoveState
        self._resolve_to_move(rng)

        return self.current_move

    def _worst_branch(self, state: MonsterState) -> str | None:
        """The hardest-hitting branch of a random state, or None to roll normally.

        WHY THE SEARCH SHOULD NOT GAMBLE. Nine of the eleven remaining move
        mispredictions `audit_dynamics` found live are `RandomBranchState`
        monsters -- Mawler's three moves all follow up into
        `new RandomBranchState("RAND")`, so there is genuinely no order to read
        off. Drawing one branch and planning as though certain is wrong roughly
        as often as the branch count implies, and the cost is ASYMMETRIC:
        unblocked damage ends a run, and block that turned out to be unnecessary
        costs one card.

        It is also the offline/live gap. Offline the simulator is the game, so
        the branch it draws is the branch that happens and the search is never
        wrong; live it is a coin flip. `boss_counterfactuals` wins 62.5% of
        live-LOST act 1 boss positions at identical settings -- free
        information, not more thinking time.

        Damage is read off the branch's own intents rather than assumed, so a
        debuff or buff branch scores 0 and an attack branch scores what it
        telegraphs. Branches whose weight has gone to zero are skipped, because
        the game will not pick those either.
        """
        if not self.assume_worst_branch:
            return None
        branches = getattr(state, "branches", None)
        if not branches:
            return None
        best_id, best_damage = None, -1.0
        for branch in branches:
            try:
                if branch.get_weight(self.state_log) <= 0:
                    continue
            except Exception:  # noqa: BLE001 - a weight callback must not break search
                continue
            move = self.states.get(branch.state_id)
            damage = 0.0
            for intent in (getattr(move, "intents", None) or ()):
                damage += ((getattr(intent, "damage", 0) or 0)
                           * max(1, getattr(intent, "hits", 1) or 1))
            if damage > best_damage:
                best_id, best_damage = branch.state_id, damage
        return best_id

    def on_move_performed(self) -> None:
        """Called after the current move has been executed."""
        self._performed_first_move = True
        state = self.states[self._current_state_id]
        if isinstance(state, MoveState):
            state._performed_at_least_once = True
        if state.should_appear_in_logs:
            self.state_log.append(self._current_state_id)
