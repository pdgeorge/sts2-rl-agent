"""Play the benchmark, and report what happened, for any agent.

One harness for every candidate so the comparison is like for like: same 200
fights, same turn cap, same accounting. Whether the actions come from a policy
net or from search is behind `CombatAgent`, which is the whole point -- the
Phase 1 gate is this file run twice.

WHAT IS MEASURED

Win rate, and HP lost. Neither alone is enough. An agent that wins every fight at
2 HP has a 100% win rate and will die on the next floor, and that is exactly the
failure the live runs show: deaths at elites after arriving hurt. HP is the
resource a run is actually made of, so a fight won at a cost is reported as a
fight won at a cost.

Losses are counted as losing every remaining point of HP, because that is what
they cost the run. Mean HP lost is therefore reported twice: over won fights,
which measures play, and over all fights, which measures consequences.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Protocol

import numpy as np

from sts2_env.gym_env.action_space import apply_combat_action, get_action_mask
from sts2_env.search.situation import CombatSituation

if TYPE_CHECKING:
    # Import-time only. Importing CombatState here for real makes this module the
    # first thing to touch sts2_env.core.combat, which the card modules import
    # back into during their own import -- a cycle that only bites when
    # benchmark is the entry point.
    from sts2_env.core.combat import CombatState

logger = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 200
DEFAULT_MAX_IDLE = 25

FLOOR_BANDS = ((1, 4), (5, 8), (9, 12), (13, 16))


class CombatAgent(Protocol):
    """Anything that can choose a combat action."""

    name: str

    def act(self, combat: "CombatState") -> int:
        ...


class RandomAgent:
    """The floor. Any agent that cannot beat this is broken, not weak."""

    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def act(self, combat: "CombatState") -> int:
        mask = get_action_mask(combat)
        valid = np.where(mask == 1)[0]
        return int(self._rng.choice(valid)) if len(valid) else 0


class ModelAgent:
    """A trained MaskablePPO policy over the 131-dim combat observation."""

    def __init__(self, model_path: str, deterministic: bool = True, device: str = "cpu"):
        from sb3_contrib import MaskablePPO

        self.model = MaskablePPO.load(model_path, device=device)
        self.deterministic = deterministic
        self.name = model_path

    def act(self, combat: "CombatState") -> int:
        from sts2_env.gym_env.observation import encode_observation

        mask = get_action_mask(combat)
        obs = encode_observation(combat)
        action, _ = self.model.predict(
            obs, action_masks=mask, deterministic=self.deterministic
        )
        return int(action)


@dataclass
class Result:
    """One fight, played out."""

    situation_id: str
    room_type: str
    floor: int
    won: bool
    hp_before: int
    hp_after: int
    turns: int
    steps: int
    timed_out: bool = False
    stalled: bool = False

    @property
    def hp_lost(self) -> int:
        return max(0, self.hp_before - self.hp_after)

    @property
    def hp_lost_cost(self) -> int:
        """What the fight cost the run: a loss costs everything left."""
        return self.hp_before if not self.won else self.hp_lost


def play(
    situation: CombatSituation,
    agent: CombatAgent,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_idle: int = DEFAULT_MAX_IDLE,
) -> Result:
    """Play one situation to the end."""
    combat = situation.to_combat()
    hp_before = combat.player.current_hp

    steps = 0
    idle = 0
    stalled = False
    while not combat.is_over and combat.turn_count <= max_turns:
        action = agent.act(combat)
        acted = apply_combat_action(combat, action)
        steps += 1

        # A rejected action changes nothing, including turn_count, so a policy
        # that keeps choosing one never terminates and never truncates. Every
        # such action should have been masked out, so a run of them is a mask
        # bug; cut the fight off rather than hang.
        idle = 0 if acted else idle + 1
        if idle >= max_idle:
            stalled = True
            logger.warning(
                "%s stalled on %s: %d rejected actions in a row (action %d)",
                agent.name, situation.situation_id, idle, action,
            )
            break

    timed_out = not combat.is_over and not stalled
    return Result(
        situation_id=situation.situation_id,
        room_type=situation.room_type,
        floor=situation.total_floor,
        won=bool(combat.is_over and combat.player_won),
        hp_before=hp_before,
        hp_after=max(0, combat.player.current_hp),
        turns=combat.turn_count,
        steps=steps,
        timed_out=timed_out,
        stalled=stalled,
    )


@dataclass
class Summary:
    agent: str
    results: list[Result] = field(default_factory=list)

    # -- headline ----------------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def win_rate(self) -> float:
        return sum(r.won for r in self.results) / self.n if self.n else 0.0

    @property
    def win_rate_se(self) -> float:
        """How much of the difference between two agents is just 200 fights."""
        if not self.n:
            return 0.0
        p = self.win_rate
        return (p * (1 - p) / self.n) ** 0.5

    @property
    def hp_lost_when_won(self) -> float:
        won = [r.hp_lost for r in self.results if r.won]
        return statistics.mean(won) if won else 0.0

    @property
    def hp_lost_overall(self) -> float:
        return statistics.mean([r.hp_lost_cost for r in self.results]) if self.n else 0.0

    @property
    def mean_turns(self) -> float:
        return statistics.mean([r.turns for r in self.results]) if self.n else 0.0

    def subset(self, predicate) -> Summary:
        return Summary(self.agent, [r for r in self.results if predicate(r)])

    # -- reporting ---------------------------------------------------------

    def report(self) -> str:
        if not self.n:
            return "No fights played."

        lines = [
            "",
            "=" * 66,
            f"agent:  {self.agent}",
            f"fights: {self.n}",
            "",
            f"  win rate          {self.win_rate:6.1%}  +/- {self.win_rate_se:.1%} (1 se)",
            f"  hp lost | won     {self.hp_lost_when_won:6.1f}",
            f"  hp lost overall   {self.hp_lost_overall:6.1f}   (a loss costs every point left)",
            f"  turns             {self.mean_turns:6.1f}",
        ]

        problems = [r for r in self.results if r.timed_out or r.stalled]
        if problems:
            lines.append(
                f"  did not finish    {len(problems):>6}   "
                f"({sum(r.stalled for r in problems)} stalled, "
                f"{sum(r.timed_out for r in problems)} hit the turn cap)"
            )

        lines += ["", "by room:"]
        for room in ("MONSTER", "ELITE", "BOSS"):
            sub = self.subset(lambda r, room=room: r.room_type == room)
            if sub.n:
                lines.append(
                    f"  {room:<8} {sub.n:>4} fights   win {sub.win_rate:6.1%}   "
                    f"hp lost {sub.hp_lost_overall:5.1f}"
                )

        lines += ["", "by floor:"]
        for lo, hi in FLOOR_BANDS:
            sub = self.subset(lambda r, lo=lo, hi=hi: lo <= r.floor <= hi)
            if sub.n:
                lines.append(
                    f"  {lo:>2}-{hi:<3} {sub.n:>4} fights   win {sub.win_rate:6.1%}   "
                    f"hp lost {sub.hp_lost_overall:5.1f}"
                )

        lines += ["=" * 66, ""]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "fights": self.n,
            "win_rate": self.win_rate,
            "win_rate_se": self.win_rate_se,
            "hp_lost_when_won": self.hp_lost_when_won,
            "hp_lost_overall": self.hp_lost_overall,
            "mean_turns": self.mean_turns,
            "did_not_finish": sum(r.timed_out or r.stalled for r in self.results),
        }


def score(
    situations: Iterable[CombatSituation],
    agent: CombatAgent,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    progress_every: int = 0,
) -> Summary:
    summary = Summary(agent=agent.name)
    for i, situation in enumerate(situations, 1):
        summary.results.append(play(situation, agent, max_turns=max_turns))
        if progress_every and i % progress_every == 0:
            logger.info("  %d fights, win rate so far %.1%%", i, summary.win_rate * 100)
    return summary


def _verdict(difference: float, standard_error: float) -> str:
    if standard_error <= 0:
        return "no spread to judge against"
    z = difference / standard_error
    if abs(z) > 2:
        return "clear" if z > 0 else "clearly worse"
    if abs(z) > 1:
        return "suggestive, not conclusive"
    return "inside the noise"


def paired_stats(a: Summary, b: Summary) -> dict[str, Any] | None:
    """Fight-by-fight differences, for two agents on the same fixture.

    Both agents face identical situations, so pairing on the situation removes
    the variance that comes from the fixture itself -- one draw containing three
    bosses and another containing none. The unpaired comparison charges that
    spread against the difference between the agents and hides real effects
    behind it.

    Returns None when the two runs are not on the same situations, because a
    paired number computed across different fights would be worse than no number.
    """
    by_id_a = {r.situation_id: r for r in a.results}
    by_id_b = {r.situation_id: r for r in b.results}
    shared = sorted(set(by_id_a) & set(by_id_b))
    if not shared or len(shared) != len(by_id_a) or len(shared) != len(by_id_b):
        return None

    win_diffs = [float(by_id_b[i].won) - float(by_id_a[i].won) for i in shared]
    hp_diffs = [float(by_id_b[i].hp_lost_cost - by_id_a[i].hp_lost_cost) for i in shared]

    def mean_and_se(values: list[float]) -> tuple[float, float]:
        n = len(values)
        mean = statistics.mean(values)
        if n < 2:
            return mean, 0.0
        return mean, statistics.stdev(values) / (n ** 0.5)

    win_mean, win_se = mean_and_se(win_diffs)
    hp_mean, hp_se = mean_and_se(hp_diffs)

    return {
        "n": len(shared),
        "win_rate_delta": win_mean,
        "win_rate_delta_se": win_se,
        "win_verdict": _verdict(win_mean, win_se),
        "hp_lost_delta": hp_mean,
        "hp_lost_delta_se": hp_se,
        # Less HP lost is better, so the sign flips before judging.
        "hp_verdict": _verdict(-hp_mean, hp_se),
        "only_b_won": sum(1 for i in shared if by_id_b[i].won and not by_id_a[i].won),
        "only_a_won": sum(1 for i in shared if by_id_a[i].won and not by_id_b[i].won),
    }


def compare(a: Summary, b: Summary) -> str:
    """Two agents on the same fixture, with the uncertainty stated.

    The question is never "is the number bigger" -- 200 fights carries a standard
    error around 3 points, so a two-point difference is nothing. It is "is it
    bigger by more than the measurement is worth".
    """
    d_win = b.win_rate - a.win_rate
    se = (a.win_rate_se ** 2 + b.win_rate_se ** 2) ** 0.5
    d_hp = b.hp_lost_overall - a.hp_lost_overall

    lines = [
        "",
        "=" * 66,
        f"{a.agent}",
        f"  vs {b.agent}",
        "",
        f"  win rate     {a.win_rate:6.1%}  ->  {b.win_rate:6.1%}   "
        f"{d_win:+.1%}  ({d_win / se if se else 0:+.1f} se)   {_verdict(d_win, se)}",
        f"  hp lost      {a.hp_lost_overall:6.1f}  ->  {b.hp_lost_overall:6.1f}   {d_hp:+.1f}",
        f"  turns        {a.mean_turns:6.1f}  ->  {b.mean_turns:6.1f}",
    ]

    paired = paired_stats(a, b)
    if paired is not None:
        lines += [
            "",
            f"  paired over the same {paired['n']} fights:",
            f"    win rate   {paired['win_rate_delta']:+.1%} "
            f"+/- {paired['win_rate_delta_se']:.1%}   {paired['win_verdict']}",
            f"    hp lost    {paired['hp_lost_delta']:+.1f} "
            f"+/- {paired['hp_lost_delta_se']:.1f}   {paired['hp_verdict']}",
            f"    won only by the challenger: {paired['only_b_won']}, "
            f"only by the baseline: {paired['only_a_won']}",
        ]

    lines += ["=" * 66, ""]
    return "\n".join(lines)
