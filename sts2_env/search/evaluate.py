"""How good is this position? One number, from named parts.

Search is only as good as this file. Enumerating every line in a turn is
worthless if the thing choosing between them cannot tell a good outcome from a
bad one, and that is most of why the earlier flat-MC teacher stalled at 9.5
floors: random rollouts score every branch as roughly equally bad.

The evaluation deliberately does not predict anything. It is called on a state
the simulator has already produced -- typically after the turn has been ended and
the enemies have acted -- so "how much damage will get through" is not estimated
from intents, it is read off the player's HP after the hit landed. The simulator
is the authority on its own rules, which is also what keeps this correct across a
game update.

UNITS

Fractions of a maximum, not raw points, so the same weights mean the same thing
in a 40 HP hallway fight and a 500 HP boss. Same convention as
`gym_env/reward_config.py`, and for the same reason: numbers you can reason about
against each other rather than juggle.

WHY PLAYER HP OUTWEIGHS ENEMY HP

Enemy HP resets every fight and yours does not. A line that kills two turns
faster while costing 15 HP has not saved anything -- it has borrowed against the
elite two floors up, which is precisely how the current models die. The default
weights price a point of your own HP at four times a point of the enemy's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sts2_env.core.enums import PowerId

if TYPE_CHECKING:
    from sts2_env.core.combat import CombatState


# Powers worth counting, and what one stack of each is worth to us. Positive
# means "good for the player". Everything not listed is left to show up through
# its effect on HP rather than guessed at here.
PLAYER_POWER_VALUES: dict[PowerId, float] = {
    PowerId.STRENGTH: 0.030,
    PowerId.DEXTERITY: 0.020,
    PowerId.ARTIFACT: 0.015,
    PowerId.VULNERABLE: -0.030,
    PowerId.WEAK: -0.020,
    PowerId.FRAIL: -0.015,
}

#: Above this, an enemy's HP is a statement rather than a quantity. The game
#: sets Waterfall Giant to 999,999,999 with `HpDisplay.InfiniteWithoutNumbers`
#: when it starts erupting -- the phase is explicitly not about damage any more.
UNKILLABLE_HP = 1_000_000


def cannot_be_killed(enemy) -> bool:
    """Is damaging this enemy incapable of achieving anything?

    Read off the HP, which is how the game says it: `SetMaxAndCurrentHp(creature,
    999999999m)` plus `HpDisplay.InfiniteWithoutNumbers`. A real fight never
    approaches this, so the test needs no per-monster knowledge and cannot go
    stale on a rebalance patch.

    MEASURED CONSEQUENCE. When the Waterfall Giant dies it erupts: HP to a
    billion, `ShouldStopCombatFromEnding` true, and one turn to block the total
    Steam Eruption it banked. Every evaluation term went flat there -- `is_over`
    unreachable so no `win`, `kill` zero forever, and `enemy_hp` scoring 22
    damage at 5e-9 against a billion-HP denominator. The only term still
    carrying weight was `block_unused` at -0.02, pushing the searcher away from
    the single move that survives. The live logs show precisely that: 9, 7 and
    22 damage dealt into the eruption with zero block held, across three
    separate boss losses.

    NOT ASKED OF THE POWERS, which was the first attempt and is wrong. Both
    `should_creature_be_removed_from_combat_after_death` and
    `should_stop_combat_ending` are already true for SteamEruptionPower *before*
    the giant dies -- and before it dies, attacking it is exactly the right move,
    because killing it is how the fight progresses. A power-shaped test tells the
    searcher to stop attacking a 240 HP boss it needs to kill.

    The same reasoning rules out treating Eye With Teeth as unkillable. It
    revives, so damage is poor value, but killing it costs it a turn healing
    instead of adding Dazed -- poor value is not no value, and the Fogmog problem
    was the index mapping, not the evaluation.
    """
    return getattr(enemy, "current_hp", 0) >= UNKILLABLE_HP


# The same, on an enemy, from our side of the table.
ENEMY_POWER_VALUES: dict[PowerId, float] = {
    PowerId.VULNERABLE: 0.020,
    PowerId.WEAK: 0.020,
    PowerId.STRENGTH: -0.030,
}


@dataclass(frozen=True)
class EvalWeights:
    """Every term, in one place, so tuning is one object rather than a hunt."""

    player_hp: float = 1.0
    """Per fraction of the player's max HP still held. The dominant term."""

    enemy_hp: float = 0.25
    """Per fraction of total enemy HP removed. A quarter of the player's own HP:
    damage is how fights end, but it is not what runs are made of."""

    kill: float = 0.10
    """Per enemy dead, on top of the HP. A dead enemy stops attacking, which is
    worth more than the same damage spread across two living ones."""

    win: float = 3.0
    """Winning the fight outright."""

    loss: float = -10.0
    """Dying. Larger than a win because a run has one of them in it."""

    turn: float = -0.02
    """Per turn taken. Enough to break ties toward finishing, not enough to make
    a setup turn look bad -- the rule is not "stalling is bad", it is "stalling
    has to be going somewhere"."""

    powers: float = 1.0
    """Scales the power table above."""

    powers_cap: float = 0.25
    """The most the power terms may contribute, either way.

    Without it the term is linear and unbounded, and that is not a theoretical
    worry: Bygone Effigy is a sleeping elite that gains tens of Strength when
    woken, which at 0.03 a stack scored -1.2 -- five times the entire range of
    the enemy-HP term and larger than most of the player-HP term. The searcher
    responded rationally by refusing to attack at all, passing turns until it
    died on the turn it finally had to.

    Capped, a power reading stays what it is meant to be -- a tiebreaker between
    lines that are otherwise close -- and never outvotes what actually happened
    to somebody's HP. Powers that matter beyond this turn are the horizon's
    problem, not a number to inflate here."""

    block_unused: float = -0.02
    """Per fraction of max HP held as block at the moment the fight ends or is
    scored mid-turn. Block that was never needed was energy spent on nothing."""


DEFAULT_WEIGHTS = EvalWeights()


def evaluate_components(
    combat: "CombatState",
    weights: EvalWeights = DEFAULT_WEIGHTS,
) -> dict[str, float]:
    """The score, broken into its named parts.

    Kept separate from `evaluate` so each term can be tested and so a puzzling
    choice can be explained by printing the breakdown rather than by guessing.
    """
    player = combat.player
    max_hp = max(1, player.max_hp)

    parts: dict[str, float] = {}

    # -- the player ---------------------------------------------------------
    parts["player_hp"] = weights.player_hp * (max(0, player.current_hp) / max_hp)

    # -- the enemies --------------------------------------------------------
    #
    # Only the ones killing can actually remove. Damage into anything else is
    # energy that bought nothing, and scoring it as progress is how the searcher
    # spent the Waterfall Giant's eruption turn attacking a creature with a
    # billion HP instead of blocking the hit that killed it. See
    # `cannot_be_killed`.
    enemies = list(combat.enemies)
    killable = [e for e in enemies if not cannot_be_killed(e)]
    total_max = sum(max(1, e.max_hp) for e in killable) if killable else 0
    if total_max:
        remaining = sum(max(0, e.current_hp) for e in killable)
        parts["enemy_hp"] = weights.enemy_hp * (1.0 - remaining / total_max)
        dead = sum(1 for e in killable if not e.is_alive)
        parts["kill"] = weights.kill * (dead / len(killable))
    else:
        # Nothing on the table can be killed this turn. There is no progress to
        # be made by attacking, so both terms go silent and the score is carried
        # by the player's own HP -- which is the correct thing to be optimising
        # when the only question left is how much of the hit you eat.
        parts["enemy_hp"] = 0.0
        parts["kill"] = 0.0

    # -- terminal -----------------------------------------------------------
    if combat.is_over:
        parts["terminal"] = weights.win if combat.player_won else weights.loss
    elif player.current_hp <= 0:
        # Not yet marked over, but there is only one way this ends.
        parts["terminal"] = weights.loss
    else:
        parts["terminal"] = 0.0

    # -- powers -------------------------------------------------------------
    power_score = 0.0
    for power_id, value in PLAYER_POWER_VALUES.items():
        power_score += value * player.get_power_amount(power_id)
    for enemy in enemies:
        if not enemy.is_alive:
            continue
        for power_id, value in ENEMY_POWER_VALUES.items():
            power_score += value * enemy.get_power_amount(power_id)
    scaled = weights.powers * power_score
    parts["powers"] = max(-weights.powers_cap, min(weights.powers_cap, scaled))

    # -- tempo --------------------------------------------------------------
    parts["turn"] = weights.turn * combat.turn_count
    parts["block_unused"] = weights.block_unused * (player.block / max_hp)

    return parts


def evaluate(
    combat: "CombatState",
    weights: EvalWeights = DEFAULT_WEIGHTS,
) -> float:
    """One number. Higher is better for the player."""
    return sum(evaluate_components(combat, weights).values())


def explain(combat: "CombatState", weights: EvalWeights = DEFAULT_WEIGHTS) -> str:
    """The breakdown as text, for working out why a line was chosen."""
    parts = evaluate_components(combat, weights)
    width = max(len(k) for k in parts)
    lines = [f"  {k:<{width}}  {v:+7.3f}" for k, v in sorted(parts.items())]
    lines.append(f"  {'TOTAL':<{width}}  {sum(parts.values()):+7.3f}")
    return "\n".join(lines)
