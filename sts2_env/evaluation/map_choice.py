"""How dangerous each room is. NOT which room is best.

READ THIS BEFORE USING IT AS A CHOOSER

An earlier version of this module ranked rooms by expected HP cost and picked the
cheapest. It picked the rest site every single time, for every deck, because rest
is the only room with negative cost and combat rewards are not modelled. "Avoid
all combat" is not a strategy -- you cannot progress, and you get no cards, gold
or relics.

So this ranks *risk*, and the caller decides what to do with it. The intended use
is a veto, not a preference: keep the existing priority order, and refuse the
rooms this says will probably end the run.

The heuristic this replaces is a fixed priority order -- boss, elite, monster,
event, unknown, treasure, shop, rest when healthy, and a reshuffled version when
below half HP. It takes elites *second*, always, whether or not the deck can beat
one.

The battery already answers that. On a ten-card starter deck the measured act 1
elite win rate is 33-42%: walking into one is closer to a coin flip on the whole
run than to a reward. A fixed order cannot express "not yet", and the HP
threshold it does have is the wrong question -- being at full HP does not make an
elite winnable, it just makes losing take longer.

PRICING

Every room is scored as expected HP lost, so lower is better:

    expected_cost = hp_lost x win_rate + death_cost x (1 - win_rate)

`death_cost` is the player's current HP, because losing a fight ends the run and
forfeits everything already spent getting there. That single term is what makes
the model refuse elites it cannot beat: at a 35% win rate an elite costs about
0.65 of the entire run, which no relic reward offsets.

Rest sites price as negative cost -- they give HP back.

WHAT IS NOT PRICED

Shops, treasures, events and unknowns get a flat neutral prior. Their value is
real but not simulatable here: a treasure's relic and an event's outcome are not
things the battery can play out. They are deliberately left at zero rather than
guessed at, which means this module ranks *combat risk* and defers on everything
else. A caller that wants those preferences should keep using the priority list
for ties, which is what the bridge does.

ELITE REWARDS ARE NOT CREDITED EITHER

An elite drops a relic, and that is worth real HP over a run. Not modelling it
biases this against elites. That is the safer direction to be wrong in at present
-- runs currently end in act 1 or 2, so the relic rarely gets to pay off, and the
measured elite win rate says most attempts are losses. Revisit when decks start
clearing elites reliably.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from sts2_env.core.constants import IRONCLAD_STARTING_HP
from sts2_env.evaluation.battery import Pilot, Tier, score_cell

logger = logging.getLogger(__name__)

NEUTRAL_COST = 0.0
"""Rooms whose value is real but not simulatable. Left at zero rather than
invented, so an unmeasured room never outranks a measured one on a made-up
number."""

REST_HEAL_FRACTION = 0.3


@dataclass(frozen=True)
class RoomOption:
    room_type: str
    expected_hp_cost: float
    index: int
    win_rate: float | None = None

    @property
    def label(self) -> str:
        if self.win_rate is None:
            return f"{self.room_type}"
        return f"{self.room_type}({self.win_rate:.0%}win)"


def _combat_cost(
    deck: Sequence, tier: Tier, pilot: Pilot, seeds: Sequence[int],
    current_hp: int, max_hp: int,
) -> tuple[float, float]:
    """(expected HP cost, win rate) for walking into this kind of fight."""
    cell = score_cell(deck, tier, pilot, seeds=seeds, max_hp=max_hp)
    hp_lost = cell.hp_lost_on_wins
    if hp_lost != hp_lost:          # NaN: never won
        hp_lost = float(max_hp)
    death_cost = float(current_hp)
    expected = hp_lost * cell.win_rate + death_cost * (1.0 - cell.win_rate)
    return expected, cell.win_rate


def elite_survivability(
    deck: Sequence,
    pilot: Pilot,
    *,
    floor: int = 1,
    max_hp: int = IRONCLAD_STARTING_HP,
    seeds: Sequence[int] = (0, 1, 2, 3),
) -> float:
    """Measured win rate against this act's elites. The number worth vetoing on.

    The priority list takes elites second, always. On a starter deck the measured
    act 1 elite win rate is 33-42%, so that is closer to a coin flip on the whole
    run than to a reward, and no HP threshold detects it -- being at full HP does
    not make an elite winnable, it only makes losing take longer.
    """
    act = 1 if floor <= 17 else (2 if floor <= 34 else 3)
    cell = score_cell(deck, Tier(act, "elite"), pilot, seeds=seeds, max_hp=max_hp)
    return cell.win_rate


def rank_rooms(
    deck: Sequence,
    rooms: Sequence[tuple[int, str]],
    pilot: Pilot,
    *,
    current_hp: int,
    max_hp: int = IRONCLAD_STARTING_HP,
    floor: int = 1,
    seeds: Sequence[int] = (0, 1, 2),
) -> list[RoomOption]:
    """Rank map options cheapest-first. `rooms` is (index, room_type) pairs."""
    import math

    act = 1 if floor <= 17 else (2 if floor <= 34 else 3)

    # Each combat kind is scored once, not once per option, because two monster
    # rooms on the same floor are the same measurement.
    cache: dict[str, tuple[float, float]] = {}

    scored: list[RoomOption] = []
    for index, raw_type in rooms:
        room = str(raw_type or "").strip().lower()

        if room in ("monster", "elite"):
            kind = "normal" if room == "monster" else "elite"
            if kind not in cache:
                cache[kind] = _combat_cost(
                    deck, Tier(act, kind), pilot, seeds, current_hp, max_hp
                )
            cost, win_rate = cache[kind]
            scored.append(RoomOption(room, cost, index, win_rate))
        elif room in ("restsite", "rest", "campfire"):
            heal = min(math.floor(max_hp * REST_HEAL_FRACTION), max(0, max_hp - current_hp))
            scored.append(RoomOption(room, -float(heal), index))
        else:
            scored.append(RoomOption(room or "unknown", NEUTRAL_COST, index))

    scored.sort(key=lambda r: r.expected_hp_cost)
    logger.info(
        "map: %s",
        "  ".join(f"{r.label}={r.expected_hp_cost:+.0f}hp" for r in scored),
    )
    return scored
