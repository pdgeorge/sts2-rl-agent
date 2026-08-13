"""Plan a whole route to the boss, instead of picking the next room greedily.

WHY THIS EXISTS
---------------
`_pick_map_node` ranks the immediately reachable nodes by room type and takes
the best one. It cannot plan a route, cannot set up an elite followed by a rest
site, and diverts to a recovery room whenever *any* visible node is
unaffordable -- so it zigzags rather than committing to a path.

The game's own beginner guide opens with the opposite advice, and this project
ignored it for weeks while measuring things the guide had already answered:

  - "Plan paths backward from the Boss", and prefer routes with intersections
  - "Fight Elites when your deck is strong enough -- they drop game-changing
    relics"
  - avoid the temptation to dodge enemies to preserve health, because "regular
    hallway fights are your primary source of card rewards, gold, and potions"

The measurements agree with all three. Across 496 live runs, 52% of runs die
before ever reaching the boss -- 133 in monster rooms, 93 in elites -- and 39%
of runs fight ZERO elites. Relics are worth 22 points of boss win, and elites
are the only relic source the routing controls.

Lowering the HP gate did NOT fix the elite count: halving it from 0.80 to 0.45
bought 0.22 elites per run, because the gate was never the binding constraint.
The greedy path simply does not pass elites. That is what this fixes.

HOW IT SCORES
-------------
Every path from here to the boss, valued by what it collects, minus what it
cannot survive. The first step of the best path is the move.

Values are per the guide's ordering, not fitted: an elite is worth more than a
hallway fight, which is worth more than nothing. They are deliberately coarse,
because the failure mode of this project has been tuning numbers that were
already roughly right while structural gaps went unfixed.

THE HP MODEL IS A BUDGET, NOT A GATE
-------------------------------------
Rather than refusing a room outright, a path carries projected HP: fights cost
their measured p90 damage, rest sites restore 30% of max. A path that runs out
of HP is scored as fatal and loses to any survivable path -- but a path that
survives is never penalised for being dangerous. That is the difference between
"don't dodge fights" and "walk into anything".
"""

from __future__ import annotations

from dataclasses import dataclass

#: What a room is worth collecting, per the guide's own ordering. Elites lead
#: because they are the only relic source routing controls, and relics measured
#: at 22 points of act 1 boss win.
ROOM_VALUE = {
    "elite": 3.0,
    "restsite": 2.0,
    "shop": 1.5,
    "treasure": 1.5,
    "monster": 1.0,
    "unknown": 1.0,
    "event": 0.8,
    "boss": 0.0,
}

#: Expected HP cost of entering a room, as a fraction of max HP. Fitted from the
#: same 116 live elite fights and monster-room death rates that produced
#: ROOM_MIN_HP_FRACTION -- an elite costs roughly a quarter of the bar, a
#: hallway fight roughly a tenth.
ROOM_HP_COST = {
    "elite": 0.26,
    "monster": 0.11,
    "unknown": 0.09,
    "event": 0.03,
    "boss": 0.0,
    "restsite": 0.0,
    "shop": 0.0,
    "treasure": 0.0,
}

#: A rest site heals exactly 30% of max HP (`rest_site.py:35`).
REST_HEAL_FRACTION = 0.30

#: A path that would arrive at the boss below this is treated as fatal. The
#: upgrade/HP grid prices boss win at ~1 point per 1% of max HP, and live
#: currently arrives at 81%; below about a third the fight is not winnable.
FATAL_HP_FRACTION = 0.34

#: Paths whose value ties are broken toward arriving healthier, because HP is
#: the only boss-side lever still standing.
HP_TIEBREAK_WEIGHT = 2.0


@dataclass(frozen=True)
class PlannedRoute:
    """The best route found, and the immediate step that starts it."""

    first_step: object
    value: float
    projected_boss_hp: float
    rooms: tuple[str, ...]

    def describe(self) -> str:
        return (f"value {self.value:.1f}, arrives at boss on "
                f"{100 * self.projected_boss_hp:.0f}% HP, via "
                + " -> ".join(self.rooms[:8]))


def _canonical(room_type) -> str:
    return str(room_type or "").strip().lower().replace("_", "").replace(" ", "")


def plan_route(
    children_of,
    start_nodes,
    room_type_of,
    hp_fraction: float,
    *,
    max_depth: int = 24,
):
    """Best route to the boss from here; returns a `PlannedRoute` or None.

    `children_of(node)` yields the nodes reachable from `node`, `room_type_of`
    gives a room-type string, and `hp_fraction` is current HP over max. The
    caller supplies these so the same planner serves the live bridge and the
    offline RunManager without either one owning the graph.

    Depth-first with memoisation on (node, banded HP). HP has to be part of the
    key: the same node reached at 90% and at 40% has genuinely different best
    continuations, and memoising on the node alone silently returns the plan for
    whichever HP arrived first. Banding keeps the table small.
    """
    memo: dict[tuple[int, int], tuple[float, float, tuple[str, ...]]] = {}

    def best_from(node, hp: float, depth: int):
        """(value, boss_hp, rooms) of the best continuation from `node`."""
        room = _canonical(room_type_of(node))
        if room == "boss" or depth >= max_depth:
            return (0.0, hp, (room,))

        key = (id(node), int(hp * 20))
        if key in memo:
            return memo[key]

        hp_after = hp - ROOM_HP_COST.get(room, 0.09)
        if room == "restsite":
            hp_after = min(1.0, hp_after + REST_HEAL_FRACTION)
        hp_after = max(0.0, hp_after)

        children = list(children_of(node) or [])
        if not children:
            result = (ROOM_VALUE.get(room, 0.5), hp_after, (room,))
            memo[key] = result
            return result

        best = None
        best_key = None
        for child in children:
            sub_value, boss_hp, rooms = best_from(child, hp_after, depth + 1)
            value = ROOM_VALUE.get(room, 0.5) + sub_value
            # SURVIVABILITY FIRST, THEN VALUE -- and never the two added
            # together. Subtracting a fixed penalty from the value looks
            # equivalent and is not: when every route is fatal, the one that
            # collects most wins, so at 40% HP the planner walked into an elite
            # it could not survive because an elite is worth more than a hallway
            # fight. Among routes that all die, the only thing worth optimising
            # is arriving as healthy as possible, since the projection is
            # approximate and HP is what buys another turn.
            survivable = boss_hp >= FATAL_HP_FRACTION
            sort_key = ((1, value + HP_TIEBREAK_WEIGHT * boss_hp)
                        if survivable else (0, boss_hp))
            if best_key is None or sort_key > best_key:
                best_key = sort_key
                best = (value, boss_hp, (room,) + rooms)
        memo[key] = best
        return best

    best_route = None
    best_key = None
    for node in (start_nodes or []):
        value, boss_hp, rooms = best_from(node, hp_fraction, 0)
        # The SAME survivability-first key as the inner loop. Ranking the start
        # nodes by raw value here while the recursion ranked by survivability
        # reproduced the exact bug this key exists to prevent, one level up:
        # at 40% HP every route was fatal and the planner still walked into the
        # elite because an elite scores higher than a hallway fight.
        survivable = boss_hp >= FATAL_HP_FRACTION
        sort_key = ((1, value + HP_TIEBREAK_WEIGHT * boss_hp)
                    if survivable else (0, boss_hp))
        if best_key is None or sort_key > best_key:
            best_key = sort_key
            best_route = PlannedRoute(first_step=node, value=value,
                                      projected_boss_hp=boss_hp, rooms=rooms)
    return best_route
