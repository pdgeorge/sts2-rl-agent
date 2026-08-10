"""Say so when the live game disagrees with the simulator.

    from sts2_env.search.parity import report_disparity, disparity_summary

The reconciliation in `situation.py` overwrites the simulator's enemy state with
whatever the bridge reports, every decision. That is correct -- the game is
ground truth -- and it is also why parity bugs are invisible: a wrong constant is
silently corrected at the root state on every single call, and the run continues
as if nothing were wrong.

It does not continue as if nothing were wrong. Only the ROOT state gets the
bridge's numbers. The search's lookahead rolls forward on the simulator's own
state machines and damage figures, so a monster the simulator models incorrectly
is planned against incorrectly for every turn past the telegraphed one.

Three real ones were found by hand in a single afternoon -- Waterfall Giant 250
vs 240, Phantasmal Gardener 28-32 vs 26-31, and a summoned Eye that had no slot
at all. Hand-checking does not scale to 83 monsters. This does: play normally,
read the log, fix what it names.

WHAT COUNTS AS A DISPARITY
--------------------------
Not "the numbers differ" -- most of them differ legitimately. Current HP differs
because the fight has progressed. Max HP differs because many monsters roll
within a range and the game's roll is not ours.

So max HP is judged against the range the simulator *can* produce, sampled from
the monster's own factory, and only a value outside that range is reported. That
is the difference between a signal and a wall of noise.

DEDUPED, AND NEVER FATAL
------------------------
Each distinct (kind, subject, expected, actual) is logged once per process. A
disparity repeats on every decision of every fight; logging each occurrence would
bury the one line that matters. Nothing here raises -- a parity check that can
end a live run is worse than the parity bug it found.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

#: {(kind, subject, expected, actual): times seen}. Module-level so a session
#: can print everything it noticed, rather than the operator having to scroll
#: back through a log for lines that appeared once each.
_SEEN: dict[tuple[str, str, str, str], int] = {}


def report_disparity(kind: str, subject: str, expected, actual) -> None:
    """Record that the game and the simulator disagree, and say so once.

    `kind` groups the finding (``max_hp``, ``intent_damage``); `subject` names
    what disagreed, usually a monster id.
    """
    key = (str(kind), str(subject), str(expected), str(actual))
    _SEEN[key] = _SEEN.get(key, 0) + 1
    if _SEEN[key] == 1:
        logger.warning(
            "DISPARITY [%s] %s: simulator says %s, game says %s -- "
            "updating to the game's value. Fix the simulator.",
            kind, subject, expected, actual,
        )


def disparity_summary() -> list[str]:
    """One line per distinct disparity, most frequent first."""
    return [
        f"{kind:<14} {subject:<26} sim={expected:<12} game={actual:<12} x{n}"
        for (kind, subject, expected, actual), n in
        sorted(_SEEN.items(), key=lambda kv: -kv[1])
    ]


def reset_disparities() -> None:
    _SEEN.clear()


#: Monsters whose HP range changes form under the same id, so the factory's
#: starting form is not the only legitimate answer.
#:
#: Tough Egg hatches: 14-18 as an egg, 19-22 once it opens. Both are modelled
#: correctly -- `TOUGH_EGG_BASE_HATCHLING_MIN_HP` and friends -- but the range
#: sampler builds the egg and never sees the hatchling, so every hatched egg
#: reported as a disparity. That was 99 of the 104 disparity occurrences in one
#: live session, drowning a real Waterfall Giant finding under a false one.
#:
#: A declared table rather than a wider tolerance, because "accept anything
#: nearby" would stop catching the off-by-a-few errors that made up most of the
#: real findings -- Phantasmal Gardener was wrong by two.
SECOND_FORM_HP_RANGES: dict[str, tuple[int, int]] = {
    "TOUGH_EGG": (19, 22),
}


@functools.lru_cache(maxsize=512)
def simulator_hp_range(monster_id: str) -> tuple[int, int] | None:
    """The (min, max) starting HP this simulator can roll for a monster.

    Sampled by building the monster rather than read from a table, because the
    table is the thing under suspicion. 64 seeds is comfortably enough for the
    ranges in this game, which span single digits.

    None when the monster cannot be built, which is its own parity gap and is
    reported where it happens rather than here.

    Monsters that change form under one id have more than one legitimate range,
    and the factory only builds the form it starts in. Those extra ranges are
    declared in `SECOND_FORM_HP_RANGES` and unioned in here.
    """
    from sts2_env.core.rng import Rng
    from sts2_env.monsters.factory import create_monster_by_id

    seen: set[int] = set()
    for seed in range(64):
        built = create_monster_by_id(monster_id, Rng(seed))
        if built is not None:
            seen.add(int(built[0].max_hp))
    if not seen:
        return None
    low, high = min(seen), max(seen)
    second = SECOND_FORM_HP_RANGES.get(str(monster_id or "").upper())
    if second is not None:
        low, high = min(low, second[0]), max(high, second[1])
    return low, high


def check_max_hp(monster_id: str, game_max_hp: int) -> None:
    """Report a max HP the simulator could not have produced.

    In-range values are silent on purpose: a monster with 26-31 HP rolling 27 in
    the game and 30 here is two RNG streams disagreeing, not a modelling error.
    Outside the range there is no roll that explains it, so the constants are
    wrong.
    """
    if not monster_id or not game_max_hp:
        return
    # A phase-change HP, not a statistic. Waterfall Giant is set to 999,999,999
    # when it starts erupting, and reporting that as a modelling error buries the
    # real findings under the one the simulator already handles correctly.
    from sts2_env.search.evaluate import UNKILLABLE_HP

    if int(game_max_hp) >= UNKILLABLE_HP:
        return
    span = simulator_hp_range(monster_id)
    if span is None:
        return
    low, high = span
    if not (low <= int(game_max_hp) <= high):
        report_disparity("max_hp", monster_id, f"{low}-{high}", game_max_hp)
