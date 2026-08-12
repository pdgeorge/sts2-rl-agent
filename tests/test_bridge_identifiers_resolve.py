"""Every identifier the game has ever sent us must resolve in this build.

PHASE_TWO.md section 3.5. This is the cheapest defence the project has against
its most expensive bug class, and it was previously a script someone had to
remember to run.

WHY THIS IS NOT VERSION DIFFING
-------------------------------
`docs/PARITY_GAPS.md` and the update pipeline compare game build N against N+1.
That answers "did the game change". This answers a different question: "can our
model resolve what the game is sending, right now".

They come apart exactly where it hurt most. FLAME_BARRIER was never a version
change -- the game always sent `FLAME_BARRIER`, we always spelled it
`FLAME_BARRIER_CARD`, one of 68 members carrying that suffix. A diff between
builds correctly reports "no change" forever while 68 cards stay unplayable, the
searcher never sees them, and a run dies holding a card it could have played.

WHAT A FAILURE HERE MEANS
-------------------------
Not necessarily a bug in this repo. A genuinely new card from a game patch also
lands here, and that is the point: it becomes a red test on the next run rather
than a silent degradation discovered weeks later by watching a replay.

The fixture is whatever the captured protocol holds. It grows as sessions are
captured, so this test gets stronger over time without anyone maintaining it.
"""

from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

import pytest

import sts2_env.cards  # noqa: F401
import sts2_env.powers  # noqa: F401
from sts2_env.core.enums import PowerId  # noqa: F401  (imported for parity with the script)

REPO = Path(__file__).resolve().parent.parent

#: Captured live protocol. Any file matching contributes its identifiers.
CAPTURE_GLOBS = (
    "output/bridge_protocol*.jsonl",
    "output/bridge_boss_fights*.jsonl",
    "output/live_journal*.jsonl",
    "output/stuck_states.jsonl",
)

#: Relics the game has and this simulator does not model. Every one is
#: RelicRarity.Ancient with out-of-combat effects (AfterObtained, AfterCombatEnd,
#: map movement), so the combat search is not wrong for lacking them -- it simply
#: does not know they exist.
#:
#: This list is a LEDGER, not permission. Shrinking it is work; growing it needs
#: a reason written next to the entry.
KNOWN_UNMODELLED_RELICS = frozenset({
    "KALEIDOSCOPE",
    "FISHING_ROD",
    "NEOWS_TALISMAN",
    "WINGED_BOOTS",
    "PHIAL_HOLSTER",
})


def _collect() -> dict[str, collections.Counter]:
    found: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)

    def walk(obj) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ("relics",) and isinstance(value, list):
                    for v in value:
                        if isinstance(v, str):
                            found["relic"][v] += 1
                elif key in ("potions", "potion_slots") and isinstance(value, list):
                    for v in value:
                        if isinstance(v, str):
                            found["potion"][v] += 1
                elif key in ("deck", "hand") and isinstance(value, list):
                    for v in value:
                        name = v.get("id") if isinstance(v, dict) else v
                        if isinstance(name, str):
                            found["card"][name] += 1
                elif key == "enemies" and isinstance(value, list):
                    for v in value:
                        if not isinstance(v, dict):
                            continue
                        if isinstance(v.get("id"), str):
                            found["enemy"][v["id"]] += 1
                        for p in (v.get("powers") or []):
                            pid = p.get("id") if isinstance(p, dict) else p
                            if isinstance(pid, str):
                                found["power"][pid] += 1
                elif key == "powers" and isinstance(value, list):
                    for p in value:
                        pid = p.get("id") if isinstance(p, dict) else p
                        if isinstance(pid, str):
                            found["power"][pid] += 1
                elif key == "room_type" and isinstance(value, str):
                    found["room_type"][value] += 1
                elif key == "encounter" and isinstance(value, str):
                    found["encounter"][value] += 1
                elif key == "potion" and isinstance(value, str):
                    found["potion"][value] += 1
                else:
                    walk(value)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    for pattern in CAPTURE_GLOBS:
        for path in glob.glob(str(REPO / pattern)):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        walk(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return found


def _resolvers():
    """The REAL resolvers, never a reimplementation of them.

    A first version of the audit script checked `name in PowerId.__members__` and
    reported 42 of 43 powers missing -- STRENGTH_POWER, VULNERABLE_POWER and the
    rest -- because `_coerce_power_id` strips the `_POWER` suffix the mod's
    slugify leaves on. An audit that reimplements the thing it audits measures
    itself.
    """
    from sts2_env.core.rng import Rng
    from sts2_env.monsters.factory import create_monster_by_id
    from sts2_env.potions.base import create_potion
    from sts2_env.relics.registry import create_relic_by_name
    from sts2_env.search.situation import (
        _MAP_POINT_TO_ROOM, _coerce_power_id, resolve_card_id, resolve_encounter,
    )

    def card(n): return resolve_card_id(n) is not None
    def power(n): return _coerce_power_id(str(n)) is not None
    def room_type(n): return str(n).upper() in _MAP_POINT_TO_ROOM

    def potion(n):
        try:
            create_potion(n, slot=0); return True
        except Exception:
            return False

    def relic(n):
        try:
            return create_relic_by_name(n) is not None
        except Exception:
            return False

    def enemy(n):
        try:
            return create_monster_by_id(str(n).upper(), Rng(1)) is not None
        except Exception:
            return False

    def encounter(n):
        try:
            resolve_encounter(n); return True
        except Exception:
            return False

    return {"card": card, "power": power, "room_type": room_type,
            "potion": potion, "relic": relic, "enemy": enemy,
            "encounter": encounter}


FOUND = _collect()
RESOLVERS = _resolvers()


@pytest.mark.parametrize("kind", sorted(RESOLVERS))
def test_every_captured_identifier_resolves(kind):
    counter = FOUND.get(kind)
    if not counter:
        pytest.skip(f"no {kind} identifiers in the captured protocol yet")

    check = RESOLVERS[kind]
    allow = KNOWN_UNMODELLED_RELICS if kind == "relic" else frozenset()
    missing = sorted(
        (n, c) for n, c in counter.items()
        if str(n).upper() not in allow and not check(n)
    )
    assert not missing, (
        f"{len(missing)} {kind} identifier(s) the game sent cannot be resolved "
        f"in this build: {[n for n, _ in missing[:10]]}. "
        "Either the simulator is missing content the game has, or a name does "
        "not match -- FLAME_BARRIER against FLAME_BARRIER_CARD cost 68 "
        "unplayable cards this way. Fix the resolver or add the content; if it "
        "is genuinely unmodelled and harmless, record it with a reason."
    )


def test_the_capture_fixture_is_not_empty():
    """A green suite because nothing was captured is not a passing audit."""
    total = sum(sum(c.values()) for c in FOUND.values())
    assert total > 0, (
        "no captured protocol found -- this audit is vacuous. Run a live "
        "session with --capture-raw, or the test suite is asserting nothing."
    )


def test_the_unmodelled_relic_ledger_is_still_accurate():
    """An entry that now resolves should leave the ledger.

    Otherwise the allowlist silently grows into a place where real gaps hide.
    """
    check = RESOLVERS["relic"]
    resolved_now = sorted(n for n in KNOWN_UNMODELLED_RELICS if check(n))
    assert not resolved_now, (
        f"these are modelled now and should be removed from "
        f"KNOWN_UNMODELLED_RELICS: {resolved_now}"
    )
