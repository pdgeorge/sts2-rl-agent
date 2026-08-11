"""Every identifier the bridge has ever sent, checked against this simulator.

    python scripts/audit_bridge_ids.py

WHY
---
The expensive bugs this week were not wrong numbers. They were names that did
not resolve, each one failing quietly in its own way:

  FLAME_BARRIER   the enum member is FLAME_BARRIER_CARD, one of 68 such. The
                  card was dropped from the hand, so the searcher could not
                  play it -- and dropping it SHIFTED every later hand index, so
                  the runner then asked for the wrong card and the game refused.
  Unknown         a `?` room. RoomType has no such member, so `RoomType[name]`
                  raised, and a raise inside to_combat hands the entire fight to
                  the trained model.

Both were invisible in the numbers and obvious in the logs, if anyone had
compared the sets. This does that comparison on every id class at once, from
the journals and captured states we already have, so the next one is found by
running a script rather than by watching a run die.

Reports names the bridge sends that this build cannot resolve. Not every miss is
a bug -- a genuinely new card from a game patch belongs here too -- but every
one is a thing the simulator is blind to while playing.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _walk(obj, out: dict[str, collections.Counter]) -> None:
    """Harvest ids by the key they appear under, at any depth."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("relics", "potions", "potion_slots") and isinstance(value, list):
                for v in value:
                    if isinstance(v, str):
                        out[key][v] += 1
            elif key == "deck" and isinstance(value, list):
                for v in value:
                    name = v.get("id") if isinstance(v, dict) else v
                    if isinstance(name, str):
                        out["deck"][name] += 1
            elif key == "hand" and isinstance(value, list):
                for v in value:
                    name = v.get("id") if isinstance(v, dict) else v
                    if isinstance(name, str):
                        out["hand"][name] += 1
            elif key == "enemies" and isinstance(value, list):
                for v in value:
                    if isinstance(v, dict):
                        if isinstance(v.get("id"), str):
                            out["enemy"][v["id"]] += 1
                        if isinstance(v.get("intent_move_id"), str):
                            out["intent_move_id"][v["intent_move_id"]] += 1
                        for p in (v.get("powers") or []):
                            pid = p.get("id") if isinstance(p, dict) else p
                            if isinstance(pid, str):
                                out["power"][pid] += 1
            elif key == "powers" and isinstance(value, list):
                for p in value:
                    pid = p.get("id") if isinstance(p, dict) else p
                    if isinstance(pid, str):
                        out["power"][pid] += 1
            elif key in ("room_type", "encounter", "potion") and isinstance(value, str):
                out[key][value] += 1
            else:
                _walk(value, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--globs", nargs="*", default=[
        "output/live_journal*.jsonl",
        "output/stuck_states.jsonl",
        "output/live_search_failures.jsonl",
        "output/bridge_protocol*.jsonl",
    ])
    args = ap.parse_args()

    found: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    files = 0
    for pattern in args.globs:
        for path in sorted(glob.glob(pattern)):
            files += 1
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        _walk(json.loads(line), found)
                    except json.JSONDecodeError:
                        continue

    import sts2_env.cards  # noqa: F401
    import sts2_env.powers  # noqa: F401
    from sts2_env.core.enums import PowerId, RoomType
    from sts2_env.monsters.factory import create_monster_by_id
    from sts2_env.potions.base import create_potion
    from sts2_env.relics.registry import create_relic_by_name
    from sts2_env.search.situation import (
        _MAP_POINT_TO_ROOM, _coerce_power_id, resolve_card_id, resolve_encounter,
    )
    from sts2_env.core.rng import Rng

    def card_ok(n): return resolve_card_id(n) is not None
    # THE REAL RESOLVER, not a guess at it. A first pass here checked
    # `name in PowerId.__members__` and reported 42 of 43 powers missing --
    # STRENGTH_POWER, VULNERABLE_POWER, the lot. They resolve fine;
    # `_coerce_power_id` strips the `_POWER` suffix the mod's slugify leaves on.
    # An audit that reimplements the thing it audits measures itself.
    def power_ok(n): return _coerce_power_id(str(n)) is not None
    def room_ok(n): return str(n).upper() in _MAP_POINT_TO_ROOM

    def potion_ok(n):
        try:
            create_potion(n, slot=0); return True
        except Exception:
            return False

    def relic_ok(n):
        try:
            return create_relic_by_name(n) is not None
        except Exception:
            return False

    def enemy_ok(n):
        try:
            return create_monster_by_id(str(n).upper(), Rng(1)) is not None
        except Exception:
            return False

    def encounter_ok(n):
        try:
            resolve_encounter(n); return True
        except Exception:
            return False

    checks = {
        "deck": card_ok, "hand": card_ok,
        "power": power_ok, "room_type": room_ok,
        "potions": potion_ok, "potion_slots": potion_ok, "potion": potion_ok,
        "relics": relic_ok, "enemy": enemy_ok, "encounter": encounter_ok,
    }

    print(f"scanned {files} files\n")
    print(f"{'class':<16}{'distinct':>9}{'resolve':>9}{'MISSING':>9}")
    print("-" * 46)
    problems: dict[str, list[tuple[str, int]]] = {}
    for cls, counter in sorted(found.items()):
        check = checks.get(cls)
        if check is None:
            continue
        missing = [(n, c) for n, c in counter.items() if not check(n)]
        problems[cls] = sorted(missing, key=lambda kv: -kv[1])
        print(f"{cls:<16}{len(counter):>9}{len(counter) - len(missing):>9}"
              f"{len(missing):>9}")

    print()
    any_missing = False
    for cls, missing in problems.items():
        if not missing:
            continue
        any_missing = True
        print(f"=== {cls}: {len(missing)} unresolvable ===")
        for name, count in missing[:25]:
            print(f"   {name:<38}seen {count}x")
        if len(missing) > 25:
            print(f"   ... and {len(missing) - 25} more")
        print()
    if not any_missing:
        print("every identifier the bridge has sent resolves in this build.")
    return 1 if any_missing else 0


if __name__ == "__main__":
    sys.exit(main())
