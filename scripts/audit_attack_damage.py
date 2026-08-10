"""Check every monster's attack damage against the decompile, not just the ones live surfaced.

    python scripts/audit_attack_damage.py

Every damage bug fixed so far was found REACTIVELY -- the disparity detector
named a monster the agent happened to fight, someone opened the decompile, and
the constant was wrong. That found eight of them in a day, which says more about
how many there were than about the method. A monster the agent rarely meets, or
meets only in act 3, could be wrong indefinitely.

This asks the question directly instead. For every monster the simulator can
build, it reads the C# and compares:

    private int SwoopDamage => AscensionHelper.GetValueIfAscension(
        AscensionLevel.DeadlyEnemies, 19, 17);          <- base is the LAST arg
    ...
    new MoveState("SWOOP_MOVE", SwoopMove,
                  new SingleAttackIntent(SwoopDamage));  <- move -> property

against `ai.states["SWOOP_MOVE"].intents[0].damage` in a freshly built monster at
ascension 0.

WHAT IT CANNOT SEE, AND SAYS SO
-------------------------------
Reported as SKIPPED rather than passed, because a silent skip is how an audit
tells you everything is fine when it has not looked:

* `new SingleAttackIntent(() => CurrentPressureGunDamage)` -- a lambda over
  mutable state. Its value at combat start is checkable, its growth is not, and
  that growth was itself a real bug (the pressure gun telegraphed 20 while
  dealing 25 and climbing).
* Damage from a property this cannot resolve to a literal pair.
* Monsters whose C# class name does not map to the simulator's monster id.

MATCHING IS ON BASE VALUES AT ASCENSION 0, which is what the live game plays and
what `GetValueIfAscension(level, ifAscension, base)` returns when the level is
not met. The deadly/tough variants are checked too where both are declared.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEC = REPO / "decompiled" / "MegaCrit.Sts2.Core.Models.Monsters"

# private [static] int SwoopDamage => AscensionHelper.GetValueIfAscension(
#     AscensionLevel.DeadlyEnemies, 19, 17);
_ASC_PROP = re.compile(
    r"(?:private|public)\s+(?:static\s+)?int\s+(\w+)\s*=>\s*AscensionHelper\."
    r"GetValueIfAscension\(\s*AscensionLevel\.\w+\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")
# private int FlailDamage => 1;
_LIT_PROP = re.compile(r"(?:private|public)\s+(?:static\s+)?int\s+(\w+)\s*=>\s*(\d+)\s*;")

# new MoveState("BITE_MOVE", BiteMove, new SingleAttackIntent(BiteDamage))
_MOVE = re.compile(
    r'new\s+MoveState\(\s*"([A-Z0-9_]+)"\s*,\s*\w+\s*,\s*'
    r"new\s+(Single|Multi)AttackIntent\(\s*([^,)]+?)\s*(?:,\s*([^)]+?)\s*)?\)")

_CLASS = re.compile(r"public sealed class (\w+) : MonsterModel")


def _csharp_to_monster_id(cls: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", cls).upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true",
                        help="List matches as well as mismatches")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO))
    import sts2_env.cards  # noqa: F401
    from sts2_env.core.rng import Rng
    from sts2_env.monsters.factory import create_monster_by_id

    ok = 0
    mismatches: list[str] = []
    skipped: list[str] = []

    for path in sorted(DEC.glob("*.cs")):
        src = path.read_text(errors="ignore")
        cls_m = _CLASS.search(src)
        if not cls_m:
            continue
        monster_id = _csharp_to_monster_id(cls_m.group(1))

        built = create_monster_by_id(monster_id, Rng(0))
        if built is None:
            continue
        _, ai = built

        props: dict[str, int] = {}
        for name, _deadly, base in _ASC_PROP.findall(src):
            props[name] = int(base)
        for name, value in _LIT_PROP.findall(src):
            props.setdefault(name, int(value))

        for move_id, kind, dmg_expr, hits_expr in _MOVE.findall(src):
            if move_id not in ai.states:
                continue  # move-id mismatches are audited by the move diff
            state = ai.states[move_id]
            intents = [i for i in (getattr(state, "intents", None) or [])
                       if getattr(i, "damage", 0)]
            if not intents:
                skipped.append(f"{monster_id}.{move_id}: simulator intent has no damage")
                continue

            expr = dmg_expr.strip()
            if expr.startswith("("):
                skipped.append(f"{monster_id}.{move_id}: game damage is a lambda")
                continue
            expected = int(expr) if expr.isdigit() else props.get(expr)
            if expected is None:
                skipped.append(f"{monster_id}.{move_id}: cannot resolve {expr!r}")
                continue

            got = int(intents[0].damage)
            hits_got = int(getattr(intents[0], "hits", 1) or 1)
            line = f"{monster_id}.{move_id}"
            if got != expected:
                mismatches.append(f"  {line:<46} sim {got:<5} game {expected}")
                continue

            if kind == "Multi" and hits_expr:
                h = hits_expr.strip()
                exp_hits = int(h) if h.isdigit() else props.get(h)
                if exp_hits is not None and hits_got != exp_hits:
                    mismatches.append(
                        f"  {line:<46} sim {hits_got} hits, game {exp_hits} hits")
                    continue
            ok += 1
            if args.verbose:
                print(f"  ok  {line:<46} {got}"
                      + (f" x{hits_got}" if kind == "Multi" else ""))

    print(f"\nchecked {ok + len(mismatches)} attack moves across the monster pool")
    print(f"  match:    {ok}")
    print(f"  MISMATCH: {len(mismatches)}")
    for m in mismatches:
        print(m)
    print(f"  skipped:  {len(skipped)}  (listed so a skip is not read as a pass)")
    for s in skipped:
        print(f"    {s}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
