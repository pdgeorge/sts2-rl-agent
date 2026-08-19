"""Every monster's HP and damage, simulator against the decompile.

    .venv/bin/python scripts/audit_monster_constants.py

WHY MONSTERS SPECIFICALLY
-------------------------
Cards are already covered -- `check_card_parity.py` runs on every game update and
`derived_values.py` builds card numbers FROM the decompile rather than copying
them. `PHASE_TWO.md` section 3.7 names what is left: "Monster and relic constants
are still hand-copied, and that is where the wrong Siphon, nine wrong HP values
and thirteen wrong damages came from."

It is still true, and it is still expensive. Every parity bug found in the last
week was a monster constant:

  - Zapbot / Stabbot / Noisebot HP, 23-28 where the game says 18-23. Every bound
    +5, with the simulator's MIN sitting on the game's MAX.
  - TorchHeadAmalgam opening on an 18-damage TACKLE where the game opens on a
    26-damage STRONG_TACKLE.
  - TheForgotten's Dread flat at 15 where the game computes base 13 plus the
    creature's own Dexterity.
  - AXEBOT at 40-44 where the game rolls 74 to 95. That one cost 51 HP of 93 in
    a single fight, in the deepest run this project has recorded, and left it
    walking into the next room at 38.

The searcher plans its whole lookahead on these numbers. A monster modelled at
half its real HP is a race the agent commits to and cannot win.

WHAT IS COMPARED
----------------
HP as a RANGE, because most monsters roll within one and a single sample proves
nothing: the game's declared `MinInitialHp`/`MaxInitialHp` against the range the
simulator's factory reports. Damage as a MULTISET of the values the game's
attack intents declare against the values the simulator's move states declare --
move names differ between the two and matching on them would report renames as
defects, where the numbers are the thing the search actually uses.

Ascension is read at the BASE value, the second argument of
`GetValueIfAscension`, because that is what live runs play at.

WHAT IT CANNOT SEE
------------------
A damage the game computes rather than declares. `TheForgotten` returns
`base + GetPowerAmount<DexterityPower>()`, and a static reading of either side
sees only the base -- that bug was found by stepping a real fight in
`audit_dynamics.py`, not here. The two audits are complements: this one reads
what is written down, that one checks what actually happens.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MONSTERS = REPO / "decompiled" / "MegaCrit.Sts2.Core.Models.Monsters"

#: `GetValueIfAscension(AscensionLevel.X, ifAscension, otherwise)` -- the base
#: value is the SECOND number, and it is the one live runs play at.
_ASCENSION = re.compile(
    r"GetValueIfAscension\(\s*AscensionLevel\.\w+\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
_PLAIN_INT = re.compile(r"^\s*(-?\d+)\s*$")

_HP_PROP = re.compile(
    r"public override int (Min|Max)InitialHp\s*=>\s*([^;]+);")
#: Damage the game declares to an intent. Both the single and multi forms, plus
#: the lambda form TheForgotten uses.
_INTENT = re.compile(
    r"(?:Single|Multi)AttackIntent\(\s*(?:\(\)\s*=>\s*)?([A-Za-z_][A-Za-z0-9_]*|\d+)")
_PROP = re.compile(r"private int ([A-Za-z_][A-Za-z0-9_]*)\s*=>\s*([^;]+);")


def _base_value(expr: str) -> int | None:
    """The value a non-ascension run sees, or None if it is not a constant."""
    m = _ASCENSION.search(expr)
    if m:
        return int(m.group(2))
    m = _PLAIN_INT.match(expr)
    return int(m.group(1)) if m else None


def _pascal_to_upper(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


def _from_decompile(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "class " not in text or "MonsterModel" not in text:
        return None

    props = {name: _base_value(expr) for name, expr in _PROP.findall(text)}

    hp: dict[str, int | None] = {}
    for which, expr in _HP_PROP.findall(text):
        expr = expr.strip()
        if expr == "MinInitialHp":
            hp["Max"] = hp.get("Min")
            continue
        value = _base_value(expr)
        if value is None and expr in props:
            value = props[expr]
        hp[which] = value

    damages: Counter = Counter()
    for token in _INTENT.findall(text):
        value = int(token) if token.isdigit() else props.get(token)
        if value:
            damages[value] += 1

    return {"class": path.stem, "min_hp": hp.get("Min"), "max_hp": hp.get("Max"),
            "damages": damages}


def _from_sim(monster_id: str) -> dict | None:
    from sts2_env.core.rng import Rng
    from sts2_env.monsters.factory import create_monster_by_id

    built = create_monster_by_id(monster_id, Rng(0))
    if built is None:
        return None
    creature, ai = built
    lo = getattr(creature, "min_initial_hp", None)
    hi = getattr(creature, "max_initial_hp", None)
    if lo is None or hi is None:
        lo = hi = getattr(creature, "max_hp", None)

    damages: Counter = Counter()
    for state in (getattr(ai, "states", None) or {}).values():
        for intent in getattr(state, "intents", None) or ():
            if getattr(intent, "damage", 0):
                damages[int(intent.damage)] += 1
    return {"min_hp": lo, "max_hp": hi, "damages": damages}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show-unmodelled", action="store_true",
                    help="also list monsters the simulator cannot build at all")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    import sts2_env.cards  # noqa: F401
    import sts2_env.powers  # noqa: F401

    if not MONSTERS.is_dir():
        print(f"no decompiled monsters at {MONSTERS}")
        return 2

    hp_bad, dmg_bad, unmodelled, checked = [], [], [], 0
    for path in sorted(MONSTERS.glob("*.cs")):
        game = _from_decompile(path)
        if not game or game["min_hp"] is None:
            continue
        monster_id = _pascal_to_upper(game["class"])
        sim = _from_sim(monster_id)
        if sim is None:
            unmodelled.append(monster_id)
            continue
        checked += 1

        if (sim["min_hp"], sim["max_hp"]) != (game["min_hp"], game["max_hp"]):
            hp_bad.append((monster_id, sim["min_hp"], sim["max_hp"],
                           game["min_hp"], game["max_hp"]))

        if game["damages"] and sim["damages"] != game["damages"]:
            only_game = sorted((game["damages"] - sim["damages"]).elements())
            only_sim = sorted((sim["damages"] - game["damages"]).elements())
            if only_game or only_sim:
                dmg_bad.append((monster_id, only_sim, only_game))

    print("=" * 74)
    print(f"{checked} monsters compared "
          f"({len(unmodelled)} in the game the simulator cannot build)")
    print("=" * 74)

    print(f"\nHP RANGE MISMATCH: {len(hp_bad)}")
    for mid, s_lo, s_hi, g_lo, g_hi in hp_bad:
        note = ""
        if s_lo and g_lo and s_hi and g_hi:
            if s_lo - g_lo == s_hi - g_hi:
                note = f"   (both bounds {s_lo - g_lo:+d})"
        print(f"    {mid:<28} sim {s_lo}-{s_hi:<8} game {g_lo}-{g_hi}{note}")

    print(f"\nDAMAGE VALUES THE TWO SIDES DO NOT SHARE: {len(dmg_bad)}")
    for mid, only_sim, only_game in dmg_bad:
        print(f"    {mid:<28} sim-only {only_sim}   game-only {only_game}")

    if args.show_unmodelled and unmodelled:
        print(f"\nIN THE GAME, NOT BUILDABLE HERE: {len(unmodelled)}")
        for mid in unmodelled:
            print(f"    {mid}")

    print("\n" + "=" * 74)
    print("A damage difference can be a rename rather than a defect -- the\n"
          "comparison is on values, not move names. An HP range difference\n"
          "cannot: the game declares those outright. Read the .cs before\n"
          "changing anything, which is how the +5 on the bots was confirmed.")
    return 1 if (hp_bad or dmg_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
