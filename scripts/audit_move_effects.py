"""Does each monster move APPLY what the decompiled move applies?

    .venv/bin/python scripts/audit_move_effects.py
    .venv/bin/python scripts/audit_move_effects.py --monster HAUNTED_SHIP

`audit_monster_constants.py` already checks the NUMBERS -- hp, damage, block --
and `audit_dynamics.py` checks which move comes NEXT. Nothing checked what a
move DOES, and that gap is not theoretical: while fixing Haunted Ship's move
ORDER I noticed its Haunt applies Weak/Frail/Vulnerable 2 each here, where
`HauntedShip.HauntMove` applies `WeakPower ... 3m` and adds Dazed cards. Wrong
debuffs on a common act 1 monster feed straight into corridor damage, which
feeds arrival HP -- the one thing that separates a boss win from a boss loss at
every act 1 boss.

HOW IT WORKS. The decompiled move method is scanned for the two commands that
change a fight beyond damage:

    PowerCmd.Apply<XPower>(..., <amount>, ...)
    CardPileCmd.AddToCombatAndPreview<Y>(..., <count>, ...)

Our move is then PERFORMED on a clean combat and the powers and cards it
actually produced are read off the resulting state. Observed against observed --
the same discipline as `evaluate.py` scoring a state instead of predicting one,
and the reason this can catch a wrong effect that a constants table cannot.

KNOWN LIMITS, stated so a clean run is not mistaken for a proven one:
  * damage is NOT compared here; `audit_monster_constants.py` owns that.
  * a move whose amount is a C# expression rather than a literal is reported as
    `expr` and only its PRESENCE is checked, not its size.
  * conditional applies (inside an `if`) are counted as applies. A move that
    only sometimes debuffs will look like one that always does.
  * targets are not distinguished: applying Weak to the player and to itself
    both read as "applies WEAK".
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DECOMP = REPO / "decompiled/MegaCrit.Sts2.Core.Models.Monsters"

_APPLY = re.compile(r"PowerCmd\.Apply<(\w+?)Power>\s*\([^;]*?,\s*([\w.]+)\s*[,)]", re.S)
_ADDCARD = re.compile(r"CardPileCmd\.Add\w*<(\w+)>\s*\([^;]*?,\s*([\w.]+)\s*[,)]", re.S)


def _cs_moves(path: Path) -> dict[str, dict]:
    """Move id -> what the decompiled method applies."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # "MOVE_ID", MethodName  ->  the method that implements it
    pairs = dict(re.findall(r'new MoveState\(\s*"(\w+)"\s*,\s*(\w+)', text))
    out = {}
    for move_id, method in pairs.items():
        m = re.search(rf"private async Task {method}\s*\([^)]*\)\s*\{{", text)
        if not m:
            continue
        start = m.end()
        depth, i = 1, start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[start:i]
        out[move_id] = {
            "powers": {p.upper(): a for p, a in _APPLY.findall(body)},
            "cards": {c.upper(): a for c, a in _ADDCARD.findall(body)},
        }
    return out


def _ours(monster_id: str) -> dict[str, dict]:
    """Perform each of our moves on a clean combat and read what it produced."""
    import sts2_env.cards  # noqa: F401
    from sts2_env.core.rng import Rng
    from sts2_env.monsters.factory import create_monster_by_id
    from sts2_env.search.situation import CardRef, CombatSituation

    out = {}
    _, probe = create_monster_by_id(monster_id, Rng(1))
    for move_id, state in probe.states.items():
        if not getattr(state, "is_move", False):
            continue
        combat = CombatSituation(
            situation_id="fx", character_id="Ironclad", current_hp=80, max_hp=80,
            deck=tuple([CardRef("STRIKE_IRONCLAD")] * 10),
            encounter="setup_corpse_slugs_normal", encounter_seed=5,
            combat_seed=3, relics=()).to_combat()
        creature, ai = create_monster_by_id(monster_id, Rng(1))
        combat.add_enemy(creature, ai)
        before_p = {str(k).split(".")[-1]: v.amount
                    for k, v in (combat.player.powers or {}).items()}
        before_e = {str(k).split(".")[-1]: v.amount
                    for k, v in (creature.powers or {}).items()}
        before_cards = len(combat.discard_pile) + len(combat.draw_pile)
        try:
            ai.states[move_id].perform(combat)
        except Exception as exc:  # noqa: BLE001 - a move that cannot run is a finding
            out[move_id] = {"error": type(exc).__name__}
            continue
        after_p = {str(k).split(".")[-1]: v.amount
                   for k, v in (combat.player.powers or {}).items()}
        after_e = {str(k).split(".")[-1]: v.amount
                   for k, v in (creature.powers or {}).items()}
        applied = {}
        for name, amount in list(after_p.items()) + list(after_e.items()):
            prior = before_p.get(name, 0) if name in after_p else before_e.get(name, 0)
            if amount != prior:
                applied[name] = abs(amount - prior)
        added = (len(combat.discard_pile) + len(combat.draw_pile)) - before_cards
        out[move_id] = {"powers": applied, "cards_added": max(0, added)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--monster", default="", help="one monster id, else every act 1 one")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO))
    import sts2_env.cards  # noqa: F401
    from sts2_env.monsters.factory import known_monster_ids

    ids = [args.monster] if args.monster else sorted(known_monster_ids())
    findings, checked, skipped = [], 0, 0

    for monster_id in ids:
        cls = "".join(p.capitalize() for p in monster_id.lower().split("_"))
        path = DECOMP / f"{cls}.cs"
        if not path.exists():
            skipped += 1
            continue
        try:
            theirs = _cs_moves(path)
            ours = _ours(monster_id)
        except Exception:
            skipped += 1
            continue
        for move_id, want in theirs.items():
            got = ours.get(move_id)
            if got is None or "error" in (got or {}):
                continue
            checked += 1
            # C# spells the class `PersonalHivePower`, we spell the enum
            # PERSONAL_HIVE. Compared without separators, or the audit reports
            # its own naming convention as a parity bug -- which the first run
            # of this did, for PERSONAL_HIVE, CHAINS_OF_BINDING and
            # PAINFUL_STABS.
            norm = lambda n: n.replace("_", "")
            want_powers = {norm(x) for x in want["powers"]}
            got_powers = {norm(x) for x in got["powers"]}
            missing = want_powers - got_powers
            extra = got_powers - want_powers
            if missing or extra:
                findings.append((monster_id, move_id, sorted(missing), sorted(extra),
                                 want["powers"], got["powers"]))

    print(f"checked {checked} moves across {len(ids) - skipped} monsters "
          f"({skipped} had no decompiled class or would not build)\n")
    if not findings:
        print("no power-application mismatches found.")
        return 0
    print(f"{len(findings)} moves apply a different SET of powers than the decompile:\n")
    for monster_id, move_id, missing, extra, want, got in findings[:40]:
        print(f"  {monster_id}.{move_id}")
        if missing:
            print(f"     game applies, we do NOT : {missing}")
        if extra:
            print(f"     we apply, game does NOT : {extra}")
        print(f"     decompile={want}  ours={got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
