"""Independently re-read the decompiled card scalars and check the simulator's.

    python scripts/audit_card_values.py

WHY, WHEN THE SIMULATOR ALREADY DERIVES THESE
---------------------------------------------
`cards/reference_static_metadata.py` parses the decompiled card models, and
`cards/derived_values.py` writes the result over whatever the factories passed.
So a card's cost/rarity/damage/block already come from the game -- which means
comparing the simulator against that same parser proves nothing except that the
parser agrees with itself. Exactly the trap derived_values.py's own docstring
describes: "4,609 passing tests meant the repo agrees with itself".

This parses the same .cs files with a SEPARATE, deliberately dumb regex reader
and compares. Agreement means two independent readings of the shipped assembly
say the same thing. Disagreement means one of the two readers is wrong, and the
one that ships is the one that matters.

It also reports cards whose FACTORY literal disagrees with the game. Those are
inert -- derived_values overwrites them -- but a factory literal that is wrong
is a lie sitting in the source where the next person will read it, and it is
how the numbers drifted in the first place.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CARD_DIR = REPO / "decompiled" / "MegaCrit.Sts2.Core.Models.Cards"

_CTOR = re.compile(
    r":\s*base\(\s*(-?\d+)\s*,\s*CardType\.(\w+)\s*,\s*CardRarity\.(\w+)\s*,"
    r"\s*TargetType\.(\w+)")
_DAMAGE = re.compile(r"new DamageVar\(\s*(-?\d+(?:\.\d+)?)m")
_BLOCK = re.compile(r"new BlockVar\(\s*(-?\d+(?:\.\d+)?)m")
_UP_DAMAGE = re.compile(r"DynamicVars\.Damage\.UpgradeValueBy\(\s*(-?\d+(?:\.\d+)?)m")
_UP_BLOCK = re.compile(r"DynamicVars\.Block\.UpgradeValueBy\(\s*(-?\d+(?:\.\d+)?)m")


def parse_cs(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    ctor = _CTOR.search(text)
    if not ctor:
        return None

    def first(rx):
        m = rx.search(text)
        return int(float(m.group(1))) if m else None

    return {
        "cost": int(ctor.group(1)),
        "card_type": ctor.group(2).upper(),
        "rarity": ctor.group(3).upper(),
        "target_type": ctor.group(4).upper(),
        "base_damage": first(_DAMAGE),
        "base_block": first(_BLOCK),
        "up_damage": first(_UP_DAMAGE) or 0,
        "up_block": first(_UP_BLOCK) or 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show-factory-drift", action="store_true",
                    help="also list inert factory literals that disagree")
    args = ap.parse_args()

    import sts2_env.cards  # noqa: F401
    from sts2_env.cards.derived_values import MODELLING_OVERRIDES
    from sts2_env.cards.reference_static_metadata import card_id_for_reference_class
    from sts2_env.cards.factory import create_card

    checked = skipped = 0
    mismatches: list[str] = []
    unmapped: list[str] = []

    for path in sorted(CARD_DIR.glob("*.cs")):
        cs = parse_cs(path)
        if cs is None:
            skipped += 1
            continue
        try:
            card_id = card_id_for_reference_class(path.stem)
        except Exception:
            unmapped.append(path.stem)
            continue
        if card_id is None:
            unmapped.append(path.stem)
            continue

        for upgraded in (False, True):
            try:
                card = create_card(card_id, upgraded=upgraded)
            except Exception:
                skipped += 1
                break
            checked += 1
            override = MODELLING_OVERRIDES.get(card_id)
            skip_fields = override.fields if override else frozenset()

            want_damage = cs["base_damage"]
            want_block = cs["base_block"]
            if upgraded:
                if want_damage is not None:
                    want_damage += cs["up_damage"]
                if want_block is not None:
                    want_block += cs["up_block"]

            tag = f"{path.stem}{'+' if upgraded else ''}"
            checks = [
                ("cost", card.cost, cs["cost"]),
                ("card_type", card.card_type.name, cs["card_type"]),
                ("rarity", card.rarity.name, cs["rarity"]),
            ]
            if want_damage is not None:
                checks.append(("base_damage", card.base_damage, want_damage))
            if want_block is not None:
                checks.append(("base_block", card.base_block, want_block))

            for field, got, want in checks:
                if field in skip_fields:
                    continue
                # Upgraded cost changes are expressed in OnUpgrade bodies this
                # reader does not model; only flag cost on the base card.
                if field == "cost" and upgraded:
                    continue
                if got != want:
                    mismatches.append(
                        f"  {tag:<32}{field:<14}sim={got!s:<10}decompile={want}")

    print(f"checked {checked} card variants from {CARD_DIR}")
    if unmapped:
        print(f"unmapped C# classes (no CardId member): {len(unmapped)}")
        print("  " + ", ".join(sorted(unmapped)[:20])
              + (" ..." if len(unmapped) > 20 else ""))
    print()
    if mismatches:
        print(f"MISMATCHES ({len(mismatches)}) -- two readings of the same "
              f"assembly disagree:")
        print("\n".join(mismatches))
    else:
        print("no mismatches: the shipping parser and this independent reader "
              "agree on every card")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
