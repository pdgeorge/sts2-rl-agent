"""For each disputed card, put our code, the reference doc and the decompile side by side.

    python scripts/triage_parity.py CONFLAGRATION SPITE DOMINATE
    python scripts/triage_parity.py --failing        # every card in a failing Ironclad test

WHY

`create_reference_card` reads `docs/CARDS_REFERENCE.md`, so every "parity" test
compares the code against a MARKDOWN DOCUMENT rather than against the game. That
document is known to drift -- `gym_env/layout.py` lists it by name as a bug this
repo has already been bitten by:

    docs/CARDS_REFERENCE.md drifting from the decompile while the tests read it
    as an oracle and stayed green

So a failing parity test means "code and doc disagree" and says NOTHING about
which one matches the game. Both directions have now been confirmed:

    DEMON_FORM      code 4, test 3    decompile says 4    -> the TEST was stale
    SPITE           code draws cards  decompile hits x2   -> the CODE is wrong

Only `decompiled/` settles it. This script does not decide anything; it lays out
the three sources so a human can, which is the whole job of a triage tool.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

DECOMPILE_DIR = Path("decompiled/MegaCrit.Sts2.Core.Models.Cards")


def _pascal_case(card_id: str) -> list[str]:
    """CardId name -> candidate decompiled class names. DRUM_OF_BATTLE_CARD -> DrumOfBattle."""
    base = card_id
    for suffix in ("_CARD",):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    words = [w.capitalize() for w in base.split("_")]
    candidates = ["".join(words)]
    if len(words) > 1:                       # STRIKE_IRONCLAD -> Strike
        candidates.append(words[0])
    return candidates


def decompiled_facts(card_id: str) -> tuple[str, list[str]]:
    for name in _pascal_case(card_id):
        path = DECOMPILE_DIR / f"{name}.cs"
        if path.is_file():
            text = path.read_text()
            lines = []
            for pattern in (r"new \w*Var[^;]*", r"base\((\d+), CardType\.\w+[^)]*\)",
                            r"UpgradeValueBy\([^)]*\)", r"DynamicVars\[[^\]]*\]",
                            r"DynamicVars\.\w+"):
                lines += [m.group(0).strip() for m in re.finditer(pattern, text)]
            return str(path), sorted(set(lines))
    return "(not found)", []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cards", nargs="*")
    ap.add_argument("--failing", action="store_true",
                    help="derive the card list from failing Ironclad parity tests")
    args = ap.parse_args()

    from sts2_env.cards.factory import create_card, create_reference_card
    from sts2_env.core.enums import CardId

    cards = [c.upper() for c in args.cards]
    if args.failing or not cards:
        out = subprocess.run(
            ["python", "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
             "tests/test_ironclad_factory_upgrade_parity.py",
             "tests/test_ironclad_combat_edge_card_model_parity.py",
             "tests/test_ironclad_exhaust_self_damage_card_model_parity.py",
             "tests/test_ironclad_core_card_model_parity.py"],
            capture_output=True, text=True,
        ).stdout
        found = set(re.findall(r"\[(\w+)\]", out))
        for line in out.splitlines():
            if line.startswith("FAILED"):
                for name in re.findall(r"test_(\w+?)_(?:deals|applies|gains|exhausts|draws|loses|makes|increments|grants)", line):
                    found.add(name.upper())
        cards = sorted(c for c in found if hasattr(CardId, c))
        print(f"# derived {len(cards)} card(s) from failing tests\n")

    for name in cards:
        try:
            card_id = CardId[name]
        except KeyError:
            print(f"{name}: not a CardId\n")
            continue

        print("=" * 72)
        print(name)
        for upgraded in (False, True):
            try:
                ours = create_card(card_id, upgraded=upgraded)
                ours_s = f"cost={ours.cost} dmg={ours.base_damage} blk={ours.base_block} {ours.effect_vars}"
            except Exception as error:  # noqa: BLE001
                ours_s = f"ERROR {error!r}"
            try:
                ref = create_reference_card(card_id, upgraded=upgraded, allow_generation=True)
                ref_s = f"cost={ref.cost} dmg={ref.base_damage} blk={ref.base_block} {ref.effect_vars}"
            except Exception as error:  # noqa: BLE001
                ref_s = f"ERROR {error!r}"
            tag = "upgraded" if upgraded else "base    "
            print(f"  {tag} code : {ours_s}")
            print(f"  {tag} doc  : {ref_s}")
        path, facts = decompiled_facts(name)
        print(f"  decompile {path}")
        for fact in facts:
            print(f"      {fact}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
