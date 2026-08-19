"""Does a card deal damage as many times as the game says it does?

    .venv/bin/python scripts/audit_card_hit_counts.py

WHY HIT COUNT, OF ALL THINGS
----------------------------
Because it is the shape both real card bugs took, and neither was visible to
anything else we had.

  CONFLAGRATION was `DamageVar(2m)` with `RepeatVar(4)`. The simulator dealt
  `8 + 2 * attacks_played_this_turn` as ONE hit -- a mechanic the card does not
  have.

  SPITE is `hitCount = LostHpThisTurn ? Repeat(2) : 1` at 5 damage. The
  simulator dealt one hit and DREW A CARD. Same trigger, different effect.

`check_card_parity.py` could not see either: it compares declared values, and
both cards' damage numbers were already right. The bug was in how many times
that number lands, which changes everything downstream -- Strength and
Vulnerable apply per hit, block depletes across hits, and Thorns, Flame Barrier
and every on-hit relic fire once per hit. Four hits of 2 with +5 Strength is 28
where one hit of 8 is 13, and takes four Thorns triggers instead of one.

HOW THE EXPECTED COUNT IS READ
------------------------------
From the .cs, and only where it is unambiguous:

  no WithHitCount at all      -> 1
  WithHitCount(4)             -> 4
  WithHitCount(...Repeat...)  -> the RepeatVar(N) in CanonicalVars

Anything computed at play time -- `ResolveEnergyXValue()`, a local whose value
depends on the board, a cast expression -- is reported as UNRESOLVED rather than
guessed. A guess here would be indistinguishable from a finding, and this file
exists because a wrong number that looks like a right one costs a week.

HOW THE ACTUAL COUNT IS MEASURED
--------------------------------
By playing the card and counting calls to `apply_damage`. The card modules do
`from sts2_env.core.damage import apply_damage`, so each module holds its own
reference and patching the source module alone would count nothing -- every
importer is patched.

The position is deliberately dull: one enemy with more HP than any card can
remove, no block, no powers, full energy. A card that kills its target early
would under-count its own hits, and a card whose damage is modified would still
be counted correctly because this counts CALLS, not damage.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CARDS_CS = REPO / "decompiled" / "MegaCrit.Sts2.Core.Models.Cards"

_ATTACK = re.compile(r"DamageCmd\.Attack\(")
_HITCOUNT = re.compile(r"WithHitCount\(([^)]*)\)")
_REPEAT_VAR = re.compile(r"new\s+RepeatVar\((\d+)\)")
#: A card that plays another card cascades damage this method cannot attribute.
#: Uproar hits twice and then auto-plays a random Attack from the draw pile; the
#: counter saw three hits and called it a mismatch, and it was the harness.
_CASCADES = re.compile(r"CardCmd\.AutoPlay|PlayCardAction|CardCmd\.Play\b")


def _expected_hits(text: str) -> tuple[int | None, str]:
    """(hits, why). None means the game computes it at play time."""
    if not _ATTACK.search(text):
        return (None, "deals no damage")

    m = _HITCOUNT.search(text)
    if not m:
        return (1, "no WithHitCount")

    arg = m.group(1).strip()
    if arg.isdigit():
        return (int(arg), f"WithHitCount({arg})")
    if "Repeat" in arg:
        r = _REPEAT_VAR.search(text)
        if r:
            return (int(r.group(1)), f"RepeatVar({r.group(1)})")
        return (None, "Repeat with no RepeatVar found")
    return (None, f"computed: WithHitCount({arg[:40]})")


def _pascal_to_upper(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


def _count_hits(card_id, patched: list) -> int | None:
    """Play the card on a dull position; count apply_damage calls."""
    from sts2_env.cards.factory import create_card
    from sts2_env.search.situation import CardRef, CombatSituation

    combat = CombatSituation(
        situation_id="hits", character_id="Ironclad", current_hp=80, max_hp=80,
        deck=tuple([CardRef("STRIKE_IRONCLAD")] * 5 + [CardRef("DEFEND_IRONCLAD")] * 5),
        encounter="setup_shrinker_beetle_weak", encounter_seed=11, combat_seed=11,
        relics=("BURNING_BLOOD",)).to_combat()
    # More HP than any card can remove, so nothing dies early and under-counts.
    for enemy in combat.enemies:
        enemy.max_hp = enemy.current_hp = 9999
        enemy.block = 0
    try:
        card = create_card(card_id)
    except Exception:  # noqa: BLE001
        return None
    combat.hand.clear()
    combat.hand.append(card)
    combat.energy = 9

    # Count only damage to an ENEMY. Hemokinesis and Breakthrough pay HP to
    # deal damage, and counting the self-damage call made them read as two hits
    # where the game declares one -- the harness inventing a finding, which is
    # the failure this whole file is meant to catch elsewhere.
    player = combat.player
    counter = {"n": 0}
    for module, original in patched:
        def wrapper(target=None, *a, _o=original, **kw):
            if target is not None and target is not player:
                counter["n"] += 1
            return _o(target, *a, **kw)
        setattr(module, "apply_damage", wrapper)
    try:
        from sts2_env.gym_env.action_space import get_action_mask
        import numpy as np
        from sts2_env.gym_env.action_space import apply_combat_action
        mask = get_action_mask(combat)
        for action in np.where(mask == 1)[0]:
            if int(action) == 0:
                continue
            before = counter["n"]
            if apply_combat_action(combat, int(action)):
                return counter["n"] - before
        return None
    except Exception:  # noqa: BLE001
        return None
    finally:
        for module, original in patched:
            setattr(module, "apply_damage", original)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show-unresolved", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    import sts2_env.cards  # noqa: F401
    import sts2_env.powers  # noqa: F401
    from sts2_env.core.enums import CardId

    # Every module holding its own `apply_damage` reference.
    import sts2_env.core.damage as damage_mod
    patched = [(damage_mod, damage_mod.apply_damage)]
    for name, module in list(sys.modules.items()):
        if not name.startswith("sts2_env.") or module is None:
            continue
        fn = getattr(module, "apply_damage", None)
        if fn is not None and module is not damage_mod:
            patched.append((module, fn))

    names = {c.name for c in CardId}
    mismatch, unresolved, inert, cascading = [], [], [], []
    checked = unbuildable = 0
    for path in sorted(CARDS_CS.glob("*.cs")):
        cid = _pascal_to_upper(path.stem)
        if cid not in names:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if _CASCADES.search(text) and _ATTACK.search(text):
            cascading.append(cid)
            continue
        want, why = _expected_hits(text)
        if want is None:
            if why != "deals no damage":
                unresolved.append((cid, why))
            continue
        got = _count_hits(CardId[cid], patched)
        if got is None:
            unbuildable += 1
            continue
        checked += 1
        if got == 0 and want > 0:
            # The card is conditional and its condition is not met by a neutral
            # board -- Pact's End wants three cards exhausted, and this position
            # has none. That is the position's fault, not the card's. Setting up
            # every condition is a different and much larger job; naming them is
            # honest and still useful.
            inert.append((cid, want, why))
            continue
        if got != want:
            mismatch.append((cid, want, got, why))

    print("=" * 74)
    print(f"{checked} damaging cards played and counted "
          f"({len(unresolved)} the game computes at play time, "
          f"{unbuildable} would not build)")
    print("=" * 74)
    if cascading:
        print(f"\nPLAYS ANOTHER CARD -- damage cannot be attributed: {len(cascading)}"
              f"  -- not checked, not a finding")
        print("    " + ", ".join(cascading))
    if inert:
        print(f"\nCONDITIONAL, DID NOT FIRE ON A NEUTRAL BOARD: {len(inert)}"
              f"  -- not checked, not a finding")
        print("    " + ", ".join(cid for cid, _w, _y in inert))
    print(f"\nHIT COUNT MISMATCH: {len(mismatch)}")
    for cid, want, got, why in mismatch:
        print(f"    {cid:<24} game {want} ({why})   simulator {got}")
    if args.show_unresolved and unresolved:
        print(f"\nCOMPUTED AT PLAY TIME -- read these by hand: {len(unresolved)}")
        for cid, why in unresolved:
            print(f"    {cid:<24} {why}")
    print("\n" + "=" * 74)
    print("A mismatch here is never cosmetic. Strength and Vulnerable apply per\n"
          "hit, block depletes across hits, and Thorns and every on-hit relic\n"
          "fire once per hit -- so 2x4 and 8x1 are the same number and a\n"
          "different card.")
    return 1 if mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
