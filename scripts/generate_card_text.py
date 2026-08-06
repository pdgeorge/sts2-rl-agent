"""Write every card out as the JSON an embedding model should read.

    python scripts/generate_card_text.py --out output/card_text.json

The Phase 5 archetype work needs a vector per card. Vectors come from text, and
this is where the text comes from.

WHY THE SIMULATOR AND NOT THE DECOMPILE
---------------------------------------
The decompile is the source of truth for what a card *is*, and `on_update.sh`
already diffs it against the simulator on every patch (step 4, card values).
Reading it again here would mean writing a second C# parser that can drift from
the first, and the existing one is deliberately conservative -- its own comment
notes the extractor reads a card's first DamageVar, "which is right for most
cards and wrong for the ones with conditional or multi-hit damage".

So this reads the **simulator**, which is the thing the agent actually plays
with, and which the decompile diff already polices. If a patch changes a card
and nobody updates `sts2_env/`, `on_update.sh` says so and this file is stale in
exactly the same way the simulator is -- one staleness to track, not two.

WHY NOT THE PUBLISHED DATASET
-----------------------------
`t22000t/slay-the-spire-2-card-embeddings` is a fine reference and is **several
game versions out of date**. Building on it would mean encoding cards as they
used to be, and depending on someone else to republish after each patch. This
plus the 0.6B encoder makes us independent: text from our simulator, vectors
regenerated on our patch day.

SHAPE
-----
Deliberately close to the published dataset's, so its vectors stay usable as a
cross-check: name, type, rarity, color, cost, description, keywords. The
description is assembled from what the simulator knows rather than the game's
localisation strings, which live in the install rather than the decompile --
"Deal 32 damage." rather than the exact shipped wording. What matters for
similarity is that mechanically alike cards read alike, and they do.
"""

from __future__ import annotations

import argparse
import json
import sys


def _describe(preview) -> str:
    """A sentence a reader -- or an encoder -- can tell cards apart by.

    Assembled from mechanics rather than copied from localisation, because the
    strings ship in the game install and not in the decompile. Ordered
    damage-then-block-then-modifiers so that two attacks read alike and an
    attack never reads like a skill.
    """
    parts: list[str] = []
    damage = preview.base_damage or 0
    block = preview.base_block or 0
    if damage:
        parts.append(f"Deal {damage} damage.")
    if block:
        parts.append(f"Gain {block} Block.")
    if preview.is_power:
        parts.append("A Power: its effect lasts for the rest of the combat.")
    if preview.exhausts:
        parts.append("Exhaust.")
    if preview.is_ethereal:
        parts.append("Ethereal: exhausted if still in hand at end of turn.")
    if preview.is_innate:
        parts.append("Innate: starts in the opening hand.")
    if preview.is_retain:
        parts.append("Retain: kept in hand between turns.")
    if preview.is_unplayable:
        parts.append("Unplayable.")
    if preview.has_energy_cost_x:
        parts.append("Costs X energy and scales with the energy spent.")
    if preview.base_replay_count:
        parts.append(f"Repeats {preview.base_replay_count} additional times.")
    if preview.affliction:
        parts.append(f"Applies {preview.affliction}.")
    for name, amount in sorted((preview.effect_vars or {}).items()):
        parts.append(f"{name}: {amount}.")
    if not parts:
        parts.append("No direct damage or block; its effect is situational.")
    return "\n".join(parts)


def card_text(card_id) -> dict:
    """One card, as the dict that gets serialised and encoded."""
    from sts2_env.bridge.card_quality import _metadata

    meta, preview = _metadata(card_id)
    keywords = sorted(str(k).split(".")[-1] for k in (preview.keywords or ()))
    tags = sorted(str(t).split(".")[-1] for t in (preview.tags or ()))
    return {
        "name": card_id.name.replace("_", " ").title(),
        "type": str(preview.card_type).split(".")[-1].title(),
        "rarity": str(preview.rarity).split(".")[-1].title(),
        "color": str(preview.visual_card_pool.value
                     if hasattr(preview.visual_card_pool, "value")
                     else preview.visual_card_pool).lower(),
        "cost": str(preview.cost),
        "target": str(preview.target_type).split(".")[-1],
        "description": _describe(preview),
        "keywords": keywords,
        "tags": tags,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="output/card_text.json")
    parser.add_argument("--pool", default=None,
                        help="Only this card pool (e.g. Ironclad).")
    args = parser.parse_args()

    import sts2_env.cards  # noqa: F401  (resolves package import order)
    from sts2_env.core.enums import CardId

    rows: dict[str, dict] = {}
    skipped: list[str] = []
    for card_id in CardId:
        try:
            row = card_text(card_id)
        except Exception as exc:
            # A card the simulator names but cannot build is a content gap, not
            # a reason to produce no corpus. Reported, not swallowed.
            skipped.append(f"{card_id.name}: {type(exc).__name__}")
            continue
        if args.pool and row["color"].lower() != args.pool.lower():
            continue
        rows[card_id.name] = row

    from pathlib import Path

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(f"wrote {len(rows)} cards to {out}")
    if skipped:
        print(f"{len(skipped)} could not be built (content gaps):")
        for line in skipped[:10]:
            print(f"  {line}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
