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
import functools
import json
import re
import sys
from pathlib import Path


DECOMPILE_CARDS = Path("decompiled/MegaCrit.Sts2.Core.Models.Cards")

_APPLY = re.compile(r"Apply<(\w+?)Power>")
_CMD = re.compile(r"\b(\w+)Cmd\.(\w+)")


@functools.lru_cache(maxsize=None)
def _effects_from_decompile(class_name: str) -> tuple[str, ...]:
    """What the card's own C# says it does, as short phrases.

    The preview fields cannot distinguish two Powers that grant nothing
    numeric: Barricade and Dark Embrace both reduce to "a Power with no damage
    or block", so they encode identically and an archetype classifier cannot
    tell a block deck from an exhaust deck. Measured: leave-one-out fell to
    7/15 on text built from preview fields alone.

    The decompile names them -- `Apply<BarricadePower>`, `Apply<DarkEmbracePower>`
    -- which is exactly the discriminating signal. Read from the decompile
    rather than hand-listed, so a rebalance patch updates it via on_update.sh
    like everything else.

    Returns () when the file is missing: the decompile is a symlink to a path
    outside the repo, and a missing one should cost description quality, not
    raise.
    """
    path = DECOMPILE_CARDS / f"{class_name}.cs"
    try:
        source = path.read_text(errors="ignore")
    except OSError:
        return ()
    phrases: list[str] = []
    for power in dict.fromkeys(_APPLY.findall(source)):
        phrases.append(f"Applies {_spaced(power)}.")
    verbs = dict.fromkeys(
        f"{_spaced(obj)} {action.lower()}"
        for obj, action in _CMD.findall(source)
        if obj not in {"Card"} or action not in {"Upgrade"}
    )
    if verbs:
        phrases.append("Uses: " + ", ".join(list(verbs)[:6]) + ".")
    return tuple(phrases)


_POWER_REF = re.compile(r"PowerId\.([A-Z_]+)")
_TAG_REF = re.compile(r"CardTag\.([A-Z_]+)")


@functools.lru_cache(maxsize=None)
def _power_descriptions() -> dict[str, str]:
    """PowerId name -> the first line of its simulator class docstring.

    This is where the semantic content comes from, and it is the whole reason
    the archetype classifier works at all.

    A description assembled from preview fields reduces every Power with no
    numeric grant to one sentence -- Barricade and Dark Embrace came out
    byte-identical -- and naming the power does not help either, because the
    encoder has no Slay the Spire knowledge and "Barricade" is an opaque token
    to it. Measured: 7/15 leave-one-out both ways.

    The docstrings are plain English about *what the effect does*: "Block is not
    removed at the start of turn", "Whenever a card is Exhausted, draw Amount
    card(s)". That is the register the published dataset's text is in, and it
    scored 14/15 on the same seeds. They are also written by whoever ported each
    power and live beside the code, so on_update.sh already covers them.
    """
    import sts2_env.powers  # noqa: F401  -- import for the side effect of
    # every power module registering itself; POWER_CLASSES in common.py is a
    # legacy alias holding only nine of them.
    from sts2_env.core.creature import _POWER_CLASSES

    out: dict[str, str] = {}
    for power_id, cls in _POWER_CLASSES.items():
        doc = (cls.__doc__ or "").strip()
        if not doc:
            continue
        first = doc.split("\n")[0].strip()
        if first:
            out[getattr(power_id, "name", str(power_id))] = first
    return out


@functools.lru_cache(maxsize=None)
def _effects_from_simulator(card_id_name: str) -> tuple[str, ...]:
    """What the card's registered effect function actually does.

    Reads the simulator rather than the decompile: the effect function is short
    and explicit -- `barricade` is one call to `apply_power_to(owner,
    PowerId.BARRICADE, 1)` -- so the powers it applies can be read off the
    source, and each power's docstring says what it means.
    """
    import inspect

    from sts2_env.cards.registry import _CARD_EFFECTS
    from sts2_env.core.enums import CardId

    card_id = getattr(CardId, card_id_name, None)
    effect = _CARD_EFFECTS.get(card_id) if card_id is not None else None
    if effect is None:
        return ()
    try:
        source = inspect.getsource(effect)
    except (OSError, TypeError):
        return ()

    described = _power_descriptions()
    phrases: list[str] = []
    for power in dict.fromkeys(_POWER_REF.findall(source)):
        text = described.get(power)
        pretty = _spaced(power.title().replace("_", ""))
        phrases.append(f"Applies {pretty}: {text}" if text else f"Applies {pretty}.")
    tags = dict.fromkeys(_TAG_REF.findall(source))
    if tags:
        phrases.append(
            "Scales with the number of "
            + ", ".join(t.title() for t in tags)
            + " cards you own."
        )
    return tuple(phrases)


def _spaced(camel: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", camel)


def _class_name(card_id) -> str:
    """CardId.BARRICADE_CARD -> "Barricade", matching the decompiled file name."""
    stem = card_id.name[:-5] if card_id.name.endswith("_CARD") else card_id.name
    return "".join(part.title() for part in stem.split("_"))


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
    # NO BOILERPLATE. A sentence every Power shares is the longest common
    # substring in the corpus, so it dominates similarity and clusters cards by
    # *type* rather than by effect -- Barricade, Corruption and Dark Embrace all
    # classified as "strength" because they shared it with Demon Form and
    # Inflame. `type` is already a field; saying it again in prose only hurts.
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
    return "\n".join(parts)


def card_text(card_id) -> dict:
    """One card, as the dict that gets serialised and encoded."""
    from sts2_env.bridge.card_quality import _metadata

    meta, preview = _metadata(card_id)
    keywords = sorted(str(k).split(".")[-1] for k in (preview.keywords or ()))
    tags = sorted(str(t).split(".")[-1] for t in (preview.tags or ()))
    effects = _effects_from_simulator(card_id.name) or _effects_from_decompile(_class_name(card_id))
    description = _describe(preview)
    if effects:
        description = description + "\n" + "\n".join(effects)
    return {
        # Strip the simulator's `_CARD` suffix: it named the card "Barricade
        # Card", which is noise in an embedding and wrong on a screen.
        "name": _spaced(_class_name(card_id)),
        "type": str(preview.card_type).split(".")[-1].title(),
        "rarity": str(preview.rarity).split(".")[-1].title(),
        "color": str(preview.visual_card_pool.value
                     if hasattr(preview.visual_card_pool, "value")
                     else preview.visual_card_pool).lower(),
        "cost": str(preview.cost),
        "target": str(preview.target_type).split(".")[-1],
        "description": description,
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
