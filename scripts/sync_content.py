"""Bring the simulator's content inventory up to the installed game. Additive only.

Run after scripts/on_update.sh has refreshed the decompile. It answers "what does
the game have that the simulator does not", then does the part of the answer that
can be done mechanically.

WHAT IT ADDS, AND WHAT IT REFUSES TO

Card values are no longer part of this. derived_values.py reads cost, type,
target, rarity, damage, block and effect_vars from the decompile at construction
time, so a new card needs no numbers written at all. What it needs is OnPlay
behaviour, and nothing can derive that from a decompiled method body.

So the split is:

  Added automatically -- enum members. CardId, PowerId, RelicId. A name is a name;
  adding one is safe, reversible, and it is what unblocks the reference parser,
  which raises on the first class it cannot map and takes the whole test suite
  with it.

  Scaffolded for review -- everything with behaviour. Written to a directory that
  is NOT imported, each file carrying the decompiled C# so the behaviour can be
  written next to its source.

  Never done -- registering a stub as a working card. A card that exists, is
  dealt, and does nothing is worse than a card that is missing: the policy learns
  around a hole and nothing reports it. Every silent failure this project has hit
  had that shape, so scaffolding stays out of the import path until a human puts
  it in.

Nothing is ever deleted. Content the game removed stays as dead code -- ripping
Doormaker out is ~90 references through core/combat.py for something already
unreachable, and the risk is all on the deletion side.

Usage:
    python scripts/sync_content.py                 # report only
    python scripts/sync_content.py --write         # add enum members
    python scripts/sync_content.py --write --scaffold  # + write review stubs
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sts2_env.cards.reference_static_metadata import reference_card_dir  # noqa: E402
from sts2_env.core.enums import CardId, PowerId  # noqa: E402
from sts2_env.relics.base import RelicId  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ENUMS_PATH = REPO / "sts2_env" / "core" / "enums.py"
RELIC_BASE_PATH = REPO / "sts2_env" / "relics" / "base.py"
DEFAULT_SCAFFOLD = REPO / "scaffold"

CAMEL_1 = re.compile(r"(.)([A-Z][a-z]+)")
CAMEL_2 = re.compile(r"([a-z0-9])([A-Z])")

# Cards whose name collides with a power get a _CARD suffix in the simulator, so
# the bare name must not be treated as absent.
CARD_SUFFIXES = ("", "_CARD", "_STATUS")


def snake_upper(name: str) -> str:
    return CAMEL_2.sub(r"\1_\2", CAMEL_1.sub(r"\1_\2", name)).upper()


@dataclass
class Surface:
    name: str
    decompiled_dir: str
    # The model class a real entry inherits from. A namespace directory also holds
    # helpers -- Models.Relics contains VakuuCardSelector : ICardSelector, which is
    # not a relic -- and counting every file made the tool propose an enum member
    # for a card-selection helper.
    base_class: str
    strip_suffix: str = ""
    enum: object = None
    enum_path: Path = ENUMS_PATH
    enum_name: str = ""
    candidates: tuple[str, ...] = ("",)
    missing: list[tuple[str, str]] = field(default_factory=list)  # (class name, enum member)


SURFACES = (
    Surface("cards", "MegaCrit.Sts2.Core.Models.Cards", base_class="CardModel", enum=CardId,
            enum_name="CardId", candidates=CARD_SUFFIXES),
    Surface("powers", "MegaCrit.Sts2.Core.Models.Powers", base_class="PowerModel", strip_suffix="Power",
            enum=PowerId, enum_name="PowerId"),
    Surface("relics", "MegaCrit.Sts2.Core.Models.Relics", base_class="RelicModel", enum=RelicId,
            enum_name="RelicId", enum_path=RELIC_BASE_PATH),
)


CLASS_DECL = re.compile(r"class\s+(\w+)\s*:\s*([\w<>]+)")


def _inherits(stem: str, bases: dict[str, str], target: str) -> bool:
    """Follow the inheritance chain, not just the immediate base.

    Powers do not all derive from PowerModel directly: FadePower extends
    TemporaryDexterityPower, and there are three such intermediates. Matching only
    the immediate base silently dropped a real power from the missing list, which
    is the failure this whole script exists to stop.
    """
    seen: set[str] = set()
    current = stem
    while current and current not in seen:
        seen.add(current)
        base = bases.get(current)
        if base == target:
            return True
        current = base
    return False


def find_missing(surface: Surface, root: Path) -> None:
    directory = root / surface.decompiled_dir
    if not directory.is_dir():
        return

    bases: dict[str, str] = {}
    for path in directory.glob("*.cs"):
        for name, base in CLASS_DECL.findall(path.read_text(encoding="utf-8", errors="replace")):
            bases.setdefault(name, base)

    members = surface.enum.__members__
    for path in sorted(directory.glob("*.cs")):
        stem = path.stem
        if "Deprecated" in stem or not _inherits(stem, bases, surface.base_class):
            continue
        base = snake_upper(stem.removesuffix(surface.strip_suffix) if surface.strip_suffix else stem)
        if any(f"{base}{suffix}" in members for suffix in surface.candidates):
            continue
        surface.missing.append((stem, base))


def add_enum_members(surface: Surface, dry_run: bool) -> int:
    """Append members to the end of the enum class body.

    Appending rather than inserting in sorted position: these enums use auto(),
    so inserting renumbers every member after the insertion point. Nothing should
    depend on those numbers, but 'should' is doing a lot of work in a file this
    widely imported, and appending cannot renumber anything.
    """
    if not surface.missing:
        return 0
    text = surface.enum_path.read_text()
    match = re.search(rf"^class {surface.enum_name}\(Enum\):\n", text, re.MULTILINE)
    if not match:
        print(f"  ! could not find 'class {surface.enum_name}(Enum):' in {surface.enum_path}")
        return 0

    # Insert after the class body's last non-blank line, leaving the blank lines
    # that separate the class from whatever follows exactly as they were. Moving
    # the insertion point back past them instead just pushed them into the tail,
    # and they reappeared below the addition -- four blank lines before the next
    # class, growing by two on every run.
    body_start = match.end()
    rest = text[body_start:]
    end_match = re.search(r"\n(?=\S)", rest)
    body_end = body_start + (end_match.start() + 1 if end_match else len(rest))
    insert_at = len(text[:body_end].rstrip())

    block = [
        "",
        "",
        "    # Present in the game, not yet implemented here. Added by",
        "    # scripts/sync_content.py so the reference parser can map every",
        "    # decompiled class; a member is a name, not an implementation.",
    ]
    block += [f"    {member} = auto()" for _stem, member in surface.missing]
    addition = "\n".join(block)

    if not dry_run:
        surface.enum_path.write_text(text[:insert_at] + addition + text[insert_at:])
    return len(surface.missing)


CS_ONPLAY = re.compile(
    r"(protected|public)\s+override\s+(async\s+)?Task\s+OnPlay\b.*?(?=\n\t(protected|public|\}))",
    re.DOTALL)


def scaffold(surface: Surface, root: Path, out_dir: Path, dry_run: bool) -> int:
    if not surface.missing:
        return 0
    target = out_dir / surface.name
    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)
    for stem, member in surface.missing:
        source = (root / surface.decompiled_dir / f"{stem}.cs")
        cs = source.read_text(encoding="utf-8", errors="replace") if source.is_file() else ""
        body_start = cs.find(f"class {stem}")
        behaviour = cs[body_start:] if body_start >= 0 else cs
        body = f'''"""Scaffold for {stem} -- NOT IMPLEMENTED, NOT IMPORTED.

This file is deliberately outside the import path. Move the implementation into
the right sts2_env module by hand; do not import this directory.

Values do not need writing: derived_values.py reads cost, type, target, rarity,
damage, block and effect_vars from the decompile at construction time. Only the
behaviour below has to be translated.

Decompiled source ({surface.decompiled_dir}/{stem}.cs):

{behaviour}
"""

# TODO: implement {member} and register it, then delete this file.
'''
        if not dry_run:
            (target / f"{snake_upper(stem).lower()}.py").write_text(body)
    return len(surface.missing)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="Actually add enum members")
    ap.add_argument("--scaffold", action="store_true",
                    help="Also write review stubs (requires --write)")
    ap.add_argument("--scaffold-dir", type=Path, default=DEFAULT_SCAFFOLD)
    args = ap.parse_args()

    root = reference_card_dir().parent
    print(f"decompile in use: {root}\n")

    total = 0
    for surface in SURFACES:
        find_missing(surface, root)
        total += len(surface.missing)
        print(f"{surface.name:<8} missing {len(surface.missing):>3}"
              + (f": {', '.join(s for s, _ in surface.missing[:6])}"
                 f"{' ...' if len(surface.missing) > 6 else ''}" if surface.missing else ""))

    if not total:
        print("\nNothing to add. The simulator names everything the game has.")
        return 0

    dry = not args.write
    print(f"\n{'WOULD ADD' if dry else 'ADDING'} enum members:")
    for surface in SURFACES:
        count = add_enum_members(surface, dry)
        if count:
            print(f"  {surface.enum_name}: +{count}  ({surface.enum_path.relative_to(REPO)})")

    if args.scaffold:
        print(f"\n{'WOULD WRITE' if dry else 'WRITING'} review stubs to {args.scaffold_dir}:")
        for surface in SURFACES:
            count = scaffold(surface, root, args.scaffold_dir, dry)
            if count:
                print(f"  {surface.name}: {count} files")
        print("  (this directory is not imported; behaviour must be moved in by hand)")

    if dry:
        print("\nDry run. Nothing written. Re-run with --write, then read `git diff`.")
    else:
        print("\nDone. Enum members are names only -- the behaviour is still missing,")
        print("and the reference audit will keep saying so until it is written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
