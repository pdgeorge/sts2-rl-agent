"""Diff two decompiled StS2 trees: what a game patch actually changed.

`check_card_parity.py` answers "is the simulator wrong?" by comparing it to one
decompile. This answers the prior question -- "what did Mega Crit change?" -- by
comparing two decompiles of the game to each other.

That distinction matters. Sim-vs-game drift is a standing backlog that never
reaches zero, so a patch's real changes are invisible inside it: 19 new cards
land in a list of 196 unimplemented ones and nothing stands out. Game-vs-game
has no such background. Every line of output is something this patch did, which
is the only list short enough to act on.

It is also the only comparison with no heuristics on either side. Both inputs
came out of the same decompiler, so a difference is a difference -- unlike the
sim side, where "the extractor could not read this literal" and "the value is
wrong" have to be told apart.

Usage:

    python scripts/diff_decompiles.py --old <decompile.prev> --new <decompile>

Normally invoked by scripts/on_update.sh, which keeps the previous tree around
precisely so this can run.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_card_parity as ccp
import parity_reference_audit as pra

# Every namespace directory the reference audit already knows about, so the two
# tools cannot drift apart on what "the surfaces" are.
SURFACE_DIRS: dict[str, str] = {
    name: Path(cfg.reference_dir).name for name, cfg in pra.SURFACES.items()
}

# If nearly everything looks changed, the two trees almost certainly came out of
# different ilspycmd versions rather than different game builds.
DECOMPILER_MISMATCH_RATIO = 0.8


@dataclass
class FieldChange:
    field: str
    old: object
    new: object


@dataclass
class SurfaceDiff:
    surface: str
    old_total: int
    new_total: int
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    # Only populated for cards, where there is a scalar extractor to diff with.
    field_changes: dict[str, list[FieldChange]] = field(default_factory=dict)
    # Names the simulator mentions. A removed name the sim knows about is work;
    # a removed name it never heard of is free.
    sim_knows: dict[str, bool] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed)


def _class_files(root: Path, dirname: str) -> dict[str, Path]:
    d = root / dirname
    if not d.is_dir():
        return {}
    return {p.stem: p for p in sorted(d.glob("*.cs"))}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _sim_alias_index(repo_root: Path) -> pra.AliasIndex:
    """One index over the whole simulator, reused for every surface."""
    sim_dirs = tuple(
        p
        for cfg in pra.SURFACES.values()
        for p in cfg.implementation_paths
    )
    return pra.alias_index(pra.collect_code_text(repo_root, sim_dirs))


def diff_surface(
    surface: str,
    old_root: Path,
    new_root: Path,
    sim_index: pra.AliasIndex | None,
) -> SurfaceDiff:
    dirname = SURFACE_DIRS[surface]
    old = _class_files(old_root, dirname)
    new = _class_files(new_root, dirname)

    diff = SurfaceDiff(surface=surface, old_total=len(old), new_total=len(new))
    diff.added = sorted(new.keys() - old.keys())
    diff.removed = sorted(old.keys() - new.keys())
    diff.changed = sorted(
        name for name in old.keys() & new.keys() if _text(old[name]) != _text(new[name])
    )

    if sim_index is not None:
        suffixes = pra.SURFACES[surface].suffixes
        explicit = pra.SURFACES[surface].explicit_aliases
        for name in diff.added + diff.removed:
            aliases = pra.aliases_for(name, suffixes, explicit)
            diff.sim_knows[name] = bool(pra.alias_hits(sim_index, aliases))

    return diff


def diff_card_values(old_root: Path, new_root: Path) -> dict[str, list[FieldChange]]:
    """Field-level card scalars, old build vs new build.

    Uses the same extractor as check_card_parity, so a card whose value it cannot
    read statically is simply absent from both sides and reports nothing -- the
    extractor's blind spots are identical on both trees and therefore cancel.
    """
    old = ccp.parse_csharp(old_root)
    new = ccp.parse_csharp(new_root)

    fields = ("cost", "card_type", "rarity", "target", "damage", "block",
              "upgrade_damage", "upgrade_block")
    out: dict[str, list[FieldChange]] = {}
    for cid in sorted(old.keys() & new.keys()):
        changes = [
            FieldChange(f, getattr(old[cid], f), getattr(new[cid], f))
            for f in fields
            if getattr(old[cid], f) != getattr(new[cid], f)
        ]
        if changes:
            out[cid] = changes
    return out


def _mark(diff: SurfaceDiff, name: str) -> str:
    known = diff.sim_knows.get(name)
    if known is None:
        return ""
    return "  [sim has it]" if known else "  [sim never had it]"


def print_report(diffs: list[SurfaceDiff], suspect_decompiler: bool,
                 show_changed: bool = False) -> None:
    if suspect_decompiler:
        print("WARNING: almost every class differs. The two trees were probably produced")
        print("         by different ilspycmd versions, not different game builds.")
        print("         Re-decompile both with the same tool before trusting 'changed'.\n")

    print(f"{'surface':<12} {'old':>5} {'new':>5} {'added':>6} {'removed':>8} "
          f"{'changed':>8} {'values':>7}")
    print("-" * 58)
    for d in diffs:
        print(f"{d.surface:<12} {d.old_total:>5} {d.new_total:>5} "
              f"{len(d.added):>6} {len(d.removed):>8} {len(d.changed):>8} "
              f"{len(d.field_changes):>7}")

    for d in diffs:
        if d.is_empty:
            continue
        print(f"\n=== {d.surface} ===")
        if d.removed:
            # Removals first: they are the dangerous class. A card the sim still
            # trains on but the game deleted poisons the training distribution
            # silently, which is worse than a card that is merely missing.
            print(f"  REMOVED ({len(d.removed)}):")
            for n in d.removed:
                print(f"    - {n}{_mark(d, n)}")
        if d.added:
            print(f"  ADDED ({len(d.added)}):")
            for n in d.added:
                print(f"    + {n}{_mark(d, n)}")
        if d.field_changes:
            print(f"  CHANGED VALUES ({len(d.field_changes)}):")
            for n, changes in sorted(d.field_changes.items()):
                print(f"    ~ {n}")
                for fc in changes:
                    print(f"        {fc.field}: {fc.old} -> {fc.new}")

        # Most "changed" classes changed only in how they call the engine --
        # `FromCard(this)` becoming `FromCard(this, cardPlay)` and the like. That
        # is real, but it is not a number the simulator can be wrong about, and
        # listing 483 of them buries the 4 that are. Counted here, named in the
        # JSON, printed only on request.
        behaviour_only = [n for n in d.changed if n not in d.field_changes]
        if behaviour_only:
            print(f"  CHANGED BEHAVIOUR ONLY ({len(behaviour_only)})"
                  f"{':' if show_changed else '  -- --show-changed to list'}")
            if show_changed:
                for n in behaviour_only:
                    print(f"    ~ {n}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, type=Path,
                    help="Previous decompile root (ilspycmd -p output)")
    ap.add_argument("--new", required=True, type=Path,
                    help="Current decompile root")
    ap.add_argument("--repo", type=Path,
                    default=Path(__file__).resolve().parent.parent,
                    help="Repository root, for checking what the simulator knows about")
    ap.add_argument("--surface", action="append", choices=sorted(SURFACE_DIRS),
                    help="Limit to one surface. Repeatable. Defaults to all.")
    ap.add_argument("--show-changed", action="store_true",
                    help="List every changed class, not just the ones whose values moved")
    ap.add_argument("--no-sim-check", action="store_true",
                    help="Skip cross-referencing names against the simulator")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    for label, root in (("old", args.old), ("new", args.new)):
        if not root.is_dir():
            print(f"{label} decompile root not found: {root}", file=sys.stderr)
            return 2

    surfaces = tuple(args.surface or sorted(SURFACE_DIRS))
    sim_index = None if args.no_sim_check else _sim_alias_index(args.repo)

    diffs = [diff_surface(s, args.old, args.new, sim_index) for s in surfaces]

    if "cards" in surfaces:
        card_changes = diff_card_values(args.old, args.new)
        for d in diffs:
            if d.surface == "cards":
                d.field_changes = card_changes

    common = sum(min(d.old_total, d.new_total) for d in diffs)
    changed = sum(len(d.changed) for d in diffs)
    suspect = common > 0 and changed / common > DECOMPILER_MISMATCH_RATIO

    if args.json:
        print(json.dumps(
            {"suspect_decompiler_mismatch": suspect,
             "surfaces": [asdict(d) for d in diffs]},
            indent=2, sort_keys=True))
    else:
        print_report(diffs, suspect, show_changed=args.show_changed)

    return 1 if any(not d.is_empty for d in diffs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
