"""Rewrite the simulator's card scalars to match the decompiled game.

Doing these edits by hand is worse than doing them by script, and not by a little.
A script's mistakes are systematic -- the same wrong rule applied 60 times, visible
as a pattern in `git diff` and fixable in one place. A human's mistakes are random:
one transposed digit at edit 43 that looks exactly like the other 59 and never
shows up until a policy has trained on it. Sixty careful edits is precisely the
kind of work where care does not help.

So this applies them, under a rule it can state: it edits a field only when the
game's value is unambiguous AND the simulator's value is a literal it can replace
exactly. Everything else is refused by name and reason -- never guessed at.

WHAT GETS REFUSED, AND WHY IT MATTERS

The motivating case is Conflagration. The game says `new DamageVar(2m)`, the
simulator says `base_damage=8`, and that looks like a 4x error until you read
OnPlay:

    DamageCmd.Attack(Damage.BaseValue).WithHitCount(Repeat.IntValue)   // 4 hits of 2

The simulator collapses the multi-hit into one 8-damage swing. It is a different
modelling choice, not drift, and "fixing" it would quarter a rare card's damage
with nothing failing and no diff worth noticing. So `WithHitCount` or a
`RepeatVar` anywhere in the card disqualifies its damage from automatic rewriting.

The full refusal list:
  - the card's type or target changed  -> it was redesigned; behaviour must be
    rewritten, and a number swap would paper over that
  - WithHitCount / RepeatVar present   -> the two sides model hits differently
  - more than one DamageVar/BlockVar   -> conditional damage; which one is base
    is a judgement call
  - the simulator's value is a name    -> it comes from a module constant that may
    be shared with other cards
  - anything the extractor could not read on either side

VERIFICATION

After writing, every touched file is re-parsed and every applied value is
re-extracted and checked against the game. If any file fails to parse or any value
did not land, the run aborts and says so. Review the result with `git diff`.

Usage:
    python scripts/apply_card_values.py --decompiled <dir>          # dry run
    python scripts/apply_card_values.py --decompiled <dir> --write
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_card_parity as ccp

# Signals in the decompiled card that mean the two sides do not model damage the
# same way, so no scalar comparison between them is meaningful.
MULTI_HIT_RE = re.compile(r"WithHitCount|RepeatVar")
DAMAGE_VAR_RE = re.compile(r"new\s+DamageVar\(")
BLOCK_VAR_RE = re.compile(r"new\s+BlockVar\(")

# game field -> simulator keyword
NUMERIC_FIELDS = {"damage": "base_damage", "block": "base_block", "cost": "cost"}


@dataclass
class SimCardSite:
    """Where a card is constructed in the simulator, with source spans."""
    card_id: str
    path: Path
    keywords: dict[str, ast.AST]
    duplicate: bool = False


@dataclass
class Edit:
    path: Path
    start: int          # byte offset
    end: int
    new_text: str


@dataclass
class Decision:
    card_id: str
    applied: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    edits: list[Edit] = field(default_factory=list)


def _line_starts(data: bytes) -> list[int]:
    starts, pos = [0], 0
    for line in data.splitlines(keepends=True):
        pos += len(line)
        starts.append(pos)
    return starts


def _span(node: ast.AST, starts: list[int]) -> tuple[int, int]:
    """Absolute byte span. ast column offsets are UTF-8 byte offsets into the
    line, so the whole rewrite is done on bytes rather than str."""
    return (starts[node.lineno - 1] + node.col_offset,
            starts[node.end_lineno - 1] + node.end_col_offset)


def collect_sim_sites(cards_dir: Path) -> dict[str, SimCardSite]:
    sites: dict[str, SimCardSite] = {}
    for path in sorted(cards_dir.glob("*.py")):
        try:
            tree = ast.parse(path.read_bytes())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if fname != "CardInstance":
                continue
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            cid = kw.get("card_id")
            if not isinstance(cid, ast.Attribute):
                continue
            card_id = cid.attr
            if card_id in sites:
                # Two constructors for one card usually means base/upgraded
                # variants written out separately. Picking one is a coin flip, so
                # take neither.
                sites[card_id].duplicate = True
                continue
            sites[card_id] = SimCardSite(card_id=card_id, path=path, keywords=kw)
    return sites


def sim_enum_members(enums_path: Path, enum_name: str) -> frozenset[str]:
    """Members the simulator's enum actually defines.

    The game can introduce a value the simulator has no name for -- v0.109.1 added
    a TOKEN rarity -- and writing `CardRarity.TOKEN` would produce a file that
    parses cleanly and raises AttributeError the first time that card is dealt.
    Adding the member is a code change, not a value swap, so those are refused.
    """
    try:
        tree = ast.parse(enums_path.read_bytes())
    except (OSError, SyntaxError):
        return frozenset()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == enum_name:
            return frozenset(
                t.id
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                for t in stmt.targets
                if isinstance(t, ast.Name)
            )
    return frozenset()


def _fmt(value: float | int) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _numeric_edits(node: ast.AST, base: float | int, upgraded: float | int | None,
                   starts: list[int], path: Path) -> tuple[list[Edit], str | None]:
    """Edits to make a literal (or `X if upgraded else Y`) match the game.

    Returns (edits, refusal_reason). The simulator stores the upgraded number
    absolutely where the game stores a delta, so the delta is resolved here rather
    than written through.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if upgraded is not None:
            return [], "game has an upgrade delta but the sim stores one flat value"
        s, e = _span(node, starts)
        return [Edit(path, s, e, _fmt(base))], None

    if isinstance(node, ast.IfExp):
        if not (isinstance(node.body, ast.Constant)
                and isinstance(node.orelse, ast.Constant)
                and isinstance(node.body.value, (int, float))
                and isinstance(node.orelse.value, (int, float))):
            return [], "sim's upgraded/base expression is not two literals"
        if upgraded is None:
            return [], "sim stores an upgraded value but the game has no upgrade delta"
        s1, e1 = _span(node.body, starts)
        s2, e2 = _span(node.orelse, starts)
        return ([Edit(path, s1, e1, _fmt(base + upgraded)),
                 Edit(path, s2, e2, _fmt(base))], None)

    if isinstance(node, ast.Name):
        return [], f"sim's value comes from the constant {node.id}, possibly shared"

    return [], "sim's value is a computed expression"


def decide(card_id: str, game: ccp.GameCard, site: SimCardSite, sim: ccp.SimCard,
           starts: list[int], known_rarities: frozenset[str]) -> Decision:
    d = Decision(card_id=card_id)

    if site.duplicate:
        d.refused.append("more than one CardInstance for this card in the sim")
        return d

    source = Path(game.source).read_text(encoding="utf-8", errors="replace")
    multi_hit = bool(MULTI_HIT_RE.search(source))
    n_damage_vars = len(DAMAGE_VAR_RE.findall(source))
    n_block_vars = len(BLOCK_VAR_RE.findall(source))

    # A redesigned card is not a numbers problem, and quietly correcting its
    # numbers would hide that its behaviour is now wrong.
    if game.card_type and sim.card_type and game.card_type != str(sim.card_type):
        d.refused.append(f"card type changed ({sim.card_type} -> {game.card_type}); "
                         "behaviour needs rewriting, not a value swap")
        return d

    for gfield, kw_name in NUMERIC_FIELDS.items():
        gval = getattr(game, gfield)
        if gval is None or kw_name not in site.keywords:
            continue
        simval = getattr(sim, gfield if gfield == "cost" else gfield)
        if isinstance(simval, (int, float)) and float(simval) == float(gval):
            continue

        if gfield == "damage":
            if multi_hit:
                d.refused.append("damage: card is multi-hit (WithHitCount/RepeatVar); "
                                 "the sim collapses the hits, so the numbers are not "
                                 "comparable")
                continue
            if n_damage_vars != 1:
                d.refused.append(f"damage: {n_damage_vars} DamageVars, cannot tell which is base")
                continue
        if gfield == "block" and n_block_vars != 1:
            d.refused.append(f"block: {n_block_vars} BlockVars, cannot tell which is base")
            continue

        upgrade = None
        if gfield == "damage":
            upgrade = game.upgrade_damage
        elif gfield == "block":
            upgrade = game.upgrade_block

        edits, reason = _numeric_edits(site.keywords[kw_name], gval, upgrade,
                                       starts, site.path)
        if reason:
            d.refused.append(f"{gfield}: {reason}")
        else:
            d.edits.extend(edits)
            shown = f"{gval}" + (f" (+{upgrade} upgraded)" if upgrade else "")
            d.applied.append(f"{kw_name} = {shown}  (was {simval})")

    # Rarity is a plain enum swap with no arithmetic, so it is safe whenever both
    # sides name a member.
    if game.rarity and sim.rarity and game.rarity != str(sim.rarity):
        node = site.keywords.get("rarity")
        if known_rarities and game.rarity not in known_rarities:
            d.refused.append(f"rarity: the game says {game.rarity}, which CardRarity "
                             "does not define -- add the enum member first")
        elif isinstance(node, ast.Attribute):
            s, e = _span(node, starts)
            d.edits.append(Edit(site.path, s, e, f"CardRarity.{game.rarity}"))
            d.applied.append(f"rarity = {game.rarity}  (was {sim.rarity})")
        else:
            d.refused.append("rarity: sim's value is not a plain enum member")

    return d


def apply_edits(edits: list[Edit]) -> set[Path]:
    """Rewrite in place. Edits are applied back-to-front so that earlier offsets
    stay valid as later ones shift."""
    by_file: dict[Path, list[Edit]] = {}
    for e in edits:
        by_file.setdefault(e.path, []).append(e)

    for path, file_edits in by_file.items():
        data = path.read_bytes()
        for e in sorted(file_edits, key=lambda x: x.start, reverse=True):
            data = data[:e.start] + e.new_text.encode() + data[e.end:]
        path.write_bytes(data)
    return set(by_file)


def verify(cards_dir: Path, decompiled: Path, expected: list[str]) -> list[str]:
    """Re-read both sides from scratch and confirm every applied value landed."""
    problems = []
    for path in sorted(cards_dir.glob("*.py")):
        try:
            ast.parse(path.read_bytes())
        except SyntaxError as ex:
            problems.append(f"{path.name} no longer parses: {ex}")
    if problems:
        return problems

    game = ccp.parse_csharp(decompiled)
    sim = ccp.parse_python(cards_dir)
    for cid in expected:
        g, s = game.get(cid), sim.get(cid)
        if not g or not s:
            problems.append(f"{cid}: disappeared from one side after the edit")
            continue
        for gfield in ("cost", "damage", "block"):
            gv, sv = getattr(g, gfield), getattr(s, gfield)
            if gv is None or not isinstance(sv, (int, float)):
                continue
            if float(gv) != float(sv):
                problems.append(f"{cid}.{gfield} still {sv}, expected {gv}")
        # Rarity is written as an enum member, so a wrong one parses fine and only
        # fails at runtime. Check it here rather than trusting the write.
        if g.rarity and s.rarity and g.rarity != str(s.rarity):
            problems.append(f"{cid}.rarity still {s.rarity}, expected {g.rarity}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decompiled", required=True, type=Path)
    ap.add_argument("--sim-cards", type=Path,
                    default=Path(__file__).resolve().parent.parent / "sts2_env" / "cards")
    ap.add_argument("--write", action="store_true",
                    help="Actually edit the files. Without this, nothing is written.")
    ap.add_argument("--show-refusals", action="store_true", default=True)
    args = ap.parse_args()

    if not args.decompiled.is_dir():
        print(f"decompiled root not found: {args.decompiled}", file=sys.stderr)
        return 2

    game = ccp.parse_csharp(args.decompiled)
    sim = ccp.parse_python(args.sim_cards)
    sites = collect_sim_sites(args.sim_cards)

    known_rarities = sim_enum_members(
        args.sim_cards.parent / "core" / "enums.py", "CardRarity")

    starts_cache: dict[Path, list[int]] = {}
    decisions: list[Decision] = []
    for cid in sorted(game.keys() & sim.keys() & sites.keys()):
        site = sites[cid]
        if site.path not in starts_cache:
            starts_cache[site.path] = _line_starts(site.path.read_bytes())
        d = decide(cid, game[cid], site, sim[cid], starts_cache[site.path], known_rarities)
        if d.applied or d.refused:
            decisions.append(d)

    to_apply = [d for d in decisions if d.edits]
    refused = [d for d in decisions if d.refused]

    print(f"{'WOULD CHANGE' if not args.write else 'CHANGING'} "
          f"{len(to_apply)} cards ({sum(len(d.edits) for d in to_apply)} values):\n")
    for d in to_apply:
        print(f"  {d.card_id}")
        for line in d.applied:
            print(f"      {line}")

    if refused and args.show_refusals:
        print(f"\nREFUSED -- decide these yourself ({len(refused)} cards):\n")
        for d in refused:
            print(f"  {d.card_id}")
            for line in d.refused:
                print(f"      {line}")

    if not args.write:
        print("\nDry run. Nothing written. Re-run with --write, then read `git diff`.")
        return 0

    all_edits = [e for d in to_apply for e in d.edits]
    if not all_edits:
        print("\nNothing to write.")
        return 0

    touched = apply_edits(all_edits)
    print(f"\nWrote {len(touched)} file(s). Verifying...")

    problems = verify(args.sim_cards, args.decompiled, [d.card_id for d in to_apply])
    if problems:
        print("\nVERIFICATION FAILED -- `git checkout` the card files and read this:",
              file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 2

    print("Verified: every applied value now matches the game.")
    try:
        diff = subprocess.run(["git", "diff", "--stat", "--"] + [str(p) for p in touched],
                              capture_output=True, text=True, cwd=args.sim_cards)
        if diff.stdout:
            print("\n" + diff.stdout.rstrip())
    except OSError:
        pass
    print("\nReview with: git diff -- sts2_env/cards/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
