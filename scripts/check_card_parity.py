"""Diff the Python simulator's card values against decompiled game source.

The simulator is a hand-written reimplementation, so it goes stale silently every
time the game patches -- no compile error, just subtly wrong numbers and therefore
wrong value estimates from anything trained on it. That is strictly worse than the
mod's failure mode, which at least breaks loudly.

This compares scalar card values only: cost, base damage, base block, and upgrade
deltas. It deliberately does NOT look at OnPlay behaviour -- effect logic is not
mechanically comparable, and pretending otherwise would produce confident nonsense.
Behavioural drift is what the bridge replay harness is for.

Ground truth is the decompiled game assembly, not a wiki. Wikis lag patches and
introduce transcription errors; the assembly is what actually shipped.

Refresh the decompile after each game patch:

    ilspycmd -p -o <outdir> "<steam>/Slay the Spire 2/data_sts2_linuxbsd_x86_64/sts2.dll"

Then:

    python scripts/check_card_parity.py --decompiled <outdir>
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# C# side
# ---------------------------------------------------------------------------

CLASS_RE = re.compile(r"public\s+(?:sealed\s+)?class\s+(\w+)\s*:\s*CardModel\b")
CTOR_RE = re.compile(r":\s*base\(\s*(\d+)\s*,\s*CardType\.(\w+)\s*,\s*CardRarity\.(\w+)\s*,\s*TargetType\.(\w+)")
DAMAGE_RE = re.compile(r"new\s+DamageVar\(\s*([\d.]+)m")
BLOCK_RE = re.compile(r"new\s+BlockVar\(\s*([\d.]+)m")
UPGRADE_DMG_RE = re.compile(r"DynamicVars\.Damage\.UpgradeValueBy\(\s*([\d.]+)m")
UPGRADE_BLK_RE = re.compile(r"DynamicVars\.Block\.UpgradeValueBy\(\s*([\d.]+)m")


def pascal_to_upper_snake(name: str) -> str:
    """Bash -> BASH, DefendIronclad -> DEFEND_IRONCLAD."""
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.upper()


def _num(text: str) -> float | int:
    value = float(text)
    return int(value) if value.is_integer() else value


@dataclass
class GameCard:
    name: str
    card_id: str
    cost: int | None = None
    card_type: str | None = None
    rarity: str | None = None
    target: str | None = None
    damage: float | None = None
    block: float | None = None
    upgrade_damage: float | None = None
    upgrade_block: float | None = None
    source: str = ""
    ambiguous: list[str] = field(default_factory=list)


def parse_csharp(root: Path) -> dict[str, GameCard]:
    cards: dict[str, GameCard] = {}
    card_dirs = [p for p in root.iterdir()
                 if p.is_dir() and p.name.endswith("Core.Models.Cards")]
    if not card_dirs:
        card_dirs = [p for p in root.rglob("*") if p.is_dir() and p.name.endswith("Models.Cards")]

    for d in card_dirs:
        for path in sorted(d.glob("*.cs")):
            text = path.read_text(encoding="utf-8", errors="replace")
            m = CLASS_RE.search(text)
            if not m:
                continue
            name = m.group(1)
            card = GameCard(name=name, card_id=pascal_to_upper_snake(name), source=str(path))

            ctor = CTOR_RE.search(text)
            if ctor:
                card.cost = int(ctor.group(1))
                card.card_type = ctor.group(2).upper()
                card.rarity = ctor.group(3).upper()
                card.target = ctor.group(4)

            for attr, regex in (("damage", DAMAGE_RE), ("block", BLOCK_RE),
                                ("upgrade_damage", UPGRADE_DMG_RE),
                                ("upgrade_block", UPGRADE_BLK_RE)):
                hits = regex.findall(text)
                if len(hits) == 1:
                    setattr(card, attr, _num(hits[0]))
                elif len(hits) > 1:
                    # More than one is real (conditional cards). Take the first but
                    # say so, rather than silently reporting a mismatch later.
                    setattr(card, attr, _num(hits[0]))
                    card.ambiguous.append(f"{attr} x{len(hits)}")

            cards[card.card_id] = card
    return cards


# ---------------------------------------------------------------------------
# Python side
# ---------------------------------------------------------------------------

@dataclass
class SimCard:
    card_id: str
    cost: int | None = None
    card_type: str | None = None
    target: str | None = None
    rarity: str | None = None
    damage: float | None = None
    block: float | None = None
    source: str = ""


def _module_constants(tree: ast.Module) -> dict[str, float | int]:
    """Module-level numeric constants.

    Cards routinely say `base_damage=BASH_DAMAGE` or
    `base_damage=X_UPGRADED if upgraded else X_DAMAGE`, so without resolving these
    the extractor sees nothing and reports every such card as missing a value.
    """
    consts: dict[str, float | int] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
            inner = value.operand
            if isinstance(inner, ast.Constant) and isinstance(inner.value, (int, float)):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        consts[t.id] = -inner.value
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    consts[t.id] = value.value
    return consts


def _const(node: ast.AST | None, consts: dict[str, float | int] | None = None):
    """Resolve a literal-ish expression. Returns None when it cannot be resolved,
    which is treated as 'unknown', never as a mismatch."""
    consts = consts or {}
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const(node.operand, consts)
        return -inner if isinstance(inner, (int, float)) else None
    if isinstance(node, ast.IfExp):
        # `X_UPGRADED if upgraded else X_BASE` -- we compare base values, so take
        # the else branch.
        return _const(node.orelse, consts)
    if isinstance(node, ast.Attribute):
        # Enum member (CardType.ATTACK) -> its name. Never a number, so callers
        # comparing numerics will ignore it.
        return node.attr
    return None


def parse_python(cards_dir: Path) -> dict[str, SimCard]:
    """Walks the AST rather than regexing -- the constructor calls are keyword-only
    and spread over several lines, which regex handles badly."""
    found: dict[str, SimCard] = {}
    for path in sorted(cards_dir.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        consts = _module_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if fname != "CardInstance":
                continue

            kw = {k.arg: k.value for k in node.keywords if k.arg}
            cid = _const(kw.get("card_id"), consts)
            if not isinstance(cid, str):
                continue

            card = SimCard(card_id=cid, source=str(path))
            card.cost = _const(kw.get("cost"), consts)
            card.card_type = _const(kw.get("card_type"), consts)
            card.target = _const(kw.get("target_type"), consts)
            card.rarity = _const(kw.get("rarity"), consts)
            card.damage = _const(kw.get("base_damage"), consts)
            card.block = _const(kw.get("base_block"), consts)
            # First definition wins; duplicates are upgrade variants.
            found.setdefault(cid, card)
    return found


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def _constructed_values() -> dict:
    """What each card actually IS once the factory has built it.

    The AST says what the source declares; this says what the game gets. They
    differ for every card whose value `apply_derived_values` rewrites from the
    decompile, which is most of them. Never raises -- a card that will not build
    simply is not cross-checked, and the literal comparison stands.
    """
    out: dict[str, dict] = {}
    try:
        import sts2_env.cards  # noqa: F401
        from sts2_env.cards.factory import create_card
        from sts2_env.core.enums import CardId
    except Exception:  # noqa: BLE001
        return out
    for member in CardId:
        try:
            card = create_card(member)
        except Exception:  # noqa: BLE001
            continue
        out[member.name] = {"cost": card.cost, "damage": card.base_damage,
                            "block": card.base_block}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decompiled", required=True, type=Path,
                    help="Root of the ilspycmd -p output for sts2.dll")
    ap.add_argument("--sim-cards", type=Path,
                    default=Path(__file__).resolve().parent.parent / "sts2_env" / "cards",
                    help="Directory of the simulator's card definitions")
    ap.add_argument("--show-missing", action="store_true",
                    help="Also list cards present in only one side")
    args = ap.parse_args()

    if not args.decompiled.is_dir():
        print(f"decompiled root not found: {args.decompiled}", file=sys.stderr)
        return 2

    game = parse_csharp(args.decompiled)
    sim = parse_python(args.sim_cards)

    print(f"game cards parsed: {len(game)}")
    print(f"sim  cards parsed: {len(sim)}\n")

    if not game:
        print("No card classes found. Is --decompiled pointing at the ilspycmd -p output?",
              file=sys.stderr)
        return 2

    built = _constructed_values()

    mismatches = 0
    unresolved = 0
    suppressed = 0
    for cid in sorted(game.keys() & sim.keys()):
        g, s = game[cid], sim[cid]
        problems = []
        unknown = []
        for label, gv, sv in (("cost", g.cost, s.cost),
                              ("damage", g.damage, s.damage),
                              ("block", g.block, s.block)):
            if gv is None:
                continue
            if not isinstance(sv, (int, float)):
                # The sim computes this rather than declaring a literal. That is a
                # limit of this extractor, NOT evidence of drift -- reporting it as a
                # mismatch would bury the real ones in noise.
                unknown.append(label)
                continue
            if float(gv) != float(sv):
                problems.append(f"{label}: game={gv} sim={sv}")

        # THE LITERAL IS NOT WHAT SHIPS. `apply_derived_values` rewrites a
        # card's cost, damage and block from the decompile when the factory
        # builds it, so a stale literal in the source is inert -- and this
        # extractor reads the AST, which is the literal.
        #
        # Measured 2026-08-19: all 20 mismatches this reported were stale
        # literals. MANGLE's source said 15 and it ships 20; PACTS_END said 17
        # and ships 18; SUNDER, HYPERBEAM, SPITE, every one of them already
        # correct at runtime. Twenty false positives is not a noisy signal, it
        # is an alarm nobody can act on, and it is how a real one gets ignored.
        #
        # So a problem is only reported when the CONSTRUCTED card still
        # disagrees. A source literal that is merely stale is reported
        # separately, as tidying rather than as drift.
        if problems and cid in built:
            live = built[cid]
            still_wrong = []
            for problem in problems:
                field = problem.split(":", 1)[0]
                value = live.get(field)
                game_value = {"cost": g.cost, "damage": g.damage, "block": g.block}[field]
                if value is None or game_value is None:
                    continue
                if float(value) != float(game_value):
                    still_wrong.append(f"{field}: game={game_value} built={value}")
            if not still_wrong:
                suppressed += 1
                if args.show_missing:
                    print(f"stale literal only (correct at runtime): {cid}  "
                          f"-- {'; '.join(problems)}")
                continue
            problems = still_wrong

        if problems:
            mismatches += 1
            note = f"  [{', '.join(g.ambiguous)}]" if g.ambiguous else ""
            print(f"MISMATCH {cid}{note}")
            for p in problems:
                print(f"    {p}")
            print(f"    game: {g.source}")
            print(f"    sim:  {s.source}")
        elif unknown:
            unresolved += 1

    only_game = sorted(game.keys() - sim.keys())
    only_sim = sorted(sim.keys() - game.keys())

    print(f"\n--- summary ---")
    print(f"  compared:            {len(game.keys() & sim.keys())}")
    print(f"  value mismatches:    {mismatches}   <- real drift, fix these")
    print(f"  stale literals only: {suppressed}   <- source says one thing, "
          f"apply_derived_values ships another; tidy, not drift (--show-missing to list)")
    print(f"  not statically known:{unresolved}   <- sim computes it; this tool cannot check")
    print(f"  in game, not in sim: {len(only_game)}   <- structural drift, needs implementing")
    print(f"  in sim, not in game: {len(only_sim)}   <- removed/renamed upstream")

    if args.show_missing:
        if only_game:
            print("\nin game but not in sim:")
            for c in only_game:
                print(f"  {c}")
        if only_sim:
            print("\nin sim but not in game:")
            for c in only_sim:
                print(f"  {c}")

    return 1 if (mismatches or only_game) else 0


if __name__ == "__main__":
    raise SystemExit(main())
