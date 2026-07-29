#!/usr/bin/env python3
"""Report decompiled gameplay classes that lack obvious Python or test references."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable


CAMEL_WORD_BOUNDARY_RE = re.compile(r"(.)([A-Z][a-z]+)")
LOWER_TO_UPPER_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")
ALNUM_UNDERSCORE_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
PYTHON_FILE_PATTERN = "*.py"
CS_FILE_PATTERN = "*.cs"
DEPRECATED_NAME_MARKER = "Deprecated"
DEFAULT_TEST_PATHS = ("tests",)
SUMMARY_COLUMNS = (
    ("surface", "Surface"),
    ("total", "Total"),
    ("missing_implementation", "Missing implementation"),
    ("missing_tests", "Missing tests"),
)
MISSING_IMPLEMENTATION_LABEL = "Missing implementation mentions"
MISSING_TESTS_LABEL = "Missing test mentions"
NONE_LABEL = "none"


@dataclass(frozen=True)
class SurfaceConfig:
    reference_dir: str
    implementation_paths: tuple[str, ...]
    suffixes: tuple[str, ...] = ()
    explicit_aliases: dict[str, tuple[str, ...]] | None = None


@dataclass(frozen=True)
class ReferenceItem:
    name: str
    path: str
    implementation_hits: tuple[str, ...]
    test_hits: tuple[str, ...]

    @property
    def has_implementation_reference(self) -> bool:
        return bool(self.implementation_hits)

    @property
    def has_test_reference(self) -> bool:
        return bool(self.test_hits)


@dataclass(frozen=True)
class SurfaceReport:
    surface: str
    total: int
    missing_implementation: tuple[str, ...]
    missing_tests: tuple[str, ...]
    items: tuple[ReferenceItem, ...]


@dataclass(frozen=True)
class AliasIndex:
    identifier_tokens: frozenset[str]
    snake_segment_tokens: frozenset[str]


SURFACES: dict[str, SurfaceConfig] = {
    "cards": SurfaceConfig(
        "decompiled/MegaCrit.Sts2.Core.Models.Cards",
        ("sts2_env/cards",),
        suffixes=("Card",),
        explicit_aliases={
            "Sloth": ("SLOTH_STATUS", "sloth_status", "make_sloth_status"),
        },
    ),
    "relics": SurfaceConfig(
        "decompiled/MegaCrit.Sts2.Core.Models.Relics",
        ("sts2_env/relics",),
    ),
    "potions": SurfaceConfig(
        "decompiled/MegaCrit.Sts2.Core.Models.Potions",
        ("sts2_env/potions",),
        suffixes=("Potion",),
    ),
    "powers": SurfaceConfig(
        "decompiled/MegaCrit.Sts2.Core.Models.Powers",
        ("sts2_env/powers",),
        suffixes=("Power",),
    ),
    "monsters": SurfaceConfig(
        "decompiled/MegaCrit.Sts2.Core.Models.Monsters",
        ("sts2_env/monsters",),
    ),
    "events": SurfaceConfig(
        "decompiled/MegaCrit.Sts2.Core.Models.Events",
        ("sts2_env/events",),
    ),
    "encounters": SurfaceConfig(
        "decompiled/MegaCrit.Sts2.Core.Models.Encounters",
        ("sts2_env/encounters",),
        suffixes=("Encounter",),
    ),
    "modifiers": SurfaceConfig(
        "decompiled/MegaCrit.Sts2.Core.Models.Modifiers",
        ("sts2_env/run/modifiers.py",),
        suffixes=("Modifier",),
    ),
}


def snake_case(name: str) -> str:
    first = CAMEL_WORD_BOUNDARY_RE.sub(r"\1_\2", name)
    return LOWER_TO_UPPER_BOUNDARY_RE.sub(r"\1_\2", first).lower()


def aliases_for(
    name: str,
    suffixes: Iterable[str],
    explicit_aliases: dict[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    aliases = {name, snake_case(name), snake_case(name).upper()}
    aliases.update((explicit_aliases or {}).get(name, ()))
    for suffix in suffixes:
        suffix_snake = snake_case(suffix)
        aliases.update(
            {
                f"{name}{suffix}",
                f"{snake_case(name)}_{suffix_snake}",
                f"{snake_case(name).upper()}_{suffix_snake.upper()}",
            }
        )
        if not name.endswith(suffix):
            continue
        stripped = name[: -len(suffix)]
        if stripped:
            aliases.update({stripped, snake_case(stripped), snake_case(stripped).upper()})
    return tuple(sorted(aliases, key=lambda value: (value.lower(), value)))


def collect_text(root: Path, paths: Iterable[str]) -> str:
    chunks: list[str] = []
    for path_text in paths:
        path = root / path_text
        if path.is_file():
            chunks.append(path.read_text(errors="ignore"))
        elif path.is_dir():
            for file_path in sorted(path.rglob(PYTHON_FILE_PATTERN)):
                chunks.append(file_path.read_text(errors="ignore"))
    return "\n".join(chunks)


def collect_code_text(root: Path, paths: Iterable[str]) -> str:
    chunks: list[str] = []
    for path_text in paths:
        path = root / path_text
        if path.is_file() and path.suffix == ".py":
            chunks.append(path.read_text(errors="ignore"))
        elif path.is_dir():
            for file_path in sorted(path.rglob(PYTHON_FILE_PATTERN)):
                if "__pycache__" not in file_path.parts:
                    chunks.append(file_path.read_text(errors="ignore"))
    return "\n".join(chunks)


def collect_test_text(root: Path, *, direct_references_only: bool = False) -> str:
    if direct_references_only:
        return collect_direct_test_reference_text(root)
    return collect_text(root, DEFAULT_TEST_PATHS)


def collect_direct_test_reference_text(root: Path) -> str:
    chunks: list[str] = []
    for path_text in DEFAULT_TEST_PATHS:
        path = root / path_text
        files = [path] if path.is_file() else sorted(path.rglob(PYTHON_FILE_PATTERN))
        for file_path in files:
            chunks.append(direct_test_reference_text(file_path.read_text(errors="ignore")))
    return "\n".join(chunks)


def direct_test_reference_text(source: str) -> str:
    try:
        module = ast.parse(source)
    except SyntaxError:
        return ""
    module_assignments = _module_level_assignments(module)
    chunks: list[str] = []
    pending_names: set[str] = set()
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            chunks.extend(_node_reference_chunks(node))
            pending_names.update(_referenced_names(node))
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            chunks.extend(_node_reference_chunks(node))
            pending_names.update(_referenced_names(node))
    chunks.extend(_module_assignment_reference_chunks(module_assignments, pending_names))
    return "\n".join(chunks)


def _module_level_assignments(module: ast.Module) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for name in _assigned_names(target):
                    assignments[name] = node
        elif isinstance(node, ast.AnnAssign):
            for name in _assigned_names(node.target):
                assignments[name] = node
    return assignments


def _assigned_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in node.elts:
            names.update(_assigned_names(element))
        return names
    return set()


def _referenced_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _module_assignment_reference_chunks(
    assignments: dict[str, ast.AST],
    initial_names: set[str],
) -> list[str]:
    chunks: list[str] = []
    pending_names = set(initial_names)
    seen_names: set[str] = set()
    seen_nodes: set[int] = set()
    while pending_names:
        name = pending_names.pop()
        if name in seen_names:
            continue
        seen_names.add(name)
        node = assignments.get(name)
        if node is None or id(node) in seen_nodes:
            continue
        seen_nodes.add(id(node))
        chunks.extend(_node_reference_chunks(node))
        pending_names.update(_referenced_names(node))
    return chunks


def _node_reference_chunks(node: ast.AST) -> list[str]:
    chunks: list[str] = []
    name = getattr(node, "name", None)
    if isinstance(name, str):
        chunks.append(name)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
        docstring = ast.get_docstring(node)
        if docstring:
            chunks.append(docstring)
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            chunks.append(child.value)
        elif isinstance(child, ast.Attribute):
            chunks.append(child.attr)
        elif isinstance(child, ast.Name):
            chunks.append(child.id)
    return chunks


def alias_index(text: str) -> AliasIndex:
    identifier_tokens = frozenset(ALNUM_UNDERSCORE_TOKEN_RE.findall(text))
    snake_tokens: set[str] = set()
    for token in identifier_tokens:
        parts = [part for part in token.split("_") if part]
        for start in range(len(parts)):
            for end in range(start + 1, len(parts) + 1):
                snake_tokens.add("_".join(parts[start:end]))
    return AliasIndex(identifier_tokens, frozenset(snake_tokens))


def alias_hits(index: AliasIndex, aliases: Iterable[str]) -> tuple[str, ...]:
    hits: list[str] = []
    for alias in aliases:
        tokens = (
            index.snake_segment_tokens
            if alias == alias.lower() or alias == alias.upper()
            else index.identifier_tokens
        )
        if alias in tokens:
            hits.append(alias)
    return tuple(hits)


def display_path(path: Path, root: Path) -> str:
    """Repo-relative when possible, absolute otherwise.

    --decompiled-root can point outside the repository (the decompile is large and
    Mega Crit's copyright, so it does not have to live here), in which case
    relative_to would raise.
    """
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def surfaces_rooted_at(decompiled_root: Path) -> dict[str, SurfaceConfig]:
    """The same surfaces, but reading from a decompile elsewhere on disk.

    The committed decompiled/ tree is a snapshot of whatever build it was taken
    from, so auditing against it silently answers a question about the past.
    """
    return {
        name: replace(cfg, reference_dir=str(decompiled_root / Path(cfg.reference_dir).name))
        for name, cfg in SURFACES.items()
    }


def reference_files(root: Path, config: SurfaceConfig, include_deprecated: bool) -> list[Path]:
    reference_dir = root / config.reference_dir
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {reference_dir}")
    files = sorted(reference_dir.glob(CS_FILE_PATTERN))
    if include_deprecated:
        return files
    return [path for path in files if DEPRECATED_NAME_MARKER not in path.stem]


def build_report(
    root: Path,
    surface: str,
    include_deprecated: bool = False,
    *,
    direct_test_references: bool = False,
    code_implementation_references: bool = False,
    surfaces: dict[str, SurfaceConfig] | None = None,
) -> SurfaceReport:
    config = (surfaces or SURFACES)[surface]
    implementation_text = (
        collect_code_text(root, config.implementation_paths)
        if code_implementation_references
        else collect_text(root, config.implementation_paths)
    )
    test_text = collect_test_text(root, direct_references_only=direct_test_references)
    implementation_index = alias_index(implementation_text)
    test_index = alias_index(test_text)
    items: list[ReferenceItem] = []

    for path in reference_files(root, config, include_deprecated):
        aliases = aliases_for(path.stem, config.suffixes, config.explicit_aliases)
        items.append(
            ReferenceItem(
                name=path.stem,
                path=display_path(path, root),
                implementation_hits=alias_hits(implementation_index, aliases),
                test_hits=alias_hits(test_index, aliases),
            )
        )

    missing_implementation = tuple(
        item.name for item in items if not item.has_implementation_reference
    )
    missing_tests = tuple(item.name for item in items if not item.has_test_reference)
    return SurfaceReport(
        surface=surface,
        total=len(items),
        missing_implementation=missing_implementation,
        missing_tests=missing_tests,
        items=tuple(items),
    )


def build_reports(
    root: Path,
    surfaces: Iterable[str],
    include_deprecated: bool = False,
    *,
    direct_test_references: bool = False,
    code_implementation_references: bool = False,
    surface_configs: dict[str, SurfaceConfig] | None = None,
) -> tuple[SurfaceReport, ...]:
    return tuple(
        build_report(
            root,
            surface,
            include_deprecated=include_deprecated,
            direct_test_references=direct_test_references,
            code_implementation_references=code_implementation_references,
            surfaces=surface_configs,
        )
        for surface in surfaces
    )


def print_text_report(reports: Iterable[SurfaceReport], show_missing: bool) -> None:
    reports = tuple(reports)
    rows = [
        {
            "surface": report.surface,
            "total": str(report.total),
            "missing_implementation": str(len(report.missing_implementation)),
            "missing_tests": str(len(report.missing_tests)),
        }
        for report in reports
    ]
    widths = {
        key: max(len(header), *(len(row[key]) for row in rows))
        for key, header in SUMMARY_COLUMNS
    }
    header = "  ".join(header.ljust(widths[key]) for key, header in SUMMARY_COLUMNS)
    divider = "  ".join("-" * widths[key] for key, _header in SUMMARY_COLUMNS)
    print(header)
    print(divider)
    for report in reports:
        row = {
            "surface": report.surface,
            "total": str(report.total),
            "missing_implementation": str(len(report.missing_implementation)),
            "missing_tests": str(len(report.missing_tests)),
        }
        print(
            "  ".join(
                row[key].rjust(widths[key]) if row[key].isdigit() else row[key].ljust(widths[key])
                for key, _header in SUMMARY_COLUMNS
            )
        )

    if not show_missing:
        return

    for report in reports:
        print()
        print(f"{report.surface}:")
        if report.missing_implementation:
            print(
                f"  {MISSING_IMPLEMENTATION_LABEL}: "
                + ", ".join(report.missing_implementation)
            )
        else:
            print(f"  {MISSING_IMPLEMENTATION_LABEL}: {NONE_LABEL}")
        if report.missing_tests:
            print(f"  {MISSING_TESTS_LABEL}: " + ", ".join(report.missing_tests))
        else:
            print(f"  {MISSING_TESTS_LABEL}: {NONE_LABEL}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan decompiled gameplay model filenames against Python implementation "
            "and test references. Missing rows are leads for audit, not proof of bugs."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root. Defaults to this script's parent repository.",
    )
    parser.add_argument(
        "--decompiled-root",
        type=Path,
        help=(
            "Read decompiled classes from here instead of the committed decompiled/ "
            "tree. Point this at a fresh decompile of the installed build; the "
            "committed one is a snapshot of whatever version it was taken from."
        ),
    )
    parser.add_argument(
        "--surface",
        action="append",
        choices=sorted(SURFACES),
        help="Surface to scan. May be provided multiple times. Defaults to all surfaces.",
    )
    parser.add_argument(
        "--include-deprecated",
        action="store_true",
        help="Include decompiled classes with Deprecated in the filename.",
    )
    parser.add_argument(
        "--direct-test-references",
        action="store_true",
        help=(
            "Only count references inside test functions/classes, their docstrings, "
            "and their literals/identifiers. This avoids module import and registry "
            "pollution, but remains a lead-finding heuristic."
        ),
    )
    parser.add_argument(
        "--code-implementation-references",
        action="store_true",
        help="Only count Python source files for implementation mentions, excluding docs and bytecode.",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Print the missing implementation and test mention names.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the text table.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    surfaces = tuple(args.surface or sorted(SURFACES))
    reports = build_reports(
        args.root,
        surfaces,
        include_deprecated=args.include_deprecated,
        direct_test_references=args.direct_test_references,
        code_implementation_references=args.code_implementation_references,
        surface_configs=(
            surfaces_rooted_at(args.decompiled_root.resolve())
            if args.decompiled_root
            else None
        ),
    )
    if args.json:
        print(json.dumps([asdict(report) for report in reports], indent=2, sort_keys=True))
    else:
        print_text_report(reports, show_missing=args.show_missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
