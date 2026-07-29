"""Which build of the game the simulator is currently describing.

A policy learns the numbers it was trained on. "Cinder deals 18" ends up as a
weight in a matrix, and no amount of deriving values at construction time can
reach back into a trained model and update it. So a model is only valid for the
build it saw, and the useful question is not "which model matches this game" --
you will always be moving forward, never running an old model against an old
patch -- but "is the thing I am about to do reading the game I am playing?"

Two places that matters:

  - Training. Reading a stale decompile produces a run that looks completely
    normal and encodes the previous patch. Nothing about the output says so.
  - Streaming. A model trained three patches ago will play confidently and
    wrongly, and the failure looks like bad play rather than a stale artifact.

Both are silent, which is the only reason this module exists.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_STEAM_DIR = Path.home() / ".local/share/Steam/steamapps/common/Slay the Spire 2"
STS2_DIR_ENV = "STS2_GAME_DIR"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class BuildFingerprint:
    """Enough to say later what a model was trained against."""

    dll_sha256: str
    informational_version: str
    decompile_root: str
    decompile_matches_installed: bool


def installed_dll() -> Path | None:
    """The sts2.dll of the installed game, or None if it cannot be found.

    The data directory is named for the platform it shipped for and there is
    exactly one, so a glob beats a per-platform lookup table.
    """
    root = Path(os.environ.get(STS2_DIR_ENV, DEFAULT_STEAM_DIR))
    if not root.is_dir():
        return None
    for data_dir in sorted(root.glob("data_sts2_*")):
        dll = data_dir / "sts2.dll"
        if dll.is_file():
            return dll
    return None


@lru_cache(maxsize=4)
def _sha256(path: str, mtime: float) -> str:
    """Hash the file. mtime is part of the cache key so a reinstall is noticed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_build_hash() -> str | None:
    dll = installed_dll()
    if dll is None:
        return None
    return _sha256(str(dll), dll.stat().st_mtime)


def active_decompile_root() -> Path:
    """Where card values are actually being read from right now."""
    from sts2_env.cards.reference_static_metadata import reference_card_dir

    return reference_card_dir().parent


def decompile_source_info(root: Path | None = None) -> dict:
    """The .source.json scripts/on_update.sh leaves beside a decompile.

    The committed decompiled/ tree has none, which is not a missing file so much
    as a missing claim: nothing records which build it came from, so it cannot be
    checked against anything.
    """
    root = root or active_decompile_root()
    stamp = root / ".source.json"
    if not stamp.is_file():
        return {}
    try:
        return json.loads(stamp.read_text())
    except (OSError, ValueError):
        return {}


def check_decompile_matches_installed() -> tuple[bool, str]:
    """(ok, human-readable reason). Never raises -- callers decide how to react."""
    root = active_decompile_root()
    installed = installed_build_hash()
    info = decompile_source_info(root)
    recorded = info.get("dll_sha256")

    if installed is None:
        return False, (f"Slay the Spire 2 not found (set {STS2_DIR_ENV}), so the "
                       f"decompile at {root} cannot be checked against anything.")
    if not recorded:
        return False, (
            f"The decompile in use ({root}) has no .source.json, so which build it "
            f"came from is unrecorded. This is normally the committed decompiled/ "
            f"tree, which is a snapshot of an older patch.\n"
            f"  Fix: STS2_DECOMPILED_ROOT=/path/to/fresh/decompile, produced by "
            f"scripts/on_update.sh.")
    if recorded != installed:
        return False, (
            f"The decompile in use is not the installed build.\n"
            f"  installed:  {installed[:16]}\n"
            f"  decompile:  {recorded[:16]}  ({root})\n"
            f"  Fix: run scripts/on_update.sh to refresh it.")
    return True, f"decompile matches installed build {installed[:16]}"


def build_fingerprint() -> BuildFingerprint:
    ok, _ = check_decompile_matches_installed()
    info = decompile_source_info()
    return BuildFingerprint(
        dll_sha256=info.get("dll_sha256") or installed_build_hash() or UNKNOWN,
        informational_version=info.get("informational_version", UNKNOWN),
        decompile_root=str(active_decompile_root()),
        decompile_matches_installed=ok,
    )


def write_fingerprint(output_dir: Path) -> Path:
    """Record the build alongside a trained model.

    Written as a sidecar rather than into the checkpoint: SB3 owns that file
    format, and a stamp that survives `model.save()` changing shape is worth more
    than one that reads more neatly.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "game_build.json"
    path.write_text(json.dumps(asdict(build_fingerprint()), indent=2, sort_keys=True))
    return path


def read_fingerprint(output_dir: Path) -> BuildFingerprint | None:
    path = Path(output_dir) / "game_build.json"
    if not path.is_file():
        return None
    try:
        return BuildFingerprint(**json.loads(path.read_text()))
    except (OSError, ValueError, TypeError):
        return None


def describe_model_staleness(output_dir: Path) -> tuple[bool, str]:
    """(ok, message) for a trained model about to be used against the live game."""
    stamp = read_fingerprint(output_dir)
    if stamp is None:
        return False, (f"{output_dir} has no game_build.json, so which build this "
                       f"model trained against is unknown. Models trained before "
                       f"this check existed will all look like this.")
    installed = installed_build_hash()
    if installed is None:
        return False, f"cannot find the installed game to compare against {output_dir}"
    if stamp.dll_sha256 != installed:
        return False, (
            f"This model trained against a different build of the game.\n"
            f"  model:      {stamp.dll_sha256[:16]}  ({stamp.informational_version})\n"
            f"  installed:  {installed[:16]}\n"
            f"  It will play confidently and wrongly. Retrain.")
    return True, f"model trained against the installed build {installed[:16]}"
