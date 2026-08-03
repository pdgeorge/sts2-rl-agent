"""What each column of the observation is, and a version that says so out loud.

THE FAILURE THIS EXISTS TO STOP

The observation is a flat vector. Nothing in it is labelled at runtime. If a block
moves, changes size, or is reordered, every downstream consumer keeps reading the
same offsets and quietly gets different quantities -- the policy reads gold where
it expects HP and plays worse, and nothing raises. This project has already been
bitten by the same shape of bug more than once:

* card identity keyed on enum *position*, so inserting one card shifted every
  card after it and scrambled what the model had learned
* ``intent_damage`` carrying base damage in training and modifier-adjusted damage
  live, same column, two meanings
* ``docs/CARDS_REFERENCE.md`` drifting from the decompile while the tests read it
  as an oracle and stayed green

Every one was silent. A size check does not catch them, because the dangerous
edits keep the size the same.

WHAT THIS MODULE GIVES

1. ``RUN_OBS_LAYOUT`` / ``COMBAT_OBS_LAYOUT`` -- the blocks, derived from the size
   constants themselves so they cannot fall out of step with the encoders.
2. ``layout_fingerprint()`` -- a hash over the *names, offsets and sizes*, not just
   the total. Reordering two equal-sized blocks changes it; growing the vector
   changes it; renaming a block changes it.
3. ``stamp_checkpoint`` / ``verify_checkpoint`` -- a sidecar written next to a
   saved model and checked before it is trusted, so a mismatched checkpoint
   refuses to load rather than misreading the game for eleven hours.

``tests/test_observation_layout.py`` pins every offset as a literal. That test is
meant to fail on any layout edit: read the diff, confirm it was intended, bump
``OBS_LAYOUT_VERSION``, update the literals. It is a speed bump by design.

BUMPING THE VERSION

Bump when the meaning of any column changes -- moved, resized, reordered, or the
same slot now carries a different quantity. Existing checkpoints stop loading,
which is the point: they were trained against a layout that no longer exists.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from sts2_env.gym_env.choice_encoding import CHOICE_OBS_SIZE
from sts2_env.gym_env.deck_features import DECK_FEATURE_SIZE
from sts2_env.gym_env.entity_encoding import ENTITY_OBS_SIZE
from sts2_env.gym_env.observation import OBS_SIZE as COMBAT_OBS_SIZE
from sts2_env.gym_env.relic_potion_encoding import RELIC_POTION_OBS_SIZE
from sts2_env.gym_env.run_level_encoding import RUN_LEVEL_SIZE

OBS_LAYOUT_VERSION = 2
"""Bump whenever any column's meaning changes. See the module docstring.

v2: card identity moved from feature hashing to frozen text embeddings. The hand
    block became per-slot (10 x 65) instead of a 256-bucket bag, and the deck
    block became a pooled 65 instead of 256 buckets. Every v1 checkpoint is
    invalid -- the columns mean different things and the vector is longer.
"""

SIDECAR_SUFFIX = ".layout.json"


class ObservationLayoutMismatch(RuntimeError):
    """A checkpoint was trained against a different observation layout."""


@dataclass(frozen=True)
class Block:
    """One named span of the observation vector."""

    name: str
    start: int
    size: int

    @property
    def stop(self) -> int:
        return self.start + self.size


def _lay_out(pairs: tuple[tuple[str, int], ...]) -> tuple[Block, ...]:
    """Turn ``(name, size)`` pairs into blocks with running offsets."""
    blocks: list[Block] = []
    offset = 0
    for name, size in pairs:
        blocks.append(Block(name, offset, size))
        offset += size
    return tuple(blocks)


COMBAT_OBS_LAYOUT: tuple[Block, ...] = _lay_out((
    ("player_state", 4),
    ("player_powers", 6),
    ("hand_cards", 50),
    ("pile_summaries", 6),
    ("enemies", 65),
))

RUN_OBS_LAYOUT: tuple[Block, ...] = _lay_out((
    ("combat", COMBAT_OBS_SIZE),
    ("entity_identity", ENTITY_OBS_SIZE),
    ("deck_features", DECK_FEATURE_SIZE),
    ("run_level", RUN_LEVEL_SIZE),
    ("choices", CHOICE_OBS_SIZE),
    ("relics_potions", RELIC_POTION_OBS_SIZE),
))


def layout_size(layout: tuple[Block, ...]) -> int:
    return layout[-1].stop if layout else 0


def describe_layout(layout: tuple[Block, ...] = RUN_OBS_LAYOUT) -> str:
    """Human-readable table. Printed by the tests when they fail, so the diff
    says which block moved rather than only that a number changed."""
    lines = [f"{'block':<20} {'start':>7} {'size':>7} {'stop':>7}"]
    for block in layout:
        lines.append(f"{block.name:<20} {block.start:>7} {block.size:>7} {block.stop:>7}")
    lines.append(f"{'TOTAL':<20} {'':>7} {layout_size(layout):>7}")
    return "\n".join(lines)


def layout_fingerprint(layout: tuple[Block, ...] = RUN_OBS_LAYOUT) -> str:
    """Stable hash over names, offsets and sizes -- not just the total.

    Two blocks of equal size swapping places keeps the total identical and is
    exactly the edit that silently breaks a checkpoint, so the total alone is not
    a safe fingerprint.
    """
    payload = json.dumps(
        {
            "version": OBS_LAYOUT_VERSION,
            "blocks": [[b.name, b.start, b.size] for b in layout],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def layout_manifest(layout: tuple[Block, ...] = RUN_OBS_LAYOUT) -> dict:
    return {
        "obs_layout_version": OBS_LAYOUT_VERSION,
        "fingerprint": layout_fingerprint(layout),
        "size": layout_size(layout),
        "blocks": [{"name": b.name, "start": b.start, "size": b.size} for b in layout],
    }


def _sidecar_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    return path.with_suffix(path.suffix + SIDECAR_SUFFIX)


def stamp_checkpoint(
    model_path: str | Path, layout: tuple[Block, ...] = RUN_OBS_LAYOUT
) -> Path:
    """Record the layout a checkpoint was trained against, beside the checkpoint.

    A sidecar rather than something inside the archive, so it works for any saver
    and an existing checkpoint can be stamped after the fact.
    """
    sidecar = _sidecar_path(model_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(layout_manifest(layout), indent=2) + "\n")
    return sidecar


def verify_checkpoint(
    model_path: str | Path,
    layout: tuple[Block, ...] = RUN_OBS_LAYOUT,
    *,
    allow_unstamped: bool = True,
) -> None:
    """Raise if a checkpoint was trained against a different layout.

    ``allow_unstamped`` defaults to True so checkpoints predating this module
    still load. They are the ones most likely to be stale, so the caller is
    warned rather than stopped -- turn it off once everything in ``output/`` has
    been stamped.
    """
    sidecar = _sidecar_path(model_path)
    if not sidecar.is_file():
        if allow_unstamped:
            return
        raise ObservationLayoutMismatch(
            f"{model_path} has no {SIDECAR_SUFFIX} sidecar, so the layout it was "
            f"trained against is unknown. Current layout is "
            f"v{OBS_LAYOUT_VERSION}/{layout_fingerprint(layout)}."
        )

    recorded = json.loads(sidecar.read_text())
    current = layout_manifest(layout)
    if recorded.get("fingerprint") == current["fingerprint"]:
        return

    raise ObservationLayoutMismatch(
        f"{model_path} was trained against observation layout "
        f"v{recorded.get('obs_layout_version')}/{recorded.get('fingerprint')} "
        f"(size {recorded.get('size')}), but this build is "
        f"v{current['obs_layout_version']}/{current['fingerprint']} "
        f"(size {current['size']}).\n\n"
        f"Loading it would read every column as something it is not. Retrain, or "
        f"check out the revision that produced it.\n\n"
        f"Current layout:\n{describe_layout(layout)}"
    )
