"""Read side of the frozen card embedding table.

The build script owns writing; this owns reading, and it is what the observation
encoder will call. Loading is cached, so the cost is paid once per process rather
than per observation -- encoding already costs ~5.6x a policy forward pass, and
the point of this change is partly to make that cheaper.

OUT OF VOCABULARY

A card with no row gets a zero vector **and** ``is_known = 0.0``. The flag is not
decoration. Without it the network cannot distinguish "I have never seen this
card" from "this card's vector legitimately sits near the origin", and those call
for opposite behaviour: the first is a reason to fall back on the scalar features
beside it, the second is real information.

This happens when the game ships content before ``sync_content`` has run. The
alternative -- refusing to start -- was considered and rejected: a single unknown
card would block a run, and the person who forgot to sync is most likely to find
out about it five minutes before going live.

WHAT THIS DOES NOT DO

No name aliasing. The table is generated from ``CardId`` members by this repo, so
keys match exactly by construction. The suffix-alias dance in ``trace_delta`` and
``sync_content`` exists for reconciling *external* naming, which is precisely the
dependency this design avoids.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_TABLE_PATH = Path("data/card_embeddings/v1")

IS_KNOWN_DIMS = 1
"""The flag rides alongside the vector, so a card contributes dims + 1 columns."""


class CardEmbeddingTableMissing(RuntimeError):
    """The table has not been built. See scripts/build_card_embeddings.py."""


@dataclass(frozen=True)
class CardEmbeddingTable:
    dims: int
    vectors: np.ndarray          # (n_cards, dims), row order matches card_ids
    index: dict[str, int]        # CardId.name -> row
    manifest: dict

    def vector(self, card_id) -> tuple[np.ndarray, float]:
        """``(vector, is_known)`` for a card. Never raises on an unknown card."""
        name = getattr(card_id, "name", str(card_id))
        row = self.index.get(name)
        if row is None:
            return np.zeros(self.dims, dtype=np.float32), 0.0
        return self.vectors[row], 1.0

    def encode(self, card_id) -> np.ndarray:
        """``dims + 1`` values: the vector with its ``is_known`` flag appended."""
        vector, known = self.vector(card_id)
        return np.concatenate([vector, np.float32([known])])

    def pooled(self, card_ids) -> np.ndarray:
        """Mean of the known vectors in a pile, plus the known fraction.

        Mean rather than sum so a 40-card deck and a 10-card deck stay on the
        same scale -- the policy should read "what kind of deck is this", with
        deck size already carried explicitly elsewhere in the observation.

        Unknown cards are excluded from the mean rather than pulled toward the
        origin, and the fraction that were known is reported so the network can
        tell a confidently-empty deck from an unrecognised one.
        """
        rows = [self.index[n] for n in
                (getattr(c, "name", str(c)) for c in card_ids) if n in self.index]
        total = sum(1 for _ in card_ids)
        if not rows:
            return np.zeros(self.dims + IS_KNOWN_DIMS, dtype=np.float32)
        mean = self.vectors[rows].mean(axis=0)
        known_fraction = len(rows) / total if total else 0.0
        return np.concatenate([mean, np.float32([known_fraction])])

    @property
    def width(self) -> int:
        return self.dims + IS_KNOWN_DIMS


@lru_cache(maxsize=4)
def load_table(path: str | Path = DEFAULT_TABLE_PATH) -> CardEmbeddingTable:
    """Load and cache a built table."""
    path = Path(path)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise CardEmbeddingTableMissing(
            f"No card embedding table at {path}. Build it with:\n"
            f"    python scripts/build_card_embeddings.py"
        )

    manifest = json.loads(manifest_path.read_text())
    vectors = np.load(path / "vectors.npy").astype(np.float32)
    names = (path / "card_ids.txt").read_text().split()

    if len(names) != vectors.shape[0]:
        raise CardEmbeddingTableMissing(
            f"{path} is inconsistent: {len(names)} ids against "
            f"{vectors.shape[0]} vectors. Rebuild it."
        )
    if vectors.shape[1] != manifest["dims"]:
        raise CardEmbeddingTableMissing(
            f"{path} manifest says {manifest['dims']} dims but vectors are "
            f"{vectors.shape[1]}. Rebuild it."
        )

    return CardEmbeddingTable(
        dims=int(manifest["dims"]),
        vectors=vectors,
        index={name: row for row, name in enumerate(names)},
        manifest=manifest,
    )
