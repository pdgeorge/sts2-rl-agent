"""Feature hashing for patch-stable observation encoding.

Replaces variable-size multi-hot vectors with fixed-size hashed vectors so that
adding or removing game content (cards, powers, relics, monsters) never changes
the observation space shape. A model trained before a patch can load after it
without retraining from scratch.

The hashing trick:
  - Each item (e.g. "STRIKE_IRONCLAD") is mapped to a small fixed number of
    buckets (e.g. 4) in a fixed-size array (e.g. 256).
  - The mapping is deterministic: same name → same buckets, forever.
  - Buckets receive a signed value (+1 or -1) so that accidental collisions
    partially cancel rather than systematically inflate.

This is the standard "hashing trick" from NLP / ad-tech: a vocabulary of 50K
words is hashed into 4K dimensions with 4 hashes per word, and the model learns
just fine.

Usage:
    hasher = FeatureHasher(n_buckets=256, n_hashes=4)
    vec = hasher.encode_scalar("POISON", amount=12.0)
    vec = hasher.encode_binary("STRIKE_IRONCLAD")
    vec = hasher.encode_set(["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"])
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _hash_bucket(key: str, i: int, seed: int, n_buckets: int) -> int:
    """Deterministic hash of (key, hash_index, seed) into [0, n_buckets)."""
    digest = hashlib.md5(f"{key}:{i}:{seed}".encode()).digest()
    val = struct.unpack("<I", digest[:4])[0]
    return (val + i * 0x9E3779B9) % n_buckets


def _hash_sign(key: str, i: int, seed: int) -> float:
    """Deterministic sign for the i-th hash of key."""
    digest = hashlib.md5(f"{key}:{i}:{seed}:sign".encode()).digest()
    val = struct.unpack("<I", digest[:4])[0]
    return 1.0 if (val & 1) == 0 else -1.0


class FeatureHasher:
    """Encode categorical items into a fixed-size float32 vector.

    Parameters
    ----------
    n_buckets : int
        Fixed size of the output vector. Must not change across patches.
    n_hashes : int
        How many independent hashes each item contributes (default 4).
        More hashes = less collision damage, slightly denser vector.
    seed : int
        Deterministic seed. Changing it shifts every item's buckets and
        invalidates trained models, so treat it as a constant.
    """

    def __init__(self, n_buckets: int, n_hashes: int = 4, seed: int = 42):
        self.n_buckets = n_buckets
        self.n_hashes = n_hashes
        self.seed = seed

    def encode_scalar(self, key: str, value: float) -> np.ndarray:
        """Hash a single key with a scalar value (e.g. power amount)."""
        out = np.zeros(self.n_buckets, dtype=np.float32)
        for i in range(self.n_hashes):
            idx = _hash_bucket(key, i, self.seed, self.n_buckets)
            out[idx] += _hash_sign(key, i, self.seed) * float(value)
        return out

    def encode_binary(self, key: str) -> np.ndarray:
        """Hash a single key with unit value (e.g. card presence)."""
        return self.encode_scalar(key, 1.0)

    def encode_set(
        self,
        items: Iterable[Any],
        get_key: Any = None,
        get_value: Any = None,
    ) -> np.ndarray:
        """Hash a collection of items, optionally with per-item values.

        get_key(item) → str name to hash. Defaults to str(item).
        get_value(item) → float scalar. Defaults to 1.0.
        """
        out = np.zeros(self.n_buckets, dtype=np.float32)
        if not items:
            return out

        for item in items:
            key = get_key(item) if get_key is not None else str(item)
            if key is None:
                continue
            value = get_value(item) if get_value is not None else 1.0
            for i in range(self.n_hashes):
                idx = _hash_bucket(key, i, self.seed, self.n_buckets)
                out[idx] += _hash_sign(key, i, self.seed) * float(value)
        return out


# ─── Pre-built hashers for each identity block ──────────────────────────────
# These sizes are FIXED. Changing any of them shifts every trained model's
# observation space. If you need more capacity, add a NEW hasher alongside
# rather than growing an existing one.

PLAYER_POWER_HASHER = FeatureHasher(n_buckets=128, n_hashes=4, seed=1)
HAND_CARD_HASHER = FeatureHasher(n_buckets=256, n_hashes=4, seed=2)
DECK_CARD_HASHER = FeatureHasher(n_buckets=256, n_hashes=4, seed=3)
ENEMY_IDENTITY_HASHER = FeatureHasher(n_buckets=64, n_hashes=4, seed=4)
ENEMY_POWER_HASHER = FeatureHasher(n_buckets=128, n_hashes=4, seed=5)
RELIC_HASHER = FeatureHasher(n_buckets=256, n_hashes=4, seed=6)
POTION_HASHER = FeatureHasher(n_buckets=128, n_hashes=4, seed=7)

# ─── Block sizes (exported for run_env.py to compute RUN_OBS_SIZE) ──────────

PLAYER_POWER_BUCKETS = PLAYER_POWER_HASHER.n_buckets       # 128
HAND_CARD_BUCKETS = HAND_CARD_HASHER.n_buckets               # 256
HAND_EXTRA_FEATURES = 9                                      # 5 types + upgraded + playable + targets_enemy + cost_x
HAND_EXTRA_BLOCK = 10 * HAND_EXTRA_FEATURES                  # 90
DECK_CARD_BUCKETS = DECK_CARD_HASHER.n_buckets               # 256
ENEMY_IDENTITY_BUCKETS = ENEMY_IDENTITY_HASHER.n_buckets     # 64
ENEMY_POWER_BUCKETS = ENEMY_POWER_HASHER.n_buckets           # 128
ENEMY_EXT_PER_SLOT = ENEMY_IDENTITY_BUCKETS + ENEMY_POWER_BUCKETS  # 192
MAX_ENEMY_SLOTS = 5
ENEMY_EXT_BLOCK = MAX_ENEMY_SLOTS * ENEMY_EXT_PER_SLOT       # 960

# --- cards: embeddings, not hashes -------------------------------------------
#
# Declared here rather than read from the built table, because the observation's
# shape must be knowable without the artifact on disk. `entity_encoding._table()`
# asserts the loaded table agrees, so a table built at another width fails loudly
# instead of silently producing a differently-shaped observation.
#
# HAND_CARD_HASHER and DECK_CARD_HASHER are kept only so old checkpoints and the
# parity suites can still describe the previous layout. Nothing encodes with them.
CARD_EMBED_DIMS = 64
CARD_SLOT_WIDTH = CARD_EMBED_DIMS + 1                        # + is_known flag
MAX_HAND_CARD_SLOTS = 10
HAND_CARD_BLOCK = MAX_HAND_CARD_SLOTS * CARD_SLOT_WIDTH      # 650, per-slot identity
DECK_CARD_BLOCK = CARD_SLOT_WIDTH                            # 65, pooled mean

ENTITY_OBS_SIZE: int = (
    PLAYER_POWER_BUCKETS
    + HAND_CARD_BLOCK
    + HAND_EXTRA_BLOCK
    + DECK_CARD_BLOCK
    + ENEMY_EXT_BLOCK
)  # = 1893

RELIC_BUCKETS = RELIC_HASHER.n_buckets                       # 256
POTION_BUCKETS = POTION_HASHER.n_buckets                     # 128
RELIC_POTION_OBS_SIZE: int = RELIC_BUCKETS + POTION_BUCKETS  # 384
