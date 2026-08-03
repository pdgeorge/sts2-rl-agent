"""Tests for the read side of the card embedding table.

Built on a synthetic table rather than the real artifact, so these run before the
table exists and do not silently start testing whatever happens to be on disk.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sts2_env.embedding.table import (
    CardEmbeddingTableMissing,
    load_table,
)


@pytest.fixture
def table_dir(tmp_path):
    """A three-card table with known contents."""
    path = tmp_path / "v1"
    path.mkdir()
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.save(path / "vectors.npy", vectors)
    (path / "card_ids.txt").write_text("BASH\nSTRIKE_IRONCLAD\nBARRICADE_CARD\n")
    (path / "manifest.json").write_text(json.dumps({"dims": 4, "model": "synthetic"}))
    return path


def test_known_card_returns_its_row_and_flag(table_dir):
    table = load_table(table_dir)
    vector, known = table.vector("BASH")
    assert known == 1.0
    assert np.allclose(vector, [1.0, 0.0, 0.0, 0.0])


def test_unknown_card_is_zero_and_flagged(table_dir):
    """The flag is the point.

    Without it the network cannot tell "never seen this card" from "this card's
    vector sits near the origin", and those call for opposite behaviour.
    """
    table = load_table(table_dir)
    vector, known = table.vector("A_CARD_FROM_NEXT_PATCH")
    assert known == 0.0
    assert np.allclose(vector, np.zeros(4))


def test_unknown_card_does_not_raise(table_dir):
    """Refusing to start was considered and rejected: one unknown card must not
    block a run."""
    table = load_table(table_dir)
    table.vector("TOTALLY_MADE_UP")  # must not raise


def test_encode_appends_the_flag(table_dir):
    table = load_table(table_dir)
    assert table.encode("BASH").shape == (5,)
    assert table.encode("BASH")[-1] == 1.0
    assert table.encode("NOPE")[-1] == 0.0
    assert table.width == 5


def test_pooling_averages_known_cards_and_reports_coverage(table_dir):
    table = load_table(table_dir)

    pooled = table.pooled(["BASH", "STRIKE_IRONCLAD"])
    assert np.allclose(pooled[:4], [0.5, 0.5, 0.0, 0.0])
    assert pooled[4] == pytest.approx(1.0)

    # An unknown card lowers coverage but must not drag the mean toward zero.
    half = table.pooled(["BASH", "UNKNOWN_CARD"])
    assert np.allclose(half[:4], [1.0, 0.0, 0.0, 0.0])
    assert half[4] == pytest.approx(0.5)


def test_pooling_is_scale_free(table_dir):
    """Mean, not sum: a 40-card deck and a 10-card deck stay comparable."""
    table = load_table(table_dir)
    small = table.pooled(["BASH", "STRIKE_IRONCLAD"])
    large = table.pooled(["BASH", "STRIKE_IRONCLAD"] * 8)
    assert np.allclose(small, large)


def test_pooling_an_empty_or_wholly_unknown_pile(table_dir):
    table = load_table(table_dir)
    assert np.allclose(table.pooled([]), np.zeros(5))
    assert np.allclose(table.pooled(["NOPE", "ALSO_NOPE"]), np.zeros(5))


def test_missing_table_says_how_to_build_it(tmp_path):
    with pytest.raises(CardEmbeddingTableMissing) as excinfo:
        load_table(tmp_path / "does_not_exist")
    assert "build_card_embeddings" in str(excinfo.value)


def test_inconsistent_table_is_rejected(tmp_path):
    """Ids and vectors disagreeing is silent misalignment -- every card would
    read as its neighbour."""
    path = tmp_path / "bad"
    path.mkdir()
    np.save(path / "vectors.npy", np.zeros((3, 4), dtype=np.float32))
    (path / "card_ids.txt").write_text("BASH\nSTRIKE_IRONCLAD\n")  # only two
    (path / "manifest.json").write_text(json.dumps({"dims": 4}))

    with pytest.raises(CardEmbeddingTableMissing):
        load_table(path)


def test_dims_mismatch_is_rejected(tmp_path):
    path = tmp_path / "dims"
    path.mkdir()
    np.save(path / "vectors.npy", np.zeros((2, 4), dtype=np.float32))
    (path / "card_ids.txt").write_text("BASH\nSTRIKE_IRONCLAD\n")
    (path / "manifest.json").write_text(json.dumps({"dims": 64}))

    with pytest.raises(CardEmbeddingTableMissing):
        load_table(path)


def test_card_id_enum_members_work_as_keys(table_dir):
    """Callers pass CardId, not strings."""
    from sts2_env.core.enums import CardId

    table = load_table(table_dir)
    _, known = table.vector(CardId.BASH)
    assert known == 1.0
