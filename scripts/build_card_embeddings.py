"""Build (or extend) the frozen card embedding table.

    python scripts/build_card_embeddings.py                 # first build
    python scripts/build_card_embeddings.py --append        # after a patch

Renders every card through the frozen template, embeds the text with
Qwen3-Embedding-0.6B, projects 1024 -> N dims through a PCA fitted exactly once,
and writes a versioned directory:

    data/card_embeddings/v1/
      manifest.json     model id + revision, template version + hash, dims, dtype
      card_ids.txt      row order, append-only
      texts.txt         exactly what was embedded, for auditing
      mean.npy          frozen centring vector
      projection.npy    frozen PCA matrix
      vectors.npy       N x dims, append-only

WHAT IS FROZEN AND WHY

Four things decide what a vector means: the model revision, the text template,
the centring mean, and the projection matrix. Change any one and every vector
moves -- silently, because a vector has no visibly wrong value. Every checkpoint
trained against the old table then misreads every card.

So ``--append`` never refits. It loads the existing mean and projection, embeds
only the cards that have no row yet, and appends. Existing rows come out
bit-identical because the projection is a pure function:

    vector = (embed(text) - frozen_mean) @ frozen_projection

The mean is the easy one to miss. PCA centres before projecting, so recomputing
it on a larger corpus shifts every existing vector by a constant.

A fresh build into a directory that already exists is refused. Rebuilding is a
legitimate thing to want, but it invalidates every model trained against the old
table, so it takes ``--force`` and a new version directory is usually the better
answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_DIMS = 64
DEFAULT_OUT = Path("data/card_embeddings/v1")


def _fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def render_all_cards() -> tuple[list[str], list[str]]:
    """Every renderable card, in a stable order, with its text.

    Ordered by ``CardId`` name rather than enum position: enum position is
    exactly what shifted under a patch and started this whole line of work.
    """
    import sts2_env.cards  # noqa: F401 -- populates the effect registry
    import sts2_env.powers  # noqa: F401 -- populates the power class registry
    from sts2_env.cards.factory import card_preview
    from sts2_env.core.enums import CardId
    from sts2_env.embedding.card_text import render_card_text

    names: list[str] = []
    texts: list[str] = []
    for card_id in sorted(CardId, key=lambda c: c.name):
        try:
            card_preview(card_id)
        except Exception:  # noqa: BLE001 -- not every enum member is a real card
            continue
        names.append(card_id.name)
        texts.append(render_card_text(card_id))
    return names, texts


def _last_token_pool(hidden, attention_mask):
    """Qwen3-Embedding pools the last non-padding token, not the mean."""
    import torch

    left_padded = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padded:
        return hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]


def embed_texts(texts: list[str], model_name: str, batch_size: int, device: str):
    """Embed with Qwen3-Embedding and L2-normalise. Returns (matrix, revision)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModel.from_pretrained(model_name, dtype=torch.float32).to(device).eval()

    revision = "unknown"
    try:  # record exactly which weights produced these vectors
        from huggingface_hub import model_info

        revision = model_info(model_name).sha or "unknown"
    except Exception:  # noqa: BLE001 -- offline is fine, the note just says so
        pass

    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            pooled = _last_token_pool(hidden, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.float().cpu().numpy())
            print(f"  embedded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)

    return np.concatenate(out, axis=0), revision


def fit_projection(matrix: np.ndarray, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit the PCA once. Returns (mean, projection)."""
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    explained = float(np.cumsum(singular**2)[dims - 1] / np.sum(singular**2))
    print(f"  PCA {matrix.shape[1]} -> {dims} dims, explained variance {explained:.1%}")
    return mean.astype(np.float32), vt[:dims].T.astype(np.float32)


def project(matrix: np.ndarray, mean: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """The pure function every vector must come from, now and after any patch."""
    return ((matrix - mean) @ projection).astype(np.float32)


def template_hash() -> str:
    """Hash the template's source, so a changed renderer is detectable."""
    from sts2_env.embedding import card_text

    source = Path(card_text.__file__).read_text()
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def write_table(out: Path, names, texts, vectors, mean, projection, manifest) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "card_ids.txt").write_text("\n".join(names) + "\n")
    (out / "texts.txt").write_text("\n\n----\n\n".join(texts) + "\n")
    np.save(out / "mean.npy", mean)
    np.save(out / "projection.npy", projection)
    np.save(out / "vectors.npy", vectors)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dims", type=int, default=DEFAULT_DIMS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--append", action="store_true",
        help="Embed only cards with no row yet, projecting through the frozen "
             "mean and matrix. This is the path sync_content uses after a patch.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Refit into an existing directory. Invalidates every checkpoint "
             "trained against it.",
    )
    args = parser.parse_args()

    from sts2_env.embedding.card_text import TEMPLATE_VERSION

    names, texts = render_all_cards()
    print(f"rendered {len(names)} cards, template v{TEMPLATE_VERSION}")

    manifest_path = args.out / "manifest.json"
    exists = manifest_path.is_file()

    if exists and not (args.append or args.force):
        _fail(
            f"{args.out} already exists. Use --append to add new cards through the "
            f"frozen projection, or --force to refit (which invalidates every "
            f"checkpoint trained against it)."
        )
    if args.append and not exists:
        _fail(f"{args.out} does not exist yet; run without --append to build it.")

    if args.append:
        existing_manifest = json.loads(manifest_path.read_text())
        existing_names = (args.out / "card_ids.txt").read_text().split()
        existing_vectors = np.load(args.out / "vectors.npy")
        mean = np.load(args.out / "mean.npy")
        projection = np.load(args.out / "projection.npy")

        if existing_manifest.get("template_hash") != template_hash():
            _fail(
                "the card text template has changed since this table was built. "
                "Appending would mix vectors from two different templates. Build a "
                "new version directory instead."
            )

        known = set(existing_names)
        new = [(n, t) for n, t in zip(names, texts) if n not in known]
        if not new:
            print("no new cards; table is current")
            return 0

        print(f"appending {len(new)} new card(s): {', '.join(n for n, _ in new[:10])}")
        new_names = [n for n, _ in new]
        new_texts = [t for _, t in new]
        raw, revision = embed_texts(new_texts, args.model, args.batch_size, args.device)
        new_vectors = project(raw, mean, projection)

        combined = np.concatenate([existing_vectors, new_vectors], axis=0)
        all_names = existing_names + new_names
        all_texts = (args.out / "texts.txt").read_text().split("\n\n----\n\n") + new_texts

        existing_manifest.update({
            "cards": len(all_names),
            "appended_model_revision": revision,
        })
        write_table(args.out, all_names, all_texts, combined, mean, projection, existing_manifest)
        print(f"appended. table now {combined.shape[0]} x {combined.shape[1]}")
        return 0

    print(f"embedding with {args.model} on {args.device} ...")
    raw, revision = embed_texts(texts, args.model, args.batch_size, args.device)
    print(f"raw embeddings {raw.shape}")

    mean, projection = fit_projection(raw, args.dims)
    vectors = project(raw, mean, projection)

    manifest = {
        "model": args.model,
        "model_revision": revision,
        "template_version": TEMPLATE_VERSION,
        "template_hash": template_hash(),
        "raw_dims": int(raw.shape[1]),
        "dims": int(args.dims),
        "cards": len(names),
        "dtype": "float32",
        "device_used": args.device,
        "pooling": "last_token",
        "normalised": True,
        "frozen": ["model_revision", "template_hash", "mean.npy", "projection.npy"],
    }
    write_table(args.out, names, texts, vectors, mean, projection, manifest)
    print(f"wrote {args.out} -- {vectors.shape[0]} x {vectors.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
