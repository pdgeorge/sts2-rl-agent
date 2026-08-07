"""Encode every card as a vector, from our own card text.

    python scripts/generate_card_text.py --out output/card_text.json
    python scripts/generate_card_embeddings.py \
        --model <path-to-Qwen3-Embedding-0.6B> \
        --card-text output/card_text.json \
        --out sts2_env/data/card_embeddings.npz

Step 2 of the Phase 5 build plan. Produces the vectors the archetype scorer
runs on.

WHY WE ENCODE OUR OWN TEXT
--------------------------
`t22000t/slay-the-spire-2-card-embeddings` is the reference this design came
from, and it is **several game versions out of date** — building on it would
encode cards as they used to be and leave us waiting on a third party after each
patch. `generate_card_text.py` writes the same JSON shape from *our simulator*,
which `on_update.sh` already polices against the decompile, so there is one
staleness to track rather than two.

THIS IS THE ONLY TIME THE MODEL RUNS
------------------------------------
Not at decision time. The scorer loads an `.npz` and takes dot products —
microseconds, numpy only. The encoder exists to regenerate that file on patch
days, which is why it lives in `scripts/` and not in `sts2_env/`.

POOLING
-------
Qwen3-Embedding pools the **last** token, not the mean, and expects
left-padding so that token is at a fixed index. Getting this wrong produces
vectors that look plausible and encode mostly padding, so it is asserted rather
than trusted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

INSTRUCTION = (
    "Represent this Slay the Spire 2 card by its mechanics, so that cards which "
    "work together in the same deck are close together."
)
"""The single most valuable thing in the encoded string. MEASURED.

An ablation over format x instruction, leave-one-out archetype classification:

                    no instruction    with instruction
        json             7/15              10/15
        pipe             6/15               9/15
        prose            6/15               9/15

Worth ~3 points in every format, where format itself is worth at most 1 and
that is inside the noise at n=15.

This was predicted backwards. The argument against it was that an identical
25-token preamble on all 577 cards is dilution -- ~85% of every string already
in common -- and that stripping scaffolding would sharpen the signal. It did the
opposite, twice. Qwen3-Embedding is instruction-tuned: the preamble is not
padding, it selects *which* similarity to encode, and without it the model
chooses its own, which is not deck synergy.

Leave it in. If it is ever changed, re-run the ablation rather than reasoning
about it."""


def card_sentence(card: dict) -> str:
    """Prettified JSON, which won the ablation above -- narrowly, over pipes and
    prose, and only because the instruction is doing the real work.

    Kept as JSON for one further reason: it is the shape the reference dataset
    used, so its vectors stay comparable with ours as a cross-check.
    """
    return json.dumps(card, indent=2, sort_keys=True)


def _last_token_pool(hidden, attention_mask):
    """Qwen3-Embedding's pooling: the final non-padding token of each sequence."""
    import torch

    left_padded = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padded:
        return hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]


def encode(texts: list[str], model_path: str, batch_size: int = 32) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    prompted = [f"Instruct: {INSTRUCTION}\nCard: {t}" for t in texts]
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(prompted), batch_size):
            batch = prompted[start:start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=512, return_tensors="pt").to(device)
            hidden = model(**encoded).last_hidden_state
            pooled = _last_token_pool(hidden, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.float().cpu().numpy())
            print(f"  {min(start + batch_size, len(prompted))}/{len(prompted)}",
                  end="\r", flush=True)
    print()
    return np.concatenate(out).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to Qwen3-Embedding-0.6B")
    parser.add_argument("--card-text", default="output/card_text.json")
    parser.add_argument("--out", default="sts2_env/data/card_embeddings.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cards = json.loads(Path(args.card_text).read_text())
    ids = sorted(cards)
    texts = [card_sentence(cards[cid]) for cid in ids]
    print(f"encoding {len(ids)} cards with {args.model}")

    vectors = encode(texts, args.model, args.batch_size)
    if vectors.shape[0] != len(ids):
        raise SystemExit(f"got {vectors.shape[0]} vectors for {len(ids)} cards")

    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-2), f"not unit-normalised: {norms.min()}..{norms.max()}"

    # A pooling bug shows up as near-identical vectors, because every sequence
    # ends up encoding its padding. Cheap to check, expensive to miss.
    spread = float(np.std(vectors @ vectors[0]))
    if spread < 0.01:
        raise SystemExit(
            f"vectors are nearly identical (similarity std {spread:.4f}) -- "
            "pooling is probably reading padding rather than the card"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ids=np.array(ids), vectors=vectors)
    print(f"wrote {vectors.shape[0]} x {vectors.shape[1]} to {out}")
    print(f"  similarity spread against card 0: std {spread:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
