#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BGE embeddings for a processed MTFM dataset."
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("data/processed_mtfm_warm_bge_weighted_log_m2_k50"),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-small-en-v1.5",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_seq_length", type=int, default=256)
    return parser.parse_args()


def load_texts(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        values = json.load(file)
    if not isinstance(values, list):
        raise TypeError(f"{path} must contain a JSON list.")
    return [str(value or "") for value in values]


def main() -> None:
    args = parse_args()
    processed_dir = args.processed_dir.resolve()

    mashup_texts = load_texts(processed_dir / "mashup_texts.json")
    api_texts = load_texts(processed_dir / "api_texts.json")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required. Install it with: "
            "pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(args.model_name, device=args.device)
    model.max_seq_length = args.max_seq_length

    print(f"Model:              {args.model_name}")
    print(f"Device:             {args.device}")
    print(f"Mashup texts:       {len(mashup_texts)}")
    print(f"API texts:          {len(api_texts)}")
    print(f"Max sequence length:{args.max_seq_length}")

    mashup_embeddings = model.encode(
        mashup_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    api_embeddings = model.encode(
        api_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    np.save(processed_dir / "mashup_text_emb.npy", mashup_embeddings)
    np.save(processed_dir / "api_text_emb.npy", api_embeddings)

    print("=" * 80)
    print("BGE embeddings completed")
    print("=" * 80)
    print(f"Mashup shape: {mashup_embeddings.shape}")
    print(f"API shape:    {api_embeddings.shape}")
    print(f"Saved:        {processed_dir / 'mashup_text_emb.npy'}")
    print(f"Saved:        {processed_dir / 'api_text_emb.npy'}")


if __name__ == "__main__":
    main()
