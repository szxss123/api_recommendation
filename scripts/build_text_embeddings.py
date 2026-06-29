from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def safe_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def split_multi_value_field(x) -> List[str]:
    if pd.isna(x):
        return []
    x = str(x).strip()
    if not x:
        return []
    return [i.strip() for i in x.split("###") if str(i).strip()]


def build_mashup_text(row) -> str:
    parts = [
        safe_text(row.get("mashups_name", "")),
        safe_text(row.get("description", "")),
        safe_text(row.get("categories", "")),
        safe_text(row.get("company", "")),
        safe_text(row.get("mashup_app_type", "")),
    ]
    return " ".join([p for p in parts if p])


def build_api_text(row) -> str:
    parts = [
        safe_text(row.get("APIName", "")),
        safe_text(row.get("Description", "")),
        safe_text(row.get("Categories", "")),
        safe_text(row.get("Versions", "")),
        safe_text(row.get("DevelopersName", "")),
    ]
    return " ".join([p for p in parts if p])


def load_node_mapping(mapping_path: Path) -> Dict:
    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)
    return mapping


def build_graph_order_mashup_texts(mashup_df: pd.DataFrame, api_df: pd.DataFrame) -> List[str]:
    """
    不依赖 node_mapping['mashup'] 的 key。
    直接复现构图时的过滤逻辑，并按过滤后的 DataFrame 行顺序生成文本。
    """
    mashup_df = mashup_df.copy()
    api_df = api_df.copy()

    mashup_df = mashup_df.dropna(subset=["mashups_name"]).copy()
    api_df = api_df.dropna(subset=["APIName"]).copy()

    mashup_df["mashups_name"] = mashup_df["mashups_name"].astype(str).str.strip()
    api_df["APIName"] = api_df["APIName"].astype(str).str.strip()

    api_name_set = set(api_df["APIName"].tolist())

    if "related_apis" not in mashup_df.columns:
        raise KeyError("mashups.xlsx must contain column 'related_apis'")

    mashup_df["related_apis_list"] = mashup_df["related_apis"].apply(split_multi_value_field)

    def filter_known_apis(api_list):
        return [a for a in api_list if a in api_name_set]

    mashup_df["related_apis_list"] = mashup_df["related_apis_list"].apply(filter_known_apis)
    mashup_df = mashup_df[mashup_df["related_apis_list"].map(len) > 0].reset_index(drop=True)

    texts = [build_mashup_text(row) for _, row in mashup_df.iterrows()]
    print(f"[Mashup graph-order build] total={len(texts)}")
    return texts


def build_aligned_api_texts(api_df: pd.DataFrame, api_mapping: Dict[str, int]) -> List[str]:
    api_df = api_df.copy()
    api_df["APIName"] = api_df["APIName"].astype(str).str.strip()

    name_to_text = {}
    for _, row in api_df.iterrows():
        name = str(row["APIName"]).strip()
        if name not in name_to_text:
            name_to_text[name] = build_api_text(row)

    texts = [""] * len(api_mapping)
    miss = 0
    for name, idx in api_mapping.items():
        key = str(name).strip()
        txt = name_to_text.get(key, "")
        if not txt:
            miss += 1
        texts[idx] = txt

    print(f"[API alignment] total={len(api_mapping)}, missing={miss}")
    return texts


def transformer_embeddings(
    texts: List[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
    device: str | None = None,
) -> np.ndarray:
    texts = [t if isinstance(t, str) and t.strip() else "[EMPTY]" for t in texts]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Embedding] loading model: {model_name}")
    print(f"[Embedding] device: {device}")

    model = SentenceTransformer(model_name, device=device)

    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Build BGE text embeddings for mashup/api.")
    parser.add_argument("--api_path", type=str, default="data/raw/apisData.xlsx")
    parser.add_argument("--mashup_path", type=str, default="data/raw/mashups.xlsx")
    parser.add_argument("--mapping_path", type=str, default="data/processed/node_mapping.pkl")
    parser.add_argument("--graph_path", type=str, default="data/processed/graph_data.pt")
    parser.add_argument("--output_dir", type=str, default="data/processed")

    parser.add_argument("--model_name", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    api_path = Path(args.api_path)
    mashup_path = Path(args.mashup_path)
    mapping_path = Path(args.mapping_path)
    graph_path = Path(args.graph_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not api_path.exists():
        raise FileNotFoundError(f"Missing file: {api_path}")
    if not mashup_path.exists():
        raise FileNotFoundError(f"Missing file: {mashup_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing file: {mapping_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing file: {graph_path}")

    api_df = pd.read_excel(api_path)
    mashup_df = pd.read_excel(mashup_path)

    api_df = api_df.dropna(subset=["APIName"]).copy()
    mashup_df = mashup_df.dropna(subset=["mashups_name"]).copy()

    mapping = load_node_mapping(mapping_path)
    if "api" not in mapping:
        raise KeyError("node_mapping.pkl must contain 'api' key")

    api_mapping = mapping["api"]

    mashup_texts = build_graph_order_mashup_texts(mashup_df, api_df)
    api_texts = build_aligned_api_texts(api_df, api_mapping)

    print("mashup sample texts:")
    for i in range(min(5, len(mashup_texts))):
        print(i, repr(mashup_texts[i][:200]))

    print("api sample texts:")
    for i in range(min(5, len(api_texts))):
        print(i, repr(api_texts[i][:200]))

    print("non-empty mashup texts:", sum(1 for t in mashup_texts if isinstance(t, str) and t.strip()))
    print("non-empty api texts:", sum(1 for t in api_texts if isinstance(t, str) and t.strip()))

    graph_data = torch.load(graph_path, map_location="cpu")
    expected_mashup = graph_data["mashup"].x.shape[0]
    expected_api = graph_data["api"].x.shape[0]

    print(f"[Graph rows] mashup={expected_mashup}, api={expected_api}")
    print(f"[Text rows ] mashup={len(mashup_texts)}, api={len(api_texts)}")

    if len(mashup_texts) != expected_mashup:
        raise ValueError(
            f"Mashup text rows do not match graph rows: texts={len(mashup_texts)}, graph={expected_mashup}"
        )
    if len(api_texts) != expected_api:
        raise ValueError(
            f"API text rows do not match graph rows: texts={len(api_texts)}, graph={expected_api}"
        )

    mashup_emb = transformer_embeddings(
        mashup_texts,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )
    api_emb = transformer_embeddings(
        api_texts,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )

    mashup_out = output_dir / "mashup_text_emb.npy"
    api_out = output_dir / "api_text_emb.npy"

    np.save(mashup_out, mashup_emb)
    np.save(api_out, api_emb)

    print("Saved text embeddings:")
    print(f"  {mashup_out} -> {mashup_emb.shape}")
    print(f"  {api_out} -> {api_emb.shape}")


if __name__ == "__main__":
    main()