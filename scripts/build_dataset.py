from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path
import sys

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.preprocess import load_api_recommendation_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build API recommendation dataset from xlsx files.")

    parser.add_argument("--api_path", type=str, default="data/raw/apisData.xlsx")
    parser.add_argument("--mashup_path", type=str, default="data/raw/mashups.xlsx")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--use_time_split", action="store_true", default=False)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="mashup_time",
        choices=["mashup_time", "mashup_random", "interaction_loo"],
        help="Dataset split mode. interaction_loo is warm-start.",
    )

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_neg_per_pos_train", type=int, default=3)
    parser.add_argument("--num_neg_per_pos_eval", type=int, default=50)
    parser.add_argument("--hard_negative_ratio", type=float, default=0.5)
    parser.add_argument("--max_text_features", type=int, default=512)

    parser.add_argument(
        "--negative_mode",
        type=str,
        default="hybrid",
        choices=["random", "hard", "hybrid"],
        help="Negative sampling mode.",
    )

    parser.add_argument(
        "--add_api_cooccur",
        action="store_true",
        help="Add API-API co-occurrence edges.",
    )
    parser.add_argument(
        "--cooccur_min_count",
        type=int,
        default=2,
        help="Minimum co-occurrence count for keeping API-API edges.",
    )
    parser.add_argument(
        "--cooccur_topk",
        type=int,
        default=50,
        help="Top-K co-occurrence neighbors kept for each API.",
    )
    parser.add_argument(
        "--cooccur_source",
        type=str,
        default="train",
        choices=["train", "all"],
        help="Use train-only or all positives to build API-API co-occurrence edges.",
    )
    parser.add_argument(
        "--cooccur_weight_mode",
        type=str,
        default="binary",
        choices=["binary", "count", "log", "norm", "pmi"],
        help=(
            "Weight mode for API-API co-occurrence edges: "
            "binary/count/log/norm/pmi. Use log as the first weighted experiment."
        ),
    )

    parser.add_argument(
        "--copy_text_emb_from",
        type=str,
        default=None,
        help="Copy mashup_text_emb.npy and api_text_emb.npy from another processed dir.",
    )

    return parser.parse_args()


def _safe_len(x):
    try:
        return len(x)
    except Exception:
        return -1


def _to_numpy_array(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _save_pairs(pos_pairs, neg_pairs, path: Path) -> None:
    obj = {
        "pos": _to_numpy_array(pos_pairs),
        "neg": _to_numpy_array(neg_pairs),
    }
    torch.save(obj, path)


def _validate_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File was not created: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError(f"File is empty: {path}")
    print(f"[OK] {path.name}: {size} bytes")


def _copy_text_embeddings(copy_from: str | None, output_dir: Path) -> None:
    if copy_from is None:
        return

    src_dir = Path(copy_from)
    if not src_dir.exists():
        raise FileNotFoundError(f"copy_text_emb_from does not exist: {src_dir}")

    src_mashup = src_dir / "mashup_text_emb.npy"
    src_api = src_dir / "api_text_emb.npy"

    if not src_mashup.exists():
        raise FileNotFoundError(f"Missing source file: {src_mashup}")
    if not src_api.exists():
        raise FileNotFoundError(f"Missing source file: {src_api}")

    dst_mashup = output_dir / "mashup_text_emb.npy"
    dst_api = output_dir / "api_text_emb.npy"

    shutil.copy2(src_mashup, dst_mashup)
    shutil.copy2(src_api, dst_api)

    print(f"[Copied] {src_mashup} -> {dst_mashup}")
    print(f"[Copied] {src_api} -> {dst_api}")


def _dump_bundle(bundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = output_dir / "graph_data.pt"
    train_pairs_path = output_dir / "train_pairs.pt"
    val_pairs_path = output_dir / "val_pairs.pt"
    test_pairs_path = output_dir / "test_pairs.pt"
    mapping_path = output_dir / "node_mapping.pkl"
    reverse_mapping_path = output_dir / "reverse_node_mapping.pkl"
    metadata_path = output_dir / "metadata.pkl"

    saved_paths = []

    print("===== before save =====")
    print("hetero_data type:", type(bundle.hetero_data))
    print("train_pos_pairs:", type(bundle.train_pos_pairs), _safe_len(bundle.train_pos_pairs))
    print("val_pos_pairs:", type(bundle.val_pos_pairs), _safe_len(bundle.val_pos_pairs))
    print("test_pos_pairs:", type(bundle.test_pos_pairs), _safe_len(bundle.test_pos_pairs))
    print("train_neg_pairs:", type(bundle.train_neg_pairs), _safe_len(bundle.train_neg_pairs))
    print("val_neg_pairs:", type(bundle.val_neg_pairs), _safe_len(bundle.val_neg_pairs))
    print("test_neg_pairs:", type(bundle.test_neg_pairs), _safe_len(bundle.test_neg_pairs))

    torch.save(bundle.hetero_data, graph_path)
    saved_paths.append(graph_path)

    _save_pairs(bundle.train_pos_pairs, bundle.train_neg_pairs, train_pairs_path)
    _save_pairs(bundle.val_pos_pairs, bundle.val_neg_pairs, val_pairs_path)
    _save_pairs(bundle.test_pos_pairs, bundle.test_neg_pairs, test_pairs_path)
    saved_paths.extend([train_pairs_path, val_pairs_path, test_pairs_path])

    _save_pickle(bundle.mappings, mapping_path)
    _save_pickle(bundle.reverse_mappings, reverse_mapping_path)
    _save_pickle(bundle.metadata, metadata_path)
    saved_paths.extend([mapping_path, reverse_mapping_path, metadata_path])

    print("===== after save =====")
    for p in saved_paths:
        _validate_file(p)


def main() -> None:
    args = parse_args()

    print("Start building dataset...")
    print(f"api_path={args.api_path}")
    print(f"mashup_path={args.mashup_path}")
    print(f"output_dir={args.output_dir}")
    print(f"split_mode={args.split_mode}")
    print(f"use_time_split={args.use_time_split}")
    print(f"negative_mode={args.negative_mode}, hard_negative_ratio={args.hard_negative_ratio}")
    print(
        f"add_api_cooccur={args.add_api_cooccur}, "
        f"cooccur_min_count={args.cooccur_min_count}, "
        f"cooccur_topk={args.cooccur_topk}, "
        f"cooccur_source={args.cooccur_source}, "
        f"cooccur_weight_mode={args.cooccur_weight_mode}"
    )

    bundle = load_api_recommendation_data(
        api_path=args.api_path,
        mashup_path=args.mashup_path,
        random_seed=args.seed,
        use_time_split=args.use_time_split,
        split_mode=args.split_mode,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_neg_per_pos_train=args.num_neg_per_pos_train,
        num_neg_per_pos_eval=args.num_neg_per_pos_eval,
        hard_negative_ratio=args.hard_negative_ratio,
        max_text_features=args.max_text_features,
        negative_mode=args.negative_mode,
        add_api_cooccur=args.add_api_cooccur,
        cooccur_min_count=args.cooccur_min_count,
        cooccur_topk=args.cooccur_topk,
        cooccur_source=args.cooccur_source,
        cooccur_weight_mode=args.cooccur_weight_mode,
    )

    print("Data bundle constructed, start dumping...")

    output_dir = Path(args.output_dir)
    _dump_bundle(bundle, output_dir)
    _copy_text_embeddings(args.copy_text_emb_from, output_dir)

    print("\nDataset build finished.")
    print(f"Saved to: {output_dir}")

    for key in [
        "split_mode",
        "graph_uses_edges_source",
        "num_graph_train_uses_edges",
        "negative_mode",
        "hard_negative_ratio",
        "add_api_cooccur",
        "cooccur_source",
        "cooccur_min_count",
        "cooccur_topk",
        "cooccur_weight_mode",
        "num_api_api_cooccur_edges",
        "cooccur_weight_min",
        "cooccur_weight_max",
        "cooccur_weight_mean",
        "num_mashups",
        "num_apis",
        "num_categories",
        "num_developers",
        "num_companies",
    ]:
        print(f"{key}: {bundle.metadata.get(key, 'unknown')}")

    print(f"train_pos_pairs: {len(bundle.train_pos_pairs)}")
    print(f"val_pos_pairs: {len(bundle.val_pos_pairs)}")
    print(f"test_pos_pairs: {len(bundle.test_pos_pairs)}")
    print(f"train_neg_pairs: {len(bundle.train_neg_pairs)}")
    print(f"val_neg_pairs: {len(bundle.val_neg_pairs)}")
    print(f"test_neg_pairs: {len(bundle.test_neg_pairs)}")


if __name__ == "__main__":
    main()
