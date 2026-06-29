#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from strict_baseline_core import (
    load_strict_data,
    load_yaml,
    normalize_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit strict new-Mashup cold-start data for exact/near duplicate "
            "Mashups, direct ground-truth API-name mentions, and nearest-neighbor "
            "interaction overlap."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/strict_leakage_audit.yaml"),
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value).lower())
    return " ".join(tokens)


def compact_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def safe_reverse_lookup(container: Any, index: int) -> str:
    if isinstance(container, Mapping):
        for key in (index, str(index)):
            if key in container:
                return str(container[key])
        raise KeyError(index)
    if isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
        return str(container[index])
    raise TypeError(f"Unsupported reverse mapping type: {type(container)!r}")


def identifier_variants(identifier: str) -> List[str]:
    base = str(identifier).strip().lower()
    candidates = {
        normalize_text(base),
        normalize_text(base.replace("_", " ").replace("-", " ")),
        compact_identifier(base),
    }
    return sorted(value for value in candidates if value)


def find_api_name_mention(text: str, api_identifier: str) -> Tuple[bool, str]:
    normalized = normalize_text(text)
    compact = compact_identifier(text)
    for variant in identifier_variants(api_identifier):
        # Space-separated variant: enforce token boundaries.
        if " " in variant:
            pattern = rf"(?:^|\s){re.escape(variant)}(?:$|\s)"
            if re.search(pattern, normalized):
                return True, variant
        else:
            # Avoid one/two-character or purely numeric false positives.
            if len(variant) >= 4 and not variant.isdigit() and variant in compact:
                return True, variant
    return False, ""


def positives_by_mashup(pairs: np.ndarray) -> Dict[int, Set[int]]:
    result: Dict[int, Set[int]] = defaultdict(set)
    for mashup_id, api_id in pairs:
        result[int(mashup_id)].add(int(api_id))
    return dict(result)


def topk_cosine_neighbors(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_ids: np.ndarray,
    topk: int,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    query = normalize_rows(query_embeddings.astype(np.float32))
    candidates = normalize_rows(candidate_embeddings.astype(np.float32))
    topk = min(topk, candidates.shape[0])

    all_indices = np.empty((query.shape[0], topk), dtype=np.int64)
    all_scores = np.empty((query.shape[0], topk), dtype=np.float32)

    for start in range(0, query.shape[0], chunk_size):
        end = min(query.shape[0], start + chunk_size)
        similarity = query[start:end] @ candidates.T
        partition = np.argpartition(
            -similarity,
            kth=topk - 1,
            axis=1,
        )[:, :topk]
        partition_scores = np.take_along_axis(similarity, partition, axis=1)
        order = np.argsort(-partition_scores, axis=1)
        selected = np.take_along_axis(partition, order, axis=1)
        selected_scores = np.take_along_axis(partition_scores, order, axis=1)

        all_indices[start:end] = candidate_ids[selected]
        all_scores[start:end] = selected_scores

    return all_indices, all_scores


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [
            str(row[column]).replace("|", r"\|").replace("\n", " ")
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    reference_config = Path(config["reference_config"])
    data = load_strict_data(reference_config)

    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_positive = positives_by_mashup(data.train_pairs)
    test_positive = positives_by_mashup(data.test_pairs)

    train_ids = np.asarray(sorted(train_positive), dtype=np.int64)
    test_ids = np.asarray(sorted(test_positive), dtype=np.int64)

    processed_dir = data.processed_dir
    reverse_path = processed_dir / str(
        config.get("reverse_mapping_filename", "reverse_node_mapping.pkl")
    )
    if not reverse_path.exists():
        raise FileNotFoundError(
            f"Reverse node mapping not found: {reverse_path}"
        )
    with reverse_path.open("rb") as file:
        reverse_mapping = pickle.load(file)

    mashup_reverse = reverse_mapping["mashup"]
    api_reverse = reverse_mapping["api"]

    train_normalized_text: Dict[str, List[int]] = defaultdict(list)
    train_normalized_name: Dict[str, List[int]] = defaultdict(list)
    for mashup_id in train_ids:
        train_normalized_text[
            normalize_text(data.mashup_texts[int(mashup_id)])
        ].append(int(mashup_id))
        train_normalized_name[
            normalize_text(safe_reverse_lookup(mashup_reverse, int(mashup_id)))
        ].append(int(mashup_id))

    neighbor_ids, neighbor_scores = topk_cosine_neighbors(
        data.mashup_text_emb[test_ids],
        data.mashup_text_emb[train_ids],
        train_ids,
        topk=int(config["topk_neighbors"]),
        chunk_size=int(config.get("chunk_size", 256)),
    )

    thresholds = [
        float(value)
        for value in config.get(
            "near_duplicate_thresholds",
            [0.90, 0.95, 0.98, 0.99],
        )
    ]

    audit_rows: List[Dict[str, Any]] = []
    neighbor_rows: List[Dict[str, Any]] = []
    mention_rows: List[Dict[str, Any]] = []

    for row_index, test_id_value in enumerate(test_ids):
        test_id = int(test_id_value)
        mashup_text = data.mashup_texts[test_id]
        mashup_identifier = safe_reverse_lookup(mashup_reverse, test_id)
        positives = test_positive[test_id]

        normalized_text = normalize_text(mashup_text)
        normalized_name = normalize_text(mashup_identifier)
        exact_text_matches = train_normalized_text.get(normalized_text, [])
        exact_name_matches = train_normalized_name.get(normalized_name, [])

        top1_train_id = int(neighbor_ids[row_index, 0])
        top1_similarity = float(neighbor_scores[row_index, 0])
        top1_apis = train_positive[top1_train_id]
        top1_overlap = positives & top1_apis

        neighbor_union: Set[int] = set()
        for rank, (train_id_value, similarity_value) in enumerate(
            zip(neighbor_ids[row_index], neighbor_scores[row_index]),
            start=1,
        ):
            train_id = int(train_id_value)
            train_apis = train_positive[train_id]
            overlap = positives & train_apis
            neighbor_union.update(train_apis)
            neighbor_rows.append(
                {
                    "test_mashup_id": test_id,
                    "test_mashup_identifier": mashup_identifier,
                    "neighbor_rank": rank,
                    "train_mashup_id": train_id,
                    "train_mashup_identifier": safe_reverse_lookup(
                        mashup_reverse,
                        train_id,
                    ),
                    "cosine_similarity": float(similarity_value),
                    "train_positive_count": len(train_apis),
                    "shared_positive_count": len(overlap),
                    "shared_positive_api_ids": " ".join(
                        str(value) for value in sorted(overlap)
                    ),
                    "shared_positive_recall": (
                        len(overlap) / len(positives) if positives else 0.0
                    ),
                }
            )

        mention_count = 0
        for api_id in sorted(positives):
            api_identifier = safe_reverse_lookup(api_reverse, int(api_id))
            mentioned, matched_variant = find_api_name_mention(
                mashup_text,
                api_identifier,
            )
            mention_count += int(mentioned)
            mention_rows.append(
                {
                    "mashup_id": test_id,
                    "mashup_identifier": mashup_identifier,
                    "api_id": int(api_id),
                    "api_identifier": api_identifier,
                    "is_directly_mentioned": int(mentioned),
                    "matched_variant": matched_variant,
                }
            )

        record: Dict[str, Any] = {
            "mashup_id": test_id,
            "mashup_identifier": mashup_identifier,
            "num_positive_apis": len(positives),
            "exact_train_text_duplicate": int(bool(exact_text_matches)),
            "exact_train_text_match_ids": " ".join(
                str(value) for value in exact_text_matches
            ),
            "exact_train_name_duplicate": int(bool(exact_name_matches)),
            "exact_train_name_match_ids": " ".join(
                str(value) for value in exact_name_matches
            ),
            "top1_train_mashup_id": top1_train_id,
            "top1_train_mashup_identifier": safe_reverse_lookup(
                mashup_reverse,
                top1_train_id,
            ),
            "top1_cosine_similarity": top1_similarity,
            "top1_shared_positive_count": len(top1_overlap),
            "top1_shared_positive_recall": (
                len(top1_overlap) / len(positives) if positives else 0.0
            ),
            "topk_neighbor_union_positive_recall": (
                len(positives & neighbor_union) / len(positives)
                if positives
                else 0.0
            ),
            "direct_api_name_mention_count": mention_count,
            "direct_api_name_mention_recall": (
                mention_count / len(positives) if positives else 0.0
            ),
        }
        for threshold in thresholds:
            record[f"top1_cosine_ge_{threshold:.2f}"] = int(
                top1_similarity >= threshold
            )
        audit_rows.append(record)

    audit_frame = pd.DataFrame(audit_rows)
    neighbors_frame = pd.DataFrame(neighbor_rows)
    mentions_frame = pd.DataFrame(mention_rows)

    audit_frame.to_csv(output_dir / "mashup_audit.csv", index=False)
    neighbors_frame.to_csv(
        output_dir / "nearest_train_neighbors.csv",
        index=False,
    )
    mentions_frame.to_csv(
        output_dir / "api_name_mentions.csv",
        index=False,
    )

    summary: Dict[str, Any] = {
        "num_test_mashups": int(len(audit_frame)),
        "exact_train_text_duplicate_count": int(
            audit_frame["exact_train_text_duplicate"].sum()
        ),
        "exact_train_name_duplicate_count": int(
            audit_frame["exact_train_name_duplicate"].sum()
        ),
        "direct_api_name_mention_mashup_count": int(
            (audit_frame["direct_api_name_mention_count"] > 0).sum()
        ),
        "direct_api_name_mention_mashup_rate": float(
            (audit_frame["direct_api_name_mention_count"] > 0).mean()
        ),
        "mean_direct_api_name_mention_recall": float(
            audit_frame["direct_api_name_mention_recall"].mean()
        ),
        "top1_neighbor_shares_positive_mashup_count": int(
            (audit_frame["top1_shared_positive_count"] > 0).sum()
        ),
        "top1_neighbor_shares_positive_mashup_rate": float(
            (audit_frame["top1_shared_positive_count"] > 0).mean()
        ),
        "mean_top1_shared_positive_recall": float(
            audit_frame["top1_shared_positive_recall"].mean()
        ),
        "mean_topk_union_positive_recall": float(
            audit_frame["topk_neighbor_union_positive_recall"].mean()
        ),
        "top1_cosine_mean": float(
            audit_frame["top1_cosine_similarity"].mean()
        ),
        "top1_cosine_median": float(
            audit_frame["top1_cosine_similarity"].median()
        ),
        "top1_cosine_max": float(
            audit_frame["top1_cosine_similarity"].max()
        ),
        "near_duplicate_threshold_counts": {
            f"{threshold:.2f}": int(
                audit_frame[f"top1_cosine_ge_{threshold:.2f}"].sum()
            )
            for threshold in thresholds
        },
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    suspicious = audit_frame.sort_values(
        [
            "exact_train_text_duplicate",
            "exact_train_name_duplicate",
            "direct_api_name_mention_recall",
            "top1_cosine_similarity",
        ],
        ascending=False,
    ).head(int(config.get("report_top_cases", 20)))

    report_summary = pd.DataFrame(
        [
            ["Test Mashups", summary["num_test_mashups"]],
            [
                "Exact train-text duplicates",
                summary["exact_train_text_duplicate_count"],
            ],
            [
                "Exact train-name duplicates",
                summary["exact_train_name_duplicate_count"],
            ],
            [
                "Mashups mentioning ≥1 ground-truth API name",
                (
                    f"{summary['direct_api_name_mention_mashup_count']} "
                    f"({summary['direct_api_name_mention_mashup_rate']:.2%})"
                ),
            ],
            [
                "Top-1 neighbor sharing ≥1 positive API",
                (
                    f"{summary['top1_neighbor_shares_positive_mashup_count']} "
                    f"({summary['top1_neighbor_shares_positive_mashup_rate']:.2%})"
                ),
            ],
            [
                "Mean Top-1 shared-positive recall",
                f"{summary['mean_top1_shared_positive_recall']:.4f}",
            ],
            [
                "Mean Top-K union positive recall",
                f"{summary['mean_topk_union_positive_recall']:.4f}",
            ],
            [
                "Mean / median / max Top-1 cosine",
                (
                    f"{summary['top1_cosine_mean']:.4f} / "
                    f"{summary['top1_cosine_median']:.4f} / "
                    f"{summary['top1_cosine_max']:.4f}"
                ),
            ],
        ],
        columns=["Item", "Value"],
    )

    suspicious_columns = [
        "mashup_id",
        "mashup_identifier",
        "exact_train_text_duplicate",
        "exact_train_name_duplicate",
        "top1_train_mashup_identifier",
        "top1_cosine_similarity",
        "top1_shared_positive_recall",
        "direct_api_name_mention_recall",
    ]
    suspicious_display = suspicious[suspicious_columns].copy()
    for column in (
        "top1_cosine_similarity",
        "top1_shared_positive_recall",
        "direct_api_name_mention_recall",
    ):
        suspicious_display[column] = suspicious_display[column].map(
            lambda value: f"{float(value):.4f}"
        )

    report = [
        "# Strict New-Mashup Cold-Start Leakage and Near-Duplicate Audit",
        "",
        "This audit separates three different phenomena:",
        "",
        "1. exact/near duplicate Mashups across train and test, which can threaten "
        "the validity of a strict split;",
        "2. direct mentions of a ground-truth API name in test Mashup text, which "
        "are legal side information only if the deployment setting exposes such text;",
        "3. API overlap between a test Mashup and its nearest training Mashups, "
        "which is the intended transfer signal of the inductive collaborative baselines "
        "and is not by itself leakage.",
        "",
        "## Summary",
        "",
        markdown_table(report_summary),
        "",
        "## Near-duplicate thresholds",
        "",
    ]
    for threshold, count in summary["near_duplicate_threshold_counts"].items():
        report.append(f"- Top-1 cosine ≥ {threshold}: {count}")
    report.extend(
        [
            "",
            "## Highest-risk cases",
            "",
            markdown_table(suspicious_display),
            "",
            "## Interpretation rules",
            "",
            "- Any exact text/name duplicate should be manually inspected.",
            "- A very high cosine score alone is not proof of leakage; inspect the text.",
            "- API-name mentions must be disclosed in the paper if names/descriptions "
            "are available to the recommender at inference time.",
            "- Shared APIs among semantic neighbors explain why inductive collaborative "
            "transfer can outperform direct text similarity.",
            "",
        ]
    )
    (output_dir / "leakage_audit_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
