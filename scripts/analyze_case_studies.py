#!/usr/bin/env python3
"""
Automatically select representative strict cold-start recommendation cases.

The script selects up to six complementary cases:
1. Graph+BGE succeeds while Graph-only fails.
2. Graph+BGE retrieves a Middle positive API.
3. Graph+BGE retrieves a Tail positive API.
4. BGE-only supplies useful semantic evidence while Graph-only fails.
5. Graph-only collapses to Popularity while Graph+BGE improves the ranking.
6. A Mashup containing an Unseen positive API remains unsolved.

Selection uses all three random seeds. Displayed recommendation lists are
consensus Top-10 rankings aggregated across seeds by reciprocal-rank voting.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import yaml


METHODS = {
    "Popularity": ("graph_bge_zscore", "Popularity"),
    "Graph-only": ("graph_only", "Ours"),
    "BGE-only": ("bge_only", "Ours"),
    "Graph+BGE": ("graph_bge_zscore", "Ours"),
}

CASE_LABELS = {
    "fusion_beats_graph": "Graph+BGE succeeds while Graph-only fails",
    "middle_success": "Graph+BGE retrieves a Middle positive API",
    "tail_success": "Graph+BGE retrieves a Tail positive API",
    "bge_semantic_support": "BGE-only provides useful semantic evidence",
    "graph_popularity_collapse": "Graph-only collapses to Popularity",
    "unseen_failure": "Unseen API remains unsolved",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/statistical_analysis/runs"),
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        default=Path("outputs/case_studies/ground_truth.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/case_studies/summary"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config used only for metadata discovery.",
    )
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def load_yaml(path: Optional[Path]) -> Dict:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data if isinstance(data, dict) else {}


def load_runs(root: Path, k: int):
    rankings: Dict[str, Dict[int, pd.DataFrame]] = defaultdict(dict)
    metrics: Dict[str, Dict[int, pd.DataFrame]] = defaultdict(dict)
    api_groups: Optional[pd.DataFrame] = None

    for method_name, (method_dir, source_method) in METHODS.items():
        for seed in (0, 1, 2):
            run_dir = root / method_dir / f"seed{seed}"
            ranking_path = run_dir / "rankings_topk.csv"
            metric_path = run_dir / "per_mashup_metrics.csv"
            group_path = run_dir / "api_groups.csv"

            for path in (ranking_path, metric_path, group_path):
                if not path.exists():
                    raise FileNotFoundError(path)

            ranking = pd.read_csv(ranking_path)
            ranking = ranking[
                (ranking["method"] == source_method)
                & (ranking["rank"] <= k)
            ].copy()
            ranking["mashup_id"] = ranking["mashup_id"].astype(int)
            ranking["api_id"] = ranking["api_id"].astype(int)
            ranking["rank"] = ranking["rank"].astype(int)
            rankings[method_name][seed] = ranking

            metric = pd.read_csv(metric_path)
            metric = metric[metric["method"] == source_method].copy()
            metric["mashup_id"] = metric["mashup_id"].astype(int)
            metrics[method_name][seed] = metric

            current_groups = pd.read_csv(group_path).sort_values("api_id")
            if api_groups is None:
                api_groups = current_groups
            else:
                columns = ["api_id", "train_frequency", "group"]
                if not api_groups[columns].reset_index(drop=True).equals(
                    current_groups[columns].reset_index(drop=True)
                ):
                    raise ValueError(
                        f"API groups differ for {method_name}, seed={seed}"
                    )

    assert api_groups is not None
    return rankings, metrics, api_groups


def average_metrics(
    metrics: Mapping[str, Mapping[int, pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    result = {}
    for method_name, seed_frames in metrics.items():
        merged = pd.concat(
            [
                frame.assign(seed=seed)
                for seed, frame in seed_frames.items()
            ],
            ignore_index=True,
        )
        numeric = [
            column
            for column in merged.columns
            if column
            not in {"method", "mashup_id", "num_positives", "seed"}
        ]
        averaged = (
            merged.groupby("mashup_id", as_index=False)
            .agg(
                num_positives=("num_positives", "first"),
                **{column: (column, "mean") for column in numeric},
            )
            .sort_values("mashup_id")
        )
        result[method_name] = averaged
    return result


def build_rank_dict(frame: pd.DataFrame) -> Dict[int, Tuple[int, ...]]:
    result = {}
    for mashup_id, group in frame.sort_values(
        ["mashup_id", "rank"]
    ).groupby("mashup_id"):
        result[int(mashup_id)] = tuple(
            int(value) for value in group["api_id"].tolist()
        )
    return result


def consensus_rankings(
    rankings: Mapping[str, Mapping[int, pd.DataFrame]],
    k: int,
) -> Tuple[
    Dict[str, Dict[int, List[int]]],
    Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
]:
    consensus: Dict[str, Dict[int, List[int]]] = defaultdict(dict)
    details: Dict[
        str, Dict[int, Dict[int, Dict[str, float]]]
    ] = defaultdict(dict)

    for method_name, seed_frames in rankings.items():
        per_seed = {
            seed: build_rank_dict(frame)
            for seed, frame in seed_frames.items()
        }
        mashup_ids = sorted(set.intersection(
            *(set(value.keys()) for value in per_seed.values())
        ))

        for mashup_id in mashup_ids:
            scores = defaultdict(float)
            ranks = defaultdict(list)
            appearances = defaultdict(int)

            for seed in (0, 1, 2):
                for rank, api_id in enumerate(
                    per_seed[seed][mashup_id],
                    start=1,
                ):
                    scores[api_id] += 1.0 / rank
                    ranks[api_id].append(rank)
                    appearances[api_id] += 1

            ordered = sorted(
                scores,
                key=lambda api_id: (
                    -scores[api_id],
                    -appearances[api_id],
                    np.mean(ranks[api_id]),
                    api_id,
                ),
            )[:k]

            consensus[method_name][mashup_id] = ordered
            details[method_name][mashup_id] = {
                api_id: {
                    "reciprocal_rank_score": float(scores[api_id]),
                    "seed_appearances": int(appearances[api_id]),
                    "mean_rank_when_present": float(
                        np.mean(ranks[api_id])
                    ),
                }
                for api_id in ordered
            }

    return consensus, details


def ranking_exact_match_counts(
    rankings: Mapping[str, Mapping[int, pd.DataFrame]],
) -> Dict[int, int]:
    counts = defaultdict(int)
    for seed in (0, 1, 2):
        graph = build_rank_dict(rankings["Graph-only"][seed])
        popularity = build_rank_dict(rankings["Popularity"][seed])
        for mashup_id in graph:
            if graph[mashup_id] == popularity[mashup_id]:
                counts[mashup_id] += 1
    return dict(counts)


def mean_metric_lookup(
    averaged: Mapping[str, pd.DataFrame],
    method: str,
    metric: str,
) -> Dict[int, float]:
    """
    Build a Mashup-to-metric lookup without using tuple attributes.

    Metric column names such as ``NDCG@10`` and ``Recall@10`` are valid
    pandas column names, but they are not valid Python attribute names.
    Therefore ``getattr(row, metric)`` cannot be used on itertuples().
    """
    frame = averaged[method]

    if "mashup_id" not in frame.columns:
        raise KeyError(
            f"Missing mashup_id column for method={method!r}"
        )
    if metric not in frame.columns:
        raise KeyError(
            f"Missing metric column {metric!r} for method={method!r}. "
            f"Available columns: {list(frame.columns)}"
        )

    return {
        int(mashup_id): float(metric_value)
        for mashup_id, metric_value in zip(
            frame["mashup_id"].to_numpy(),
            frame[metric].to_numpy(),
        )
    }


def positive_sets(
    ground_truth: pd.DataFrame,
) -> Tuple[Dict[int, Set[int]], Dict[int, Dict[str, Set[int]]]]:
    all_positive: Dict[int, Set[int]] = defaultdict(set)
    by_group: Dict[int, Dict[str, Set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in ground_truth.itertuples(index=False):
        mashup_id = int(row.mashup_id)
        api_id = int(row.api_id)
        group = str(row.api_group)
        all_positive[mashup_id].add(api_id)
        by_group[mashup_id][group].add(api_id)
    return dict(all_positive), by_group


def consensus_hits(
    consensus: Mapping[str, Mapping[int, Sequence[int]]],
    positives: Mapping[int, Set[int]],
) -> Dict[str, Dict[int, Set[int]]]:
    result = defaultdict(dict)
    for method, rankings in consensus.items():
        for mashup_id, api_ids in rankings.items():
            result[method][mashup_id] = (
                set(api_ids) & positives[mashup_id]
            )
    return result


def choose_best(
    candidates: Iterable[int],
    score_fn,
    used: Set[int],
) -> Tuple[Optional[int], bool]:
    candidates = list(dict.fromkeys(int(value) for value in candidates))
    unused = [value for value in candidates if value not in used]
    relaxed = False
    pool = unused
    if not pool:
        pool = candidates
        relaxed = True
    if not pool:
        return None, relaxed
    return max(pool, key=score_fn), relaxed


def discover_mapping(
    search_roots: Sequence[Path],
    entity: str,
) -> Dict[int, Dict[str, str]]:
    """
    Best-effort metadata discovery. Failure is harmless: reports still use IDs.
    """
    candidates: List[Path] = []
    keywords = ("metadata", "mapping", "names", "id_map", "items")
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            lower = path.name.lower()
            if (
                path.is_file()
                and entity in lower
                and path.suffix.lower() in {".csv", ".json"}
                and any(keyword in lower for keyword in keywords)
            ):
                candidates.append(path)

    id_aliases = [
        f"{entity}_id",
        f"{entity}_idx",
        "id",
        "index",
        "idx",
    ]
    name_aliases = [
        f"{entity}_name",
        "name",
        "title",
        "display_name",
    ]
    description_aliases = [
        "description",
        "desc",
        "summary",
        "text",
    ]

    for path in sorted(candidates, key=lambda value: len(str(value))):
        try:
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
                lower_map = {
                    str(column).lower(): column
                    for column in frame.columns
                }
                id_column = next(
                    (
                        lower_map[alias]
                        for alias in id_aliases
                        if alias in lower_map
                    ),
                    None,
                )
                name_column = next(
                    (
                        lower_map[alias]
                        for alias in name_aliases
                        if alias in lower_map
                    ),
                    None,
                )
                desc_column = next(
                    (
                        lower_map[alias]
                        for alias in description_aliases
                        if alias in lower_map
                    ),
                    None,
                )
                if id_column is None or name_column is None:
                    continue

                mapping = {}
                for row in frame.itertuples(index=False):
                    record = row._asdict()
                    try:
                        internal_id = int(record[id_column])
                    except Exception:
                        continue
                    mapping[internal_id] = {
                        "name": str(record.get(name_column, "")),
                        "description": str(
                            record.get(desc_column, "")
                            if desc_column is not None
                            else ""
                        ),
                    }
                if mapping:
                    print(f"[Metadata] loaded {entity} metadata from {path}")
                    return mapping

            else:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                mapping = {}
                for key, value in data.items():
                    try:
                        internal_id = int(key)
                        direct = True
                    except Exception:
                        direct = False

                    if direct:
                        if isinstance(value, dict):
                            name = (
                                value.get("name")
                                or value.get("title")
                                or value.get("display_name")
                                or ""
                            )
                            description = (
                                value.get("description")
                                or value.get("desc")
                                or value.get("summary")
                                or ""
                            )
                        else:
                            name = value
                            description = ""
                        mapping[internal_id] = {
                            "name": str(name),
                            "description": str(description),
                        }
                    else:
                        try:
                            internal_id = int(value)
                        except Exception:
                            continue
                        mapping[internal_id] = {
                            "name": str(key),
                            "description": "",
                        }
                if mapping:
                    print(f"[Metadata] loaded {entity} metadata from {path}")
                    return mapping
        except Exception:
            continue

    return {}


def md_escape(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ").strip()


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            + " | ".join(md_escape(row[column]) for column in columns)
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = pd.read_csv(args.ground_truth.resolve())
    required_gt = {
        "mashup_id",
        "api_id",
        "api_group",
        "train_frequency",
    }
    missing = required_gt - set(ground_truth.columns)
    if missing:
        raise ValueError(
            f"Ground-truth file is missing: {sorted(missing)}"
        )

    rankings, metrics, api_groups = load_runs(root, args.k)
    averaged = average_metrics(metrics)
    consensus, consensus_details = consensus_rankings(rankings, args.k)
    exact_match_counts = ranking_exact_match_counts(rankings)
    positives, positives_by_group = positive_sets(ground_truth)
    hits = consensus_hits(consensus, positives)

    mashup_ids = sorted(positives)
    ndcg = {
        method: mean_metric_lookup(averaged, method, "NDCG@10")
        for method in METHODS
    }
    recall = {
        method: mean_metric_lookup(averaged, method, "Recall@10")
        for method in METHODS
    }
    hitrate = {
        method: mean_metric_lookup(averaged, method, "HitRate@10")
        for method in METHODS
    }

    used: Set[int] = set()
    selected: List[Dict] = []

    # 1. Fusion succeeds while Graph-only fails.
    candidates = [
        m
        for m in mashup_ids
        if hitrate["Graph+BGE"][m] >= 2.0 / 3.0
        and hitrate["Graph-only"][m] == 0.0
    ]
    chosen, relaxed = choose_best(
        candidates,
        lambda m: (
            ndcg["Graph+BGE"][m] - ndcg["Graph-only"][m],
            recall["Graph+BGE"][m],
        ),
        used,
    )
    if chosen is not None:
        used.add(chosen)
        selected.append(
            {
                "case_type": "fusion_beats_graph",
                "mashup_id": chosen,
                "selection_relaxed": relaxed,
                "reason": (
                    "Graph+BGE hits relevant APIs in at least two seeds, "
                    "while Graph-only misses in all seeds."
                ),
            }
        )

    # 2. Middle success.
    candidates = [
        m
        for m in mashup_ids
        if hits["Graph+BGE"][m]
        & positives_by_group[m].get("Middle", set())
    ]
    chosen, relaxed = choose_best(
        candidates,
        lambda m: (
            len(
                hits["Graph+BGE"][m]
                & positives_by_group[m].get("Middle", set())
            ),
            ndcg["Graph+BGE"][m],
            ndcg["Graph+BGE"][m] - ndcg["Graph-only"][m],
        ),
        used,
    )
    if chosen is not None:
        used.add(chosen)
        selected.append(
            {
                "case_type": "middle_success",
                "mashup_id": chosen,
                "selection_relaxed": relaxed,
                "reason": (
                    "The consensus Graph+BGE Top-10 contains a relevant "
                    "Middle API."
                ),
            }
        )

    # 3. Tail success.
    candidates = [
        m
        for m in mashup_ids
        if hits["Graph+BGE"][m]
        & positives_by_group[m].get("Tail", set())
    ]
    chosen, relaxed = choose_best(
        candidates,
        lambda m: (
            len(
                hits["Graph+BGE"][m]
                & positives_by_group[m].get("Tail", set())
            ),
            ndcg["Graph+BGE"][m],
        ),
        used,
    )
    if chosen is not None:
        used.add(chosen)
        selected.append(
            {
                "case_type": "tail_success",
                "mashup_id": chosen,
                "selection_relaxed": relaxed,
                "reason": (
                    "The consensus Graph+BGE Top-10 contains a relevant "
                    "Tail API."
                ),
            }
        )

    # 4. BGE semantic support.
    candidates = [
        m
        for m in mashup_ids
        if hitrate["BGE-only"][m] > hitrate["Graph-only"][m]
        and hitrate["Graph+BGE"][m] > 0.0
    ]
    chosen, relaxed = choose_best(
        candidates,
        lambda m: (
            hitrate["BGE-only"][m] - hitrate["Graph-only"][m],
            ndcg["Graph+BGE"][m] - ndcg["Graph-only"][m],
            ndcg["Graph+BGE"][m],
        ),
        used,
    )
    if chosen is not None:
        used.add(chosen)
        selected.append(
            {
                "case_type": "bge_semantic_support",
                "mashup_id": chosen,
                "selection_relaxed": relaxed,
                "reason": (
                    "BGE-only finds relevant semantic evidence that "
                    "Graph-only misses; fusion retains useful evidence."
                ),
            }
        )

    # 5. Graph collapses to popularity.
    candidates = [
        m
        for m in mashup_ids
        if exact_match_counts.get(m, 0) == 3
        and ndcg["Graph+BGE"][m] > ndcg["Graph-only"][m]
    ]
    chosen, relaxed = choose_best(
        candidates,
        lambda m: (
            ndcg["Graph+BGE"][m] - ndcg["Graph-only"][m],
            ndcg["Graph+BGE"][m],
        ),
        used,
    )
    if chosen is not None:
        used.add(chosen)
        selected.append(
            {
                "case_type": "graph_popularity_collapse",
                "mashup_id": chosen,
                "selection_relaxed": relaxed,
                "reason": (
                    "Graph-only exactly matches Popularity Top-10 in all "
                    "three seeds, while fusion improves NDCG@10."
                ),
            }
        )

    # 6. Unseen failure.
    candidates = [
        m
        for m in mashup_ids
        if positives_by_group[m].get("Unseen", set())
        and all(hitrate[method][m] == 0.0 for method in METHODS)
    ]
    chosen, relaxed = choose_best(
        candidates,
        lambda m: (
            len(positives_by_group[m].get("Unseen", set())),
            len(positives[m]),
        ),
        used,
    )
    if chosen is not None:
        used.add(chosen)
        selected.append(
            {
                "case_type": "unseen_failure",
                "mashup_id": chosen,
                "selection_relaxed": relaxed,
                "reason": (
                    "The Mashup contains an Unseen positive API and all "
                    "methods miss every positive API at Top-10."
                ),
            }
        )

    if not selected:
        raise RuntimeError("No case could be selected")

    config = load_yaml(args.config.resolve() if args.config else None)
    search_roots = [Path.cwd()]
    processed_dir = config.get("processed_dir")
    if processed_dir:
        search_roots.insert(0, Path(processed_dir))
    search_roots.extend(
        [
            Path("data_unified/mtfm_pw"),
            Path("data_unified"),
            Path("data"),
        ]
    )
    mashup_meta = discover_mapping(search_roots, "mashup")
    api_meta = discover_mapping(search_roots, "api")

    api_group_lookup = {
        int(row.api_id): str(row.group)
        for row in api_groups.itertuples(index=False)
    }
    api_frequency_lookup = {
        int(row.api_id): int(row.train_frequency)
        for row in api_groups.itertuples(index=False)
    }

    selected_rows = []
    ranking_rows = []
    ground_truth_rows = []

    for selected_case in selected:
        case_type = selected_case["case_type"]
        mashup_id = int(selected_case["mashup_id"])
        meta = mashup_meta.get(
            mashup_id,
            {"name": "", "description": ""},
        )

        row = dict(selected_case)
        row.update(
            {
                "case_label": CASE_LABELS[case_type],
                "mashup_name": meta.get("name", ""),
                "mashup_description": meta.get("description", ""),
                "num_positives": len(positives[mashup_id]),
                "positive_api_ids": ";".join(
                    str(value)
                    for value in sorted(positives[mashup_id])
                ),
                "positive_groups": ";".join(
                    sorted(
                        {
                            api_group_lookup[value]
                            for value in positives[mashup_id]
                        }
                    )
                ),
                "graph_bge_recall10": recall["Graph+BGE"][mashup_id],
                "graph_bge_ndcg10": ndcg["Graph+BGE"][mashup_id],
                "graph_only_recall10": recall["Graph-only"][mashup_id],
                "graph_only_ndcg10": ndcg["Graph-only"][mashup_id],
                "bge_only_recall10": recall["BGE-only"][mashup_id],
                "bge_only_ndcg10": ndcg["BGE-only"][mashup_id],
                "popularity_recall10": recall["Popularity"][mashup_id],
                "popularity_ndcg10": ndcg["Popularity"][mashup_id],
                "graph_popularity_exact_seed_count": exact_match_counts.get(
                    mashup_id,
                    0,
                ),
            }
        )
        selected_rows.append(row)

        for api_id in sorted(positives[mashup_id]):
            api_metadata = api_meta.get(
                api_id,
                {"name": "", "description": ""},
            )
            ground_truth_rows.append(
                {
                    "case_type": case_type,
                    "case_label": CASE_LABELS[case_type],
                    "mashup_id": mashup_id,
                    "mashup_name": meta.get("name", ""),
                    "api_id": api_id,
                    "api_name": api_metadata.get("name", ""),
                    "api_description": api_metadata.get(
                        "description",
                        "",
                    ),
                    "api_group": api_group_lookup[api_id],
                    "train_frequency": api_frequency_lookup[api_id],
                }
            )

        for method in METHODS:
            for rank, api_id in enumerate(
                consensus[method][mashup_id],
                start=1,
            ):
                detail = consensus_details[method][mashup_id][api_id]
                api_metadata = api_meta.get(
                    api_id,
                    {"name": "", "description": ""},
                )
                ranking_rows.append(
                    {
                        "case_type": case_type,
                        "case_label": CASE_LABELS[case_type],
                        "mashup_id": mashup_id,
                        "mashup_name": meta.get("name", ""),
                        "method": method,
                        "rank": rank,
                        "api_id": api_id,
                        "api_name": api_metadata.get("name", ""),
                        "api_description": api_metadata.get(
                            "description",
                            "",
                        ),
                        "api_group": api_group_lookup[api_id],
                        "train_frequency": api_frequency_lookup[api_id],
                        "is_positive": int(api_id in positives[mashup_id]),
                        "reciprocal_rank_score": detail[
                            "reciprocal_rank_score"
                        ],
                        "seed_appearances": detail["seed_appearances"],
                        "mean_rank_when_present": detail[
                            "mean_rank_when_present"
                        ],
                    }
                )

    selected_df = pd.DataFrame(selected_rows)
    rankings_df = pd.DataFrame(ranking_rows)
    case_ground_truth_df = pd.DataFrame(ground_truth_rows)

    selected_df.to_csv(output_dir / "cases_selected.csv", index=False)
    rankings_df.to_csv(output_dir / "case_rankings.csv", index=False)
    case_ground_truth_df.to_csv(
        output_dir / "case_ground_truth.csv",
        index=False,
    )

    report: List[str] = [
        "# Strict Cold-Start Recommendation Case Studies",
        "",
        "Recommendation lists are consensus Top-10 rankings aggregated over "
        "seeds 0/1/2 using reciprocal-rank voting.",
        "",
    ]

    for selected_case in selected_rows:
        case_type = selected_case["case_type"]
        mashup_id = int(selected_case["mashup_id"])
        title = selected_case.get("mashup_name") or f"Mashup {mashup_id}"

        report += [
            f"## {CASE_LABELS[case_type]}",
            "",
            f"**Mashup:** {md_escape(title)} (`{mashup_id}`)",
            "",
        ]
        description = str(
            selected_case.get("mashup_description", "")
        ).strip()
        if description:
            report += [
                f"**Description:** {md_escape(description)}",
                "",
            ]

        report += [
            f"**Selection reason:** {selected_case['reason']}",
            "",
            (
                "**Mean metrics:** "
                f"Graph+BGE R@10={selected_case['graph_bge_recall10']:.4f}, "
                f"NDCG@10={selected_case['graph_bge_ndcg10']:.4f}; "
                f"Graph-only R@10={selected_case['graph_only_recall10']:.4f}, "
                f"NDCG@10={selected_case['graph_only_ndcg10']:.4f}; "
                f"BGE-only R@10={selected_case['bge_only_recall10']:.4f}, "
                f"NDCG@10={selected_case['bge_only_ndcg10']:.4f}."
            ),
            "",
            "### Ground truth",
            "",
        ]

        gt_table = case_ground_truth_df[
            case_ground_truth_df["case_type"] == case_type
        ][
            [
                "api_id",
                "api_name",
                "api_group",
                "train_frequency",
            ]
        ].rename(
            columns={
                "api_id": "API ID",
                "api_name": "API name",
                "api_group": "Group",
                "train_frequency": "Train frequency",
            }
        )
        report += [markdown_table(gt_table), ""]

        for method in METHODS:
            method_table = rankings_df[
                (rankings_df["case_type"] == case_type)
                & (rankings_df["method"] == method)
            ][
                [
                    "rank",
                    "api_id",
                    "api_name",
                    "api_group",
                    "train_frequency",
                    "is_positive",
                    "seed_appearances",
                ]
            ].rename(
                columns={
                    "rank": "Rank",
                    "api_id": "API ID",
                    "api_name": "API name",
                    "api_group": "Group",
                    "train_frequency": "Train freq.",
                    "is_positive": "Hit",
                    "seed_appearances": "Seeds",
                }
            )
            method_table["Hit"] = method_table["Hit"].map(
                {1: "✓", 0: ""}
            )
            report += [
                f"### {method}",
                "",
                markdown_table(method_table),
                "",
            ]

    report += [
        "## Reading guide",
        "",
        "- `Seeds` is the number of random seeds in which an API appeared "
        "in that method's Top-10.",
        "- A check mark means the recommended API belongs to the Mashup's "
        "ground-truth positive set.",
        "- Empty API names mean no compatible internal-ID metadata mapping "
        "was discovered; IDs and quantitative conclusions remain valid.",
        "",
    ]

    (output_dir / "case_study_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\nSelected cases:")
    display_columns = [
        "case_type",
        "mashup_id",
        "mashup_name",
        "positive_groups",
        "graph_bge_ndcg10",
        "graph_only_ndcg10",
        "bge_only_ndcg10",
    ]
    print(selected_df[display_columns].to_string(index=False))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
