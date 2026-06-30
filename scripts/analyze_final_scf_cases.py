#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from strict_baseline_core import load_strict_data, load_yaml


METHOD_ORDER = [
    "BGE-only",
    "Graph+BGE",
    "Inductive LightGCN",
    "SCF-LightGCN+BGE",
]

CASE_LABELS = {
    "middle_rescue": "Middle API 语义协同补救案例",
    "tail_rescue": "Tail API 语义协同补救案例",
    "unseen_rescue": "训练交互未见 API 零样式补救案例",
    "complex_failure": "复杂多 API 失败案例",
}

GROUP_ORDER = {
    "Head": 0,
    "Middle": 1,
    "Tail": 2,
    "Unseen": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select final paper-ready SCF cases from all three seeds. "
            "The script uses reciprocal-rank consensus rather than cherry-picking "
            "a single seed."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final_scf_case_study.yaml"),
    )
    return parser.parse_args()


def resolve_path(value: str, description: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def slug_to_display(value: str, fallback: str) -> str:
    text = str(value).strip()
    if not text:
        return fallback
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return fallback
    return " ".join(
        token if any(char.isupper() for char in token[1:]) else token.capitalize()
        for token in text.split()
    )


def safe_reverse_lookup(container: Any, index: int, fallback: str) -> str:
    if isinstance(container, Mapping):
        for key in (index, str(index)):
            if key in container:
                return str(container[key])
        # Some saved maps are original-id -> internal-id.
        for key, value in container.items():
            try:
                if int(value) == index:
                    return str(key)
            except Exception:
                continue
        return fallback
    if isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
        if 0 <= index < len(container):
            return str(container[index])
    return fallback


def load_entity_names(processed_dir: Path, num_mashups: int, num_apis: int):
    reverse_path = processed_dir / "reverse_node_mapping.pkl"
    mashup_names = {
        index: f"Mashup {index}" for index in range(num_mashups)
    }
    api_names = {
        index: f"API {index}" for index in range(num_apis)
    }

    if not reverse_path.exists():
        return mashup_names, api_names, None

    with reverse_path.open("rb") as file:
        reverse = pickle.load(file)

    mashup_container = None
    api_container = None
    if isinstance(reverse, Mapping):
        mashup_container = (
            reverse.get("mashup")
            or reverse.get("mashups")
            or reverse.get("mashup_reverse")
        )
        api_container = (
            reverse.get("api")
            or reverse.get("apis")
            or reverse.get("api_reverse")
        )

    if mashup_container is not None:
        for index in range(num_mashups):
            raw = safe_reverse_lookup(
                mashup_container,
                index,
                f"Mashup {index}",
            )
            mashup_names[index] = slug_to_display(
                raw,
                f"Mashup {index}",
            )

    if api_container is not None:
        for index in range(num_apis):
            raw = safe_reverse_lookup(
                api_container,
                index,
                f"API {index}",
            )
            api_names[index] = slug_to_display(
                raw,
                f"API {index}",
            )

    return mashup_names, api_names, reverse_path


def normalize_method(value: str) -> str:
    text = str(value).strip()
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())

    if compact.startswith("scf") or (
        "lightgcn" in compact and "bge" in compact
    ):
        return "SCF-LightGCN+BGE"
    if "lightgcn" in compact:
        return "Inductive LightGCN"
    if compact in {"bge", "bgeonly"}:
        return "BGE-only"
    if (
        "graphbge" in compact
        or ("graph" in compact and "bge" in compact)
        or "zscore" in compact
        or "fusion" in compact
    ):
        return "Graph+BGE"
    if compact == "ours":
        return "Ours"
    return text


def load_rankings(
    *,
    method: str,
    seed: int,
    config: Mapping[str, Any],
    k: int,
) -> pd.DataFrame:
    if method == "BGE-only":
        template = str(config["bge_rankings_template"])
        source_label = "Ours"
    elif method == "Graph+BGE":
        template = str(config["graph_bge_rankings_template"])
        source_label = "Ours"
    elif method == "Inductive LightGCN":
        template = str(config["lightgcn_rankings_template"])
        source_label = "Inductive LightGCN"
    elif method == "SCF-LightGCN+BGE":
        template = str(config["scf_rankings_template"])
        source_label = None
    else:
        raise ValueError(f"Unsupported method: {method}")

    path = resolve_path(
        template.format(seed=seed),
        f"{method} seed-{seed} ranking file",
    )
    frame = pd.read_csv(path)
    required = {"mashup_id", "rank", "api_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    if "method" in frame.columns:
        normalized = frame["method"].map(normalize_method)
        if source_label is not None:
            source_normalized = normalize_method(source_label)
            selected = frame[
                (normalized == method)
                | (normalized == source_normalized)
            ].copy()
        else:
            selected = frame[
                normalized == method
            ].copy()
            if selected.empty:
                selected = frame.copy()
        frame = selected

    if frame.empty:
        raise ValueError(f"No rows for {method} in {path}")

    frame["mashup_id"] = frame["mashup_id"].astype(int)
    frame["rank"] = frame["rank"].astype(int)
    frame["api_id"] = frame["api_id"].astype(int)
    frame = frame[frame["rank"] <= k].copy()
    frame["method"] = method
    frame["seed"] = seed

    duplicates = frame.duplicated(
        ["mashup_id", "rank"],
        keep=False,
    )
    if duplicates.any():
        raise ValueError(
            f"Duplicate Mashup/rank rows found in {path}"
        )

    return frame[
        ["method", "seed", "mashup_id", "rank", "api_id"]
    ].sort_values(["mashup_id", "rank"])


def build_seed_rank_lookup(
    rankings: Mapping[str, Mapping[int, pd.DataFrame]],
) -> Dict[str, Dict[int, Dict[int, Dict[int, int]]]]:
    result: Dict[str, Dict[int, Dict[int, Dict[int, int]]]] = defaultdict(dict)
    for method, seed_frames in rankings.items():
        for seed, frame in seed_frames.items():
            mashup_map: Dict[int, Dict[int, int]] = {}
            for mashup_id, group in frame.groupby("mashup_id", sort=False):
                mashup_map[int(mashup_id)] = {
                    int(row.api_id): int(row.rank)
                    for row in group.itertuples(index=False)
                }
            result[method][seed] = mashup_map
    return result


def build_consensus(
    rankings: Mapping[str, Mapping[int, pd.DataFrame]],
    k: int,
):
    consensus: Dict[str, Dict[int, List[int]]] = defaultdict(dict)
    details: Dict[
        str,
        Dict[int, Dict[int, Dict[str, float]]],
    ] = defaultdict(dict)

    seed_lookup = build_seed_rank_lookup(rankings)

    for method in METHOD_ORDER:
        seed_maps = seed_lookup[method]
        common_mashups = set.intersection(
            *(set(seed_maps[seed]) for seed in (0, 1, 2))
        )

        for mashup_id in sorted(common_mashups):
            rr_score: Dict[int, float] = defaultdict(float)
            ranks: Dict[int, List[int]] = defaultdict(list)

            for seed in (0, 1, 2):
                for api_id, rank in seed_maps[seed][mashup_id].items():
                    rr_score[api_id] += 1.0 / rank
                    ranks[api_id].append(rank)

            ordered = sorted(
                rr_score,
                key=lambda api_id: (
                    -rr_score[api_id],
                    -len(ranks[api_id]),
                    float(np.mean(ranks[api_id])),
                    api_id,
                ),
            )[:k]

            consensus[method][mashup_id] = ordered
            details[method][mashup_id] = {
                api_id: {
                    "consensus_rank": rank,
                    "reciprocal_rank_score": float(rr_score[api_id]),
                    "seed_appearances": int(len(ranks[api_id])),
                    "mean_rank_when_present": float(
                        np.mean(ranks[api_id])
                    ),
                    "best_seed_rank": int(min(ranks[api_id])),
                }
                for rank, api_id in enumerate(ordered, start=1)
            }

    return consensus, details, seed_lookup


def positives_by_mashup(pairs: np.ndarray) -> Dict[int, Set[int]]:
    result: Dict[int, Set[int]] = defaultdict(set)
    for mashup_id, api_id in pairs:
        result[int(mashup_id)].add(int(api_id))
    return dict(result)


def load_api_groups(path: Path, num_apis: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"api_id", "train_frequency", "group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    frame = frame[["api_id", "train_frequency", "group"]].copy()
    frame["api_id"] = frame["api_id"].astype(int)
    frame["train_frequency"] = frame["train_frequency"].astype(int)
    frame["group"] = frame["group"].astype(str)

    if frame["api_id"].nunique() != num_apis:
        raise ValueError(
            f"API group rows ({frame['api_id'].nunique()}) "
            f"do not match catalog size ({num_apis})"
        )

    return frame.sort_values("api_id").reset_index(drop=True)


def load_strict_clean_ids(
    audit_path: Optional[Path],
    threshold: float,
) -> Optional[Set[int]]:
    if audit_path is None:
        return None
    audit = pd.read_csv(audit_path)
    required = {
        "mashup_id",
        "exact_train_text_duplicate",
        "exact_train_name_duplicate",
        "direct_api_name_mention_count",
        "top1_cosine_similarity",
    }
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(
            f"{audit_path} is missing columns: {sorted(missing)}"
        )

    mask = (
        (audit["exact_train_text_duplicate"] == 0)
        & (audit["exact_train_name_duplicate"] == 0)
        & (audit["direct_api_name_mention_count"] == 0)
        & (audit["top1_cosine_similarity"] < threshold)
    )
    return set(
        audit.loc[mask, "mashup_id"].astype(int).tolist()
    )


def find_positive_rank(
    *,
    method: str,
    mashup_id: int,
    api_id: int,
    details,
) -> Optional[int]:
    record = details.get(method, {}).get(mashup_id, {}).get(api_id)
    if record is None:
        return None
    return int(record["consensus_rank"])


def seed_appearance_count(
    *,
    method: str,
    mashup_id: int,
    api_id: int,
    seed_lookup,
) -> int:
    return sum(
        int(api_id in seed_lookup[method][seed].get(mashup_id, {}))
        for seed in (0, 1, 2)
    )


def best_rank_across_seeds(
    *,
    method: str,
    mashup_id: int,
    api_id: int,
    seed_lookup,
) -> Optional[int]:
    ranks = [
        seed_lookup[method][seed][mashup_id][api_id]
        for seed in (0, 1, 2)
        if mashup_id in seed_lookup[method][seed]
        and api_id in seed_lookup[method][seed][mashup_id]
    ]
    return min(ranks) if ranks else None



GENERIC_API_NAME_TOKENS = {
    "api", "apis", "service", "services", "web", "app", "application",
    "platform", "official", "developer", "developers",
}


def normalize_name_tokens(value: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(value).lower())
        if token not in GENERIC_API_NAME_TOKENS
    ]


def direct_api_name_match(
    *,
    mashup_name: str,
    mashup_text: str,
    api_name: str,
) -> Tuple[bool, str]:
    """Conservative exact-name cue check, including short names such as Moo."""
    tokens = normalize_name_tokens(api_name)
    if not tokens:
        return False, ""

    haystack = f"{mashup_name} {mashup_text}".lower()

    if len(tokens) == 1:
        token = tokens[0]
        if len(token) < 3:
            return False, ""
        pattern = rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])"
        match = re.search(pattern, haystack)
        return (match is not None, match.group(0) if match else "")

    separator = r"[\s_\-:/\.]*"
    pattern = (
        r"(?<![a-z0-9])"
        + separator.join(re.escape(token) for token in tokens)
        + r"(?![a-z0-9])"
    )
    match = re.search(pattern, haystack)
    if match:
        return True, match.group(0)

    compact_name = "".join(tokens)
    compact_haystack = re.sub(r"[^a-z0-9]+", "", haystack)
    if len(compact_name) >= 4 and compact_name in compact_haystack:
        return True, compact_name

    return False, ""


def candidate_rescues(
    *,
    target_group: str,
    mashup_ids: Iterable[int],
    positives: Mapping[int, Set[int]],
    group_by_api: Mapping[int, str],
    details,
    seed_lookup,
    strict_clean_ids: Optional[Set[int]],
    mashup_names: Mapping[int, str],
    api_names: Mapping[int, str],
    mashup_texts: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for mashup_id in mashup_ids:
        for api_id in positives[mashup_id]:
            if group_by_api[api_id] != target_group:
                continue

            direct_mention, direct_match = direct_api_name_match(
                mashup_name=mashup_names[mashup_id],
                mashup_text=mashup_texts[mashup_id],
                api_name=api_names[api_id],
            )

            scf_appearances = seed_appearance_count(
                method="SCF-LightGCN+BGE",
                mashup_id=mashup_id,
                api_id=api_id,
                seed_lookup=seed_lookup,
            )
            lightgcn_appearances = seed_appearance_count(
                method="Inductive LightGCN",
                mashup_id=mashup_id,
                api_id=api_id,
                seed_lookup=seed_lookup,
            )
            graph_appearances = seed_appearance_count(
                method="Graph+BGE",
                mashup_id=mashup_id,
                api_id=api_id,
                seed_lookup=seed_lookup,
            )
            bge_appearances = seed_appearance_count(
                method="BGE-only",
                mashup_id=mashup_id,
                api_id=api_id,
                seed_lookup=seed_lookup,
            )

            if scf_appearances == 0:
                continue

            scf_rank = find_positive_rank(
                method="SCF-LightGCN+BGE",
                mashup_id=mashup_id,
                api_id=api_id,
                details=details,
            )
            lightgcn_rank = find_positive_rank(
                method="Inductive LightGCN",
                mashup_id=mashup_id,
                api_id=api_id,
                details=details,
            )
            graph_rank = find_positive_rank(
                method="Graph+BGE",
                mashup_id=mashup_id,
                api_id=api_id,
                details=details,
            )
            bge_rank = find_positive_rank(
                method="BGE-only",
                mashup_id=mashup_id,
                api_id=api_id,
                details=details,
            )

            is_clean = (
                strict_clean_ids is not None
                and mashup_id in strict_clean_ids
            )

            # Selection levels, lower is stronger/more defensible.
            if (
                is_clean
                and scf_appearances >= 2
                and lightgcn_appearances == 0
                and graph_appearances == 0
            ):
                level = 0
                rule = (
                    "严格清洗子集；SCF 至少 2/3 seed 命中；"
                    "LightGCN 与 Graph+BGE 均未命中"
                )
            elif (
                is_clean
                and scf_appearances >= 2
                and lightgcn_appearances == 0
            ):
                level = 1
                rule = (
                    "严格清洗子集；SCF 至少 2/3 seed 命中；"
                    "LightGCN 未命中"
                )
            elif (
                scf_appearances >= 2
                and lightgcn_appearances == 0
                and graph_appearances == 0
            ):
                level = 2
                rule = (
                    "完整测试集；SCF 至少 2/3 seed 命中；"
                    "LightGCN 与 Graph+BGE 均未命中"
                )
            elif (
                scf_appearances >= 2
                and lightgcn_appearances == 0
            ):
                level = 3
                rule = (
                    "完整测试集；SCF 至少 2/3 seed 命中；"
                    "LightGCN 未命中"
                )
            elif lightgcn_rank is None or (
                scf_rank is not None
                and lightgcn_rank is not None
                and scf_rank < lightgcn_rank
            ):
                level = 4
                rule = (
                    "SCF 共识排名优于 LightGCN，"
                    "但稳定性或清洗条件已放宽"
                )
            else:
                continue

            rows.append(
                {
                    "mashup_id": mashup_id,
                    "mashup_name": mashup_names[mashup_id],
                    "api_id": api_id,
                    "api_name": api_names[api_id],
                    "api_group": target_group,
                    "direct_api_name_mention": int(direct_mention),
                    "direct_api_name_match": direct_match,
                    "selection_level": level,
                    "selection_rule": rule,
                    "strict_clean": int(is_clean),
                    "scf_seed_appearances": scf_appearances,
                    "lightgcn_seed_appearances": lightgcn_appearances,
                    "graph_bge_seed_appearances": graph_appearances,
                    "bge_seed_appearances": bge_appearances,
                    "scf_consensus_rank": scf_rank,
                    "lightgcn_consensus_rank": lightgcn_rank,
                    "graph_bge_consensus_rank": graph_rank,
                    "bge_consensus_rank": bge_rank,
                    "scf_best_seed_rank": best_rank_across_seeds(
                        method="SCF-LightGCN+BGE",
                        mashup_id=mashup_id,
                        api_id=api_id,
                        seed_lookup=seed_lookup,
                    ),
                    "num_positives": len(positives[mashup_id]),
                }
            )

    return rows


def failure_candidates(
    *,
    mashup_ids: Iterable[int],
    positives: Mapping[int, Set[int]],
    group_by_api: Mapping[int, str],
    seed_lookup,
    strict_clean_ids: Optional[Set[int]],
    min_positives: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for mashup_id in mashup_ids:
        relevant = positives[mashup_id]
        if len(relevant) < min_positives:
            continue

        hit_counts: Dict[str, int] = {}
        for method in METHOD_ORDER:
            total_hits = 0
            for seed in (0, 1, 2):
                ranked = set(
                    seed_lookup[method][seed]
                    .get(mashup_id, {})
                    .keys()
                )
                total_hits += len(ranked & relevant)
            hit_counts[method] = total_hits

        is_clean = (
            strict_clean_ids is not None
            and mashup_id in strict_clean_ids
        )
        scf_hits = hit_counts["SCF-LightGCN+BGE"]
        all_method_hits = sum(hit_counts.values())

        if is_clean and scf_hits == 0 and all_method_hits == 0:
            level = 0
            rule = "严格清洗子集；四种方法三个 seed 均未命中"
        elif scf_hits == 0 and all_method_hits == 0:
            level = 1
            rule = "完整测试集；四种方法三个 seed 均未命中"
        elif is_clean and scf_hits == 0:
            level = 2
            rule = "严格清洗子集；SCF 三个 seed 均未命中"
        elif scf_hits == 0:
            level = 3
            rule = "完整测试集；SCF 三个 seed 均未命中"
        else:
            continue

        group_counts = defaultdict(int)
        for api_id in relevant:
            group_counts[group_by_api[api_id]] += 1

        rows.append(
            {
                "mashup_id": mashup_id,
                "api_id": -1,
                "api_group": "Mixed",
                "selection_level": level,
                "selection_rule": rule,
                "strict_clean": int(is_clean),
                "num_positives": len(relevant),
                "head_positives": group_counts["Head"],
                "middle_positives": group_counts["Middle"],
                "tail_positives": group_counts["Tail"],
                "unseen_positives": group_counts["Unseen"],
                "scf_total_hits_over_3_seeds": scf_hits,
                "all_methods_total_hits_over_3_seeds": all_method_hits,
            }
        )

    return rows


def choose_rescue_candidate(
    frame: pd.DataFrame,
    used_mashups: Set[int],
) -> Optional[pd.Series]:
    if frame.empty:
        return None

    eligible = frame[
        frame["direct_api_name_mention"].fillna(0).astype(int) == 0
    ].copy()
    if eligible.empty:
        return None

    available = eligible[
        ~eligible["mashup_id"].isin(used_mashups)
    ].copy()
    if available.empty:
        available = eligible.copy()

    for column in (
        "scf_consensus_rank",
        "lightgcn_consensus_rank",
        "graph_bge_consensus_rank",
        "bge_consensus_rank",
    ):
        available[column] = available[column].fillna(999)

    available = available.sort_values(
        [
            "selection_level",
            "scf_seed_appearances",
            "bge_seed_appearances",
            "scf_consensus_rank",
            "num_positives",
            "mashup_id",
        ],
        ascending=[True, False, False, True, False, True],
    )
    return available.iloc[0]


def choose_failure_candidate(
    frame: pd.DataFrame,
    used_mashups: Set[int],
) -> Optional[pd.Series]:
    if frame.empty:
        return None

    available = frame[
        ~frame["mashup_id"].isin(used_mashups)
    ].copy()
    if available.empty:
        available = frame.copy()

    available = available.sort_values(
        [
            "selection_level",
            "num_positives",
            "unseen_positives",
            "tail_positives",
            "middle_positives",
            "mashup_id",
        ],
        ascending=[True, False, False, False, False, True],
    )
    return available.iloc[0]


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with",
    "is", "are", "be", "by", "from", "this", "that", "api", "service",
    "application", "user", "users", "using", "use", "allows", "provides",
    "data", "web", "site", "mashup",
}


def content_tokens(text: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) >= 3
        and token not in STOPWORDS
        and not token.isdigit()
    }


def shared_keywords(mashup_text: str, api_text: str, limit: int = 8) -> str:
    shared = sorted(
        content_tokens(mashup_text) & content_tokens(api_text),
        key=lambda token: (-len(token), token),
    )
    return ", ".join(shared[:limit])


def truncate_text(value: str, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def rank_display(value: Any) -> str:
    if value is None:
        return "未进入共识Top-10"
    try:
        if pd.isna(value):
            return "未进入共识Top-10"
    except Exception:
        pass
    return str(int(value))


def md_escape(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ").strip()


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_无数据_"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            + " | ".join(
                md_escape(row[column]) for column in columns
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config.resolve())

    output_dir = Path(config["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_config = resolve_path(
        str(config["reference_config"]),
        "strict reference configuration",
    )
    data = load_strict_data(reference_config)

    mashup_names, api_names, reverse_path = load_entity_names(
        data.processed_dir,
        data.num_mashups,
        data.num_apis,
    )

    api_groups_path = resolve_path(
        str(config["api_groups_source"]),
        "API group metadata",
    )
    api_groups = load_api_groups(
        api_groups_path,
        data.num_apis,
    )
    group_by_api = api_groups.set_index("api_id")["group"].to_dict()
    frequency_by_api = (
        api_groups.set_index("api_id")["train_frequency"].to_dict()
    )

    audit_value = str(config.get("audit_csv", "")).strip()
    audit_path = (
        resolve_path(audit_value, "leakage audit CSV")
        if audit_value
        else None
    )
    strict_clean_ids = load_strict_clean_ids(
        audit_path,
        float(config["strict_clean_cosine_threshold"]),
    )

    k = int(config.get("k", 10))
    rankings: Dict[str, Dict[int, pd.DataFrame]] = defaultdict(dict)
    for method in METHOD_ORDER:
        for seed in (0, 1, 2):
            rankings[method][seed] = load_rankings(
                method=method,
                seed=seed,
                config=config,
                k=k,
            )

    consensus, details, seed_lookup = build_consensus(
        rankings,
        k,
    )
    positives = positives_by_mashup(data.test_pairs)
    mashup_ids = sorted(positives)

    candidate_frames: Dict[str, pd.DataFrame] = {}
    for case_type, target_group in (
        ("middle_rescue", "Middle"),
        ("tail_rescue", "Tail"),
        ("unseen_rescue", "Unseen"),
    ):
        candidate_frames[case_type] = pd.DataFrame(
            candidate_rescues(
                target_group=target_group,
                mashup_ids=mashup_ids,
                positives=positives,
                group_by_api=group_by_api,
                details=details,
                seed_lookup=seed_lookup,
                strict_clean_ids=strict_clean_ids,
                mashup_names=mashup_names,
                api_names=api_names,
                mashup_texts=data.mashup_texts,
            )
        )

    candidate_frames["complex_failure"] = pd.DataFrame(
        failure_candidates(
            mashup_ids=mashup_ids,
            positives=positives,
            group_by_api=group_by_api,
            seed_lookup=seed_lookup,
            strict_clean_ids=strict_clean_ids,
            min_positives=int(
                config.get("minimum_failure_positives", 3)
            ),
        )
    )

    candidate_pool_rows: List[pd.DataFrame] = []
    for case_type, frame in candidate_frames.items():
        if frame.empty:
            continue
        copied = frame.copy()
        copied.insert(0, "case_type", case_type)
        copied.insert(1, "case_label", CASE_LABELS[case_type])
        candidate_pool_rows.append(copied)
    if candidate_pool_rows:
        pd.concat(
            candidate_pool_rows,
            ignore_index=True,
            sort=False,
        ).to_csv(
            output_dir / "case_candidate_pool.csv",
            index=False,
        )

    selected: List[Dict[str, Any]] = []
    used_mashups: Set[int] = set()

    for case_type in (
        "middle_rescue",
        "tail_rescue",
        "unseen_rescue",
    ):
        chosen = choose_rescue_candidate(
            candidate_frames[case_type],
            used_mashups,
        )
        if chosen is None:
            print(f"WARNING: no candidate selected for {case_type}")
            continue
        mashup_id = int(chosen["mashup_id"])
        used_mashups.add(mashup_id)
        record = chosen.to_dict()
        record["case_type"] = case_type
        record["case_label"] = CASE_LABELS[case_type]
        selected.append(record)

    failure = choose_failure_candidate(
        candidate_frames["complex_failure"],
        used_mashups,
    )
    if failure is not None:
        mashup_id = int(failure["mashup_id"])
        used_mashups.add(mashup_id)
        record = failure.to_dict()
        record["case_type"] = "complex_failure"
        record["case_label"] = CASE_LABELS["complex_failure"]
        selected.append(record)
    else:
        print("WARNING: no complex-failure candidate selected")

    if len(selected) < 4:
        raise RuntimeError(
            f"Only {len(selected)} case categories were selected. "
            "Inspect case_candidate_pool.csv and relax the config/criteria."
        )

    selected_frame = pd.DataFrame(selected)
    selected_frame["mashup_name"] = selected_frame["mashup_id"].map(
        mashup_names
    )
    selected_frame["mashup_description"] = selected_frame[
        "mashup_id"
    ].map(
        lambda index: data.mashup_texts[int(index)]
    )
    selected_frame.to_csv(
        output_dir / "selected_cases.csv",
        index=False,
    )

    ground_truth_rows: List[Dict[str, Any]] = []
    ranking_rows: List[Dict[str, Any]] = []
    method_summary_rows: List[Dict[str, Any]] = []

    for selected_case in selected:
        case_type = str(selected_case["case_type"])
        case_label = CASE_LABELS[case_type]
        mashup_id = int(selected_case["mashup_id"])
        relevant = positives[mashup_id]

        for api_id in sorted(
            relevant,
            key=lambda value: (
                GROUP_ORDER.get(group_by_api[value], 99),
                value,
            ),
        ):
            row: Dict[str, Any] = {
                "case_type": case_type,
                "case_label": case_label,
                "mashup_id": mashup_id,
                "mashup_name": mashup_names[mashup_id],
                "api_id": api_id,
                "api_name": api_names[api_id],
                "api_group": group_by_api[api_id],
                "train_frequency": int(
                    frequency_by_api[api_id]
                ),
                "api_description": data.api_texts[api_id],
                "shared_keywords": shared_keywords(
                    data.mashup_texts[mashup_id],
                    data.api_texts[api_id],
                ),
            }
            for method in METHOD_ORDER:
                prefix = (
                    method.lower()
                    .replace("+", "_")
                    .replace("-", "_")
                    .replace(" ", "_")
                )
                row[f"{prefix}_consensus_rank"] = find_positive_rank(
                    method=method,
                    mashup_id=mashup_id,
                    api_id=api_id,
                    details=details,
                )
                row[f"{prefix}_seed_appearances"] = (
                    seed_appearance_count(
                        method=method,
                        mashup_id=mashup_id,
                        api_id=api_id,
                        seed_lookup=seed_lookup,
                    )
                )
            ground_truth_rows.append(row)

        for method in METHOD_ORDER:
            ranked = consensus[method][mashup_id]
            hits = [api_id for api_id in ranked if api_id in relevant]
            method_summary_rows.append(
                {
                    "case_type": case_type,
                    "case_label": case_label,
                    "mashup_id": mashup_id,
                    "mashup_name": mashup_names[mashup_id],
                    "method": method,
                    "num_positives": len(relevant),
                    "hit_count": len(hits),
                    "consensus_recall@10": (
                        len(hits) / len(relevant)
                        if relevant
                        else 0.0
                    ),
                    "first_hit_rank": (
                        min(
                            details[method][mashup_id][api_id][
                                "consensus_rank"
                            ]
                            for api_id in hits
                        )
                        if hits
                        else np.nan
                    ),
                    "hit_api_ids": ";".join(
                        str(api_id) for api_id in hits
                    ),
                    "hit_api_names": "; ".join(
                        api_names[api_id] for api_id in hits
                    ),
                }
            )

            for rank, api_id in enumerate(ranked, start=1):
                detail = details[method][mashup_id][api_id]
                ranking_rows.append(
                    {
                        "case_type": case_type,
                        "case_label": case_label,
                        "mashup_id": mashup_id,
                        "mashup_name": mashup_names[mashup_id],
                        "method": method,
                        "consensus_rank": rank,
                        "api_id": api_id,
                        "api_name": api_names[api_id],
                        "api_group": group_by_api[api_id],
                        "train_frequency": int(
                            frequency_by_api[api_id]
                        ),
                        "is_ground_truth": int(api_id in relevant),
                        "seed_appearances": int(
                            detail["seed_appearances"]
                        ),
                        "mean_rank_when_present": float(
                            detail["mean_rank_when_present"]
                        ),
                        "reciprocal_rank_score": float(
                            detail["reciprocal_rank_score"]
                        ),
                    }
                )

    ground_truth_frame = pd.DataFrame(ground_truth_rows)
    ranking_frame = pd.DataFrame(ranking_rows)
    method_summary_frame = pd.DataFrame(method_summary_rows)

    ground_truth_frame.to_csv(
        output_dir / "case_ground_truth.csv",
        index=False,
    )
    ranking_frame.to_csv(
        output_dir / "case_rankings.csv",
        index=False,
    )
    method_summary_frame.to_csv(
        output_dir / "case_method_summary.csv",
        index=False,
    )

    compact_rows: List[Dict[str, Any]] = []
    for selected_case in selected:
        mashup_id = int(selected_case["mashup_id"])
        case_type = str(selected_case["case_type"])
        for method in METHOD_ORDER:
            subset = ranking_frame[
                (ranking_frame["mashup_id"] == mashup_id)
                & (ranking_frame["method"] == method)
            ].sort_values("consensus_rank")
            compact_rows.append(
                {
                    "case_type": case_type,
                    "mashup_id": mashup_id,
                    "mashup_name": mashup_names[mashup_id],
                    "method": method,
                    "top10": "；".join(
                        (
                            f"{int(row.consensus_rank)}."
                            f"{row.api_name}"
                            f"[{row.api_group}]"
                            f"{'✓' if int(row.is_ground_truth) else ''}"
                        )
                        for row in subset.itertuples(index=False)
                    ),
                }
            )
    pd.DataFrame(compact_rows).to_csv(
        output_dir / "case_top10_compact.csv",
        index=False,
    )

    report: List[str] = [
        "# SCF 最终案例分析",
        "",
        "## 选择原则",
        "",
        "- 使用 seed 0/1/2 的倒数排名投票生成共识 Top-10，避免只展示单个随机种子的偶然结果。",
        "- Middle、Tail 和训练交互未见 API 案例优先从严格清洗子集中选择。",
        "- 成功案例会额外排除 Mashup 名称或描述中直接出现真实 API 名称的样本，包括 Moo 这类三字符短名称。",
        "- 优先选择 SCF 至少在 2/3 个 seed 中命中、而 Inductive LightGCN 未命中的样本。",
        "- “Unseen”仅表示该 API 在训练交互中的频次为 0，不表示预训练文本编码器从未接触相关概念。",
        "",
    ]

    for order, selected_case in enumerate(selected, start=1):
        case_type = str(selected_case["case_type"])
        case_label = CASE_LABELS[case_type]
        mashup_id = int(selected_case["mashup_id"])
        mashup_name = mashup_names[mashup_id]
        selection_rule = str(selected_case["selection_rule"])
        strict_clean = bool(int(selected_case.get("strict_clean", 0)))

        report.extend(
            [
                f"## 案例 {order}：{case_label}",
                "",
                f"**Mashup：** {mashup_name}（ID={mashup_id}）",
                "",
                f"**是否属于严格清洗子集：** {'是' if strict_clean else '否'}",
                "",
                f"**自动选择规则：** {selection_rule}",
                "",
                f"**Mashup 描述：** {truncate_text(data.mashup_texts[mashup_id], 500)}",
                "",
                "### 真实 API",
                "",
            ]
        )

        gt = ground_truth_frame[
            ground_truth_frame["mashup_id"] == mashup_id
        ].copy()
        gt_display = gt[
            [
                "api_name",
                "api_group",
                "train_frequency",
                "shared_keywords",
            ]
        ].rename(
            columns={
                "api_name": "真实 API",
                "api_group": "分组",
                "train_frequency": "训练频次",
                "shared_keywords": "与 Mashup 共享关键词",
            }
        )
        report.extend(
            [
                markdown_table(gt_display),
                "",
                "### 各方法共识 Top-10 表现",
                "",
            ]
        )

        summary = method_summary_frame[
            method_summary_frame["mashup_id"] == mashup_id
        ].copy()
        summary["first_hit_rank"] = summary[
            "first_hit_rank"
        ].map(
            lambda value: (
                "-"
                if pd.isna(value)
                else str(int(value))
            )
        )
        summary_display = summary[
            [
                "method",
                "hit_count",
                "consensus_recall@10",
                "first_hit_rank",
                "hit_api_names",
            ]
        ].rename(
            columns={
                "method": "方法",
                "hit_count": "命中数",
                "consensus_recall@10": "共识 Recall@10",
                "first_hit_rank": "首个命中排名",
                "hit_api_names": "命中的真实 API",
            }
        )
        summary_display["共识 Recall@10"] = summary_display[
            "共识 Recall@10"
        ].map(lambda value: f"{float(value):.4f}")
        report.extend(
            [
                markdown_table(summary_display),
                "",
                "### 共识推荐列表",
                "",
            ]
        )

        compact = pd.DataFrame(compact_rows)
        compact = compact[
            compact["mashup_id"] == mashup_id
        ][["method", "top10"]].rename(
            columns={
                "method": "方法",
                "top10": "Top-10（✓表示真实 API）",
            }
        )
        report.extend(
            [
                markdown_table(compact),
                "",
                "### 分析",
                "",
            ]
        )

        if case_type in {
            "middle_rescue",
            "tail_rescue",
            "unseen_rescue",
        }:
            target_api_id = int(selected_case["api_id"])
            target_name = api_names[target_api_id]
            target_group = group_by_api[target_api_id]
            target_freq = int(frequency_by_api[target_api_id])
            scf_rank = find_positive_rank(
                method="SCF-LightGCN+BGE",
                mashup_id=mashup_id,
                api_id=target_api_id,
                details=details,
            )
            lightgcn_rank = find_positive_rank(
                method="Inductive LightGCN",
                mashup_id=mashup_id,
                api_id=target_api_id,
                details=details,
            )
            bge_rank = find_positive_rank(
                method="BGE-only",
                mashup_id=mashup_id,
                api_id=target_api_id,
                details=details,
            )
            appearances = seed_appearance_count(
                method="SCF-LightGCN+BGE",
                mashup_id=mashup_id,
                api_id=target_api_id,
                seed_lookup=seed_lookup,
            )
            keywords = shared_keywords(
                data.mashup_texts[mashup_id],
                data.api_texts[target_api_id],
            )

            report.append(
                f"- 目标真实 API 为 **{target_name}**，属于 "
                f"**{target_group}**，训练交互频次为 **{target_freq}**。"
            )
            report.append(
                f"- SCF 在 {appearances}/3 个 seed 中将其放入 Top-10，"
                f"共识排名为 **{rank_display(scf_rank)}**；"
                f"Inductive LightGCN 为 **{rank_display(lightgcn_rank)}**。"
            )
            report.append(
                f"- BGE-only 共识排名为 **{rank_display(bge_rank)}**。"
            )
            if keywords:
                report.append(
                    f"- Mashup 与该 API 文本的可见共享关键词包括："
                    f"**{keywords}**。这些词只是解释线索，不等同于完整的 BGE 语义机制。"
                )
            if case_type == "middle_rescue":
                report.append(
                    "- 该案例表明，直接 Mashup–API 语义匹配可以补充"
                    "历史 Mashup 协同迁移对中频 API 的遗漏。"
                )
            elif case_type == "tail_rescue":
                report.append(
                    "- 该案例表明，SCF 能在不完全依赖热门协同信号的情况下"
                    "恢复低频 Tail API，提高长尾服务的可发现性。"
                )
            else:
                report.append(
                    "- LightGCN 无法从训练交互中学习该 API 的有效协同表示，"
                    "SCF 主要依靠直接 BGE 文本分支完成零样式检索。"
                )
        else:
            groups = [
                group_by_api[api_id]
                for api_id in positives[mashup_id]
            ]
            report.append(
                f"- 该 Mashup 共有 **{len(positives[mashup_id])}** 个真实 API，"
                f"分组构成为：{', '.join(groups)}。"
            )
            report.append(
                "- SCF 的三个 seed 均未完整恢复真实服务组合，说明模型仍难以处理"
                "多意图、描述信息不足或多个 API 功能互补的复杂组合场景。"
            )
            report.append(
                "- 该失败案例应作为局限性展示，而不是通过继续查看测试集后调整"
                "0.35/0.65 融合权重来修复。"
            )

        report.append("")

    report.extend(
        [
            "## 人工复核清单",
            "",
            "1. 对照原始数据确认自动恢复的 Mashup/API 名称是否正确。",
            "2. 检查案例描述中是否直接出现真实 API 名称；脚本已增加短名称过滤，但仍需人工复核。",
            "3. 对 Unseen API 使用“训练交互未见”措辞，不使用“模型从未见过”。",
            "4. 论文正文建议放 3 个成功案例和 1 个失败案例；完整 Top-10 可放附录。",
            "5. 不根据案例结果重新调整模型权重或重新选择测试样本阈值。",
            "",
        ]
    )

    report_path = output_dir / "case_study_analysis_cn.md"
    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    metadata = {
        "method_order": METHOD_ORDER,
        "case_labels": CASE_LABELS,
        "selection_uses_three_seed_consensus": True,
        "direct_api_name_filter_enabled": True,
        "direct_api_name_filter_scope": "Mashup title plus description",
        "consensus_rule": (
            "sum reciprocal ranks; then seed appearances; then mean rank"
        ),
        "k": k,
        "strict_clean_cosine_threshold": float(
            config["strict_clean_cosine_threshold"]
        ),
        "reverse_mapping_path": (
            str(reverse_path) if reverse_path else None
        ),
        "selected_case_count": len(selected),
    }
    (output_dir / "case_study_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(selected_frame[
        [
            "case_type",
            "mashup_id",
            "mashup_name",
            "selection_level",
            "strict_clean",
        ]
    ].to_string(index=False))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
