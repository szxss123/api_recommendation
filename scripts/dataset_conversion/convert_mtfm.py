#!/usr/bin/env python3
"""
Convert the public MTFM ProgrammableWeb dataset into a unified, human-readable format.

Default behavior follows the official MTFM candidate setting:
only APIs listed in used_api_list.json are retained as recommendation candidates.

Outputs
-------
apis.jsonl
mashups.jsonl
interactions.csv
api_categories.csv
mashup_categories.csv
api_quality_features.csv
api_id_map.json
mashup_id_map.json
metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REQUIRED_JSON_FILES = (
    "api_name.json",
    "api_description.json",
    "api_category.json",
    "mashup_name.json",
    "mashup_description.json",
    "mashup_category.json",
    "mashup_used_api.json",
    "used_api_list.json",
    "category_list.json",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(value: Any) -> str:
    """Convert string/token-list style descriptions to plain text."""
    if value is None:
        return ""

    if isinstance(value, str):
        return " ".join(value.split())

    if isinstance(value, (list, tuple)):
        parts: List[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                text = item.strip()
            else:
                text = str(item).strip()
            if text:
                parts.append(text)
        return " ".join(parts)

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    return str(value).strip()


def normalize_categories(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]
    elif not isinstance(value, (list, tuple, set)):
        value = [value]

    result: List[str] = []
    seen = set()
    for item in value:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def ensure_list(name: str, value: Any) -> List[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON list, but got {type(value).__name__}.")
    return value


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def safe_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def parse_quality_file(path: Path) -> Tuple[List[str], Dict[str, List[float | None]], List[List[float | None]]]:
    """
    Parse api_quality_feature.dat.

    Returns:
      feature_names,
      API-name -> feature vector,
      vectors in file order (fallback for imperfect API-name matching).
    """
    if not path.exists():
        return [], {}, []

    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as file:
        reader = csv.reader(file)
        rows = list(reader)

    if not rows:
        return [], {}, []

    header = [cell.strip() for cell in rows[0]]
    if not header:
        return [], {}, []

    feature_names = header[1:]
    by_name: Dict[str, List[float | None]] = {}
    vectors_in_order: List[List[float | None]] = []

    for row in rows[1:]:
        if not row:
            continue

        # Pad/truncate to the declared header length.
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header)]

        api_name = row[0].strip()
        vector = [safe_float(cell) for cell in row[1:]]
        vectors_in_order.append(vector)

        if api_name:
            by_name[api_name] = vector

    return feature_names, by_name, vectors_in_order


def calculate_coverage(api_counter: Counter[int], total_interactions: int, k: int) -> float:
    if total_interactions == 0:
        return 0.0
    covered = sum(count for _, count in api_counter.most_common(k))
    return covered / total_interactions


def validate_required_files(input_dir: Path) -> None:
    missing = [name for name in REQUIRED_JSON_FILES if not (input_dir / name).exists()]
    if missing:
        missing_text = "\n".join(f"  - {name}" for name in missing)
        raise FileNotFoundError(f"Missing required MTFM files:\n{missing_text}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MTFM ProgrammableWeb data to a unified dataset format."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("external/MTFM/data"),
        help="Directory containing the original MTFM data files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data_unified/mtfm_pw"),
        help="Directory in which converted files will be written.",
    )
    parser.add_argument(
        "--candidate_scope",
        choices=("used", "all"),
        default="used",
        help=(
            "'used' keeps only APIs in used_api_list.json and matches the official "
            "MTFM recommendation candidate space; 'all' keeps all APIs."
        ),
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    validate_required_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_names = ensure_list("api_name.json", load_json(input_dir / "api_name.json"))
    api_descriptions = ensure_list(
        "api_description.json", load_json(input_dir / "api_description.json")
    )
    api_categories = ensure_list("api_category.json", load_json(input_dir / "api_category.json"))

    mashup_names = ensure_list("mashup_name.json", load_json(input_dir / "mashup_name.json"))
    mashup_descriptions = ensure_list(
        "mashup_description.json", load_json(input_dir / "mashup_description.json")
    )
    mashup_categories = ensure_list(
        "mashup_category.json", load_json(input_dir / "mashup_category.json")
    )
    mashup_used_apis = ensure_list(
        "mashup_used_api.json", load_json(input_dir / "mashup_used_api.json")
    )

    used_api_names = ensure_list("used_api_list.json", load_json(input_dir / "used_api_list.json"))
    category_list = ensure_list("category_list.json", load_json(input_dir / "category_list.json"))

    if not (len(api_names) == len(api_descriptions) == len(api_categories)):
        raise ValueError(
            "API arrays have inconsistent lengths: "
            f"name={len(api_names)}, description={len(api_descriptions)}, "
            f"category={len(api_categories)}"
        )

    if not (
        len(mashup_names)
        == len(mashup_descriptions)
        == len(mashup_categories)
        == len(mashup_used_apis)
    ):
        raise ValueError(
            "Mashup arrays have inconsistent lengths: "
            f"name={len(mashup_names)}, description={len(mashup_descriptions)}, "
            f"category={len(mashup_categories)}, used_api={len(mashup_used_apis)}"
        )

    all_api_name_to_index: Dict[str, int] = {}
    duplicate_all_api_names: List[str] = []
    for index, raw_name in enumerate(api_names):
        name = str(raw_name).strip()
        if name in all_api_name_to_index:
            duplicate_all_api_names.append(name)
        else:
            all_api_name_to_index[name] = index

    if duplicate_all_api_names:
        examples = duplicate_all_api_names[:10]
        raise ValueError(
            "api_name.json contains duplicate API names, so name-based mapping is ambiguous. "
            f"Examples: {examples}"
        )

    if args.candidate_scope == "used":
        candidate_names = [str(name).strip() for name in used_api_names]
    else:
        candidate_names = [str(name).strip() for name in api_names]

    duplicate_candidates = [
        name for name, count in Counter(candidate_names).items() if count > 1
    ]
    if duplicate_candidates:
        raise ValueError(
            "Candidate API list contains duplicate names. "
            f"Examples: {duplicate_candidates[:10]}"
        )

    unknown_candidate_names = [
        name for name in candidate_names if name not in all_api_name_to_index
    ]
    if unknown_candidate_names:
        raise ValueError(
            "Some candidate APIs do not occur in api_name.json. "
            f"Examples: {unknown_candidate_names[:10]}"
        )

    candidate_name_to_id = {name: index for index, name in enumerate(candidate_names)}

    quality_feature_names, quality_by_name, quality_in_order = parse_quality_file(
        input_dir / "api_quality_feature.dat"
    )

    api_rows: List[Dict[str, Any]] = []
    quality_rows: List[Dict[str, Any]] = []
    missing_quality_count = 0

    for api_id, api_name in enumerate(candidate_names):
        original_index = all_api_name_to_index[api_name]
        description = normalize_text(api_descriptions[original_index])
        categories = normalize_categories(api_categories[original_index])

        api_rows.append(
            {
                "api_id": api_id,
                "name": api_name,
                "description": description,
                "categories": categories,
                "source": "mtfm_programmableweb",
                "original_api_index": original_index,
            }
        )

        vector = quality_by_name.get(api_name)
        quality_match = "name"

        if vector is None and original_index < len(quality_in_order):
            # The original file is expected to align with api_name.json. This fallback
            # keeps conversion usable if URL formatting differs slightly.
            vector = quality_in_order[original_index]
            quality_match = "row_index_fallback"

        if vector is None:
            vector = [None] * len(quality_feature_names)
            quality_match = "missing"
            missing_quality_count += 1

        row: Dict[str, Any] = {
            "api_id": api_id,
            "name": api_name,
            "original_api_index": original_index,
            "quality_match": quality_match,
        }
        for feature_name, feature_value in zip(quality_feature_names, vector):
            row[feature_name] = feature_value
        quality_rows.append(row)

    mashup_rows: List[Dict[str, Any]] = []
    interactions_set: set[Tuple[int, int]] = set()
    unknown_relation_apis: Counter[str] = Counter()
    empty_relation_mashups = 0

    for mashup_id, raw_name in enumerate(mashup_names):
        mashup_name = str(raw_name).strip()
        description = normalize_text(mashup_descriptions[mashup_id])
        categories = normalize_categories(mashup_categories[mashup_id])

        mashup_rows.append(
            {
                "mashup_id": mashup_id,
                "name": mashup_name,
                "description": description,
                "categories": categories,
                "source": "mtfm_programmableweb",
            }
        )

        used_values = mashup_used_apis[mashup_id]
        if used_values is None:
            used_values = []
        elif isinstance(used_values, str):
            used_values = [used_values]
        elif not isinstance(used_values, (list, tuple, set)):
            used_values = [used_values]

        mapped_count = 0
        for raw_api_name in used_values:
            api_name = str(raw_api_name).strip()
            api_id = candidate_name_to_id.get(api_name)
            if api_id is None:
                unknown_relation_apis[api_name] += 1
                continue
            interactions_set.add((mashup_id, api_id))
            mapped_count += 1

        if mapped_count == 0:
            empty_relation_mashups += 1

    if unknown_relation_apis and args.candidate_scope == "used":
        examples = unknown_relation_apis.most_common(10)
        raise ValueError(
            "mashup_used_api.json contains API names not present in used_api_list.json. "
            f"Examples with counts: {examples}"
        )

    interactions = sorted(interactions_set)

    # Write entity files.
    write_jsonl(output_dir / "apis.jsonl", api_rows)
    write_jsonl(output_dir / "mashups.jsonl", mashup_rows)

    with (output_dir / "interactions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["mashup_id", "api_id"])
        writer.writerows(interactions)

    with (output_dir / "api_categories.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["api_id", "category"])
        for row in api_rows:
            for category in row["categories"]:
                writer.writerow([row["api_id"], category])

    with (output_dir / "mashup_categories.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["mashup_id", "category"])
        for row in mashup_rows:
            for category in row["categories"]:
                writer.writerow([row["mashup_id"], category])

    quality_columns = [
        "api_id",
        "name",
        "original_api_index",
        "quality_match",
        *quality_feature_names,
    ]
    with (output_dir / "api_quality_features.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=quality_columns)
        writer.writeheader()
        writer.writerows(quality_rows)

    write_json(
        output_dir / "api_id_map.json",
        {
            "id_to_name": candidate_names,
            "name_to_id": candidate_name_to_id,
        },
    )
    write_json(
        output_dir / "mashup_id_map.json",
        {
            "id_to_name": [str(name).strip() for name in mashup_names],
        },
    )

    mashup_interaction_counter = Counter(mashup_id for mashup_id, _ in interactions)
    api_interaction_counter = Counter(api_id for _, api_id in interactions)

    interaction_counts = [
        mashup_interaction_counter.get(mashup_id, 0)
        for mashup_id in range(len(mashup_rows))
    ]

    nonempty_interaction_counts = [count for count in interaction_counts if count > 0]
    total_interactions = len(interactions)
    denominator = len(mashup_rows) * len(api_rows)
    sparsity = 1.0 - (total_interactions / denominator) if denominator else 0.0
    density = total_interactions / denominator if denominator else 0.0

    api_description_missing = sum(not row["description"] for row in api_rows)
    mashup_description_missing = sum(not row["description"] for row in mashup_rows)
    api_category_missing = sum(not row["categories"] for row in api_rows)
    mashup_category_missing = sum(not row["categories"] for row in mashup_rows)

    metadata = {
        "dataset": "MTFM ProgrammableWeb",
        "source": "whale-ynu/MTFM",
        "candidate_scope": args.candidate_scope,
        "all_api_count": len(api_names),
        "candidate_api_count": len(api_rows),
        "mashup_count": len(mashup_rows),
        "interaction_count": total_interactions,
        "unique_interacting_api_count": len(api_interaction_counter),
        "mashups_without_mapped_interactions": empty_relation_mashups,
        "average_apis_per_mashup": (
            total_interactions / len(mashup_rows) if mashup_rows else 0.0
        ),
        "median_apis_per_mashup": (
            statistics.median(interaction_counts) if interaction_counts else 0.0
        ),
        "maximum_apis_per_mashup": max(interaction_counts, default=0),
        "mashup_single_api_ratio": (
            sum(count == 1 for count in interaction_counts) / len(interaction_counts)
            if interaction_counts
            else 0.0
        ),
        "mashup_two_or_fewer_api_ratio": (
            sum(count <= 2 for count in interaction_counts) / len(interaction_counts)
            if interaction_counts
            else 0.0
        ),
        "interaction_density": density,
        "interaction_sparsity": sparsity,
        "api_description_missing_count": api_description_missing,
        "api_description_missing_ratio": (
            api_description_missing / len(api_rows) if api_rows else 0.0
        ),
        "mashup_description_missing_count": mashup_description_missing,
        "mashup_description_missing_ratio": (
            mashup_description_missing / len(mashup_rows) if mashup_rows else 0.0
        ),
        "api_category_missing_count": api_category_missing,
        "api_category_missing_ratio": (
            api_category_missing / len(api_rows) if api_rows else 0.0
        ),
        "mashup_category_missing_count": mashup_category_missing,
        "mashup_category_missing_ratio": (
            mashup_category_missing / len(mashup_rows) if mashup_rows else 0.0
        ),
        "quality_feature_count": len(quality_feature_names),
        "quality_missing_count": missing_quality_count,
        "category_vocabulary_size": len(category_list),
        "top_api_coverage": {
            f"top_{k}": calculate_coverage(api_interaction_counter, total_interactions, k)
            for k in (5, 10, 20, 50, 100)
        },
        "most_popular_apis": [
            {
                "api_id": api_id,
                "name": candidate_names[api_id],
                "interaction_count": count,
                "coverage": count / total_interactions if total_interactions else 0.0,
            }
            for api_id, count in api_interaction_counter.most_common(20)
        ],
        "files": {
            "apis": "apis.jsonl",
            "mashups": "mashups.jsonl",
            "interactions": "interactions.csv",
            "api_categories": "api_categories.csv",
            "mashup_categories": "mashup_categories.csv",
            "api_quality_features": "api_quality_features.csv",
            "api_id_map": "api_id_map.json",
            "mashup_id_map": "mashup_id_map.json",
        },
    }

    write_json(output_dir / "metadata.json", metadata)

    print("=" * 80)
    print("MTFM conversion completed")
    print("=" * 80)
    print(f"Input directory:       {input_dir}")
    print(f"Output directory:      {output_dir}")
    print(f"All APIs:              {len(api_names)}")
    print(f"Candidate APIs:        {len(api_rows)}")
    print(f"Mashups:               {len(mashup_rows)}")
    print(f"Interactions:          {total_interactions}")
    print(f"Interacting APIs:      {len(api_interaction_counter)}")
    print(f"Average APIs/mashup:   {metadata['average_apis_per_mashup']:.4f}")
    print(f"Density:               {density:.8f}")
    print(f"Top-10 API coverage:   {metadata['top_api_coverage']['top_10']:.4f}")
    print(f"Missing API texts:     {api_description_missing}")
    print(f"Missing mashup texts:  {mashup_description_missing}")
    print(f"Missing quality rows:  {missing_quality_count}")
    print()
    print("Generated files:")
    for filename in metadata["files"].values():
        print(f"  - {output_dir / filename}")
    print(f"  - {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
