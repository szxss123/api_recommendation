from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Any, List

import pandas as pd


def clean_text(x: Any) -> str:
    if pd.isna(x):
        return ""

    text = str(x)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = text.replace("###", ", ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_multi_value(x: Any) -> List[str]:
    if pd.isna(x):
        return []

    text = str(x).strip()
    if not text:
        return []

    if "###" in text:
        parts = text.split("###")
    elif "," in text:
        parts = text.split(",")
    else:
        parts = [text]

    cleaned = []
    for p in parts:
        p = clean_text(p)
        if p:
            cleaned.append(p)

    return list(dict.fromkeys(cleaned))


def join_multi_value(xs: List[str]) -> str:
    return "###".join(xs)


def clean_api_df(api_df: pd.DataFrame, drop_empty_desc: bool = False) -> pd.DataFrame:
    api_df = api_df.copy()

    required_cols = ["APIName"]
    for col in required_cols:
        if col not in api_df.columns:
            raise KeyError(f"apisData.xlsx missing required column: {col}")

    text_cols = ["APIName", "Description", "Categories", "Versions", "DevelopersName"]
    for col in text_cols:
        if col in api_df.columns:
            api_df[col] = api_df[col].apply(clean_text)
        else:
            api_df[col] = ""

    api_df = api_df[api_df["APIName"].str.len() > 0].copy()

    if drop_empty_desc:
        api_df = api_df[api_df["Description"].str.len() > 0].copy()

    api_df["Categories"] = api_df["Categories"].apply(lambda x: join_multi_value(split_multi_value(x)))
    api_df["DevelopersName"] = api_df["DevelopersName"].apply(lambda x: join_multi_value(split_multi_value(x)))

    api_df = api_df.drop_duplicates(subset=["APIName"], keep="first").reset_index(drop=True)
    return api_df


def clean_mashup_df(
    mashup_df: pd.DataFrame,
    valid_api_names: set[str],
    drop_empty_desc: bool = False,
    min_related_apis: int = 1,
) -> pd.DataFrame:
    mashup_df = mashup_df.copy()

    required_cols = ["mashups_name", "related_apis"]
    for col in required_cols:
        if col not in mashup_df.columns:
            raise KeyError(f"mashups.xlsx missing required column: {col}")

    text_cols = [
        "mashups_name",
        "description",
        "categories",
        "company",
        "mashup_app_type",
        "related_apis",
        "SubmittedDate",
    ]

    for col in text_cols:
        if col in mashup_df.columns:
            mashup_df[col] = mashup_df[col].apply(clean_text)
        else:
            mashup_df[col] = ""

    mashup_df = mashup_df[mashup_df["mashups_name"].str.len() > 0].copy()

    if drop_empty_desc:
        mashup_df = mashup_df[mashup_df["description"].str.len() > 0].copy()

    mashup_df["categories"] = mashup_df["categories"].apply(lambda x: join_multi_value(split_multi_value(x)))

    def filter_related_apis(x: Any) -> str:
        apis = split_multi_value(x)
        apis = [a for a in apis if a in valid_api_names]
        return join_multi_value(apis)

    mashup_df["related_apis"] = mashup_df["related_apis"].apply(filter_related_apis)
    mashup_df["related_api_count"] = mashup_df["related_apis"].apply(lambda x: len(split_multi_value(x)))

    mashup_df = mashup_df[mashup_df["related_api_count"] >= min_related_apis].copy()
    mashup_df = mashup_df.drop_duplicates(subset=["mashups_name"], keep="first").reset_index(drop=True)

    return mashup_df


def parse_args():
    parser = argparse.ArgumentParser(description="Clean raw API recommendation xlsx dataset.")
    parser.add_argument("--api_path", type=str, default="data/raw/apisData.xlsx")
    parser.add_argument("--mashup_path", type=str, default="data/raw/mashups.xlsx")
    parser.add_argument("--output_dir", type=str, default="data/cleaned")

    parser.add_argument("--drop_empty_desc", action="store_true")
    parser.add_argument("--min_related_apis", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    api_path = Path(args.api_path)
    mashup_path = Path(args.mashup_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not api_path.exists():
        raise FileNotFoundError(f"Missing file: {api_path}")
    if not mashup_path.exists():
        raise FileNotFoundError(f"Missing file: {mashup_path}")

    print("[Load] reading xlsx files...")
    api_df = pd.read_excel(api_path)
    mashup_df = pd.read_excel(mashup_path)

    print(f"[Raw] APIs: {len(api_df)}")
    print(f"[Raw] Mashups: {len(mashup_df)}")

    api_df = clean_api_df(api_df, drop_empty_desc=args.drop_empty_desc)
    valid_api_names = set(api_df["APIName"].tolist())

    mashup_df = clean_mashup_df(
        mashup_df,
        valid_api_names=valid_api_names,
        drop_empty_desc=args.drop_empty_desc,
        min_related_apis=args.min_related_apis,
    )

    api_out = output_dir / "apisData_cleaned.xlsx"
    mashup_out = output_dir / "mashups_cleaned.xlsx"

    api_df.to_excel(api_out, index=False)
    mashup_df.to_excel(mashup_out, index=False)

    print("[Done] cleaned dataset saved.")
    print(f"  APIs: {len(api_df)} -> {api_out}")
    print(f"  Mashups: {len(mashup_df)} -> {mashup_out}")
    print(f"  Avg related APIs per mashup: {mashup_df['related_api_count'].mean():.4f}")
    print(f"  Min related APIs per mashup: {mashup_df['related_api_count'].min()}")
    print(f"  Max related APIs per mashup: {mashup_df['related_api_count'].max()}")


if __name__ == "__main__":
    main()