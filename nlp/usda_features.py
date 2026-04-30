"""USDA FoodData Central feature preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import yaml


NUTRIENT_IDS: List[int] = [
    1008, 1003, 1004, 1005, 1079, 1093, 1253, 1258, 1087,
    1089, 1162, 1106, 1109, 1110, 1180, 1175, 1178,
]


def _csv_path(dataset_dir: str, filename: str) -> Path:
    path = Path(dataset_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected USDA file not found: {path}")
    return path


def _read_food_nutrient(dataset_dir: str, source_name: str, priority: int) -> pd.DataFrame:
    path = _csv_path(dataset_dir, "food_nutrient.csv")
    df = pd.read_csv(path, low_memory=False)
    required = {"fdc_id", "nutrient_id", "amount"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df[["fdc_id", "nutrient_id", "amount"]].copy()
    df["source"] = source_name
    df["source_priority"] = priority
    return df


def _read_food_descriptions(dataset_dir: str, source_name: str, priority: int) -> pd.DataFrame:
    path = _csv_path(dataset_dir, "food.csv")
    df = pd.read_csv(path, low_memory=False)
    required = {"fdc_id", "description"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df[["fdc_id", "description"]].copy()
    df["source"] = source_name
    df["source_priority"] = priority
    return df


def _normalize_feature_matrix(matrix: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    scaler: Dict[str, Dict[str, float]] = {}
    normalized = matrix.astype(float).copy()

    for col in normalized.columns:
        upper = normalized[col].quantile(0.99)
        normalized[col] = normalized[col].clip(upper=upper)
        min_value = float(normalized[col].min())
        max_value = float(normalized[col].max())
        scaler[str(col)] = {"min": min_value, "max": max_value}
        if max_value > min_value:
            normalized[col] = (normalized[col] - min_value) / (max_value - min_value)
        else:
            normalized[col] = 0.0

    return normalized, scaler


def build_usda_feature_matrix(
    sr_legacy_dir: str,
    foundation_dir: str,
    output_dir: str,
    nutrient_ids: Iterable[int] | None = None,
) -> None:
    """Build normalized USDA nutrient matrix and description lookup artifacts."""
    nutrient_ids = list(nutrient_ids or NUTRIENT_IDS)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    sr_nutrients = _read_food_nutrient(sr_legacy_dir, "sr_legacy", priority=0)
    foundation_nutrients = _read_food_nutrient(foundation_dir, "foundation", priority=1)
    nutrients = pd.concat([sr_nutrients, foundation_nutrients], ignore_index=True)
    nutrients = nutrients.sort_values("source_priority", ascending=False)
    nutrients = nutrients.drop_duplicates(["fdc_id", "nutrient_id"], keep="first")
    nutrients = nutrients[nutrients["nutrient_id"].isin(nutrient_ids)]

    matrix = nutrients.pivot_table(
        index="fdc_id",
        columns="nutrient_id",
        values="amount",
        aggfunc="first",
    )
    matrix = matrix.reindex(columns=nutrient_ids).fillna(0.0).sort_index()
    normalized, scaler = _normalize_feature_matrix(matrix)
    normalized.columns = [str(col) for col in normalized.columns]

    with (output / "usda_scaler.json").open("w", encoding="utf-8") as f:
        json.dump(scaler, f, indent=2, sort_keys=True)

    sr_food = _read_food_descriptions(sr_legacy_dir, "sr_legacy", priority=0)
    foundation_food = _read_food_descriptions(foundation_dir, "foundation", priority=1)
    foods = pd.concat([sr_food, foundation_food], ignore_index=True)
    foods = foods.sort_values("source_priority", ascending=False)
    foods = foods.drop_duplicates("fdc_id", keep="first")
    descriptions = {
        str(int(row.fdc_id)): str(row.description)
        for row in foods.itertuples(index=False)
        if pd.notna(row.description)
    }

    with (output / "usda_descriptions.json").open("w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=2, sort_keys=True)

    normalized.to_parquet(output / "usda_features.parquet")


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build USDA nutrient feature artifacts.")
    parser.add_argument("--config", default="configs/nlp.yaml", help="Path to NLP YAML config.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    build_usda_feature_matrix(
        sr_legacy_dir=cfg["paths"]["sr_legacy_dir"],
        foundation_dir=cfg["paths"]["foundation_dir"],
        output_dir=cfg["paths"]["processed_dir"],
        nutrient_ids=cfg["usda"]["nutrient_ids"],
    )


if __name__ == "__main__":
    main()
