"""Standalone ingredient embedding inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

try:
    from .cleaner import IngredientCleaner
    from .encoder import BERTIngredientEncoder
    from .ner import IngredientNER
    from .usda_lookup import DEFAULT_NUTRIENT_IDS, USDALookup
except ImportError:  # pragma: no cover
    from cleaner import IngredientCleaner
    from encoder import BERTIngredientEncoder
    from ner import IngredientNER
    from usda_lookup import DEFAULT_NUTRIENT_IDS, USDALookup


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_usda_lookup(cfg: dict) -> USDALookup | None:
    usda_cfg = cfg["usda"]
    required_paths = [usda_cfg["features_path"], usda_cfg["descriptions_path"], usda_cfg["scaler_path"]]
    if not all(Path(path).exists() for path in required_paths):
        return None
    return USDALookup(
        features_path=usda_cfg["features_path"],
        descriptions_path=usda_cfg["descriptions_path"],
        scaler_path=usda_cfg["scaler_path"],
        score_cutoff=usda_cfg["score_cutoff"],
        cache_size=usda_cfg.get("cache_size", 2000),
    )


def _load_model_weights(model: BERTIngredientEncoder, model_path: str) -> None:
    if not model_path or not Path(model_path).exists():
        return
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "encoder_state_dict" in checkpoint:
        state_dict = checkpoint["encoder_state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)


def _match_breakdown(usda_lookup: USDALookup | None, ingredients: List[str]) -> tuple[List[dict], List[float]]:
    nutrient_dim = len(DEFAULT_NUTRIENT_IDS)
    matches = []
    vectors = []

    for ingredient in ingredients:
        if usda_lookup is None:
            vector = np.zeros(nutrient_dim, dtype=np.float32)
            info = {
                "ingredient": ingredient,
                "matched_food": None,
                "fdc_id": None,
                "match_score": 0.0,
                "nutrient_vector": vector.tolist(),
            }
        else:
            match = usda_lookup.get_match_info(ingredient)
            vector = match["vector"].astype(np.float32)
            info = {
                "ingredient": ingredient,
                "matched_food": match["matched_description"],
                "fdc_id": match["fdc_id"],
                "match_score": match["score"],
                "nutrient_vector": vector.tolist(),
            }
        vectors.append(vector)
        matches.append(info)

    nutrition_context = (
        np.stack(vectors).mean(axis=0).astype(np.float32)
        if vectors
        else np.zeros(nutrient_dim, dtype=np.float32)
    )
    return matches, nutrition_context.tolist()


def encode_ingredients(
    raw_ingredient_str: str,
    model_path: str,
    config_path: str,
) -> dict:
    """Return embedding, normalized ingredient list, USDA matches, and context."""
    cfg = _load_config(config_path)
    cleaner = IngredientCleaner()
    ner = IngredientNER(model=cfg["ner"]["spacy_model"])
    cleaned = cleaner.clean(raw_ingredient_str)
    ingredients = ner.extract(cleaned)
    usda_lookup = _build_usda_lookup(cfg)

    model = BERTIngredientEncoder(
        bert_model_name=cfg["model"]["bert_model_name"],
        usda_lookup=usda_lookup,
        freeze_bert=cfg["model"]["freeze_bert"],
        output_dim=cfg["model"]["output_dim"],
        dropout=cfg["model"]["dropout"],
    )
    _load_model_weights(model, model_path)
    model.eval()

    with torch.no_grad():
        embedding_tensor = model(cleaned, ingredients)
    embedding = embedding_tensor.squeeze(0).cpu().numpy().astype(float)
    matches, nutrition_context = _match_breakdown(usda_lookup, ingredients)

    return {
        "raw_input": raw_ingredient_str,
        "cleaned_input": cleaned,
        "extracted_ingredients": ingredients,
        "usda_matches": matches,
        "nutrition_context": nutrition_context,
        "embedding": embedding.tolist(),
        "embedding_norm": float(np.linalg.norm(embedding)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode ingredient text into a 512-d embedding.")
    parser.add_argument("--ingredients", required=True)
    parser.add_argument("--model", default="models/nlp_best.pth")
    parser.add_argument("--config", default="configs/nlp.yaml")
    args = parser.parse_args()

    result = encode_ingredients(args.ingredients, args.model, args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
