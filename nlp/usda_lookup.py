"""Fuzzy USDA nutrient lookup for normalized ingredient names."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process


DEFAULT_NUTRIENT_IDS = [
    1008, 1003, 1004, 1005, 1079, 1093, 1253, 1258, 1087,
    1089, 1162, 1106, 1109, 1110, 1180, 1175, 1178,
]


class USDALookup:
    """Map ingredient strings to normalized 17-dimensional nutrient vectors."""

    def __init__(
        self,
        features_path: str,
        descriptions_path: str,
        scaler_path: str,
        score_cutoff: int = 70,
        cache_size: int = 2000,
    ) -> None:
        self.features_path = Path(features_path)
        self.descriptions_path = Path(descriptions_path)
        self.scaler_path = Path(scaler_path)
        self.score_cutoff = score_cutoff
        self.feature_columns = [str(nutrient_id) for nutrient_id in DEFAULT_NUTRIENT_IDS]

        self.features = pd.read_parquet(self.features_path)
        self.features.index = self.features.index.map(str)
        self.features.columns = self.features.columns.map(str)
        for column in self.feature_columns:
            if column not in self.features.columns:
                self.features[column] = 0.0
        self.features = self.features[self.feature_columns]

        with self.descriptions_path.open("r", encoding="utf-8") as f:
            self.descriptions: Dict[str, str] = json.load(f)
        with self.scaler_path.open("r", encoding="utf-8") as f:
            self.scaler = json.load(f)

        self._fdc_ids = list(self.descriptions.keys())
        self._choices = [self.descriptions[fdc_id] for fdc_id in self._fdc_ids]
        self._cached_lookup = lru_cache(maxsize=cache_size)(self._lookup_uncached)

    def lookup(self, ingredient_name: str) -> np.ndarray:
        """Return a normalized nutrient vector with graceful zero fallback."""
        key = self._normalize_key(ingredient_name)
        return self._cached_lookup(key).copy()

    def batch_lookup(self, ingredient_list: List[str]) -> np.ndarray:
        """Return stacked lookup vectors with shape (N, 17)."""
        if not ingredient_list:
            return np.zeros((0, len(self.feature_columns)), dtype=np.float32)
        return np.stack([self.lookup(ingredient) for ingredient in ingredient_list]).astype(np.float32)

    def get_match_info(self, ingredient_name: str) -> dict:
        """Return fuzzy match metadata and nutrient vector for debugging."""
        key = self._normalize_key(ingredient_name)
        match = self._match(key)
        if match is None:
            vector = np.zeros(len(self.feature_columns), dtype=np.float32)
            return {
                "ingredient": ingredient_name,
                "matched_description": None,
                "fdc_id": None,
                "score": 0.0,
                "vector": vector,
            }

        description, score, fdc_id = match
        vector = self._vector_for_fdc_id(fdc_id)
        return {
            "ingredient": ingredient_name,
            "matched_description": description,
            "fdc_id": int(fdc_id) if str(fdc_id).isdigit() else fdc_id,
            "score": float(score),
            "vector": vector,
        }

    def cache_info(self):
        """Expose LRU cache statistics for tests and monitoring."""
        return self._cached_lookup.cache_info()

    def clear_cache(self) -> None:
        self._cached_lookup.cache_clear()

    def _lookup_uncached(self, normalized_name: str) -> np.ndarray:
        match = self._match(normalized_name)
        if match is None:
            return np.zeros(len(self.feature_columns), dtype=np.float32)
        _, _, fdc_id = match
        return self._vector_for_fdc_id(fdc_id)

    def _match(self, normalized_name: str) -> Optional[Tuple[str, float, str]]:
        if not normalized_name or not self._choices:
            return None
        result = process.extractOne(
            normalized_name,
            self._choices,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=self.score_cutoff,
        )
        if result is None:
            return None
        description, score, index = result
        return description, float(score), self._fdc_ids[index]

    def _vector_for_fdc_id(self, fdc_id: str) -> np.ndarray:
        key = str(fdc_id)
        if key not in self.features.index:
            return np.zeros(len(self.feature_columns), dtype=np.float32)
        return self.features.loc[key, self.feature_columns].to_numpy(dtype=np.float32)

    @staticmethod
    def _normalize_key(ingredient_name: str) -> str:
        return " ".join(str(ingredient_name or "").lower().strip().split())
