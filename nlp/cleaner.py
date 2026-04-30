"""Ingredient text cleaning utilities."""

from __future__ import annotations

import re
from typing import List


class IngredientCleaner:
    """Normalize raw ingredient strings before NER and BERT encoding."""

    UNIT_WORDS = {
        "cup", "cups", "tbsp", "tablespoon", "tablespoons", "tsp",
        "teaspoon", "teaspoons", "oz", "ounce", "ounces", "lb", "pound",
        "pounds", "g", "gram", "grams", "kg", "ml", "liter", "liters",
        "pinch", "handful", "clove", "cloves", "slice", "slices",
        "piece", "pieces", "can", "cans", "bunch", "head",
    }

    _UNICODE_FRACTIONS = "¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞"
    _QUANTITY_RE = re.compile(
        rf"\b\d+\s*/\s*\d+\b|\b\d+(?:\.\d+)?\b|[{_UNICODE_FRACTIONS}]"
    )
    _PAREN_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")
    _UNIT_RE = re.compile(
        r"\b(?:" + "|".join(re.escape(unit) for unit in sorted(UNIT_WORDS, key=len, reverse=True)) + r")\b"
    )
    _PUNCT_RE = re.compile(r"[^a-z,\s]")
    _SPACE_RE = re.compile(r"\s+")

    def clean(self, raw_str: str) -> str:
        """Clean a raw ingredient string following the module contract."""
        if not raw_str:
            return ""

        text = raw_str.lower()
        text = self._QUANTITY_RE.sub(" ", text)
        text = self._UNIT_RE.sub(" ", text)
        text = self._PAREN_RE.sub(" ", text)
        text = self._PUNCT_RE.sub(" ", text)
        text = re.sub(r"\s*,\s*", ", ", text)
        text = self._SPACE_RE.sub(" ", text)
        text = re.sub(r"(?:,\s*)+", ", ", text)
        return text.strip(" ,")

    def split_ingredients(self, cleaned_str: str) -> List[str]:
        """Split a cleaned string on commas and drop empty tokens."""
        if not cleaned_str:
            return []
        return [token.strip() for token in cleaned_str.split(",") if token.strip()]
