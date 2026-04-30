"""spaCy-based ingredient extraction with deterministic food phrase support."""

from __future__ import annotations

from typing import Dict, List

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span


class IngredientNER:
    """Extract ingredient names from cleaned ingredient text."""

    FOOD_PHRASES = [
        "chicken", "chicken breast", "beef", "ground beef", "pork", "bacon",
        "salmon", "tuna", "shrimp", "egg", "eggs", "milk", "butter",
        "cream", "heavy cream", "cheese", "yogurt", "rice", "basmati rice",
        "brown rice", "pasta", "noodles", "bread", "wheat flour",
        "all purpose flour", "sugar", "brown sugar", "salt", "black pepper",
        "pepper", "olive oil", "vegetable oil", "cooking oil", "canola oil",
        "sesame oil", "garlic", "onion", "tomato", "potato", "carrot",
        "celery", "spinach", "lettuce", "broccoli", "cauliflower", "peas",
        "corn", "beans", "lentils", "chickpeas", "mushroom", "ginger",
        "chili pepper", "bell pepper", "lemon", "lime", "apple", "banana",
        "orange", "strawberry", "cilantro", "parsley", "basil", "oregano",
        "thyme", "cumin", "turmeric", "paprika", "cinnamon", "soy sauce",
        "vinegar", "honey", "mayonnaise", "mustard",
    ]

    VARIANT_MAP: Dict[str, str] = {
        "chili": "chili pepper",
        "chilli": "chili pepper",
        "chiles": "chili pepper",
        "veg oil": "vegetable oil",
        "veggie oil": "vegetable oil",
        "cooking oil": "vegetable oil",
        "evoo": "olive oil",
        "extra virgin olive oil": "olive oil",
        "heavy cream": "cream",
        "whipping cream": "cream",
        "all-purpose flour": "wheat flour",
        "all purpose flour": "wheat flour",
        "ap flour": "wheat flour",
        "plain flour": "wheat flour",
        "caster sugar": "sugar",
        "granulated sugar": "sugar",
        "confectioners sugar": "powdered sugar",
        "icing sugar": "powdered sugar",
        "scallion": "green onion",
        "scallions": "green onion",
        "spring onion": "green onion",
        "spring onions": "green onion",
        "garbanzo beans": "chickpeas",
        "garbanzo bean": "chickpeas",
        "cilantro leaves": "cilantro",
        "coriander leaves": "cilantro",
        "aubergine": "eggplant",
        "courgette": "zucchini",
        "bell peppers": "bell pepper",
        "capsicum": "bell pepper",
        "minced beef": "ground beef",
        "beef mince": "ground beef",
        "chicken breasts": "chicken breast",
        "boneless chicken breast": "chicken breast",
        "basmati": "basmati rice",
        "black peppercorn": "black pepper",
        "peppercorns": "black pepper",
        "kosher salt": "salt",
        "sea salt": "salt",
        "sodium chloride": "salt",
    }

    VALID_ENTITY_LABELS = {"FOOD", "PRODUCT", "ORG"}

    def __init__(self, model: str = "en_core_web_sm") -> None:
        try:
            self.nlp = spacy.load(model)
        except OSError:
            self.nlp = spacy.blank("en")

        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(phrase) for phrase in self.FOOD_PHRASES]
        self.matcher.add("FOOD", patterns)

    def extract(self, cleaned_str: str) -> List[str]:
        """Return normalized ingredient names, preserving first occurrence order."""
        if not cleaned_str:
            return []

        doc = self.nlp(cleaned_str)
        spans: List[Span] = [
            ent for ent in doc.ents if ent.label_ in self.VALID_ENTITY_LABELS
        ]
        spans.extend(Span(doc, start, end, label="FOOD") for _, start, end in self.matcher(doc))
        spans = spacy.util.filter_spans(spans)

        if spans:
            candidates = [span.text for span in spans]
        else:
            candidates = [token.strip() for token in cleaned_str.split(",") if token.strip()]

        seen = set()
        ingredients: List[str] = []
        for candidate in candidates:
            normalized = self.normalize_ingredient(candidate)
            if normalized and normalized not in seen:
                ingredients.append(normalized)
                seen.add(normalized)
        return ingredients

    def normalize_ingredient(self, name: str) -> str:
        """Map common ingredient variants to canonical names."""
        key = " ".join(name.lower().replace("-", " ").split())
        return self.VARIANT_MAP.get(key, key)
