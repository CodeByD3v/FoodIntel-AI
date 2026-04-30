"""Ingredient Extraction Module for NLP Pipeline.

Extracts ingredient names from raw text using NER and pattern matching.
"""

import re
from typing import List, Tuple
import pandas as pd


class IngredientExtractor:
    """Extract and normalize ingredient names from text."""
    
    COMMON_INGREDIENTS = {
        'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck', 'fish', 'shrimp',
        'salmon', 'tuna', 'cod', 'tilapia', 'crab', 'lobster', 'egg',
        'rice', 'pasta', 'bread', 'flour', 'noodle', 'couscous', 'quinoa',
        'potato', 'sweet potato', 'carrot', 'onion', 'garlic', 'tomato',
        'spinach', 'lettuce', 'cabbage', 'broccoli', 'cauliflower', 'celery',
        'pepper', 'bell pepper', 'chili', 'cucumber', 'zucchini', 'mushroom',
        'apple', 'banana', 'orange', 'lemon', 'lime', 'berry', 'grape',
        'milk', 'cream', 'butter', 'cheese', 'yogurt', 'sour cream',
        'oil', 'olive oil', 'vegetable oil', 'coconut oil', 'sesame oil',
        'salt', 'pepper', 'sugar', 'honey', 'vinegar', 'soy sauce',
        'garlic powder', 'onion powder', 'paprika', 'cumin', 'coriander',
        'turmeric', 'ginger', 'cinnamon', 'oregano', 'basil', 'thyme',
        'chicken broth', 'beef broth', 'vegetable broth', 'stock',
        'corn', 'bean', 'lentil', 'chickpea', 'peas', 'bacon', 'ham',
        'sausage', 'peanut butter', 'almond', 'walnut', 'cashew', 'pecan',
        'avocado', 'coconut', 'mango', 'pineapple', 'watermelon', 'strawberry'
    }
    
    QUANTITY_PATTERNS = [
        r'\d+\s*(?:cup|cups|tbsp|tsp|oz|ounce|ounces|lb|pound|pounds|g|gram|grams|kg|ml|l)',
        r'\d+\s*(?:piece|pieces|slice|slices|clove|cloves|pinch|dash)',
        r'\d+\/\d+',
        r'\d+\s*-\s*\d+',
    ]
    
    def __init__(self):
        self.common_ingredients = {ing.lower() for ing in self.COMMON_INGREDIENTS}
    
    def extract(self, text: str) -> List[str]:
        """Extract ingredients from text."""
        text = text.lower()
        
        text = re.sub(r'[^\w\s,]', ' ', text)
        
        found_ingredients = []
        
        for ingredient in self.common_ingredients:
            if ingredient in text:
                found_ingredients.append(ingredient)
        
        parts = text.split(',')
        for part in parts:
            part = part.strip()
            if part and len(part) > 2:
                words = part.split()
                if words[0].isdigit():
                    part = ' '.join(words[1:])
                if part and part not in found_ingredients:
                    if len(part) > 3:
                        found_ingredients.append(part)
        
        seen = set()
        unique_ingredients = []
        for ing in found_ingredients:
            ing_clean = ing.strip()
            if ing_clean and ing_clean not in seen:
                seen.add(ing_clean)
                unique_ingredients.append(ing_clean)
        
        return unique_ingredients
    
    def normalize_ingredient(self, ingredient: str) -> str:
        """Normalize ingredient name for KG matching."""
        ingredient = ingredient.lower().strip()
        ingredient = re.sub(r'\s+', ' ', ingredient)
        
        replacements = {
            'chicken breast': 'chicken',
            'ground beef': 'beef',
            'ground pork': 'pork',
            'bell pepper': 'pepper',
            'green onion': 'onion',
            'spring onion': 'onion',
            'garlic clove': 'garlic',
            'onion powder': 'garlic',
            'garlic powder': 'garlic',
            'olive oil': 'oil',
            'vegetable oil': 'oil',
            'coconut oil': 'oil',
            'sea salt': 'salt',
            'table salt': 'salt',
        }
        
        return replacements.get(ingredient, ingredient)


def load_usda_ingredients(usda_path: str = None) -> dict:
    """Load USDA food names for matching."""
    food_lookup = {}
    
    return food_lookup


def extract_ingredients_from_dataframe(df: pd.DataFrame, text_column: str = 'ingredients') -> pd.DataFrame:
    """Extract ingredients from DataFrame."""
    extractor = IngredientExtractor()
    
    df['extracted_ingredients'] = df[text_column].apply(
        lambda x: extractor.extract(str(x)) if pd.notna(x) else []
    )
    
    df['normalized_ingredients'] = df['extracted_ingredients'].apply(
        lambda ingredients: [extractor.normalize_ingredient(ing) for ing in ingredients]
    )
    
    return df


if __name__ == "__main__":
    test_text = "2 cups chicken breast, 1/2 cup olive oil, salt and pepper, 3 cloves garlic"
    
    extractor = IngredientExtractor()
    ingredients = extractor.extract(test_text)
    
    print(f"Original: {test_text}")
    print(f"Extracted: {ingredients}")
    
    normalized = [extractor.normalize_ingredient(ing) for ing in ingredients]
    print(f"Normalized: {normalized}")