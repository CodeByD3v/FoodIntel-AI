"""
Comprehensive Data Cleaning Module for Food Intelligence System NLP Pipeline

This module implements all standard NLP preprocessing techniques:
1. Lowercasing
2. Remove HTML Tags
3. Remove URLs
4. Remove Punctuation
5. Chat word treatment
6. Spelling Correction
7. Removing Stop words
8. Handling Emojis
9. Tokenization
10. Stemming
11. Lemmatization

Author: Food Intelligence System Team
"""

import re
import string
import unicodedata
from typing import List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# Third-party imports
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import emoji
import contractions

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


class FoodTextCleaner:
    """
    Comprehensive text cleaning pipeline for food ingredient data.
    
    Features:
    - Configurable cleaning steps
    - Food-specific preprocessing
    - Batch processing support
    - Preserves important food terms
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_punctuation: bool = False,  # Keep for food measurements
        expand_contractions: bool = True,
        correct_spelling: bool = False,  # Expensive, use cautiously
        remove_stopwords: bool = False,  # Keep for context
        handle_emojis: bool = True,
        tokenize: bool = False,
        apply_stemming: bool = False,
        apply_lemmatization: bool = False,
        preserve_food_terms: bool = True,
    ):
        """
        Initialize the text cleaner with configurable options.
        
        Args:
            lowercase: Convert text to lowercase
            remove_html: Remove HTML tags
            remove_urls: Remove URLs
            remove_punctuation: Remove punctuation (careful with measurements)
            expand_contractions: Expand contractions (don't -> do not)
            correct_spelling: Apply spelling correction (slow)
            remove_stopwords: Remove stop words
            handle_emojis: Convert or remove emojis
            tokenize: Tokenize text into words
            apply_stemming: Apply stemming
            apply_lemmatization: Apply lemmatization
            preserve_food_terms: Keep important food-related terms
        """
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_punctuation = remove_punctuation
        self.expand_contractions = expand_contractions
        self.correct_spelling = correct_spelling
        self.remove_stopwords = remove_stopwords
        self.handle_emojis = handle_emojis
        self.tokenize = tokenize
        self.apply_stemming = apply_stemming
        self.apply_lemmatization = apply_lemmatization
        self.preserve_food_terms = preserve_food_terms
        
        # Initialize NLP tools
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Food-specific terms to preserve (don't remove as stopwords)
        self.food_terms = {
            'oil', 'salt', 'pepper', 'sugar', 'butter', 'cream', 'milk',
            'water', 'flour', 'egg', 'rice', 'meat', 'fish', 'chicken',
            'beef', 'pork', 'cheese', 'bread', 'pasta', 'sauce', 'spice'
        }
        
        # Common chat words / slang in food context
        self.chat_word_dict = {
            'yum': 'yummy',
            'delish': 'delicious',
            'nom': 'eat',
            'noms': 'food',
            'veggies': 'vegetables',
            'sammie': 'sandwich',
            'sammy': 'sandwich',
            'za': 'pizza',
            'apps': 'appetizers',
            'entree': 'main course',
        }
    
    def clean_text(self, text: str) -> str:
        """
        Apply all configured cleaning steps to input text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or not text:
            return ""
        
        text = str(text)
        
        # Step 1: Remove HTML tags
        if self.remove_html:
            text = self._remove_html_tags(text)
        
        # Step 2: Remove URLs
        if self.remove_urls:
            text = self._remove_urls(text)
        
        # Step 3: Handle emojis
        if self.handle_emojis:
            text = self._handle_emojis(text)
        
        # Step 4: Expand contractions
        if self.expand_contractions:
            text = self._expand_contractions(text)
        
        # Step 5: Chat word treatment
        text = self._replace_chat_words(text)
        
        # Step 6: Lowercasing
        if self.lowercase:
            text = text.lower()
        
        # Step 7: Remove extra whitespace
        text = self._remove_extra_whitespace(text)
        
        # Step 8: Remove punctuation (optional)
        if self.remove_punctuation:
            text = self._remove_punctuation(text)
        
        # Step 9: Spelling correction (expensive)
        if self.correct_spelling:
            text = self._correct_spelling(text)
        
        # Step 10: Tokenization
        if self.tokenize or self.remove_stopwords or self.apply_stemming or self.apply_lemmatization:
            tokens = self._tokenize(text)
            
            # Step 11: Remove stopwords
            if self.remove_stopwords:
                tokens = self._remove_stopwords(tokens)
            
            # Step 12: Stemming
            if self.apply_stemming:
                tokens = self._apply_stemming(tokens)
            
            # Step 13: Lemmatization
            if self.apply_lemmatization:
                tokens = self._apply_lemmatization(tokens)
            
            text = ' '.join(tokens)
        
        return text.strip()
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of texts.
        
        Args:
            texts: List of raw text strings
            
        Returns:
            List of cleaned text strings
        """
        return [self.clean_text(text) for text in texts]
    
    # ========================================================================
    # Individual Cleaning Methods
    # ========================================================================
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags using BeautifulSoup."""
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text(separator=' ')
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        # Pattern matches http://, https://, www., and common TLDs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        
        # Remove www. URLs
        www_pattern = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(www_pattern, '', text)
        
        return text
    
    def _handle_emojis(self, text: str, mode: str = 'remove') -> str:
        """
        Handle emojis in text.
        
        Args:
            text: Input text
            mode: 'remove' to delete emojis, 'demojize' to convert to text
            
        Returns:
            Text with emojis handled
        """
        if mode == 'demojize':
            # Convert emojis to text descriptions
            return emoji.demojize(text, delimiters=(" ", " "))
        else:
            # Remove emojis
            return emoji.replace_emoji(text, replace='')
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions (don't -> do not)."""
        try:
            return contractions.fix(text)
        except:
            return text
    
    def _replace_chat_words(self, text: str) -> str:
        """Replace chat words and slang with proper words."""
        words = text.split()
        replaced = [self.chat_word_dict.get(word.lower(), word) for word in words]
        return ' '.join(replaced)
    
    def _remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace, tabs, newlines."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation but preserve important food measurements.
        
        Note: Be careful with this for food data as it removes:
        - Fractions: 1/2, 3/4
        - Measurements: 2.5 cups
        - Ranges: 2-3 tablespoons
        """
        # Keep numbers and letters, remove most punctuation
        # But preserve: / (fractions), . (decimals), - (ranges)
        text = re.sub(r'[^\w\s/.\-]', '', text)
        return text
    
    def _correct_spelling(self, text: str) -> str:
        """
        Correct spelling using TextBlob.
        
        Warning: This is slow and may incorrectly "fix" food terms.
        Use with caution on food ingredient data.
        """
        try:
            blob = TextBlob(text)
            corrected = str(blob.correct())
            return corrected
        except:
            return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return word_tokenize(text)
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words but preserve food-related terms.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Filtered list of tokens
        """
        if self.preserve_food_terms:
            # Don't remove food-related terms even if they're stopwords
            return [
                token for token in tokens
                if token.lower() not in self.stop_words or token.lower() in self.food_terms
            ]
        else:
            return [token for token in tokens if token.lower() not in self.stop_words]
    
    def _apply_stemming(self, tokens: List[str]) -> List[str]:
        """
        Apply Porter Stemming to tokens.
        
        Example: cooking -> cook, chopped -> chop
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def _apply_lemmatization(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to tokens.
        
        Example: cooking -> cook, better -> good
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]


# ============================================================================
# Convenience Functions
# ============================================================================

def clean_ingredient_text(
    text: str,
    mode: str = 'light'
) -> str:
    """
    Quick cleaning function with preset configurations.
    
    Args:
        text: Raw ingredient text
        mode: Cleaning mode
            - 'light': Minimal cleaning (recommended for BERT)
            - 'medium': Moderate cleaning
            - 'heavy': Aggressive cleaning (for structured extraction)
    
    Returns:
        Cleaned text
    """
    if mode == 'light':
        # Minimal cleaning - good for BERT embeddings
        cleaner = FoodTextCleaner(
            lowercase=True,
            remove_html=True,
            remove_urls=True,
            remove_punctuation=False,
            expand_contractions=True,
            correct_spelling=False,
            remove_stopwords=False,
            handle_emojis=True,
            tokenize=False,
            apply_stemming=False,
            apply_lemmatization=False,
        )
    elif mode == 'medium':
        # Moderate cleaning
        cleaner = FoodTextCleaner(
            lowercase=True,
            remove_html=True,
            remove_urls=True,
            remove_punctuation=False,
            expand_contractions=True,
            correct_spelling=False,
            remove_stopwords=True,
            handle_emojis=True,
            tokenize=True,
            apply_stemming=False,
            apply_lemmatization=True,
        )
    else:  # heavy
        # Aggressive cleaning - for structured extraction
        # NOTE: Spelling correction disabled due to performance (would take ~100+ hours)
        cleaner = FoodTextCleaner(
            lowercase=True,
            remove_html=True,
            remove_urls=True,
            remove_punctuation=True,
            expand_contractions=True,
            correct_spelling=False,  # Disabled: too slow for large datasets
            remove_stopwords=True,
            handle_emojis=True,
            tokenize=True,
            apply_stemming=False,
            apply_lemmatization=True,
        )
    
    return cleaner.clean_text(text)


def clean_dataframe_column(
    df: pd.DataFrame,
    column: str,
    mode: str = 'light',
    output_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Clean a specific column in a DataFrame.
    
    Args:
        df: Input DataFrame
        column: Column name to clean
        mode: Cleaning mode ('light', 'medium', 'heavy')
        output_column: Name for cleaned column (default: column_cleaned)
    
    Returns:
        DataFrame with cleaned column added
    """
    if output_column is None:
        output_column = f"{column}_cleaned"
    
    df[output_column] = df[column].apply(lambda x: clean_ingredient_text(x, mode=mode))
    return df


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Basic cleaning
    raw_text = "2 cups chopped chicken breast, 1/2 cup cooking oil, salt & pepper to taste 😋"
    
    print("=" * 80)
    print("FOOD TEXT CLEANING EXAMPLES")
    print("=" * 80)
    print(f"\nOriginal: {raw_text}")
    
    # Light cleaning (recommended for BERT)
    light = clean_ingredient_text(raw_text, mode='light')
    print(f"\nLight cleaning: {light}")
    
    # Medium cleaning
    medium = clean_ingredient_text(raw_text, mode='medium')
    print(f"\nMedium cleaning: {medium}")
    
    # Heavy cleaning
    heavy = clean_ingredient_text(raw_text, mode='heavy')
    print(f"\nHeavy cleaning: {heavy}")
    
    # Example 2: Custom cleaning
    print("\n" + "=" * 80)
    print("CUSTOM CLEANING EXAMPLE")
    print("=" * 80)
    
    cleaner = FoodTextCleaner(
        lowercase=True,
        remove_html=True,
        remove_urls=True,
        handle_emojis=True,
        expand_contractions=True,
        tokenize=True,
        apply_lemmatization=True,
    )
    
    test_texts = [
        "Don't forget the <b>garlic</b>! 🧄",
        "Visit www.recipe.com for more yummy recipes",
        "2-3 tbsp of veggies, chopped finely",
    ]
    
    for text in test_texts:
        cleaned = cleaner.clean_text(text)
        print(f"\nOriginal: {text}")
        print(f"Cleaned:  {cleaned}")
