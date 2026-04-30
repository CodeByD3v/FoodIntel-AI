"""Reusable data-cleaning utilities for the existing NLP pipeline.

This file is the Python module version of ``data_cleaning_comprehensive.ipynb``.
It keeps the notebook's cleaning steps available to scripts:

1. HTML removal
2. URL removal
3. emoji handling
4. contraction expansion
5. chat-word normalization
6. measurement normalization
7. lowercasing
8. punctuation / special-character cleaning
9. optional spelling correction
10. tokenization
11. stopword removal, stemming, lemmatization, duplicate removal
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

warnings.filterwarnings("ignore")

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency fallback
    BeautifulSoup = None

try:
    import emoji
except ImportError:  # pragma: no cover - optional dependency fallback
    emoji = None

try:
    import contractions
except ImportError:  # pragma: no cover - optional dependency fallback
    contractions = None

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError:  # pragma: no cover - optional dependency fallback
    nltk = None
    stopwords = None
    PorterStemmer = None
    WordNetLemmatizer = None
    word_tokenize = None

try:
    from textblob import TextBlob
except ImportError:  # pragma: no cover - optional dependency fallback
    TextBlob = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIRST_PART = PROJECT_ROOT / "datasets" / "first_part.csv"
DEFAULT_SECOND_PART = PROJECT_ROOT / "datasets" / "second_part.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "cleaned_ingredients.csv"


FOOD_TERMS = {
    "oil", "salt", "pepper", "sugar", "butter", "cream", "milk", "water",
    "flour", "egg", "rice", "meat", "fish", "chicken", "beef", "pork",
    "cheese", "bread", "pasta", "sauce", "spice", "onion", "garlic",
    "tomato", "potato", "carrot", "celery", "honey", "vinegar",
    "mustard", "ketchup", "mayo", "lettuce", "spinach", "cabbage",
    "broccoli", "cauliflower", "apple", "banana", "orange", "lemon",
    "lime", "berry", "lamb", "turkey", "duck", "shrimp", "crab",
    "bacon", "ham", "sausage", "corn", "bean", "pea", "basil", "thyme",
}


CHAT_WORD_DICT = {
    "yum": "delicious",
    "delish": "delicious",
    "nom": "eat",
    "noms": "food",
    "veggies": "vegetables",
    "sammie": "sandwich",
    "sammy": "sandwich",
    "za": "pizza",
    "apps": "appetizers",
    "entree": "main course",
    "tasty": "delicious",
    "yummy": "delicious",
    "scrummy": "delicious",
    "gr8": "great",
    "luv": "love",
    "wanna": "want to",
    "gonna": "going to",
    "gotta": "got to",
    "imo": "in my opinion",
    "omg": "oh my god",
}


CONTRACTION_FALLBACKS = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "won't": "will not",
    "i'm": "i am",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "i'll": "i will",
    "you'll": "you will",
}


FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with", "or", "some", "into", "then", "than", "this",
}


class FoodTextCleaner:
    """Configurable cleaner for recipe and ingredient text."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_html: bool = True,
        remove_urls_flag: bool = True,
        handle_emojis_flag: bool = True,
        expand_contractions_flag: bool = True,
        replace_chat_words_flag: bool = True,
        normalize_measurements_flag: bool = False,
        remove_punctuation_flag: bool = False,
        remove_special_chars_flag: bool = False,
        correct_spelling_flag: bool = False,
        remove_stopwords_flag: bool = False,
        apply_stemming_flag: bool = False,
        apply_lemmatization_flag: bool = False,
        remove_duplicates_flag: bool = False,
        preserve_food_terms: bool = True,
        download_nltk: bool = False,
    ) -> None:
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_urls_flag = remove_urls_flag
        self.handle_emojis_flag = handle_emojis_flag
        self.expand_contractions_flag = expand_contractions_flag
        self.replace_chat_words_flag = replace_chat_words_flag
        self.normalize_measurements_flag = normalize_measurements_flag
        self.remove_punctuation_flag = remove_punctuation_flag
        self.remove_special_chars_flag = remove_special_chars_flag
        self.correct_spelling_flag = correct_spelling_flag
        self.remove_stopwords_flag = remove_stopwords_flag
        self.apply_stemming_flag = apply_stemming_flag
        self.apply_lemmatization_flag = apply_lemmatization_flag
        self.remove_duplicates_flag = remove_duplicates_flag
        self.preserve_food_terms = preserve_food_terms

        if self._needs_nltk() and download_nltk:
            ensure_nltk_data(download=True)

        self.stop_words = load_stopwords()
        self.stemmer = PorterStemmer() if PorterStemmer else None
        self.lemmatizer = WordNetLemmatizer() if WordNetLemmatizer else None

    def clean_text(self, text: object) -> str:
        """Apply the configured cleaning pipeline to one text value."""
        if is_missing_text(text):
            return ""

        cleaned = str(text)
        if self.remove_html:
            cleaned = remove_html_tags(cleaned)
        if self.remove_urls_flag:
            cleaned = remove_urls(cleaned)
        if self.handle_emojis_flag:
            cleaned = handle_emojis(cleaned)
        if self.expand_contractions_flag:
            cleaned = expand_contractions(cleaned)
        if self.replace_chat_words_flag:
            cleaned = replace_chat_words(cleaned)
        if self.normalize_measurements_flag:
            cleaned = normalize_measurements(cleaned)
        if self.lowercase:
            cleaned = lowercase_text(cleaned)
        cleaned = clean_whitespace(cleaned)
        if self.remove_punctuation_flag:
            cleaned = remove_punctuation(cleaned)
        if self.remove_special_chars_flag:
            cleaned = remove_special_chars(cleaned)
        cleaned = remove_extra_spaces(cleaned)

        if self.correct_spelling_flag:
            cleaned = correct_spelling(cleaned)

        if self._needs_token_processing():
            tokens = tokenize_text(cleaned)
            if self.remove_stopwords_flag:
                tokens = remove_stopwords(
                    tokens,
                    stop_words_set=self.stop_words,
                    preserve_food_terms=self.preserve_food_terms,
                )
            if self.apply_stemming_flag:
                tokens = apply_stemming(tokens, stemmer=self.stemmer)
            if self.apply_lemmatization_flag:
                tokens = apply_lemmatization(tokens, lemmatizer=self.lemmatizer)
            cleaned = " ".join(tokens)

        if self.remove_duplicates_flag:
            cleaned = remove_duplicates(cleaned)

        return cleaned.strip()

    def clean_batch(self, texts: Iterable[object]) -> List[str]:
        """Clean an iterable of text values."""
        return [self.clean_text(text) for text in texts]

    def _needs_nltk(self) -> bool:
        return self._needs_token_processing()

    def _needs_token_processing(self) -> bool:
        return (
            self.remove_stopwords_flag
            or self.apply_stemming_flag
            or self.apply_lemmatization_flag
        )


def is_missing_text(text: object) -> bool:
    """Return True for None, NaN, or empty strings."""
    if text is None:
        return True
    try:
        if pd.isna(text):
            return True
    except (TypeError, ValueError):
        return False
    return str(text).strip() == ""


def ensure_nltk_data(download: bool = False) -> None:
    """Check or optionally download NLTK resources used by this module."""
    if nltk is None:
        return

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if download:
                nltk.download(package_name, quiet=True)


def load_stopwords() -> set[str]:
    """Load stopwords from NLTK when available, otherwise use a local fallback."""
    if stopwords is None:
        return set(FALLBACK_STOPWORDS)
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return set(FALLBACK_STOPWORDS)


def remove_html_tags(text: str) -> str:
    """Remove HTML markup while preserving visible text."""
    if BeautifulSoup is not None:
        return BeautifulSoup(text, "html.parser").get_text(separator=" ")
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Remove http(s) and www URLs."""
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    return text


def handle_emojis(text: str) -> str:
    """Remove emojis from text."""
    if emoji is not None:
        return emoji.replace_emoji(text, replace="")
    return re.sub(
        "[\U0001f300-\U0001faff\U00002700-\U000027bf\U00002600-\U000026ff]",
        "",
        text,
    )


def expand_contractions(text: str) -> str:
    """Expand common contractions."""
    if contractions is not None:
        try:
            return contractions.fix(text)
        except Exception:
            return text

    expanded = text
    for short, long_form in CONTRACTION_FALLBACKS.items():
        expanded = re.sub(rf"\b{re.escape(short)}\b", long_form, expanded, flags=re.IGNORECASE)
    return expanded


def replace_chat_words(text: str) -> str:
    """Replace common food and chat abbreviations with normalized terms."""
    return " ".join(CHAT_WORD_DICT.get(word.lower(), word) for word in text.split())


def lowercase_text(text: str) -> str:
    """Lowercase text."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Remove punctuation while preserving fractions, decimals, and ranges."""
    return re.sub(r"[^\w\s/.\-]", " ", text)


def correct_spelling(text: str) -> str:
    """Correct spelling with TextBlob when installed."""
    if TextBlob is None:
        return text
    try:
        return str(TextBlob(text).correct())
    except Exception:
        return text


def tokenize_text(text: str) -> List[str]:
    """Tokenize text using NLTK when available, otherwise regex tokenization."""
    if word_tokenize is not None:
        try:
            return word_tokenize(text)
        except LookupError:
            return re.findall(r"[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)?", text)
    return re.findall(r"[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)?", text)


def remove_stopwords(
    tokens: List[str],
    stop_words_set: Optional[set[str]] = None,
    preserve_food_terms: bool = True,
) -> List[str]:
    """Remove stopwords, preserving important food terms by default."""
    active_stopwords = stop_words_set if stop_words_set is not None else load_stopwords()
    return [
        token
        for token in tokens
        if token.lower() not in active_stopwords
        or (preserve_food_terms and token.lower() in FOOD_TERMS)
    ]


def apply_stemming(tokens: List[str], stemmer: object = None) -> List[str]:
    """Apply Porter stemming when NLTK is available."""
    active_stemmer = stemmer or (PorterStemmer() if PorterStemmer else None)
    if active_stemmer is None:
        return tokens
    return [active_stemmer.stem(token) for token in tokens]


def apply_lemmatization(tokens: List[str], lemmatizer: object = None) -> List[str]:
    """Apply WordNet lemmatization when NLTK is available."""
    active_lemmatizer = lemmatizer or (WordNetLemmatizer() if WordNetLemmatizer else None)
    if active_lemmatizer is None:
        return tokens
    try:
        return [active_lemmatizer.lemmatize(token) for token in tokens]
    except LookupError:
        return tokens


def clean_whitespace(text: str) -> str:
    """Collapse repeated whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def clean_numbers(text: str) -> str:
    """Remove numeric characters."""
    return re.sub(r"\d+", "", text)


def remove_special_chars(text: str) -> str:
    """Keep only alphabetic characters and spaces."""
    return re.sub(r"[^a-zA-Z\s]", " ", text)


def remove_extra_spaces(text: str) -> str:
    """Collapse repeated spaces and strip boundaries."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_measurements(text: str) -> str:
    """Expand common measurement abbreviations."""
    replacements = {
        r"\btsp\b": "teaspoon",
        r"\btbsp\b": "tablespoon",
        r"\bc\b": "cup",
        r"\bpkg\b": "package",
        r"\boz\b": "ounce",
        r"\blb\b": "pound",
        r"\bg\b": "gram",
        r"\bkg\b": "kilogram",
        r"\bml\b": "milliliter",
    }
    normalized = text
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


def remove_duplicates(text: str) -> str:
    """Remove repeated words while preserving first-occurrence order."""
    seen = set()
    result = []
    for word in text.split():
        key = word.lower()
        if key not in seen:
            seen.add(key)
            result.append(word)
    return " ".join(result)


def clean_text(text: object, apply_spelling: bool = False) -> str:
    """Notebook-compatible standard cleaning function."""
    cleaner = FoodTextCleaner(
        lowercase=True,
        remove_html=True,
        remove_urls_flag=True,
        handle_emojis_flag=True,
        expand_contractions_flag=True,
        replace_chat_words_flag=True,
        remove_punctuation_flag=True,
        correct_spelling_flag=apply_spelling,
        remove_stopwords_flag=True,
        apply_stemming_flag=True,
        apply_lemmatization_flag=True,
        preserve_food_terms=True,
    )
    return cleaner.clean_text(text)


def complete_clean(text: object, apply_spelling: bool = False) -> str:
    """Notebook-compatible enhanced cleaning pipeline."""
    cleaner = FoodTextCleaner(
        lowercase=True,
        remove_html=True,
        remove_urls_flag=True,
        handle_emojis_flag=True,
        expand_contractions_flag=True,
        replace_chat_words_flag=True,
        normalize_measurements_flag=True,
        remove_special_chars_flag=True,
        correct_spelling_flag=apply_spelling,
        remove_stopwords_flag=True,
        apply_stemming_flag=True,
        apply_lemmatization_flag=True,
        remove_duplicates_flag=True,
        preserve_food_terms=True,
    )
    return cleaner.clean_text(text)


def clean_ingredient_text(text: object, mode: str = "light", apply_spelling: bool = False) -> str:
    """Clean one ingredient text with a named preset.

    ``light`` is best for BERT inputs because it keeps measurements and syntax.
    ``medium`` removes stopwords and lemmatizes.
    ``heavy`` mirrors the notebook's ``complete_clean`` behavior.
    """
    mode = mode.lower()
    if mode == "light":
        cleaner = FoodTextCleaner(
            lowercase=True,
            remove_html=True,
            remove_urls_flag=True,
            handle_emojis_flag=True,
            expand_contractions_flag=True,
            replace_chat_words_flag=True,
            remove_punctuation_flag=False,
            correct_spelling_flag=apply_spelling,
        )
        return cleaner.clean_text(text)
    if mode == "medium":
        cleaner = FoodTextCleaner(
            lowercase=True,
            remove_html=True,
            remove_urls_flag=True,
            handle_emojis_flag=True,
            expand_contractions_flag=True,
            replace_chat_words_flag=True,
            normalize_measurements_flag=True,
            remove_punctuation_flag=True,
            correct_spelling_flag=apply_spelling,
            remove_stopwords_flag=True,
            apply_lemmatization_flag=True,
        )
        return cleaner.clean_text(text)
    if mode in {"heavy", "complete"}:
        return complete_clean(text, apply_spelling=apply_spelling)
    raise ValueError(f"Unsupported cleaning mode: {mode}")


def clean_dataframe_column(
    df: pd.DataFrame,
    column: str,
    mode: str = "complete",
    output_column: Optional[str] = None,
    apply_spelling: bool = False,
) -> pd.DataFrame:
    """Add a cleaned version of a DataFrame text column."""
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")
    output_column = output_column or f"{column}_cleaned"
    cleaned_df = df.copy()
    cleaned_df[output_column] = cleaned_df[column].apply(
        lambda value: clean_ingredient_text(value, mode=mode, apply_spelling=apply_spelling)
    )
    return cleaned_df


def process_dataset(
    input_csv: str | Path,
    output_csv: str | Path,
    text_column: str = "ingredients",
    mode: str = "complete",
    apply_spelling: bool = False,
) -> pd.DataFrame:
    """Clean one CSV file and write a new CSV with ``ingredients_cleaned``."""
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    df = pd.read_csv(input_path)
    cleaned = clean_dataframe_column(
        df,
        column=text_column,
        mode=mode,
        output_column="ingredients_cleaned",
        apply_spelling=apply_spelling,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return cleaned


def process_recipe_parts(
    first_csv: str | Path = DEFAULT_FIRST_PART,
    second_csv: str | Path = DEFAULT_SECOND_PART,
    output_csv: str | Path = DEFAULT_OUTPUT,
    text_column: str = "ingredients",
    mode: str = "complete",
    apply_spelling: bool = False,
) -> pd.DataFrame:
    """Replicate the notebook flow: concat first/second parts and clean them."""
    frames = [pd.read_csv(first_csv), pd.read_csv(second_csv)]
    df = pd.concat(frames, ignore_index=True)
    cleaned = clean_dataframe_column(
        df,
        column=text_column,
        mode=mode,
        output_column="ingredients_cleaned",
        apply_spelling=apply_spelling,
    )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return cleaned


def demo_cleaning_steps() -> None:
    """Print the notebook's cleaning-step demonstration."""
    tests = [
        ("LOWERCASING", "2 Cups of Fresh CHICKEN Breast", lowercase_text),
        ("REMOVE HTML TAGS", "<b>Garlic</b> powder and <i>onion</i>", remove_html_tags),
        ("REMOVE URLs", "Get salt from https://example.com and www.spices.org", remove_urls),
        ("REMOVE PUNCTUATION", "Salt, pepper! & (garlic) - 1/2 cup.", remove_punctuation),
        ("CHAT WORD TREATMENT", "This sammy is so yum and delish", lambda value: replace_chat_words(expand_contractions(value))),
        ("SPELLING CORRECTION", "1 cup of chiken broth with vegatables", correct_spelling),
        ("REMOVE STOP WORDS", "add a cup of the rice and water", lambda value: " ".join(remove_stopwords(tokenize_text(value)))),
        ("HANDLE EMOJIS", "Fresh tomatoes and basil for pizza", handle_emojis),
        ("TOKENIZATION", "salt, pepper, and olive oil", lambda value: str(tokenize_text(value))),
        ("STEMMING", "chopped cooking chickens", lambda value: " ".join(apply_stemming(tokenize_text(value)))),
        ("LEMMATIZATION", "leaves and potatoes cooked", lambda value: " ".join(apply_lemmatization(tokenize_text(value)))),
    ]

    print("DEMONSTRATION OF ALL 11 CLEANING STEPS")
    print("=" * 60)
    for step, input_text, func in tests:
        print(f"\n{step}:")
        print(f"  Input:  {input_text}")
        print(f"  Output: {func(input_text)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean recipe ingredient text.")
    parser.add_argument("--input", type=str, help="Input CSV. If omitted, first/second recipe parts are concatenated.")
    parser.add_argument("--second-input", type=str, default=str(DEFAULT_SECOND_PART), help="Second CSV for notebook-style concatenation.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output CSV path.")
    parser.add_argument("--text-column", type=str, default="ingredients", help="Column containing raw ingredients.")
    parser.add_argument("--mode", choices=["light", "medium", "heavy", "complete"], default="complete")
    parser.add_argument("--apply-spelling", action="store_true", help="Enable slow TextBlob spelling correction.")
    parser.add_argument("--download-nltk", action="store_true", help="Download required NLTK resources before cleaning.")
    parser.add_argument("--demo", action="store_true", help="Print cleaning-step demo and exit.")
    args = parser.parse_args()

    if args.download_nltk:
        ensure_nltk_data(download=True)
    if args.demo:
        demo_cleaning_steps()
        return

    if args.input:
        df = process_dataset(
            input_csv=args.input,
            output_csv=args.output,
            text_column=args.text_column,
            mode=args.mode,
            apply_spelling=args.apply_spelling,
        )
    else:
        df = process_recipe_parts(
            first_csv=DEFAULT_FIRST_PART,
            second_csv=args.second_input,
            output_csv=args.output,
            text_column=args.text_column,
            mode=args.mode,
            apply_spelling=args.apply_spelling,
        )

    print(f"Cleaned {len(df)} rows")
    print(f"Saved cleaned dataset to: {args.output}")


if __name__ == "__main__":
    main()
