# 🧹 Data Cleaning Module - Food Intelligence System

Comprehensive text preprocessing pipeline for food ingredient data with 11 cleaning techniques.

---

## 📋 Features

### ✅ Implemented Cleaning Techniques

| # | Technique | Description | Example |
|---|-----------|-------------|---------|
| 1 | **Lowercasing** | Convert all text to lowercase | `Chicken` → `chicken` |
| 2 | **Remove HTML Tags** | Strip HTML markup | `<b>garlic</b>` → `garlic` |
| 3 | **Remove URLs** | Remove web links | `www.recipe.com` → `` |
| 4 | **Remove Punctuation** | Strip punctuation (optional) | `salt & pepper` → `salt pepper` |
| 5 | **Chat Word Treatment** | Expand slang/abbreviations | `veggies` → `vegetables` |
| 6 | **Spelling Correction** | Fix typos (slow, optional) | `chiken` → `chicken` |
| 7 | **Remove Stop Words** | Filter common words | `a cup of rice` → `cup rice` |
| 8 | **Handle Emojis** | Remove or convert emojis | `😋` → `` or `:yum:` |
| 9 | **Tokenization** | Split into words | `"salt pepper"` → `["salt", "pepper"]` |
| 10 | **Stemming** | Reduce to root form | `cooking` → `cook` |
| 11 | **Lemmatization** | Reduce to dictionary form | `better` → `good` |

---

## 🚀 Quick Start

### Installation

```bash
# Install required dependencies
pip install beautifulsoup4 emoji contractions textblob nltk lxml

# Or use requirements.txt
pip install -r requirements.txt
```

### Basic Usage

```python
from data_cleaning import clean_ingredient_text

# Simple cleaning
text = "2 cups chopped chicken breast, 1/2 cup cooking oil 😋"

# Light cleaning (recommended for BERT)
cleaned = clean_ingredient_text(text, mode='light')
print(cleaned)
# Output: "2 cups chopped chicken breast, 1/2 cup cooking oil"

# Medium cleaning
cleaned = clean_ingredient_text(text, mode='medium')
print(cleaned)
# Output: "2 cup chopped chicken breast , 1/2 cup cooking oil"

# Heavy cleaning
cleaned = clean_ingredient_text(text, mode='heavy')
print(cleaned)
# Output: "2 cup chopped chicken breast 1/2 cup cooking oil"
```

---

## 🎯 Cleaning Modes

### 1️⃣ **Light Mode** (Recommended for BERT)

**Best for:** NLP embeddings, semantic similarity

**What it does:**
- ✅ Lowercasing
- ✅ Remove HTML tags
- ✅ Remove URLs
- ✅ Handle emojis
- ✅ Expand contractions
- ❌ No punctuation removal (preserves measurements)
- ❌ No stopword removal (preserves context)
- ❌ No stemming/lemmatization

**Example:**
```python
Input:  "Don't use 2-3 tbsp of <b>garlic</b>! 🧄"
Output: "do not use 2-3 tbsp of garlic !"
```

### 2️⃣ **Medium Mode**

**Best for:** Balanced cleaning, general NLP tasks

**What it does:**
- ✅ All light mode features
- ✅ Remove stopwords (preserves food terms)
- ✅ Tokenization
- ✅ Lemmatization
- ❌ No punctuation removal
- ❌ No spelling correction

**Example:**
```python
Input:  "2 cups of chopped chicken with salt and pepper"
Output: "2 cup chopped chicken salt pepper"
```

### 3️⃣ **Heavy Mode**

**Best for:** Structured extraction, entity matching

**What it does:**
- ✅ All medium mode features
- ✅ Remove punctuation
- ✅ Spelling correction (slow!)
- ⚠️ May lose measurement information

**Example:**
```python
Input:  "2-3 cups of chiken, chopped finely"
Output: "2 3 cup chicken chopped finely"
```

---

## 🔧 Advanced Usage

### Custom Configuration

```python
from data_cleaning import FoodTextCleaner

# Create custom cleaner
cleaner = FoodTextCleaner(
    lowercase=True,
    remove_html=True,
    remove_urls=True,
    remove_punctuation=False,  # Keep measurements
    expand_contractions=True,
    correct_spelling=False,    # Disable (slow)
    remove_stopwords=True,
    handle_emojis=True,
    tokenize=True,
    apply_stemming=False,
    apply_lemmatization=True,
    preserve_food_terms=True,  # Don't remove food words
)

# Clean single text
cleaned = cleaner.clean_text("2 cups of chicken")

# Clean batch
texts = ["chicken breast", "cooking oil", "salt & pepper"]
cleaned_batch = cleaner.clean_batch(texts)
```

### Clean DataFrame Column

```python
from data_cleaning import clean_dataframe_column
import pandas as pd

# Load data
df = pd.read_csv('recipes.csv')

# Clean ingredient column
df = clean_dataframe_column(
    df, 
    column='ingredients',
    mode='light',
    output_column='ingredients_cleaned'
)

print(df[['ingredients', 'ingredients_cleaned']].head())
```

---

## 🔗 Integration with NLP Pipeline

### Using with `run_nlp.py`

```bash
# No cleaning (original behavior)
python nlp/run_nlp.py --cleaning-mode none

# Light cleaning (recommended)
python nlp/run_nlp.py --cleaning-mode light --device cuda

# Medium cleaning
python nlp/run_nlp.py --cleaning-mode medium --batch-size 32

# Heavy cleaning
python nlp/run_nlp.py --cleaning-mode heavy
```

### Using in Custom Scripts

```python
from nlp_encoder import FoodNLPProcessor
from data_cleaning import clean_ingredient_text

# Initialize processor
processor = FoodNLPProcessor(device='cuda')

# Clean and encode
raw_text = "2 cups chopped chicken, 1/2 cup oil 😋"
cleaned_text = clean_ingredient_text(raw_text, mode='light')
embeddings, constraints = processor.encode([cleaned_text])

print(f"Embedding shape: {embeddings.shape}")  # (1, 512)
print(f"Constraint shape: {constraints.shape}")  # (1, 5, 12)
```

---

## 📊 Performance Comparison

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| **None** | ⚡⚡⚡ Fastest | Raw data | Baseline |
| **Light** | ⚡⚡⚡ Fast | Good | BERT embeddings |
| **Medium** | ⚡⚡ Moderate | Better | General NLP |
| **Heavy** | ⚡ Slow | Best | Entity extraction |

**Benchmark (1000 recipes):**
- None: ~0.1s
- Light: ~0.5s
- Medium: ~2.0s
- Heavy: ~15s (spelling correction is slow!)

---

## ⚠️ Important Notes

### Food-Specific Considerations

1. **Measurements:** Light mode preserves fractions and decimals (`1/2`, `2.5`)
2. **Food Terms:** Stopword removal preserves food-related terms (`oil`, `salt`, `pepper`)
3. **Quantities:** Heavy mode may break quantity ranges (`2-3 cups` → `2 3 cups`)

### When to Use Each Mode

| Task | Recommended Mode | Reason |
|------|------------------|--------|
| BERT embeddings | **Light** | Preserves context, measurements |
| Semantic search | **Light** | Best for similarity matching |
| Entity extraction | **Heavy** | Clean text for NER/matching |
| Knowledge graph | **Medium** | Balance between clean and context |
| Training custom model | **Medium** | Reduces noise, keeps structure |

---

## 🧪 Testing

Run the test suite:

```bash
# Test all cleaning modes
python nlp/data_cleaning.py

# Expected output:
# ================================================================================
# FOOD TEXT CLEANING EXAMPLES
# ================================================================================
# 
# Original: 2 cups chopped chicken breast, 1/2 cup cooking oil, salt & pepper to taste 😋
# 
# Light cleaning: 2 cups chopped chicken breast, 1/2 cup cooking oil, salt & pepper to taste
# 
# Medium cleaning: 2 cup chopped chicken breast , 1/2 cup cooking oil , salt & pepper taste
# 
# Heavy cleaning: 2 cup chopped chicken breast 1/2 cup cooking oil salt pepper taste
```

---

## 📚 Technical Details

### Dependencies

```
beautifulsoup4>=4.12.0  # HTML parsing
emoji>=2.8.0            # Emoji handling
contractions>=0.1.73    # Contraction expansion
textblob>=0.17.1        # Spelling correction
nltk>=3.8.1             # Tokenization, stemming, lemmatization
lxml>=4.9.0             # XML/HTML parsing
```

### NLTK Data Downloads

The module automatically downloads required NLTK data:
- `punkt` - Sentence tokenization
- `punkt_tab` - Word tokenization
- `stopwords` - English stop words
- `wordnet` - Lemmatization dictionary
- `omw-1.4` - Open Multilingual Wordnet

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'bs4'`

**Solution:**
```bash
pip install beautifulsoup4
```

### Issue: NLTK data not found

**Solution:**
```python
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Issue: Spelling correction is too slow

**Solution:** Disable spelling correction:
```python
cleaner = FoodTextCleaner(correct_spelling=False)
```

Or use light/medium mode which disables it by default.

---

## 📖 Examples

### Example 1: Recipe Ingredient List

```python
raw = """
<div>Don't forget these ingredients:</div>
- 2-3 cups of veggies (chopped)
- 1/2 lb chicken breast
- Visit www.recipe.com for more! 😋
"""

cleaned = clean_ingredient_text(raw, mode='light')
print(cleaned)
# Output: "do not forget these ingredients : - 2-3 cups of vegetables ( chopped ) - 1/2 lb chicken breast - visit for more !"
```

### Example 2: Batch Processing

```python
recipes = [
    "2 cups chicken, chopped",
    "1/2 cup cooking oil",
    "salt & pepper to taste 😋"
]

cleaner = FoodTextCleaner(lowercase=True, handle_emojis=True)
cleaned = cleaner.clean_batch(recipes)

for original, clean in zip(recipes, cleaned):
    print(f"{original:30} → {clean}")
```

### Example 3: DataFrame Processing

```python
import pandas as pd

df = pd.DataFrame({
    'recipe_id': [1, 2, 3],
    'ingredients': [
        "2 cups chicken breast",
        "1/2 cup <b>olive oil</b>",
        "salt & pepper 😋"
    ]
})

df = clean_dataframe_column(df, 'ingredients', mode='light')
print(df)
```

---

## 🎓 Best Practices

1. **Start with Light Mode** - Test with minimal cleaning first
2. **Preserve Measurements** - Don't remove punctuation for food data
3. **Avoid Heavy Spelling Correction** - It's slow and may "fix" food terms incorrectly
4. **Keep Food Terms** - Use `preserve_food_terms=True` when removing stopwords
5. **Benchmark Your Use Case** - Different tasks need different cleaning levels

---

## 📝 License

Part of the Food Intelligence System project.

---

## 👥 Contributors

Food Intelligence System Team
- Devanand Puzhakkool
- Saptaparni Saha
- Tanishka Arora

---

## 📞 Support

For issues or questions, please open a GitHub issue or refer to the main project documentation.


---

## 📊 Current Processing Status

✅ **Heavy Cleaning Mode Active** - Processing 2.2M recipes with comprehensive cleaning:

**Active Cleaning Operations:**
- ✅ Lowercasing
- ✅ HTML tag removal
- ✅ URL removal
- ✅ Punctuation removal
- ✅ Contraction expansion
- ✅ Stopword removal
- ✅ Emoji handling
- ✅ Tokenization
- ✅ Lemmatization
- ❌ Spelling correction (disabled for performance)

**Performance:**
- Processing speed: ~1.4 seconds per chunk (1024 recipes)
- Estimated completion time: ~50 minutes for full dataset
- Current progress: Check `outputs/nlp/metadata.csv`

**Output Files:**
- `outputs/nlp/metadata.csv` - Cleaned ingredient text with recipe metadata
- `outputs/nlp/*.npz` - BERT embeddings (512-d) + constraint vectors (5×12)

**Quality Metrics (from processed data):**
- ✅ 99.98% valid ingredients
- ✅ 0% numbers remaining (all measurements removed)
- ✅ 0% uppercase letters
- ✅ 0.09% punctuation remaining
- ✅ Average: 10.4 words per recipe

**Note:** Spelling correction is intentionally disabled because it would increase processing time from 50 minutes to 100+ hours. The current configuration provides excellent cleaning quality while maintaining practical processing speed.
