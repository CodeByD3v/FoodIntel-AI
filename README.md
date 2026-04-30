# NLP Module - Food Intelligence System

Comprehensive Natural Language Processing pipeline for food ingredient text encoding and analysis using BERT-based embeddings with constraint vectors.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Module Structure](#module-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Data Cleaning](#data-cleaning)
- [Usage Examples](#usage-examples)
- [Performance](#performance)
- [Output Format](#output-format)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Overview

The NLP module processes food ingredient text to generate:
- **512-dimensional BERT embeddings** for semantic understanding
- **5×12 constraint vectors** for nutritional and categorical information
- **Cleaned text** with configurable preprocessing options

This module is a critical component of the Food Intelligence System, enabling:
- Semantic search and similarity matching
- Recipe recommendation
- Ingredient substitution
- Nutritional analysis
- Knowledge graph construction

---

## Features

### Text Encoding
- BERT-based semantic embeddings (512-dimensional)
- Custom constraint vectors (5×12) for food-specific attributes
- Batch processing support
- GPU acceleration (CUDA)
- Mixed precision training (AMP)

### Data Cleaning
- 11 preprocessing techniques
- 3 preset modes (light, medium, heavy)
- Food-specific term preservation
- Configurable cleaning pipeline
- Batch and DataFrame support

### Quality Assurance
- Output verification tools
- Data quality metrics
- Visualization utilities
- Progress tracking

---

## Module Structure

```
nlp/
├── README.md                              # This file
├── README_DATA_CLEANING_PROFESSIONAL.md   # Detailed cleaning documentation
│
├── Core Processing Files
│   ├── nlp_encoder.py                     # BERT encoder class
│   ├── run_nlp.py                         # Batch processing script
│   └── verify_nlp_output.py               # Output verification
│
├── Data Cleaning
│   ├── data_cleaning.py                   # Cleaning module
│   ├── data_cleaning.ipynb                # Interactive cleaning notebook
│   └── data_cleaning_detailed.ipynb       # Detailed examples
│
└── Notebooks
    └── nlp_text_encoder.ipynb             # Interactive encoding tutorial
```

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space

### Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install NLP-specific packages
pip install torch transformers pandas numpy tqdm
pip install beautifulsoup4 emoji contractions textblob nltk lxml
```

### Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## Quick Start

### 1. Basic Text Encoding

```python
from nlp_encoder import FoodNLPProcessor

# Initialize processor
processor = FoodNLPProcessor(device='cuda')

# Encode single ingredient
text = "2 cups chopped chicken breast"
embeddings, constraints = processor.encode([text])

print(f"Embedding shape: {embeddings.shape}")    # (1, 512)
print(f"Constraint shape: {constraints.shape}")  # (1, 5, 12)
```

### 2. Batch Processing

```bash
# Process entire dataset with light cleaning
python nlp/run_nlp.py --device cuda --batch-size 32 --cleaning-mode light

# Process with heavy cleaning
python nlp/run_nlp.py --device cuda --batch-size 32 --cleaning-mode heavy
```

### 3. Data Cleaning

```python
from data_cleaning import clean_ingredient_text

# Clean text with different modes
text = "2 cups chopped chicken, 1/2 cup oil"

light = clean_ingredient_text(text, mode='light')
medium = clean_ingredient_text(text, mode='medium')
heavy = clean_ingredient_text(text, mode='heavy')
```

---

## Core Components

### 1. nlp_encoder.py - BERT Encoder

**Purpose**: Core encoding engine using BERT for semantic embeddings.

**Key Features**:
- Pre-trained BERT-base-uncased model
- Custom constraint vector generation
- Batch processing with progress tracking
- GPU acceleration support

**Usage**:
```python
from nlp_encoder import FoodNLPProcessor

processor = FoodNLPProcessor(
    device='cuda',           # 'cuda' or 'cpu'
    batch_size=32,          # Batch size for encoding
    max_length=128          # Max token length
)

# Encode texts
embeddings, constraints = processor.encode(texts)
```

**Output**:
- `embeddings`: (N, 512) - BERT embeddings
- `constraints`: (N, 5, 12) - Constraint vectors

---

### 2. run_nlp.py - Batch Processing Script

**Purpose**: Process large datasets (2.2M+ recipes) with cleaning and encoding.

**Key Features**:
- Chunked processing for memory efficiency
- Progress tracking with tqdm
- Automatic output management
- Cleaning mode integration
- Resume capability

**Usage**:
```bash
# Basic usage
python nlp/run_nlp.py

# With options
python nlp/run_nlp.py \
    --device cuda \
    --batch-size 32 \
    --cleaning-mode light \
    --overwrite
```

**Parameters**:
- `--device`: Processing device (cuda/cpu)
- `--batch-size`: Batch size for encoding
- `--cleaning-mode`: Text cleaning mode (none/light/medium/heavy)
- `--no-amp`: Disable mixed precision
- `--overwrite`: Replace existing output

**Output**:
- `outputs/nlp/metadata.csv` - Cleaned text and metadata
- `outputs/nlp/*_shard_*.npz` - Embeddings and constraints

---

### 3. data_cleaning.py - Text Cleaning Module

**Purpose**: Comprehensive text preprocessing with 11 cleaning techniques.

**Key Features**:
- 11 cleaning techniques (lowercasing, HTML removal, etc.)
- 3 preset modes (light, medium, heavy)
- Food-specific term preservation
- Batch and DataFrame support
- Configurable pipeline

**Usage**:
```python
from data_cleaning import FoodTextCleaner, clean_ingredient_text

# Quick cleaning
cleaned = clean_ingredient_text(text, mode='light')

# Custom configuration
cleaner = FoodTextCleaner(
    lowercase=True,
    remove_html=True,
    remove_stopwords=True,
    apply_lemmatization=True
)
cleaned = cleaner.clean_text(text)
```

**Cleaning Modes**:
- **Light**: Minimal cleaning, preserves context (recommended for BERT)
- **Medium**: Balanced cleaning with stopword removal
- **Heavy**: Aggressive cleaning for structured extraction

See [README_DATA_CLEANING_PROFESSIONAL.md](README_DATA_CLEANING_PROFESSIONAL.md) for details.

---

### 4. verify_nlp_output.py - Output Verification

**Purpose**: Verify and analyze processed NLP outputs.

**Key Features**:
- Load and inspect embeddings
- Validate constraint vectors
- Quality metrics
- Visualization utilities

**Usage**:
```bash
python nlp/verify_nlp_output.py
```

---

## Data Cleaning

### Cleaning Techniques

| # | Technique | Description | Example |
|---|-----------|-------------|---------|
| 1 | Lowercasing | Convert to lowercase | `Chicken` → `chicken` |
| 2 | HTML Removal | Strip HTML tags | `<b>garlic</b>` → `garlic` |
| 3 | URL Removal | Remove web links | `www.recipe.com` → `` |
| 4 | Punctuation | Remove punctuation | `salt & pepper` → `salt pepper` |
| 5 | Chat Words | Expand slang | `veggies` → `vegetables` |
| 6 | Spelling | Fix typos (disabled) | `chiken` → `chicken` |
| 7 | Stopwords | Filter common words | `a cup of rice` → `cup rice` |
| 8 | Emojis | Remove emojis | (emoji) → `` |
| 9 | Tokenization | Split into words | `"salt pepper"` → `["salt", "pepper"]` |
| 10 | Stemming | Root form | `cooking` → `cook` |
| 11 | Lemmatization | Dictionary form | `better` → `good` |

### Cleaning Modes Comparison

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| **None** | Fastest | Raw | Baseline |
| **Light** | Fast | Good | BERT embeddings |
| **Medium** | Moderate | Better | General NLP |
| **Heavy** | Moderate | Best | Entity extraction |

### When to Use Each Mode

| Task | Mode | Reason |
|------|------|--------|
| BERT embeddings | Light | Preserves context |
| Semantic search | Light | Best similarity |
| Entity extraction | Heavy | Clean text |
| Knowledge graph | Medium | Balance |
| Custom models | Medium | Reduces noise |

---

## Usage Examples

### Example 1: Single Recipe Encoding

```python
from nlp_encoder import FoodNLPProcessor

# Initialize
processor = FoodNLPProcessor(device='cuda')

# Encode
recipe = "2 cups chicken breast, 1 tbsp olive oil, salt and pepper"
embeddings, constraints = processor.encode([recipe])

print(f"Embedding: {embeddings[0][:5]}...")  # First 5 dimensions
print(f"Constraints shape: {constraints.shape}")
```

### Example 2: Batch Processing with Cleaning

```python
from nlp_encoder import FoodNLPProcessor
from data_cleaning import clean_ingredient_text

# Initialize
processor = FoodNLPProcessor(device='cuda')

# Prepare data
recipes = [
    "2 cups chicken, chopped",
    "1/2 cup olive oil",
    "salt & pepper to taste"
]

# Clean
cleaned = [clean_ingredient_text(r, mode='light') for r in recipes]

# Encode
embeddings, constraints = processor.encode(cleaned)

print(f"Processed {len(recipes)} recipes")
print(f"Embeddings shape: {embeddings.shape}")
```

### Example 3: DataFrame Processing

```python
import pandas as pd
from data_cleaning import clean_dataframe_column
from nlp_encoder import FoodNLPProcessor

# Load data
df = pd.read_csv('recipes.csv')

# Clean
df = clean_dataframe_column(df, 'ingredients', mode='light')

# Encode
processor = FoodNLPProcessor(device='cuda')
embeddings, constraints = processor.encode(df['ingredients_cleaned'].tolist())

# Add to DataFrame
df['embedding'] = list(embeddings)
df['constraints'] = list(constraints)
```

### Example 4: Large Dataset Processing

```bash
# Process 2.2M recipes with progress tracking
python nlp/run_nlp.py \
    --device cuda \
    --batch-size 32 \
    --cleaning-mode light \
    --overwrite

# Monitor progress
tail -f outputs/nlp/processing.log
```

---

## Performance

### Processing Speed

| Configuration | Speed | Dataset | Time |
|---------------|-------|---------|------|
| CPU, batch=16 | ~5s/chunk | 1000 recipes | ~5s |
| GPU, batch=32 | ~1.4s/chunk | 1000 recipes | ~1.4s |
| GPU, batch=64 | ~1.2s/chunk | 1000 recipes | ~1.2s |

**Full Dataset (2.2M recipes)**:
- Light cleaning: ~50 minutes (GPU, batch=32)
- Medium cleaning: ~60 minutes (GPU, batch=32)
- Heavy cleaning: ~55 minutes (GPU, batch=32)

### Memory Requirements

| Component | CPU | GPU |
|-----------|-----|-----|
| BERT Model | 500MB | 1.5GB |
| Batch (32) | 200MB | 500MB |
| Total | ~1GB | ~2GB |

### Optimization Tips

1. **Use GPU**: 3-5x faster than CPU
2. **Increase batch size**: Better GPU utilization
3. **Enable AMP**: 20-30% speedup with minimal quality loss
4. **Use light cleaning**: Fastest mode for BERT
5. **Process in chunks**: Better memory management

---

## Output Format

### 1. Metadata CSV (`outputs/nlp/metadata.csv`)

```csv
source_file,row_id,title,ingredients,shard,offset
datasets/first_part.csv,0,Recipe Name,cleaned ingredients,shard_00000.npz,0
```

**Columns**:
- `source_file`: Original CSV file
- `row_id`: Row index in source file
- `title`: Recipe title
- `ingredients`: Cleaned ingredient text
- `shard`: NPZ file containing embeddings
- `offset`: Index within shard

### 2. NPZ Shards (`outputs/nlp/*_shard_*.npz`)

```python
import numpy as np

# Load shard
data = np.load('outputs/nlp/first_part_shard_00000.npz')

embeddings = data['embeddings']    # (N, 512)
constraints = data['constraints']  # (N, 5, 12)
texts = data['texts']              # (N,) - cleaned text
```

**Arrays**:
- `embeddings`: BERT embeddings (float32)
- `constraints`: Constraint vectors (float32)
- `texts`: Cleaned ingredient text (string)

---

## Best Practices

### 1. Text Cleaning

**Recommended**:
- Use light mode for BERT embeddings
- Preserve measurements and quantities
- Keep food-specific terms
- Test on sample data first

**Not Recommended**:
- Use heavy cleaning for semantic tasks
- Remove all punctuation for food data
- Enable spelling correction (too slow)
- Skip cleaning entirely

### 2. Encoding

**Recommended**:
- Use GPU when available
- Process in batches
- Enable mixed precision (AMP)
- Monitor memory usage

**Not Recommended**:
- Process one-by-one
- Use batch size > GPU memory
- Ignore CUDA errors
- Skip verification

### 3. Large Datasets

**Recommended**:
- Use chunked processing
- Save intermediate results
- Track progress
- Verify outputs

**Not Recommended**:
- Load entire dataset in memory
- Skip error handling
- Ignore failed chunks
- Overwrite without backup

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**:
```bash
# Reduce batch size
python nlp/run_nlp.py --batch-size 16

# Or use CPU
python nlp/run_nlp.py --device cpu
```

### Issue: Slow Processing

**Solution**:
```bash
# Enable GPU
python nlp/run_nlp.py --device cuda

# Increase batch size
python nlp/run_nlp.py --batch-size 64

# Use light cleaning
python nlp/run_nlp.py --cleaning-mode light
```

### Issue: Import Errors

**Solution**:
```bash
# Install missing packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('all')"
```

### Issue: Encoding Errors

**Solution**:
```python
# Check input text
print(f"Text length: {len(text)}")
print(f"Text type: {type(text)}")

# Clean text first
from data_cleaning import clean_ingredient_text
cleaned = clean_ingredient_text(text, mode='light')
```

---

## Additional Resources

### Documentation
- [Data Cleaning Guide](README_DATA_CLEANING_PROFESSIONAL.md)
- [BERT Documentation](https://huggingface.co/bert-base-uncased)
- [Main Project README](../README.md)

### Notebooks
- `nlp_text_encoder.ipynb` - Interactive encoding tutorial
- `data_cleaning.ipynb` - Cleaning examples
- `data_cleaning_detailed.ipynb` - Advanced cleaning

### Scripts
- `run_nlp.py` - Batch processing
- `verify_nlp_output.py` - Output verification
- `check_cleaned_data.py` - Data quality check (in root)

---

## Contributing

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints
- Write unit tests

### Testing
```bash
# Test cleaning module
python nlp/data_cleaning.py

# Test encoder
python nlp/nlp_encoder.py

# Verify outputs
python nlp/verify_nlp_output.py
```

### Pull Requests
1. Fork the repository
2. Create feature branch
3. Add tests
4. Update documentation
5. Submit PR

---

## License

Part of the Food Intelligence System project.

---

## Team

**Food Intelligence System Team**
- Devanand Puzhakkool
- Saptaparni Saha
- Tanishka Arora

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review [Additional Resources](#additional-resources)
3. Open a GitHub issue
4. Contact the team

---

## Version History

### v1.0.0 (Current)
- BERT-based encoding
- 11 cleaning techniques
- 3 preset modes
- Batch processing
- GPU acceleration
- Comprehensive documentation

### Roadmap
- Multi-language support
- Custom BERT fine-tuning
- Real-time processing API
- Advanced constraint vectors
- Integration with knowledge graph

---

**Last Updated**: 2024  
**Module Version**: 1.0.0  
**Python Version**: 3.8+
