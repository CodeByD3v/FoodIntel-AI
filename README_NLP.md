# NLP Module - Food Intelligence System

## Overview

This module implements the Natural Language Processing (NLP) component of the Food Intelligence System, which processes ingredient lists to generate semantic embeddings for nutritional analysis.

## Features

- **BERT-based Text Encoding**: Converts ingredient lists to 512-dimensional embeddings
- **Constraint Vector Generation**: Creates 5×12 semantic feature matrices
- **GPU Acceleration**: Processes 459 recipes/second on NVIDIA RTX 4050
- **Batch Processing**: Handles millions of recipes efficiently
- **Comprehensive Verification**: Automated testing and visualization

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv food_venv

# Activate (Windows)
food_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Process recipes (1000 per file)
python run_nlp.py --device cuda --batch-size 32 --rows 1000

# Verify outputs
python verify_nlp_output.py
```

### Load Embeddings

```python
import numpy as np

# Load processed data
data = np.load('outputs/nlp_full/first_part_shard_00000.npz')

text_embeddings = data['embedding']      # (N, 512)
constraint_vectors = data['constraint']  # (N, 5, 12)
titles = data['title']
ingredients = data['ingredients']
```

## Architecture

### Text Embedding Pipeline
```
Ingredient Text → BERT Tokenizer → BERT Encoder (12 layers)
→ [CLS] Token (768-d) → Linear Projection (512-d) → L2 Normalize
```

### Constraint Vector (5×12 Matrix)
- **Row 0**: Ingredient identity
- **Row 1**: Quantity/proportion
- **Row 2**: Preparation method
- **Row 3**: Nutritional signal
- **Row 4**: Contextual interaction
- **Columns**: Top-12 attended token positions

## Files

### Core Implementation
- `nlp_encoder.py` - BERT-based encoder classes
- `run_nlp.py` - Batch processing script
- `verify_nlp_output.py` - Verification and visualization

### Notebooks
- `nlp_text_encoder.ipynb` - Interactive tutorial

### Documentation
- `Food_Intelligence_System_Report.md` - Complete system specification
- `README_NLP.md` - This file

## Performance

| Metric | Value |
|--------|-------|
| Throughput (GPU) | 459 recipes/sec |
| Throughput (CPU) | 15 recipes/sec |
| Batch Size | 32 |
| Model Size | 440 MB |
| GPU Memory | ~2 GB |

## Dataset

- **Source**: Recipe1M+ dataset
- **Total Recipes**: 2,231,142
- **Files**: 
  - `datasets/first_part.csv` (1,115,571 recipes)
  - `datasets/second_part.csv` (1,115,571 recipes)
- **Storage**: Git LFS (Large File Storage)

## Output Format

### NPZ Files
Each shard contains:
- `embedding`: (N, 512) text embeddings
- `constraint`: (N, 5, 12) constraint vectors
- `title`: Recipe names
- `ingredients`: Ingredient lists
- `row_id`: Original CSV row IDs

### Metadata CSV
- `source_file`: Original CSV path
- `row_id`: Row identifier
- `title`: Recipe name
- `ingredients`: Full ingredient list
- `shard`: NPZ file name
- `offset`: Position in shard

## Requirements

- Python 3.11+
- PyTorch 2.6.0+
- Transformers 5.6.2
- CUDA 12.4+ (for GPU acceleration)
- 8GB+ RAM
- 6GB+ GPU VRAM (recommended)

## Troubleshooting

### Matplotlib DLL Error (Windows)
```bash
pip install matplotlib==3.9.0 --force-reinstall
```

### CUDA Not Available
```bash
python run_nlp.py --device cpu --batch-size 8
```

### Out of Memory
```bash
python run_nlp.py --batch-size 16  # Reduce batch size
```

## Citation

```bibtex
@article{foodintel2025,
  title={Food Intelligence System: Multimodal AI Pipeline for Nutritional Analysis},
  author={Puzhakkool, Devanand and Saha, Saptaparni and Arora, Tanishka},
  year={2025}
}
```

## License

This project is part of the Food Intelligence System research.

## Contact

For questions or issues, please open a GitHub issue.
