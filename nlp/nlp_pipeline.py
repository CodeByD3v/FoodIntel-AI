"""Complete NLP Pipeline for Food Intelligence System.

Integrates:
1. Data preprocessing (from data_cleaning_comprehensive.ipynb)
2. BERT text encoding (from nlp_encoder.py)
3. Ingredient extraction (from ingredient_extractor.py)
4. Knowledge Graph matching (from kg_matcher.py)
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json

import torch
import torch.nn as nn
from torch import Tensor
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from cleaner import IngredientCleaner
from nlp_encoder import FoodNLPEncoder, resolve_device
from ingredient_extractor import IngredientExtractor
from kg_matcher import FoodKGMatcher, GraphBuilder


class NLPPipeline:
    """Complete NLP processing pipeline."""
    
    def __init__(
        self,
        model_path: str = None,
        device: str = "auto",
        use_bert: bool = True
    ):
        self.device = resolve_device(device)
        self.use_bert = use_bert
        
        self.cleaner = IngredientCleaner()
        self.extractor = IngredientExtractor()
        self.matcher = FoodKGMatcher()
        self.graph_builder = GraphBuilder(self.matcher)
        
        if self.use_bert:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = FoodNLPEncoder(
                model_name="bert-base-uncased",
                embed_dim=512,
                freeze_layers=6
            ).to(self.device)
            
            if model_path and os.path.exists(model_path):
                self.load_model(model_path)
            
            self.model.eval()
    
    def load_model(self, path: str):
        """Load trained model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded model from {path}")
    
    def process_text(self, text: str) -> Dict:
        """Process text through full NLP pipeline."""
        result = {
            'original_text': text,
            'cleaned_text': '',
            'ingredients': [],
            'normalized_ingredients': [],
            'text_embedding': None,
            'graph_data': None,
            'diet_tags': [],
            'disease_risks': {}
        }

        cleaned_text = self.cleaner.clean(text)
        result['cleaned_text'] = cleaned_text
        
        ingredients = self.extractor.extract(cleaned_text)
        result['ingredients'] = ingredients
        
        normalized = [self.extractor.normalize_ingredient(ing) for ing in ingredients]
        result['normalized_ingredients'] = normalized
        
        if self.use_bert and self.model:
            embedding = self.encode_text(cleaned_text)
            result['text_embedding'] = embedding
        
        if normalized:
            graph_data = self.graph_builder.build_ingredient_graph(normalized)
            result['graph_data'] = graph_data
            
            result['diet_tags'] = graph_data.get('diet_tags', [])
            result['disease_risks'] = graph_data.get('disease_risks', {})
        
        return result
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding."""
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            text_embedding, _ = self.model(input_ids, attention_mask)
        
        return text_embedding.cpu().numpy()
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'ingredients') -> pd.DataFrame:
        """Process entire DataFrame."""
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            text = str(row.get(text_column, ''))
            result = self.process_text(text)
            results.append(result)
        
        df['nlp_ingredients'] = [r['ingredients'] for r in results]
        df['nlp_diet_tags'] = [r['diet_tags'] for r in results]
        df['nlp_disease_risks'] = [r['disease_risks'] for r in results]
        
        if 'text_embedding' in results[0] and results[0]['text_embedding'] is not None:
            embeddings = np.array([r['text_embedding'] for r in results])
            df['nlp_embedding'] = list(embeddings)
        
        return df
    
    def batch_encode(self, texts: List[str]) -> Tensor:
        """Batch encode texts."""
        if not self.use_bert or not self.model:
            raise RuntimeError("BERT model not initialized")
        
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            text_embeddings, _ = self.model(input_ids, attention_mask)
        
        return text_embeddings


class NLPDataset:
    """Dataset for NLP training."""
    
    def __init__(self, csv_path: str, tokenizer, max_length: int = 128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if 'ingredients_cleaned' in self.df.columns:
            self.texts = self.df['ingredients_cleaned'].fillna('').tolist()
        else:
            self.texts = self.df['ingredients'].fillna('').tolist()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }


def process_dataset(
    input_csv: str,
    output_csv: str = None,
    model_path: str = None,
    device: str = "auto"
) -> pd.DataFrame:
    """Process dataset through NLP pipeline."""
    print(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} records")
    
    pipeline = NLPPipeline(model_path=model_path, device=device)
    
    text_column = 'ingredients_cleaned' if 'ingredients_cleaned' in df.columns else 'ingredients'
    
    df = pipeline.process_dataframe(df, text_column=text_column)
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved processed data to {output_csv}")
    
    return df


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NLP Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input CSV")
    parser.add_argument("--output", type=str, help="Output CSV")
    parser.add_argument("--model", type=str, help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    process_dataset(
        input_csv=args.input,
        output_csv=args.output,
        model_path=args.model,
        device=args.device
    )


if __name__ == "__main__":
    main()
