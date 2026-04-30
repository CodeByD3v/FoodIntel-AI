"""NLP Module Training Script.

Trains the BERT-based NLP encoder on recipe ingredient data.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.append(str(Path(__file__).parent))
from nlp_encoder import FoodNLPEncoder, NLPBatch, resolve_device


class IngredientDataset(Dataset):
    """Dataset for ingredient text encoding."""
    
    def __init__(self, csv_path: str, tokenizer, max_length: int = 128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if 'ingredients_cleaned' in self.df.columns:
            self.texts = self.df['ingredients_cleaned'].fillna('').tolist()
        elif 'ingredients' in self.df.columns:
            self.texts = self.df['ingredients'].fillna('').tolist()
        else:
            self.texts = [''] * len(self.df)
        
        if 'NER' in self.df.columns:
            self.ner_labels = self.df['NER'].fillna('[]').tolist()
        else:
            self.ner_labels = ['[]'] * len(self.df)
    
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
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'text': text
        }


class NLPTrainingConfig:
    """Configuration for NLP training."""
    
    def __init__(self):
        self.model_name = "bert-base-uncased"
        self.embed_dim = 512
        self.freeze_layers = 6
        self.batch_size = 32
        self.epochs = 10
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.max_length = 128
        self.device = "auto"
        self.checkpoint_dir = "../checkpoints"
        
        self.seed = 42
        self.use_amp = True


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: FoodNLPEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    use_amp: bool = True
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                text_embedding, constraint = model(input_ids, attention_mask)
                loss = nn.functional.mse_loss(
                    text_embedding.mean(),
                    torch.zeros_like(text_embedding.mean())
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            text_embedding, constraint = model(input_ids, attention_mask)
            loss = nn.functional.mse_loss(
                text_embedding.mean(),
                torch.zeros_like(text_embedding.mean())
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
    
    scheduler.step()
    return total_loss / len(dataloader)


def validate(model: FoodNLPEncoder, dataloader: DataLoader, device: torch.device) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            text_embedding, constraint = model(input_ids, attention_mask)
            loss = nn.functional.mse_loss(
                text_embedding.mean(),
                torch.zeros_like(text_embedding.mean())
            )
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_nlp(
    train_csv: str,
    val_csv: str = None,
    config: NLPTrainingConfig = None
):
    """Main training function."""
    if config is None:
        config = NLPTrainingConfig()
    
    set_seed(config.seed)
    device = resolve_device(config.device)
    
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    train_dataset = IngredientDataset(train_csv, tokenizer, config.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = None
    if val_csv:
        val_dataset = IngredientDataset(val_csv, tokenizer, config.max_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2
        )
    
    model = FoodNLPEncoder(
        model_name=config.model_name,
        embed_dim=config.embed_dim,
        freeze_layers=config.freeze_layers
    ).to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    best_loss = float('inf')
    
    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Training samples: {len(train_dataset)}")
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, config.use_amp
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        if val_loader:
            val_loss = validate(model, val_loader, device)
            print(f"Val Loss: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint_path = os.path.join(config.checkpoint_dir, 'nlp_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
    
    final_path = os.path.join(config.checkpoint_dir, 'nlp_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train NLP encoder")
    parser.add_argument("--train_csv", type=str, default="../datasets/cleaned_ingredients.csv",
                        help="Path to training CSV")
    parser.add_argument("--val_csv", type=str, default=None,
                        help="Path to validation CSV")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    config = NLPTrainingConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.device = args.device
    
    train_nlp(args.train_csv, args.val_csv, config)


if __name__ == "__main__":
    main()