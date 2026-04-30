"""BERT + USDA ingredient encoder."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:
    from .usda_lookup import DEFAULT_NUTRIENT_IDS, USDALookup
except ImportError:  # pragma: no cover - enables direct script execution
    from usda_lookup import DEFAULT_NUTRIENT_IDS, USDALookup


class BERTIngredientEncoder(nn.Module):
    """Encode ingredient text into a 512-dimensional normalized embedding."""

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        usda_lookup: Optional[USDALookup] = None,
        freeze_bert: bool = True,
        output_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.usda_lookup = usda_lookup
        self.nutrient_dim = len(DEFAULT_NUTRIENT_IDS)
        hidden_size = int(getattr(self.bert.config, "hidden_size", 768))

        if freeze_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(hidden_size + self.nutrient_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
        )

    def forward(self, ingredient_str: str, ingredient_list: List[str]) -> torch.Tensor:
        """Encode one ingredient string into shape (1, 512)."""
        device = next(self.parameters()).device
        tokens = self.tokenizer(
            ingredient_str,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        bert_outputs = self.bert(**tokens)
        cls_embedding = bert_outputs.last_hidden_state[:, 0, :]
        usda_context = self._nutrition_context(ingredient_list, device=device).unsqueeze(0)
        combined = torch.cat([cls_embedding, usda_context], dim=1)
        output = self.projection(combined)
        return F.normalize(output, p=2, dim=1)

    def encode_batch(self, samples: List[Dict]) -> torch.Tensor:
        """Efficiently encode a batch of samples into shape (B, 512)."""
        if not samples:
            return torch.empty((0, self.projection[-1].out_features), device=next(self.parameters()).device)

        device = next(self.parameters()).device
        texts = [sample["ingredient_str"] for sample in samples]
        tokens = self.tokenizer(
            texts,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        bert_outputs = self.bert(**tokens)
        cls_embeddings = bert_outputs.last_hidden_state[:, 0, :]
        contexts = [
            self._nutrition_context(sample.get("ingredient_list", []), device=device)
            for sample in samples
        ]
        usda_context = torch.stack(contexts, dim=0)
        combined = torch.cat([cls_embeddings, usda_context], dim=1)
        output = self.projection(combined)
        return F.normalize(output, p=2, dim=1)

    def _nutrition_context(self, ingredient_list: List[str], device: torch.device) -> torch.Tensor:
        if self.usda_lookup is None or not ingredient_list:
            vector = np.zeros(self.nutrient_dim, dtype=np.float32)
        else:
            matrix = self.usda_lookup.batch_lookup(ingredient_list)
            vector = matrix.mean(axis=0).astype(np.float32) if matrix.size else np.zeros(self.nutrient_dim, dtype=np.float32)
        return torch.tensor(vector, dtype=torch.float32, device=device)
