"""GPU-aware NLP encoder for the Food Intelligence System.

The module mirrors the NLP notebook: BERT produces a 512-dimensional text
embedding plus a structured 5x12 constraint matrix for each ingredient list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


BERT_MODEL = "bert-base-uncased"
BERT_DIM = 768
EMBED_DIM = 512
NLP_ROWS = 5
NLP_COLS = 12
MAX_SEQ = 128


def resolve_device(requested: str = "auto") -> torch.device:
    """Return the requested torch device, preferring CUDA for ``auto``."""
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(requested)


class NLPConstraintVector(nn.Module):
    """Project attended token states into a structured 5x12 NLP matrix."""

    def __init__(self, bert_dim: int = BERT_DIM, n_rows: int = NLP_ROWS, n_cols: int = NLP_COLS):
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.token_attention = nn.Sequential(
            nn.Linear(bert_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.feature_projector = nn.Sequential(
            nn.Linear(bert_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_rows),
        )
        self.layer_norm = nn.LayerNorm([n_rows, n_cols])

    def forward(self, token_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.token_attention(token_hidden).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, torch.info(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)

        k = min(self.n_cols, token_hidden.size(1))
        _, top_idx = torch.topk(weights, k=k, dim=-1)
        top_idx, _ = torch.sort(top_idx, dim=-1)
        idx_expanded = top_idx.unsqueeze(-1).expand(-1, -1, token_hidden.size(-1))
        selected = torch.gather(token_hidden, 1, idx_expanded)

        features = self.feature_projector(selected).permute(0, 2, 1)
        if features.size(-1) < self.n_cols:
            features = F.pad(features, (0, self.n_cols - features.size(-1)))
        return self.layer_norm(features)


class FoodNLPEncoder(nn.Module):
    """BERT ingredient encoder returning ``(text_embedding, constraint_vector)``."""

    def __init__(
        self,
        model_name: str = BERT_MODEL,
        embed_dim: int = EMBED_DIM,
        freeze_layers: int = 6,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.bert.config.hidden_size)

        encoder_layers = getattr(getattr(self.bert, "encoder", None), "layer", [])
        for i, layer in enumerate(encoder_layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.constraint = NLPConstraintVector(bert_dim=hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_hidden = output.last_hidden_state
        cls_vec = token_hidden[:, 0, :]
        text_embedding = F.normalize(self.projector(cls_vec), dim=-1)
        constraint_vector = self.constraint(token_hidden, attention_mask)
        return text_embedding, constraint_vector


@dataclass(frozen=True)
class NLPBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device: torch.device) -> "NLPBatch":
        return NLPBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
        )


class FoodNLPProcessor:
    """High-level tokenizer/model wrapper for batched GPU inference."""

    def __init__(
        self,
        model_name: str = BERT_MODEL,
        device: str = "auto",
        max_length: int = MAX_SEQ,
        freeze_layers: int = 6,
        use_amp: bool = True,
    ) -> None:
        self.device = resolve_device(device)
        self.max_length = max_length
        self.use_amp = use_amp and self.device.type == "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = FoodNLPEncoder(
            model_name=model_name,
            freeze_layers=freeze_layers,
        ).to(self.device)
        self.model.eval()

    def tokenize(self, texts: Iterable[str]) -> NLPBatch:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return NLPBatch(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.tokenize(texts).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
            embeddings, constraints = self.model(batch.input_ids, batch.attention_mask)
        return embeddings.float().cpu(), constraints.float().cpu()
