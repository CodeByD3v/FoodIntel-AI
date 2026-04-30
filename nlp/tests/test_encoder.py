from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import encoder as encoder_module
from encoder import BERTIngredientEncoder


class FakeTokenizer:
    def __call__(self, texts, max_length, padding, truncation, return_tensors):
        batch_size = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.ones((batch_size, 8), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 8), dtype=torch.long),
        }


class FakeBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=768)
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        hidden = torch.ones((batch_size, seq_len, 768), dtype=torch.float32, device=input_ids.device)
        return SimpleNamespace(last_hidden_state=hidden * self.weight)


class MockUSDALookup:
    def lookup(self, ingredient):
        return np.ones(17, dtype=np.float32) * 0.25

    def batch_lookup(self, ingredients):
        return np.stack([self.lookup(ingredient) for ingredient in ingredients]).astype(np.float32)


@pytest.fixture(autouse=True)
def patch_transformers(monkeypatch):
    monkeypatch.setattr(encoder_module.AutoModel, "from_pretrained", lambda *_args, **_kwargs: FakeBERT())
    monkeypatch.setattr(encoder_module.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: FakeTokenizer())


def test_encoder_output_shape_and_norm():
    model = BERTIngredientEncoder(usda_lookup=MockUSDALookup(), freeze_bert=True)
    output = model("chicken, rice, oil", ["chicken", "rice", "oil"])
    assert output.shape == (1, 512)
    assert torch.allclose(output.norm(p=2, dim=1), torch.ones(1), atol=1e-5)


def test_freeze_bert_true_disables_grad():
    model = BERTIngredientEncoder(usda_lookup=MockUSDALookup(), freeze_bert=True)
    assert all(not parameter.requires_grad for parameter in model.bert.parameters())


def test_freeze_bert_false_enables_grad():
    model = BERTIngredientEncoder(usda_lookup=MockUSDALookup(), freeze_bert=False)
    assert all(parameter.requires_grad for parameter in model.bert.parameters())

