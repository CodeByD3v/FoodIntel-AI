"""Module 4 -- Multimodal Fusion & Prediction Heads.

Fuses CNN visual embedding (v in R^512) and NLP text embedding (t in R^512)
via cross-attention into a shared 256-d representation, then branches into
three task-specific prediction heads:

  1. Nutrition Regression   -- calories, protein, fat, carbs  (MSE)
  2. Health Risk Classification -- cardiovascular, diabetes, hypertension (CE)
  3. Dietary Compatibility  -- vegan, keto, high-protein, gluten-free (BCE)

Architecture reference: Food Intelligence System Report S2.4, S5.5, Appendix C.4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants (aligned with CNN and NLP modules)
# ---------------------------------------------------------------------------
VISUAL_DIM = 512
TEXT_DIM = 512
NUM_MODALITIES = 2
FUSED_DIM = 256
NUM_HEADS = 8

NLP_CONSTRAINT_ROWS = 5
NLP_CONSTRAINT_COLS = 12

NUM_NUTRITION_TARGETS = 4
NUM_RISK_CLASSES = 3
NUM_DIET_TAGS = 4

NUTRITION_NAMES = ["calories", "protein", "fat", "carbohydrates"]
RISK_NAMES = ["cardiovascular", "type2_diabetes", "hypertension"]
DIET_NAMES = ["vegan", "keto_compatible", "high_protein", "gluten_free"]


# ---------------------------------------------------------------------------
# NLP Constraint Injector (optional -- toggleable via use_constraint)
# ---------------------------------------------------------------------------
class NLPConstraintInjector(nn.Module):
    """Project the 5x12 NLP constraint matrix into the fusion space.

    The NLP module produces two outputs: a 512-d [CLS] embedding and a
    structured 5x12 constraint matrix encoding ingredient identity, quantity,
    preparation, nutritional signal, and contextual interaction features.

    This injector flattens the matrix (60-d), projects it to ``fusion_dim``,
    and adds it as a gated residual to the fused representation.  The gate
    is initialised to zero so the constraint signal is learned gradually.

    NOTE: The report S2.4 only shows the 512-d ``t`` entering cross-attention.
    This injector is an optional enhancement.  Toggle via use_constraint=True.
    """

    def __init__(
        self,
        n_rows: int = NLP_CONSTRAINT_ROWS,
        n_cols: int = NLP_CONSTRAINT_COLS,
        fusion_dim: int = FUSED_DIM,
    ) -> None:
        super().__init__()
        flat_dim = n_rows * n_cols  # 60
        self.projector = nn.Sequential(
            nn.Linear(flat_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, fused: torch.Tensor, constraint: torch.Tensor) -> torch.Tensor:
        """Add gated constraint residual to fused representation."""
        flat = constraint.reshape(constraint.size(0), -1)
        projected = self.projector(flat)
        return fused + torch.sigmoid(self.gate) * projected


# ---------------------------------------------------------------------------
# Cross-Attention Fusion Layer
# ---------------------------------------------------------------------------
class MultimodalFusion(nn.Module):
    """Cross-attention fusion of visual and text embeddings.

    Stacks v and t into (B, 2, 512), applies multi-head self-attention so
    each modality can attend to the other, then flattens and projects to
    a shared out_dim-d space.
    """

    def __init__(
        self,
        embed_dim: int = VISUAL_DIM,
        num_heads: int = NUM_HEADS,
        out_dim: int = FUSED_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        concat_dim = embed_dim * NUM_MODALITIES  # 512 * 2 = 1024
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),  # 1024 -> 512
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(concat_dim // 2, out_dim),      # 512 -> 256
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self, visual_emb: torch.Tensor, text_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse visual and text embeddings.

        Returns:
            fused:   (B, out_dim) shared representation.
            weights: (B, 2, 2)   per-modality attention weights.
        """
        modalities = torch.stack([visual_emb, text_emb], dim=1)  # (B, 2, 512)
        attended, weights = self.attention(modalities, modalities, modalities)
        flat = attended.reshape(attended.size(0), -1)             # (B, 1024)
        out = self.norm(self.mlp(flat))                           # (B, 256)
        return out, weights


# ---------------------------------------------------------------------------
# Prediction Heads
# ---------------------------------------------------------------------------
class NutritionRegressionHead(nn.Module):
    """2-layer MLP predicting calories, protein, fat, carbohydrates (MSE)."""

    def __init__(self, in_dim: int = FUSED_DIM, out_dim: int = NUM_NUTRITION_TARGETS) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class HealthRiskClassificationHead(nn.Module):
    """Softmax classifier over health-risk categories (CrossEntropy).

    DESIGN NOTE: The report specifies softmax over 3 risk classes with CE,
    but example output in S5.5 shows independent probabilities that do NOT
    sum to 1, implying the three risks are NOT mutually exclusive.  A more
    semantically correct formulation might use 3 independent sigmoid
    classifiers with BCEWithLogitsLoss.  This follows the report literally.
    """

    def __init__(self, in_dim: int = FUSED_DIM, num_classes: int = NUM_RISK_CLASSES) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # raw logits


class DietaryCompatibilityHead(nn.Module):
    """Independent sigmoid head for multi-label dietary tags (BCE)."""

    def __init__(self, in_dim: int = FUSED_DIM, num_tags: int = NUM_DIET_TAGS) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim // 2, num_tags),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # raw logits


# ---------------------------------------------------------------------------
# Top-level FNN Model
# ---------------------------------------------------------------------------
@dataclass
class LossWeights:
    """Per-task loss multipliers for multi-task training."""
    nutrition: float = 1.0
    risk: float = 1.0
    diet: float = 1.0


class FoodIntelFNN(nn.Module):
    """Food Intelligence System -- Multimodal Fusion Network.

    Combines CNN visual embedding and NLP text embedding through
    cross-attention, then predicts nutritional content, health risk,
    and dietary compatibility in parallel.
    """

    def __init__(
        self,
        embed_dim: int = VISUAL_DIM,
        fusion_dim: int = FUSED_DIM,
        num_heads: int = NUM_HEADS,
        use_constraint: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fusion = MultimodalFusion(embed_dim, num_heads, fusion_dim, dropout)
        self.constraint_injector: Optional[NLPConstraintInjector] = None
        if use_constraint:
            self.constraint_injector = NLPConstraintInjector(fusion_dim=fusion_dim)
        self.nutrition_head = NutritionRegressionHead(fusion_dim, NUM_NUTRITION_TARGETS)
        self.risk_head = HealthRiskClassificationHead(fusion_dim, NUM_RISK_CLASSES)
        self.diet_head = DietaryCompatibilityHead(fusion_dim, NUM_DIET_TAGS)

    def forward(
        self,
        visual_emb: torch.Tensor,
        text_emb: torch.Tensor,
        constraint: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Run the full fusion + prediction pipeline.

        Args:
            visual_emb: (B, 512) CNN visual embedding v.
            text_emb:   (B, 512) NLP text embedding t.
            constraint: (B, 5, 12) or None -- NLP constraint matrix.

        Returns:
            Dict with nutrition (B,4), risk (B,3), diet (B,4),
            attention_weights (B,2,2), fused (B,256).
        """
        fused, attn_weights = self.fusion(visual_emb, text_emb)
        if self.constraint_injector is not None and constraint is not None:
            fused = self.constraint_injector(fused, constraint)
        return {
            "nutrition": self.nutrition_head(fused),
            "risk": self.risk_head(fused),
            "diet": self.diet_head(fused),
            "attention_weights": attn_weights,
            "fused": fused,
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        weights: Optional[LossWeights] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute weighted multi-task loss.

        targets must contain: nutrition (B,4), risk (B,) long, diet (B,4).
        """
        if weights is None:
            weights = LossWeights()
        loss_nut = F.mse_loss(outputs["nutrition"], targets["nutrition"])
        loss_risk = F.cross_entropy(outputs["risk"], targets["risk"])
        loss_diet = F.binary_cross_entropy_with_logits(outputs["diet"], targets["diet"])
        total = (
            weights.nutrition * loss_nut
            + weights.risk * loss_risk
            + weights.diet * loss_diet
        )
        return total, {
            "total": total.detach(),
            "nutrition": loss_nut.detach(),
            "risk": loss_risk.detach(),
            "diet": loss_diet.detach(),
        }
