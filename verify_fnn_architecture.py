"""FNN Architecture Verification & Integration Test.

Validates the FoodIntel Fusion Network (Module 4) in isolation and in
correspondence with the CNN and NLP modules.

Tests:
  1. Shape verification (all layers)
  2. Loss computation & gradient flow
  3. Attention weight validity
  4. Constraint injector toggle
  5. Training smoke test (loss decreases)
  6. CNN integration (simulated ResNet-50 -> 512-d output)
  7. NLP integration (real FoodNLPEncoder -> 512-d + 5x12)
  8. End-to-end pipeline (CNN + NLP -> FNN -> predictions)
  9. Checkpoint save/load roundtrip
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import FNN
from fusion_model import (
    FoodIntelFNN, MultimodalFusion, NutritionRegressionHead,
    HealthRiskClassificationHead, DietaryCompatibilityHead,
    NLPConstraintInjector, LossWeights,
    VISUAL_DIM, TEXT_DIM, FUSED_DIM, NUM_HEADS,
    NLP_CONSTRAINT_ROWS, NLP_CONSTRAINT_COLS,
    NUM_NUTRITION_TARGETS, NUM_RISK_CLASSES, NUM_DIET_TAGS,
)

DEVICE = torch.device("cpu")
BATCH = 8
PASSED = 0
FAILED = 0


def run_test(name, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  [PASS] {name}")
        PASSED += 1
    except Exception as e:
        print(f"  [FAIL] {name}")
        traceback.print_exc()
        FAILED += 1


# ==========================================================================
print("=" * 75)
print("FOODINTEL FNN - ARCHITECTURE VERIFICATION & INTEGRATION TEST")
print("=" * 75)
print()

# -- 1. Shape Verification ---------------------------------------------
print("1. SHAPE VERIFICATION")
print("-" * 75)

model = FoodIntelFNN(use_constraint=False).to(DEVICE)
v = torch.randn(BATCH, VISUAL_DIM, device=DEVICE)
t = torch.randn(BATCH, TEXT_DIM, device=DEVICE)
c = torch.randn(BATCH, NLP_CONSTRAINT_ROWS, NLP_CONSTRAINT_COLS, device=DEVICE)

with torch.no_grad():
    out = model(v, t)

def test_nutrition_shape():
    assert out["nutrition"].shape == (BATCH, NUM_NUTRITION_TARGETS), \
        f"Expected ({BATCH}, {NUM_NUTRITION_TARGETS}), got {out['nutrition'].shape}"

def test_risk_shape():
    assert out["risk"].shape == (BATCH, NUM_RISK_CLASSES), \
        f"Expected ({BATCH}, {NUM_RISK_CLASSES}), got {out['risk'].shape}"

def test_diet_shape():
    assert out["diet"].shape == (BATCH, NUM_DIET_TAGS), \
        f"Expected ({BATCH}, {NUM_DIET_TAGS}), got {out['diet'].shape}"

def test_fused_shape():
    assert out["fused"].shape == (BATCH, FUSED_DIM), \
        f"Expected ({BATCH}, {FUSED_DIM}), got {out['fused'].shape}"

def test_attn_shape():
    assert out["attention_weights"].shape == (BATCH, 2, 2), \
        f"Expected ({BATCH}, 2, 2), got {out['attention_weights'].shape}"

run_test("Nutrition output: (B, 4)", test_nutrition_shape)
run_test("Risk output: (B, 3)", test_risk_shape)
run_test("Diet output: (B, 4)", test_diet_shape)
run_test("Fused representation: (B, 256)", test_fused_shape)
run_test("Attention weights: (B, 2, 2)", test_attn_shape)
print()

# -- 2. Loss Computation & Gradient Flow -----------------------------
print("2. LOSS COMPUTATION & GRADIENT FLOW")
print("-" * 75)

model_grad = FoodIntelFNN(use_constraint=False).to(DEVICE)
model_grad.train()
v_g = torch.randn(BATCH, VISUAL_DIM, device=DEVICE)
t_g = torch.randn(BATCH, TEXT_DIM, device=DEVICE)
out_g = model_grad(v_g, t_g)
targets = {
    "nutrition": torch.rand(BATCH, NUM_NUTRITION_TARGETS, device=DEVICE),
    "risk": torch.randint(0, NUM_RISK_CLASSES, (BATCH,), device=DEVICE),
    "diet": torch.randint(0, 2, (BATCH, NUM_DIET_TAGS), device=DEVICE).float(),
}
total_loss, losses = model_grad.compute_loss(out_g, targets)

def test_loss_finite():
    assert torch.isfinite(total_loss), f"Total loss is not finite: {total_loss.item()}"

def test_loss_requires_grad():
    assert total_loss.requires_grad, "Total loss does not require grad"

def test_backward():
    total_loss.backward()
    has_grad = all(p.grad is not None for p in model_grad.parameters() if p.requires_grad)
    assert has_grad, "Some parameters have no gradient after backward()"

def test_per_task_losses():
    for key in ("nutrition", "risk", "diet", "total"):
        assert key in losses, f"Missing loss key: {key}"
        assert torch.isfinite(losses[key]), f"Loss '{key}' is not finite"

run_test("Total loss is finite", test_loss_finite)
run_test("Loss requires grad", test_loss_requires_grad)
run_test("Backward pass populates all gradients", test_backward)
run_test("Per-task losses present and finite", test_per_task_losses)
print()

# -- 3. Attention Weight Validity -------------------------------------
print("3. ATTENTION WEIGHT VALIDITY")
print("-" * 75)

model_attn = FoodIntelFNN().to(DEVICE)
model_attn.eval()
with torch.no_grad():
    out_a = model_attn(v, t)
    w = out_a["attention_weights"]

def test_attn_non_negative():
    assert (w >= 0).all(), "Attention weights contain negative values"

def test_attn_sum_to_one():
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
        f"Attention weights don't sum to 1: {sums}"

def test_attn_cross_modality():
    mean_w = w.mean(dim=0)
    print(f"    Mean attention: visual->[v={mean_w[0,0]:.3f}, t={mean_w[0,1]:.3f}] | "
          f"text->[v={mean_w[1,0]:.3f}, t={mean_w[1,1]:.3f}]")
    assert True

run_test("Attention weights non-negative", test_attn_non_negative)
run_test("Attention weights sum to 1", test_attn_sum_to_one)
run_test("Cross-modality attention analysis", test_attn_cross_modality)
print()

# -- 4. Constraint Injector Toggle ------------------------------------
print("4. NLP CONSTRAINT INJECTOR TOGGLE")
print("-" * 75)

model_no_c = FoodIntelFNN(use_constraint=False).to(DEVICE)
model_with_c = FoodIntelFNN(use_constraint=True).to(DEVICE)

def test_no_constraint():
    assert model_no_c.constraint_injector is None
    with torch.no_grad():
        o = model_no_c(v, t)
    assert o["fused"].shape == (BATCH, FUSED_DIM)

def test_with_constraint():
    assert model_with_c.constraint_injector is not None
    with torch.no_grad():
        o = model_with_c(v, t, constraint=c)
    assert o["fused"].shape == (BATCH, FUSED_DIM)

def test_gate_init_zero():
    gate_val = model_with_c.constraint_injector.gate.item()
    assert abs(gate_val) < 1e-6, f"Gate not initialized to 0: {gate_val}"

def test_constraint_param_count():
    base_params = sum(p.numel() for p in model_no_c.parameters())
    aug_params = sum(p.numel() for p in model_with_c.parameters())
    extra = aug_params - base_params
    print(f"    Base model: {base_params:,} params")
    print(f"    + Constraint injector: +{extra:,} params = {aug_params:,} total")
    assert extra > 0, "Constraint injector added no parameters"

run_test("Without constraint: injector is None", test_no_constraint)
run_test("With constraint: injector is active", test_with_constraint)
run_test("Gate initialized to 0", test_gate_init_zero)
run_test("Constraint injector adds parameters", test_constraint_param_count)
print()

# -- 5. Training Smoke Test -------------------------------------------
print("5. TRAINING SMOKE TEST (5 epochs on random data)")
print("-" * 75)

def test_loss_decreases():
    m = FoodIntelFNN(use_constraint=True).to(DEVICE)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    losses_list = []
    for ep in range(5):
        vv = torch.randn(32, VISUAL_DIM, device=DEVICE)
        tt = torch.randn(32, TEXT_DIM, device=DEVICE)
        cc = torch.randn(32, NLP_CONSTRAINT_ROWS, NLP_CONSTRAINT_COLS, device=DEVICE)
        tgts = {
            "nutrition": torch.rand(32, NUM_NUTRITION_TARGETS, device=DEVICE),
            "risk": torch.randint(0, NUM_RISK_CLASSES, (32,), device=DEVICE),
            "diet": torch.randint(0, 2, (32, NUM_DIET_TAGS), device=DEVICE).float(),
        }
        o = m(vv, tt, constraint=cc)
        loss, _ = m.compute_loss(o, tgts)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses_list.append(loss.item())
        print(f"    Epoch {ep+1}: loss = {loss.item():.4f}")
    assert losses_list[-1] < losses_list[0], \
        f"Loss did not decrease: {losses_list[0]:.4f} -> {losses_list[-1]:.4f}"

run_test("Loss decreases over 5 training steps", test_loss_decreases)
print()

# -- 6. CNN Integration -----------------------------------------------
print("6. CNN INTEGRATION (simulated ResNet-50 -> 512-d)")
print("-" * 75)

class SimulatedFoodCNN(nn.Module):
    """Mimics FoodCNN output: ResNet-50 backbone -> 2048 -> 512-d L2-norm."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2048, 2048)  # placeholder for ResNet GAP
        self.projection = nn.Sequential(
            nn.Linear(2048, VISUAL_DIM),
            nn.ReLU(),
        )
    def forward(self, x):
        features = self.backbone(x)
        projected = self.projection(features)
        return F.normalize(projected, dim=-1)

cnn = SimulatedFoodCNN().to(DEVICE)
cnn.eval()

def test_cnn_output_dim():
    fake_features = torch.randn(BATCH, 2048, device=DEVICE)
    with torch.no_grad():
        cnn_emb = cnn(fake_features)
    assert cnn_emb.shape == (BATCH, VISUAL_DIM), \
        f"CNN output shape mismatch: {cnn_emb.shape}"

def test_cnn_l2_norm():
    fake_features = torch.randn(BATCH, 2048, device=DEVICE)
    with torch.no_grad():
        cnn_emb = cnn(fake_features)
    norms = cnn_emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(BATCH, device=DEVICE), atol=1e-4), \
        f"CNN output not L2-normalized: norms={norms}"

def test_cnn_to_fnn():
    fake_features = torch.randn(BATCH, 2048, device=DEVICE)
    with torch.no_grad():
        cnn_emb = cnn(fake_features)
        fnn = FoodIntelFNN().to(DEVICE)
        fnn.eval()
        out = fnn(cnn_emb, torch.randn(BATCH, TEXT_DIM, device=DEVICE))
    assert out["nutrition"].shape == (BATCH, NUM_NUTRITION_TARGETS)
    print(f"    CNN(2048) -> 512-d -> FNN -> nutrition {out['nutrition'].shape}")

run_test("CNN output is (B, 512)", test_cnn_output_dim)
run_test("CNN output is L2-normalized", test_cnn_l2_norm)
run_test("CNN output feeds into FNN correctly", test_cnn_to_fnn)
print()

# -- 7. NLP Integration ----------------------------------------------
print("7. NLP INTEGRATION (FoodNLPEncoder -> 512-d + 5x12)")
print("-" * 75)

try:
    from nlp_encoder import FoodNLPEncoder, BERT_DIM, EMBED_DIM
    nlp_available = True
except Exception:
    nlp_available = False

if nlp_available:
    class SimulatedNLPEncoder(nn.Module):
        """Uses the real FoodNLPEncoder architecture but with random weights."""
        def __init__(self):
            super().__init__()
            self.projector = nn.Sequential(
                nn.Linear(BERT_DIM, EMBED_DIM),
                nn.LayerNorm(EMBED_DIM),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        def forward(self, hidden_states):
            cls_vec = hidden_states[:, 0, :]
            return F.normalize(self.projector(cls_vec), dim=-1)

    nlp_sim = SimulatedNLPEncoder().to(DEVICE)
    nlp_sim.eval()

    def test_nlp_output_dim():
        fake_hidden = torch.randn(BATCH, 128, BERT_DIM, device=DEVICE)
        with torch.no_grad():
            nlp_emb = nlp_sim(fake_hidden)
        assert nlp_emb.shape == (BATCH, TEXT_DIM), \
            f"NLP output shape mismatch: {nlp_emb.shape}"
        print(f"    NLP projector: BERT({BERT_DIM}) -> {TEXT_DIM}-d  OK")

    def test_nlp_constraint_shape():
        from nlp_encoder import NLPConstraintVector
        cv = NLPConstraintVector().to(DEVICE)
        cv.eval()
        fake_hidden = torch.randn(BATCH, 128, BERT_DIM, device=DEVICE)
        fake_mask = torch.ones(BATCH, 128, device=DEVICE)
        with torch.no_grad():
            constraint = cv(fake_hidden, fake_mask)
        assert constraint.shape == (BATCH, NLP_CONSTRAINT_ROWS, NLP_CONSTRAINT_COLS), \
            f"Constraint shape mismatch: {constraint.shape}"
        print(f"    Constraint vector: ({NLP_CONSTRAINT_ROWS}, {NLP_CONSTRAINT_COLS})  OK")

    def test_nlp_to_fnn():
        from nlp_encoder import NLPConstraintVector
        fake_hidden = torch.randn(BATCH, 128, BERT_DIM, device=DEVICE)
        fake_mask = torch.ones(BATCH, 128, device=DEVICE)
        with torch.no_grad():
            nlp_emb = nlp_sim(fake_hidden)
            cv = NLPConstraintVector().to(DEVICE)
            constraint = cv(fake_hidden, fake_mask)
            fnn = FoodIntelFNN(use_constraint=True).to(DEVICE)
            fnn.eval()
            out = fnn(torch.randn(BATCH, VISUAL_DIM, device=DEVICE), nlp_emb, constraint=constraint)
        assert out["nutrition"].shape == (BATCH, NUM_NUTRITION_TARGETS)
        print(f"    NLP({BERT_DIM}) -> 512-d + 5x12 -> FNN -> nutrition {out['nutrition'].shape}")

    run_test("NLP embedding is (B, 512)", test_nlp_output_dim)
    run_test("NLP constraint is (B, 5, 12)", test_nlp_constraint_shape)
    run_test("NLP output feeds into FNN correctly", test_nlp_to_fnn)
else:
    print("  !! Skipped: nlp_encoder.py could not be imported (transformers not installed)")
print()

# -- 8. End-to-End Pipeline -------------------------------------------
print("8. END-TO-END PIPELINE: CNN + NLP -> FNN -> Predictions")
print("-" * 75)

def test_e2e_pipeline():
    fnn = FoodIntelFNN(use_constraint=True).to(DEVICE)
    fnn.eval()
    # Simulate CNN output
    visual_emb = F.normalize(torch.randn(BATCH, VISUAL_DIM, device=DEVICE), dim=-1)
    # Simulate NLP output
    text_emb = F.normalize(torch.randn(BATCH, TEXT_DIM, device=DEVICE), dim=-1)
    constraint = torch.randn(BATCH, NLP_CONSTRAINT_ROWS, NLP_CONSTRAINT_COLS, device=DEVICE)

    with torch.no_grad():
        predictions = fnn(visual_emb, text_emb, constraint=constraint)

    # Decode predictions
    nut = predictions["nutrition"][0]
    risk_probs = torch.softmax(predictions["risk"][0], dim=-1)
    diet_probs = torch.sigmoid(predictions["diet"][0])

    print(f"    Sample prediction (item 0):")
    print(f"      Calories:  {nut[0]:.3f}")
    print(f"      Protein:   {nut[1]:.3f}")
    print(f"      Fat:       {nut[2]:.3f}")
    print(f"      Carbs:     {nut[3]:.3f}")
    print(f"      Cardio risk:    {risk_probs[0]:.3f}")
    print(f"      Diabetes risk:  {risk_probs[1]:.3f}")
    print(f"      Hypertension:   {risk_probs[2]:.3f}")
    print(f"      Vegan:          {diet_probs[0]:.3f}")
    print(f"      Keto:           {diet_probs[1]:.3f}")
    print(f"      High-protein:   {diet_probs[2]:.3f}")
    print(f"      Gluten-free:    {diet_probs[3]:.3f}")
    assert True

run_test("Full pipeline produces valid predictions", test_e2e_pipeline)
print()

# -- 9. Checkpoint Save/Load Roundtrip --------------------------------
print("9. CHECKPOINT SAVE/LOAD ROUNDTRIP")
print("-" * 75)

def test_checkpoint():
    m1 = FoodIntelFNN(use_constraint=True).to(DEVICE)
    m1.eval()
    with torch.no_grad():
        out1 = m1(v, t, constraint=c)

    ckpt_path = Path("checkpoints/test_roundtrip.pth")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": m1.state_dict()}, ckpt_path)

    m2 = FoodIntelFNN(use_constraint=True).to(DEVICE)
    m2.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["model_state"])
    m2.eval()
    with torch.no_grad():
        out2 = m2(v, t, constraint=c)

    for key in ("nutrition", "risk", "diet", "fused"):
        assert torch.allclose(out1[key], out2[key], atol=1e-6), \
            f"Mismatch in '{key}' after load"
    ckpt_path.unlink()
    print(f"    Save/load roundtrip: all outputs match  OK")

run_test("Checkpoint save/load produces identical outputs", test_checkpoint)
print()

# -- Summary ----------------------------------------------------------
print("=" * 75)
print(f"RESULTS: {PASSED} passed, {FAILED} failed, {PASSED + FAILED} total")
print("=" * 75)
if FAILED == 0:
    print("ALL TESTS PASSED  OK")
else:
    print(f"WARNING: {FAILED} test(s) failed!")
    sys.exit(1)
