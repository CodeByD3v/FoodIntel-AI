"""Fine-tune the NLP projection layer and optional BERT layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml

try:
    from .cleaner import IngredientCleaner
    from .encoder import BERTIngredientEncoder
    from .ner import IngredientNER
    from .usda_lookup import USDALookup
except ImportError:  # pragma: no cover
    from cleaner import IngredientCleaner
    from encoder import BERTIngredientEncoder
    from ner import IngredientNER
    from usda_lookup import USDALookup


class IngredientCategoryDataset(Dataset):
    """Food-101 style ingredient text classification dataset."""

    def __init__(self, samples: Sequence[Dict], label_to_id: Dict[str, int]) -> None:
        self.samples = list(samples)
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        return {
            "ingredient_str": sample["ingredient_str"],
            "ingredient_list": sample["ingredient_list"],
            "label": self.label_to_id[sample["label"]],
        }


class IngredientClassifier(nn.Module):
    """Projection encoder plus classification head."""

    def __init__(self, encoder: BERTIngredientEncoder, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        output_dim = encoder.projection[-1].out_features
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, batch: List[Dict]) -> torch.Tensor:
        embeddings = self.encoder.encode_batch(batch)
        return self.classifier(embeddings)


def load_food101_ingredient_samples(meta_json_path: str) -> List[Dict]:
    """Load class-labelled ingredient strings from Food-101 style JSON metadata."""
    cleaner = IngredientCleaner()
    ner = IngredientNER()
    with open(meta_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples: List[Dict] = []
    if isinstance(raw, dict):
        iterable = raw.items()
    elif isinstance(raw, list):
        iterable = ((row.get("label") or row.get("class") or row.get("category"), [row]) for row in raw)
    else:
        raise ValueError("Food-101 ingredient metadata must be a dict or a list of records.")

    for label, entries in iterable:
        if not label:
            continue
        if not isinstance(entries, list):
            entries = [entries]
        for entry in entries:
            text = _entry_to_text(entry)
            cleaned = cleaner.clean(text)
            ingredients = ner.extract(cleaned)
            if cleaned and ingredients:
                samples.append({"ingredient_str": cleaned, "ingredient_list": ingredients, "label": str(label)})
    if not samples:
        raise ValueError("No usable ingredient samples found in metadata.")
    return samples


def _entry_to_text(entry) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, list):
        return ", ".join(str(item) for item in entry)
    if isinstance(entry, dict):
        for key in ("ingredients", "ingredient_list", "ingredient_str", "text"):
            if key in entry:
                value = entry[key]
                return ", ".join(value) if isinstance(value, list) else str(value)
    return str(entry)


def collate_batch(rows: List[Dict]) -> tuple[List[Dict], torch.Tensor]:
    batch = [
        {"ingredient_str": row["ingredient_str"], "ingredient_list": row["ingredient_list"]}
        for row in rows
    ]
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)
    return batch, labels


def build_usda_lookup(cfg: dict) -> USDALookup | None:
    usda_cfg = cfg["usda"]
    paths = [usda_cfg["features_path"], usda_cfg["descriptions_path"], usda_cfg["scaler_path"]]
    if not all(Path(path).exists() for path in paths):
        return None
    return USDALookup(
        features_path=usda_cfg["features_path"],
        descriptions_path=usda_cfg["descriptions_path"],
        scaler_path=usda_cfg["scaler_path"],
        score_cutoff=usda_cfg["score_cutoff"],
        cache_size=usda_cfg.get("cache_size", 2000),
    )


def train_epoch(model, loader, optimizer, criterion, device, max_norm: float) -> float:
    model.train()
    total_loss = 0.0
    for batch, labels in tqdm(loader, desc="train", leave=False):
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    for batch, labels in tqdm(loader, desc="val", leave=False):
        labels = labels.to(device)
        logits = model(batch)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += labels.size(0)
    return total_loss / max(count, 1), correct / max(count, 1)


def freeze_bert(model: IngredientClassifier) -> None:
    for parameter in model.encoder.bert.parameters():
        parameter.requires_grad = False


def unfreeze_last_bert_layers(model: IngredientClassifier, num_layers: int = 2) -> None:
    freeze_bert(model)
    bert = model.encoder.bert
    if hasattr(bert, "encoder") and hasattr(bert.encoder, "layer"):
        for layer in bert.encoder.layer[-num_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    if hasattr(bert, "pooler") and bert.pooler is not None:
        for parameter in bert.pooler.parameters():
            parameter.requires_grad = True


def save_checkpoint(path: str, model: IngredientClassifier, label_to_id: Dict[str, int], val_accuracy: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "classifier_state_dict": model.classifier.state_dict(),
            "label_to_id": label_to_id,
            "val_accuracy": val_accuracy,
        },
        path,
    )


def run_training(config_path: str, food101_meta: str, phase: str = "both") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    samples = load_food101_ingredient_samples(food101_meta)
    labels = sorted({sample["label"] for sample in samples})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.15,
        random_state=42,
        stratify=[sample["label"] for sample in samples] if len(labels) > 1 else None,
    )

    train_loader = DataLoader(
        IngredientCategoryDataset(train_samples, label_to_id),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        IngredientCategoryDataset(val_samples, label_to_id),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = BERTIngredientEncoder(
        bert_model_name=cfg["model"]["bert_model_name"],
        usda_lookup=build_usda_lookup(cfg),
        freeze_bert=cfg["model"]["freeze_bert"],
        output_dim=cfg["model"]["output_dim"],
        dropout=cfg["model"]["dropout"],
    )
    model = IngredientClassifier(encoder, num_classes=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    checkpoint = cfg["paths"]["checkpoint"]

    if phase in {"1", "phase1", "both"}:
        freeze_bert(model)
        optimizer = torch.optim.AdamW(
            list(model.encoder.projection.parameters()) + list(model.classifier.parameters()),
            lr=cfg["training"]["lr_head"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        for epoch in range(1, cfg["training"]["phase1_epochs"] + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, cfg["training"]["max_norm"])
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
            print(f"phase=1 epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}")
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                save_checkpoint(checkpoint, model, label_to_id, best_accuracy)

    if phase in {"2", "phase2", "both"}:
        unfreeze_last_bert_layers(model, num_layers=2)
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in model.encoder.bert.parameters() if p.requires_grad], "lr": cfg["training"]["lr_bert"]},
                {"params": model.encoder.projection.parameters(), "lr": cfg["training"]["lr_projection"]},
                {"params": model.classifier.parameters(), "lr": cfg["training"]["lr_projection"]},
            ],
            weight_decay=cfg["training"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["phase2_epochs"])
        for epoch in range(1, cfg["training"]["phase2_epochs"] + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, cfg["training"]["max_norm"])
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            print(f"phase=2 epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}")
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                save_checkpoint(checkpoint, model, label_to_id, best_accuracy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Food Intelligence NLP encoder.")
    parser.add_argument("--config", default="configs/nlp.yaml")
    parser.add_argument("--food101-meta", required=True, help="Path to Food-101 ingredient train JSON.")
    parser.add_argument("--phase", default="both", choices=["1", "2", "phase1", "phase2", "both"])
    args = parser.parse_args()
    run_training(args.config, args.food101_meta, args.phase)


if __name__ == "__main__":
    main()
