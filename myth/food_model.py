from __future__ import annotations

import io
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, AutoModelForImageClassification


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "nateraw/food"


class FoodImageClassifier:
    """Load and run the local food classifier with a local cache."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, cache_root: Path | None = None) -> None:
        self.model_id = model_id
        self.cache_root = cache_root or (REPO_ROOT / "myth")
        self.model_dir = self.cache_root / model_id.replace("/", "-")
        self.device = self._choose_device()
        self.processor = None
        self.model = None
        self.loaded = False
        self.load_error: str | None = None
        self._load()

    def _choose_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _ensure_cached(self) -> Path:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        marker = self.model_dir / "config.json"
        if marker.exists():
            return self.model_dir
        snapshot_download(
            repo_id=self.model_id,
            local_dir=str(self.model_dir),
            local_dir_use_symlinks=False,
        )
        return self.model_dir

    def _load(self) -> None:
        try:
            model_path = self._ensure_cached()
            self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
        except Exception as exc:
            self.loaded = False
            self.load_error = str(exc)

    def _open_image(self, image_bytes: bytes) -> Image.Image:
        try:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError("Invalid image format") from exc

    def predict_top_k(self, image_bytes: bytes, top_k: int = 3) -> list[dict[str, float | str]]:
        if not self.loaded or not self.processor or not self.model:
            return []
        if not image_bytes:
            return []

        image = self._open_image(image_bytes)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            probabilities = torch.softmax(logits, dim=-1)[0]
            top_count = min(top_k, probabilities.shape[-1])
            scores, indices = torch.topk(probabilities, k=top_count)

        results: list[dict[str, float | str]] = []
        for score, index in zip(scores, indices):
            label = self.model.config.id2label.get(int(index.item()), str(index.item()))
            results.append({"label": label.replace("_", " "), "score": float(score.item())})
        return results