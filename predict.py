from __future__ import annotations

import argparse
from pathlib import Path

from myth.food_model import FoodImageClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a food label from an image.")
    parser.add_argument("image", nargs="?", default="food.jpg", help="Path to the food image.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)

    if not image_path.exists():
        print(f"Error: missing image file: {image_path}")
        return 1

    try:
        image_bytes = image_path.read_bytes()
        classifier = FoodImageClassifier(model_id="nateraw/food", cache_root=Path(__file__).resolve().parent / "myth")
        predictions = classifier.predict_top_k(image_bytes, top_k=1)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:
        print(f"Error: failed to run prediction: {exc}")
        return 1

    if not predictions:
        message = classifier.load_error or "model unavailable"
        print(f"Error: {message}")
        return 1

    best = predictions[0]
    print(f"Predicted food: {best['label']} ({best['score']:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())