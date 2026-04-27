"""Run FoodIntel NLP embedding extraction on Recipe CSV files.

Examples:
    python run_nlp.py --device cuda --batch-size 32
    python run_nlp.py --input datasets/first_part.csv --rows 1000 --device auto
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from nlp_encoder import BERT_MODEL, MAX_SEQ, FoodNLPProcessor


DEFAULT_INPUTS = ("datasets/first_part.csv", "datasets/second_part.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU NLP embedding extraction for FoodIntel-AI.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=list(DEFAULT_INPUTS),
        help="CSV files to process. Defaults to the two Recipe CSVs in datasets/.",
    )
    parser.add_argument("--output-dir", default="outputs/nlp", help="Directory for NPZ shards and metadata.")
    parser.add_argument("--model-name", default=BERT_MODEL, help="Hugging Face encoder model name.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device.")
    parser.add_argument("--batch-size", type=int, default=32, help="Texts per GPU forward pass.")
    parser.add_argument("--chunk-size", type=int, default=1024, help="CSV rows per pandas chunk.")
    parser.add_argument("--rows", type=int, default=None, help="Optional max rows per input file for quick tests.")
    parser.add_argument("--max-length", type=int, default=MAX_SEQ, help="Tokenizer max sequence length.")
    parser.add_argument("--text-column", default="NER", help="Preferred ingredient column.")
    parser.add_argument("--title-column", default="title", help="Recipe title column.")
    parser.add_argument("--id-column", default="Unnamed: 0", help="Stable row id column.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def ingredients_to_text(value: object) -> str:
    """Normalize Recipe1M-style list strings into comma-separated ingredient text."""
    if pd.isna(value):
        return ""
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())

    text = str(value).strip()
    if not text:
        return ""
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
    if isinstance(parsed, list):
        return ", ".join(str(item).strip() for item in parsed if str(item).strip())
    return str(parsed)


def batched(items: list[str], batch_size: int) -> Iterator[tuple[int, list[str]]]:
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        for child in path.glob("*"):
            if child.is_file():
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def resolve_text_column(frame: pd.DataFrame, preferred: str) -> str:
    if preferred in frame.columns:
        return preferred
    if "ingredients" in frame.columns:
        return "ingredients"
    raise KeyError(f"Could not find text column {preferred!r} or fallback 'ingredients'.")


def process_file(
    csv_path: Path,
    processor: FoodNLPProcessor,
    output_dir: Path,
    args: argparse.Namespace,
    manifest_rows: list[dict[str, object]],
) -> int:
    rows_done = 0
    shard_idx = 0
    metadata_path = output_dir / "metadata.csv"
    write_header = not metadata_path.exists()

    reader = pd.read_csv(csv_path, chunksize=args.chunk_size, nrows=args.rows)
    with metadata_path.open("a", newline="", encoding="utf-8") as meta_f:
        meta_writer = csv.DictWriter(
            meta_f,
            fieldnames=["source_file", "row_id", "title", "ingredients", "shard", "offset"],
        )
        if write_header:
            meta_writer.writeheader()

        for chunk in tqdm(reader, desc=csv_path.name, unit="chunk"):
            text_column = resolve_text_column(chunk, args.text_column)
            texts = [ingredients_to_text(value) for value in chunk[text_column].tolist()]
            titles = (
                chunk[args.title_column].fillna("").astype(str).tolist()
                if args.title_column in chunk.columns
                else [""] * len(chunk)
            )
            row_ids = (
                chunk[args.id_column].astype(str).tolist()
                if args.id_column in chunk.columns
                else [str(rows_done + i) for i in range(len(chunk))]
            )

            emb_parts = []
            constraint_parts = []
            for _, text_batch in batched(texts, args.batch_size):
                embeddings, constraints = processor.encode(text_batch)
                emb_parts.append(embeddings.numpy())
                constraint_parts.append(constraints.numpy())

            embedding_array = np.concatenate(emb_parts, axis=0)
            constraint_array = np.concatenate(constraint_parts, axis=0)
            shard_name = f"{csv_path.stem}_shard_{shard_idx:05d}.npz"
            shard_path = output_dir / shard_name
            np.savez_compressed(
                shard_path,
                row_id=np.array(row_ids),
                title=np.array(titles),
                ingredients=np.array(texts),
                embedding=embedding_array,
                constraint=constraint_array,
            )

            for offset, (row_id, title, ingredients) in enumerate(zip(row_ids, titles, texts)):
                meta_writer.writerow(
                    {
                        "source_file": str(csv_path),
                        "row_id": row_id,
                        "title": title,
                        "ingredients": ingredients,
                        "shard": shard_name,
                        "offset": offset,
                    }
                )

            manifest_rows.append(
                {
                    "source_file": str(csv_path),
                    "shard": shard_name,
                    "rows": int(len(chunk)),
                    "embedding_shape": list(embedding_array.shape),
                    "constraint_shape": list(constraint_array.shape),
                }
            )
            rows_done += len(chunk)
            shard_idx += 1

    return rows_done


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)

    device_before = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable. Install a CUDA-enabled PyTorch build or use --device cpu.")

    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    processor = FoodNLPProcessor(
        model_name=args.model_name,
        device=args.device,
        max_length=args.max_length,
        use_amp=not args.no_amp,
    )
    print(f"Using device: {processor.device}")

    manifest_rows: list[dict[str, object]] = []
    total_rows = 0
    start = perf_counter()
    for input_name in args.input:
        csv_path = Path(input_name)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        total_rows += process_file(csv_path, processor, output_dir, args, manifest_rows)

    elapsed = perf_counter() - start
    manifest = {
        "model_name": args.model_name,
        "device_requested": args.device,
        "device_used": str(processor.device),
        "cuda_was_available": device_before == "cuda",
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "max_length": args.max_length,
        "total_rows": total_rows,
        "elapsed_seconds": round(elapsed, 3),
        "rows_per_second": round(total_rows / elapsed, 3) if elapsed > 0 else None,
        "shards": manifest_rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Processed {total_rows} rows into {output_dir}")


if __name__ == "__main__":
    main()
