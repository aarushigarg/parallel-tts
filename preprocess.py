from __future__ import annotations

import csv
import json
from pathlib import Path


def _load_texts_from_transcript(path: Path) -> list[str]:
    texts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 4:
                continue
            text = parts[1].strip()
            if text:
                texts.append(text)
    return texts


def _load_texts_from_csv(path: Path) -> list[str]:
    texts = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            if text:
                texts.append(text)
    return texts


def _load_texts_from_json(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        texts: list[str] = []
        for items in payload.values():
            if isinstance(items, list):
                texts.extend(str(item).strip() for item in items if str(item).strip())
        return texts
    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]
    raise ValueError(f"Unsupported JSON prompt format in {path}")


def _load_texts_from_plaintext(path: Path) -> list[str]:
    texts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                texts.append(text)
    return texts


def load_texts(dataset_root: Path, input_file: str | Path | None = None):
    if input_file is not None:
        path = Path(input_file)
    else:
        path = Path(dataset_root) / "transcript.txt"

    if not path.exists():
        raise FileNotFoundError(f"Input text file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        texts = _load_texts_from_csv(path)
    elif suffix == ".json":
        texts = _load_texts_from_json(path)
    elif suffix == ".txt":
        if path.name == "transcript.txt":
            texts = _load_texts_from_transcript(path)
        else:
            texts = _load_texts_from_plaintext(path)
    else:
        raise ValueError(f"Unsupported input file type for {path}")

    if not texts:
        raise RuntimeError(f"No valid text found in {path}")

    return texts


def select_n_evenly(items, n):
    if not items or n <= 0:
        return []

    if n == 1:
        return [items[len(items) // 2]]

    if n >= len(items):
        return items

    selected = []
    for i in range(n):
        idx = round(i * (len(items) - 1) / (n - 1))
        selected.append(items[idx])

    return selected


def get_length_buckets(dataset_root, n=None, input_file: str | Path | None = None):
    texts = load_texts(Path(dataset_root), input_file=input_file)
    texts = sorted(texts, key=len)

    total = len(texts)
    third = total // 3

    short_bucket = texts[:third]
    medium_bucket = texts[third : 2 * third]
    long_bucket = texts[2 * third :]

    if n is None:
        short_list = short_bucket
        medium_list = medium_bucket
        long_list = long_bucket
    else:
        short_list = select_n_evenly(short_bucket, n)
        medium_list = select_n_evenly(medium_bucket, n)
        long_list = select_n_evenly(long_bucket, n)

    return short_list, medium_list, long_list


def save_buckets_to_csv(short_list, medium_list, long_list, output_file):
    rows = []

    for label, lst in [
        ("short", short_list),
        ("medium", medium_list),
        ("long", long_list),
    ]:
        for text in lst:
            rows.append([label, text, len(text)])

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["length_category", "text", "num_characters"])
        writer.writerows(rows)
