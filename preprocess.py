from pathlib import Path
import csv

def load_texts(dataset_root: Path):
    transcript_path = dataset_root / "transcript.txt"

    if not transcript_path.exists():
        raise FileNotFoundError(f"transcript.txt not found in {dataset_root}")

    texts = []

    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 4:
                continue

            # normalized script (better for TTS)
            text = parts[1].strip()

            if text:
                texts.append(text)

    if not texts:
        raise RuntimeError("No valid text found in transcript.txt")

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


def get_length_buckets(dataset_root, n):
    """
    Returns:
        short_list, medium_list, long_list
        each containing n phrases
    """

    texts = load_texts(Path(dataset_root))

    # sort by character length
    texts = sorted(texts, key=len)

    total = len(texts)
    third = total // 3

    short_bucket = texts[:third]
    medium_bucket = texts[third:2 * third]
    long_bucket = texts[2 * third:]

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
            rows.append([
                label,
                text,
                len(text)
            ])

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["length_category", "text", "num_characters"])
        writer.writerows(rows)