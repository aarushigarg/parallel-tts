from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import zipfile
from pathlib import Path


PROMPTS_FILE = Path(__file__).resolve().parent / "benchmark_prompts.json"
DEFAULT_OUTPUT_ROOT = Path("datasets")


def _read_prompts() -> dict[str, list[str]]:
    with PROMPTS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _clean_cc100_line(line: str) -> str:
    text = re.sub(r"\s+", "", line.strip())
    text = re.sub(r"[^\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff々ー。、！？・「」（）『』\dA-Za-z]", "", text)
    return text


def _bucket_name(length: int) -> str:
    if length < 40:
        return "short"
    if length < 120:
        return "medium"
    return "long"


def _write_prompt_exports(output_root: Path) -> None:
    prompts = _read_prompts()
    prompt_dir = output_root / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    with (prompt_dir / "benchmark_prompts.json").open("w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    with (prompt_dir / "benchmark_prompts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bucket", "index", "num_characters", "text"])
        for bucket, items in prompts.items():
            for index, text in enumerate(items, start=1):
                writer.writerow([bucket, index, len(text), text])

    for bucket, items in prompts.items():
        with (prompt_dir / f"{bucket}.txt").open("w", encoding="utf-8") as f:
            for text in items:
                f.write(text)
                f.write("\n")


def _extract_kaggle_transcript(kaggle_zip: Path, output_root: Path) -> Path:
    dest_dir = output_root / "kaggle_jsss"
    dest_dir.mkdir(parents=True, exist_ok=True)
    transcript_out = dest_dir / "transcript.txt"

    with zipfile.ZipFile(kaggle_zip) as zf:
        for member in zf.namelist():
            if member.endswith("transcript.txt"):
                with zf.open(member) as src, transcript_out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                return transcript_out
    raise FileNotFoundError("Could not find transcript.txt inside the Kaggle zip archive.")


def _copy_kaggle_transcript(kaggle_transcript: Path, output_root: Path) -> Path:
    dest_dir = output_root / "kaggle_jsss"
    dest_dir.mkdir(parents=True, exist_ok=True)
    transcript_out = dest_dir / "transcript.txt"
    shutil.copyfile(kaggle_transcript, transcript_out)
    return transcript_out


def _prepare_cc100_samples(cc100_input: Path, output_root: Path, max_lines_per_bucket: int) -> Path:
    dest_dir = output_root / "cc100_ja"
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, int, str]] = []
    bucket_counts = {"short": 0, "medium": 0, "long": 0}

    with cc100_input.open("r", encoding="utf-8") as f:
        for raw_line in f:
            cleaned = _clean_cc100_line(raw_line)
            if len(cleaned) < 12:
                continue
            bucket = _bucket_name(len(cleaned))
            if bucket_counts[bucket] >= max_lines_per_bucket:
                continue
            rows.append((bucket, len(cleaned), cleaned))
            bucket_counts[bucket] += 1
            if all(count >= max_lines_per_bucket for count in bucket_counts.values()):
                break

    output_csv = dest_dir / "cc100_ja_eval_samples.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bucket", "num_characters", "text"])
        writer.writerows(rows)
    return output_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare benchmark prompts and dataset files.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where prepared prompts and dataset files should be written.",
    )
    parser.add_argument(
        "--kaggle-zip",
        default=None,
        help="Optional path to japanese-single-speaker-speech-dataset.zip.",
    )
    parser.add_argument(
        "--kaggle-transcript",
        default=None,
        help="Optional path to an existing transcript.txt extracted from the Kaggle dataset.",
    )
    parser.add_argument(
        "--cc100-input",
        default=None,
        help="Optional path to a CC100 Japanese text file.",
    )
    parser.add_argument(
        "--cc100-lines-per-bucket",
        type=int,
        default=50,
        help="Maximum number of cleaned CC100 samples to keep in each length bucket.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    _write_prompt_exports(output_root)

    manifest: dict[str, object] = {
        "prompts": {
            "json": str((output_root / "prompts" / "benchmark_prompts.json").resolve()),
            "csv": str((output_root / "prompts" / "benchmark_prompts.csv").resolve()),
        }
    }

    if args.kaggle_zip:
        manifest["kaggle_transcript"] = str(
            _extract_kaggle_transcript(Path(args.kaggle_zip), output_root).resolve()
        )
    elif args.kaggle_transcript:
        manifest["kaggle_transcript"] = str(
            _copy_kaggle_transcript(Path(args.kaggle_transcript), output_root).resolve()
        )

    if args.cc100_input:
        manifest["cc100_eval_samples"] = str(
            _prepare_cc100_samples(
                Path(args.cc100_input),
                output_root,
                max_lines_per_bucket=int(args.cc100_lines_per_bucket),
            ).resolve()
        )

    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
