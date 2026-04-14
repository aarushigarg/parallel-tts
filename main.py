import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

from pipeline import PipelineTTSRunner, RFSplitPipelineTTSRunner
from preprocess import get_length_buckets, save_buckets_to_csv
from serial import run_serial_tts

OUTPUT_DIR = Path("outputs")

def parse_args():
    parser = argparse.ArgumentParser(description="Run TTS timing experiments.")
    parser.add_argument(
        "methods",
        nargs="+",
        choices=["serial", "pipeline", "pipeline1", "pipeline2"],
        help="Inference runner(s) to use.",
    )
    parser.add_argument(
        "-n",
        "--samples-per-bucket",
        type=int,
        default=1,
        help="Number of evenly selected texts to run from each length bucket.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every valid transcript.txt row, still grouped into length buckets.",
    )
    parser.add_argument(
        "--dataset-root",
        default=".",
        help="Directory containing transcript.txt.",
    )
    return parser.parse_args()


def clear_output_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def run_method(method, categorized_texts):
    method_output_dir = OUTPUT_DIR / method
    clear_output_dir(method_output_dir)
    csv_file = method_output_dir / f"{method}_results.csv"
    results = []
    sample_index = 1

    if method == "serial":
        run_tts = run_serial_tts
    elif method in {"pipeline", "pipeline1"}:
        pipeline_runner = PipelineTTSRunner()
        run_tts = pipeline_runner.run
    elif method == "pipeline2":
        pipeline_runner = RFSplitPipelineTTSRunner()
        run_tts = pipeline_runner.run
    else:
        raise ValueError(f"Unsupported method: {method}")

    program_start = datetime.now()
    program_start_str = program_start.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nRunning {method} -> {method_output_dir}")

    for length_label, text_list in categorized_texts:
        for text in text_list:
            output_wav = method_output_dir / f"{length_label}_{sample_index}.wav"
            num_characters = len(text)

            try:
                runtime = run_tts(text, output_wav)

                print(f"[{sample_index}] ({length_label}) Saved: {output_wav}")
                print(f"[{sample_index}] Time: {runtime:.2f}s")

                results.append([
                    sample_index,
                    method,
                    program_start_str,
                    length_label,
                    num_characters,
                    text,
                    str(output_wav),
                    f"{runtime:.4f}",
                ])

            except Exception as e:
                print(f"[{sample_index}] ({length_label}) Failed: {e}")

                results.append([
                    sample_index,
                    method,
                    program_start_str,
                    length_label,
                    num_characters,
                    text,
                    str(output_wav),
                    "ERROR",
                ])

            sample_index += 1

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "method",
            "program_start",
            "length_category",
            "num_characters",
            "text",
            "output_wav",
            "runtime_sec",
        ])
        writer.writerows(results)

    print(f"\nSaved CSV results to {csv_file}")


def main():
    args = parse_args()
    methods = list(dict.fromkeys(args.methods))

    dataset_root = Path(args.dataset_root)
    samples_per_bucket = None if args.all else int(args.samples_per_bucket)
    if samples_per_bucket is not None and samples_per_bucket <= 0:
        raise ValueError("--samples-per-bucket must be > 0.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    short_list, medium_list, long_list = get_length_buckets(dataset_root, samples_per_bucket)

    categorized_texts = [
        ("short", short_list),
        ("medium", medium_list),
        ("long", long_list),
    ]

    bucket_csv = OUTPUT_DIR / "buckets.csv"
    save_buckets_to_csv(short_list, medium_list, long_list, bucket_csv)
    print(f"Saved bucket data to {bucket_csv}")

    for method in methods:
        run_method(method, categorized_texts)


if __name__ == "__main__":
    main()
