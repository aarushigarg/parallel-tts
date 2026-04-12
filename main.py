import csv
from pathlib import Path
from datetime import datetime

from preprocess import get_length_buckets, save_buckets_to_csv
from serial import run_serial_tts

OUTPUT_DIR = Path("outputs")
CSV_FILE = OUTPUT_DIR / "serial_results.csv"


def main():
    DATASET_ROOT = Path(".")
    N = 1                   

    # setup
    program_start = datetime.now()
    program_start_str = program_start.strftime("%Y-%m-%d %H:%M:%S")

    #clears outputs folder
    OUTPUT_DIR.mkdir(exist_ok=True)
    for item in OUTPUT_DIR.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    short_list, medium_list, long_list = get_length_buckets(DATASET_ROOT, N)

    categorized_texts = [
        ("short", short_list),
        ("medium", medium_list),
        ("long", long_list),
    ]

    short_list, medium_list, long_list = get_length_buckets(DATASET_ROOT, N)

    #saves inputs to csv for easy viewing
    BUCKET_CSV = OUTPUT_DIR / "buckets.csv"
    save_buckets_to_csv(short_list, medium_list, long_list, BUCKET_CSV)

    print(f"Saved bucket data to {BUCKET_CSV}")

    results = []
    sample_index = 1

    #run serial
    for length_label, text_list in categorized_texts:
        for text in text_list:
            output_wav = OUTPUT_DIR / f"{length_label}_{sample_index}.wav"
            num_characters = len(text)

            try:
                runtime = run_serial_tts(text, output_wav)

                print(f"[{sample_index}] ({length_label}) Saved: {output_wav}")
                print(f"[{sample_index}] Time: {runtime:.2f}s")

                results.append([
                    sample_index,
                    "serial",
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
                    "serial",
                    program_start_str,
                    length_label,
                    num_characters,
                    text,
                    str(output_wav),
                    "ERROR",
                ])

            sample_index += 1

    #write csv
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
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

    print(f"\nSaved CSV results to {CSV_FILE}")


if __name__ == "__main__":
    main()