import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from benchmarking import (
    GPUMonitor,
    aggregate_results,
    flatten_benchmark_result,
    write_csv_rows,
    write_json,
)
from chunk import ChunkParallelTTSRunner
from pipeline import PipelineTTSRunner, RFSplitPipelineTTSRunner
from preprocess import get_length_buckets, save_buckets_to_csv
from serial import run_serial_tts

OUTPUT_DIR = Path("outputs")


def parse_args():
    parser = argparse.ArgumentParser(description="Run TTS timing experiments.")
    parser.add_argument(
        "methods",
        nargs="+",
        choices=["serial", "pipeline", "pipeline1", "pipeline2", "chunk"],
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
    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional .txt, .csv, or .json file containing benchmark prompts.",
    )
    parser.add_argument(
        "--gpu-poll-interval",
        type=float,
        default=0.25,
        help="Polling interval in seconds for nvidia-smi GPU utilization logging.",
    )
    parser.add_argument(
        "--chunk-crossfade-ms",
        type=float,
        default=0.0,
        help="Crossfade duration in ms at chunk boundaries (chunk method only). 0 uses silence instead.",
    )
    parser.add_argument(
        "--chunk-max-workers",
        type=int,
        default=None,
        help="Max parallel worker threads for chunk method. Defaults to one thread per segment. Start with 2-4 on CUDA to limit GPU memory pressure.",
    )
    return parser.parse_args()


def clear_output_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def create_runner(method, *, chunk_crossfade_ms=0.0, chunk_max_workers=None):
    if method == "serial":
        return run_serial_tts
    if method in {"pipeline", "pipeline1"}:
        return PipelineTTSRunner().run
    if method == "pipeline2":
        return RFSplitPipelineTTSRunner().run
    if method == "chunk":
        return ChunkParallelTTSRunner(crossfade_ms=chunk_crossfade_ms, max_workers=chunk_max_workers).run
    raise ValueError(f"Unsupported method: {method}")


def _print_sample_summary(index, method, length_label, result, speedup_vs_serial):
    print(f"[{index}] ({length_label}) Saved: {result.output_wav}")
    print(f"[{index}] total_synthesis_sec: {result.total_synthesis_sec:.2f}")
    if result.time_to_first_audio_sec is not None:
        print(f"[{index}] time_to_first_audio_sec: {result.time_to_first_audio_sec:.2f}")
    if result.audio_duration_sec is not None:
        print(f"[{index}] audio_duration_sec: {result.audio_duration_sec:.2f}")
    if result.audio_seconds_per_sec is not None:
        print(f"[{index}] throughput_audio_seconds_per_sec: {result.audio_seconds_per_sec:.3f}")
    if speedup_vs_serial is not None:
        print(f"[{index}] speedup_vs_serial: {speedup_vs_serial:.3f}x")
    if result.gpu_stats.sample_count > 0:
        print(
            f"[{index}] gpu_avg_utilization_pct: "
            f"{result.gpu_stats.avg_gpu_utilization_pct:.2f}"
        )
    elif result.gpu_stats.monitor_error:
        print(f"[{index}] gpu_monitor_error: {result.gpu_stats.monitor_error}")


def run_method(method, categorized_texts, *, gpu_poll_interval, serial_baselines, chunk_crossfade_ms=0.0, chunk_max_workers=None):
    method_output_dir = OUTPUT_DIR / method
    clear_output_dir(method_output_dir)
    csv_file = method_output_dir / f"{method}_results.csv"
    summary_file = method_output_dir / f"{method}_summary.json"
    results = []
    sample_index = 1
    run_tts = create_runner(method, chunk_crossfade_ms=chunk_crossfade_ms, chunk_max_workers=chunk_max_workers)

    program_start = datetime.now()
    program_start_str = program_start.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nRunning {method} -> {method_output_dir}")

    for length_label, text_list in categorized_texts:
        for text in text_list:
            output_wav = method_output_dir / f"{length_label}_{sample_index}.wav"
            gpu_csv = method_output_dir / f"{length_label}_{sample_index}_gpu.csv"

            monitor = GPUMonitor(poll_interval_sec=gpu_poll_interval)
            try:
                try:
                    monitor.start()
                    result = run_tts(text, output_wav)
                    result.method = method
                    result.num_characters = len(text)
                finally:
                    gpu_stats = monitor.stop()
                    if gpu_stats.sample_count > 0:
                        monitor.write_csv(gpu_csv)
                result.gpu_stats = gpu_stats

                speedup_vs_serial = None
                if method == "serial":
                    serial_baselines[text] = result.total_synthesis_sec
                else:
                    baseline = serial_baselines.get(text)
                    if baseline not in {None, 0.0}:
                        speedup_vs_serial = baseline / result.total_synthesis_sec
                    else:
                        result.notes.append(
                            "speedup_vs_serial unavailable because serial baseline was not run earlier in this command"
                        )

                _print_sample_summary(sample_index, method, length_label, result, speedup_vs_serial)
                results.append(
                    flatten_benchmark_result(
                        result,
                        index=sample_index,
                        program_start=program_start_str,
                        length_category=length_label,
                        text=text,
                        speedup_vs_serial=speedup_vs_serial,
                    )
                )
            except Exception as exc:
                print(f"[{sample_index}] ({length_label}) Failed: {exc}")
                results.append(
                    {
                        "index": sample_index,
                        "method": method,
                        "program_start": program_start_str,
                        "length_category": length_label,
                        "num_characters": len(text),
                        "text": text,
                        "output_wav": str(output_wav),
                        "total_synthesis_sec": "ERROR",
                        "time_to_first_audio_sec": "",
                        "audio_duration_sec": "",
                        "chars_per_sec": "",
                        "audio_seconds_per_sec": "",
                        "real_time_factor": "",
                        "speedup_vs_serial": "",
                        "gpu_sample_count": 0,
                        "gpu_device_count": 0,
                        "gpu_avg_utilization_pct": "",
                        "gpu_max_utilization_pct": "",
                        "gpu_avg_memory_utilization_pct": "",
                        "gpu_max_memory_utilization_pct": "",
                        "gpu_peak_memory_used_mb": "",
                        "gpu_monitor_error": "",
                        "notes_json": json.dumps([str(exc)], ensure_ascii=False),
                        "stage_timings_json": "{}",
                    }
                )
            finally:
                sample_index += 1

    write_csv_rows(csv_file, results)
    write_json(
        summary_file,
        {
            "method": method,
            "program_start": program_start_str,
            "sample_count": len(results),
            "aggregates": aggregate_results(results),
            "gpu_monitor_available": GPUMonitor(poll_interval_sec=gpu_poll_interval).available,
            "results": results,
        },
    )
    print(f"\nSaved CSV results to {csv_file}")
    print(f"Saved JSON summary to {summary_file}")
    return results


def write_run_summary(all_results):
    summary_rows = []
    by_method = defaultdict(list)
    for row in all_results:
        by_method[str(row["method"])].append(row)

    serial_rows = by_method.get("serial", [])
    serial_times = {}
    for row in serial_rows:
        try:
            serial_times[str(row["text"])] = float(row["total_synthesis_sec"])
        except (TypeError, ValueError):
            continue

    for method, rows in by_method.items():
        enriched_rows = []
        for row in rows:
            copied = dict(row)
            if method != "serial":
                baseline = serial_times.get(str(row["text"]))
                try:
                    current = float(row["total_synthesis_sec"])
                except (TypeError, ValueError):
                    current = None
                copied["speedup_vs_serial"] = (
                    ""
                    if baseline in {None, 0.0} or current in {None, 0.0}
                    else f"{baseline / current:.4f}"
                )
            enriched_rows.append(copied)

        aggregates = aggregate_results(enriched_rows)
        summary_rows.append(
            {
                "method": method,
                "sample_count": len(rows),
                "avg_total_synthesis_sec": ""
                if aggregates["avg_total_synthesis_sec"] is None
                else f"{aggregates['avg_total_synthesis_sec']:.4f}",
                "avg_time_to_first_audio_sec": ""
                if aggregates["avg_time_to_first_audio_sec"] is None
                else f"{aggregates['avg_time_to_first_audio_sec']:.4f}",
                "avg_audio_duration_sec": ""
                if aggregates["avg_audio_duration_sec"] is None
                else f"{aggregates['avg_audio_duration_sec']:.4f}",
                "avg_chars_per_sec": ""
                if aggregates["avg_chars_per_sec"] is None
                else f"{aggregates['avg_chars_per_sec']:.4f}",
                "avg_audio_seconds_per_sec": ""
                if aggregates["avg_audio_seconds_per_sec"] is None
                else f"{aggregates['avg_audio_seconds_per_sec']:.4f}",
                "avg_real_time_factor": ""
                if aggregates["avg_real_time_factor"] is None
                else f"{aggregates['avg_real_time_factor']:.4f}",
                "avg_speedup_vs_serial": ""
                if aggregates["avg_speedup_vs_serial"] is None
                else f"{aggregates['avg_speedup_vs_serial']:.4f}",
                "avg_gpu_utilization_pct": ""
                if aggregates["avg_gpu_utilization_pct"] is None
                else f"{aggregates['avg_gpu_utilization_pct']:.2f}",
                "avg_gpu_memory_utilization_pct": ""
                if aggregates["avg_gpu_memory_utilization_pct"] is None
                else f"{aggregates['avg_gpu_memory_utilization_pct']:.2f}",
            }
        )

    write_csv_rows(OUTPUT_DIR / "run_summary.csv", summary_rows)
    write_json(OUTPUT_DIR / "run_summary.json", {"methods": summary_rows, "results": all_results})


def main():
    args = parse_args()
    methods = list(dict.fromkeys(args.methods))

    dataset_root = Path(args.dataset_root)
    samples_per_bucket = None if args.all else int(args.samples_per_bucket)
    if samples_per_bucket is not None and samples_per_bucket <= 0:
        raise ValueError("--samples-per-bucket must be > 0.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    short_list, medium_list, long_list = get_length_buckets(
        dataset_root,
        samples_per_bucket,
        input_file=args.input_file,
    )

    categorized_texts = [
        ("short", short_list),
        ("medium", medium_list),
        ("long", long_list),
    ]

    bucket_csv = OUTPUT_DIR / "buckets.csv"
    save_buckets_to_csv(short_list, medium_list, long_list, bucket_csv)
    print(f"Saved bucket data to {bucket_csv}")

    all_results = []
    serial_baselines: dict[str, float] = {}
    for method in methods:
        method_rows = run_method(
            method,
            categorized_texts,
            gpu_poll_interval=float(args.gpu_poll_interval),
            serial_baselines=serial_baselines,
            chunk_crossfade_ms=float(args.chunk_crossfade_ms),
            chunk_max_workers=args.chunk_max_workers,
        )
        all_results.extend(method_rows)

    write_run_summary(all_results)
    print(f"Saved run summary to {OUTPUT_DIR / 'run_summary.csv'}")


if __name__ == "__main__":
    main()
