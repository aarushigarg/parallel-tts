from __future__ import annotations

import csv
import json
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def detect_nvidia_smi() -> str | None:
    return shutil.which("nvidia-smi")


def _audio_duration_seconds(output_wav: str | Path | None, sample_rate: int | None = None) -> float | None:
    if output_wav is None:
        return None
    path = Path(output_wav)
    if not path.is_file():
        return None

    if sample_rate is not None:
        try:
            import torchaudio

            waveform, _ = torchaudio.load(str(path))
            if waveform.ndim < 2:
                return None
            return float(waveform.shape[-1]) / float(sample_rate)
        except Exception:
            pass

    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.frames > 0 and info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        return None
    return None


@dataclass
class GPUSample:
    relative_time_sec: float
    index: int
    name: str
    utilization_gpu_pct: float | None
    utilization_memory_pct: float | None
    memory_used_mb: float | None
    memory_total_mb: float | None


@dataclass
class GPUStatsSummary:
    sample_count: int = 0
    device_count: int = 0
    avg_gpu_utilization_pct: float | None = None
    max_gpu_utilization_pct: float | None = None
    avg_memory_utilization_pct: float | None = None
    max_memory_utilization_pct: float | None = None
    peak_memory_used_mb: float | None = None
    monitor_error: str | None = None


@dataclass
class BenchmarkResult:
    method: str
    total_synthesis_sec: float
    time_to_first_audio_sec: float | None
    audio_duration_sec: float | None
    num_characters: int
    output_wav: str
    notes: list[str] = field(default_factory=list)
    gpu_stats: GPUStatsSummary = field(default_factory=GPUStatsSummary)
    stage_timings: dict[str, float] = field(default_factory=dict)

    @property
    def chars_per_sec(self) -> float | None:
        if self.total_synthesis_sec <= 0:
            return None
        return float(self.num_characters) / float(self.total_synthesis_sec)

    @property
    def audio_seconds_per_sec(self) -> float | None:
        if self.total_synthesis_sec <= 0 or self.audio_duration_sec is None:
            return None
        return float(self.audio_duration_sec) / float(self.total_synthesis_sec)

    @property
    def real_time_factor(self) -> float | None:
        throughput = self.audio_seconds_per_sec
        if throughput in {None, 0.0}:
            return None
        return 1.0 / throughput


class GPUMonitor:
    def __init__(self, poll_interval_sec: float = 0.25):
        self.poll_interval_sec = max(0.1, float(poll_interval_sec))
        self._binary = detect_nvidia_smi()
        self._samples: list[GPUSample] = []
        self._error: str | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float | None = None

    @property
    def available(self) -> bool:
        return self._binary is not None

    def start(self) -> None:
        if not self.available:
            return
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> GPUStatsSummary:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=self.poll_interval_sec + 1.0)
        return summarize_gpu_samples(self._samples, self._error)

    def write_csv(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "relative_time_sec",
                    "gpu_index",
                    "gpu_name",
                    "utilization_gpu_pct",
                    "utilization_memory_pct",
                    "memory_used_mb",
                    "memory_total_mb",
                ]
            )
            for sample in self._samples:
                writer.writerow(
                    [
                        f"{sample.relative_time_sec:.4f}",
                        sample.index,
                        sample.name,
                        "" if sample.utilization_gpu_pct is None else f"{sample.utilization_gpu_pct:.2f}",
                        ""
                        if sample.utilization_memory_pct is None
                        else f"{sample.utilization_memory_pct:.2f}",
                        "" if sample.memory_used_mb is None else f"{sample.memory_used_mb:.2f}",
                        "" if sample.memory_total_mb is None else f"{sample.memory_total_mb:.2f}",
                    ]
                )

    def _run(self) -> None:
        assert self._binary is not None
        query = (
            "index,name,utilization.gpu,utilization.memory,memory.used,memory.total"
        )
        while not self._stop_event.is_set():
            try:
                completed = subprocess.run(
                    [
                        self._binary,
                        f"--query-gpu={query}",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                now = time.perf_counter()
                rel_time = 0.0 if self._start_time is None else now - self._start_time
                for line in completed.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != 6:
                        continue
                    memory_used = _safe_float(parts[4])
                    memory_total = _safe_float(parts[5])
                    memory_util = None
                    if memory_used is not None and memory_total not in {None, 0.0}:
                        memory_util = (memory_used / memory_total) * 100.0
                    self._samples.append(
                        GPUSample(
                            relative_time_sec=rel_time,
                            index=int(parts[0]),
                            name=parts[1],
                            utilization_gpu_pct=_safe_float(parts[2]),
                            utilization_memory_pct=memory_util
                            if memory_util is not None
                            else _safe_float(parts[3]),
                            memory_used_mb=memory_used,
                            memory_total_mb=memory_total,
                        )
                    )
            except Exception as exc:
                self._error = str(exc)
                return
            self._stop_event.wait(self.poll_interval_sec)


def summarize_gpu_samples(samples: list[GPUSample], error: str | None = None) -> GPUStatsSummary:
    if not samples:
        return GPUStatsSummary(sample_count=0, device_count=0, monitor_error=error)

    gpu_utils = [value for value in (sample.utilization_gpu_pct for sample in samples) if value is not None]
    mem_utils = [
        value for value in (sample.utilization_memory_pct for sample in samples) if value is not None
    ]
    mem_used = [value for value in (sample.memory_used_mb for sample in samples) if value is not None]
    device_ids = {sample.index for sample in samples}

    return GPUStatsSummary(
        sample_count=len(samples),
        device_count=len(device_ids),
        avg_gpu_utilization_pct=(sum(gpu_utils) / len(gpu_utils)) if gpu_utils else None,
        max_gpu_utilization_pct=max(gpu_utils) if gpu_utils else None,
        avg_memory_utilization_pct=(sum(mem_utils) / len(mem_utils)) if mem_utils else None,
        max_memory_utilization_pct=max(mem_utils) if mem_utils else None,
        peak_memory_used_mb=max(mem_used) if mem_used else None,
        monitor_error=error,
    )


def make_benchmark_result(
    *,
    method: str,
    total_synthesis_sec: float,
    time_to_first_audio_sec: float | None,
    output_wav: str | Path,
    num_characters: int,
    audio_duration_sec: float | None = None,
    sample_rate: int | None = None,
    notes: list[str] | None = None,
    gpu_stats: GPUStatsSummary | None = None,
    stage_timings: dict[str, float] | None = None,
) -> BenchmarkResult:
    output_wav_str = str(Path(output_wav))
    resolved_audio_duration = (
        audio_duration_sec
        if audio_duration_sec is not None
        else _audio_duration_seconds(output_wav_str, sample_rate=sample_rate)
    )
    return BenchmarkResult(
        method=method,
        total_synthesis_sec=float(total_synthesis_sec),
        time_to_first_audio_sec=None
        if time_to_first_audio_sec is None
        else float(time_to_first_audio_sec),
        audio_duration_sec=resolved_audio_duration,
        num_characters=int(num_characters),
        output_wav=output_wav_str,
        notes=[] if notes is None else list(notes),
        gpu_stats=GPUStatsSummary() if gpu_stats is None else gpu_stats,
        stage_timings={} if stage_timings is None else dict(stage_timings),
    )


def flatten_benchmark_result(
    result: BenchmarkResult,
    *,
    index: int,
    program_start: str,
    length_category: str,
    text: str,
    speedup_vs_serial: float | None,
) -> dict[str, object]:
    row = {
        "index": index,
        "method": result.method,
        "program_start": program_start,
        "length_category": length_category,
        "num_characters": result.num_characters,
        "text": text,
        "output_wav": result.output_wav,
        "total_synthesis_sec": f"{result.total_synthesis_sec:.4f}",
        "time_to_first_audio_sec": ""
        if result.time_to_first_audio_sec is None
        else f"{result.time_to_first_audio_sec:.4f}",
        "audio_duration_sec": ""
        if result.audio_duration_sec is None
        else f"{result.audio_duration_sec:.4f}",
        "chars_per_sec": ""
        if result.chars_per_sec is None
        else f"{result.chars_per_sec:.4f}",
        "audio_seconds_per_sec": ""
        if result.audio_seconds_per_sec is None
        else f"{result.audio_seconds_per_sec:.4f}",
        "real_time_factor": ""
        if result.real_time_factor is None
        else f"{result.real_time_factor:.4f}",
        "speedup_vs_serial": ""
        if speedup_vs_serial is None
        else f"{speedup_vs_serial:.4f}",
        "gpu_sample_count": result.gpu_stats.sample_count,
        "gpu_device_count": result.gpu_stats.device_count,
        "gpu_avg_utilization_pct": ""
        if result.gpu_stats.avg_gpu_utilization_pct is None
        else f"{result.gpu_stats.avg_gpu_utilization_pct:.2f}",
        "gpu_max_utilization_pct": ""
        if result.gpu_stats.max_gpu_utilization_pct is None
        else f"{result.gpu_stats.max_gpu_utilization_pct:.2f}",
        "gpu_avg_memory_utilization_pct": ""
        if result.gpu_stats.avg_memory_utilization_pct is None
        else f"{result.gpu_stats.avg_memory_utilization_pct:.2f}",
        "gpu_max_memory_utilization_pct": ""
        if result.gpu_stats.max_memory_utilization_pct is None
        else f"{result.gpu_stats.max_memory_utilization_pct:.2f}",
        "gpu_peak_memory_used_mb": ""
        if result.gpu_stats.peak_memory_used_mb is None
        else f"{result.gpu_stats.peak_memory_used_mb:.2f}",
        "gpu_monitor_error": result.gpu_stats.monitor_error or "",
        "notes_json": json.dumps(result.notes, ensure_ascii=False),
        "stage_timings_json": json.dumps(result.stage_timings, ensure_ascii=False, sort_keys=True),
    }
    return row


def aggregate_results(rows: list[dict[str, object]]) -> dict[str, float | None]:
    def _mean(key: str) -> float | None:
        values: list[float] = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
            elif isinstance(value, str) and value.strip() != "":
                try:
                    values.append(float(value))
                except ValueError:
                    continue
        if not values:
            return None
        return sum(values) / len(values)

    return {
        "avg_total_synthesis_sec": _mean("total_synthesis_sec"),
        "avg_time_to_first_audio_sec": _mean("time_to_first_audio_sec"),
        "avg_audio_duration_sec": _mean("audio_duration_sec"),
        "avg_chars_per_sec": _mean("chars_per_sec"),
        "avg_audio_seconds_per_sec": _mean("audio_seconds_per_sec"),
        "avg_real_time_factor": _mean("real_time_factor"),
        "avg_speedup_vs_serial": _mean("speedup_vs_serial"),
        "avg_gpu_utilization_pct": _mean("gpu_avg_utilization_pct"),
        "avg_gpu_memory_utilization_pct": _mean("gpu_avg_memory_utilization_pct"),
    }


def write_json(path: str | Path, payload: object) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def gpu_stats_to_dict(stats: GPUStatsSummary) -> dict[str, object]:
    return asdict(stats)
