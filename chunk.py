import os
import time
from copy import copy

from benchmarking import make_benchmark_result
from pipelined_infer import (
    create_pipeline_runtime,
    parse_pipeline_args,
    prepare_segments,
    synthesize_chunks_parallel,
)


def _configure_thread_env():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


class ChunkParallelTTSRunner:
    def __init__(self, max_workers=None, crossfade_ms=0.0):
        _configure_thread_env()
        self.max_workers = max_workers
        self.base_args = parse_pipeline_args(
            [
                "--hf-checkpoint",
                "Aratako/Irodori-TTS-500M-v2",
                "--no-ref",
            ]
        )
        self.base_args.chunk_crossfade_ms = float(crossfade_ms)
        print("[chunk] loading InferenceRuntime once")
        self.runtime = create_pipeline_runtime(self.base_args)
        print("[chunk] InferenceRuntime ready")

    def run(self, text, output_wav):
        args = copy(self.base_args)
        args.text = text
        args.text_file = None
        args.text_file_lines = None
        args.output_wav = str(output_wav.resolve())

        segments = prepare_segments(args)
        start = time.time()
        metrics = synthesize_chunks_parallel(
            self.runtime, segments, args, max_workers=self.max_workers
        )
        end = time.time()
        notes = [f"segmented into {metrics.segment_count} chunk(s)"]
        return make_benchmark_result(
            method="chunk",
            total_synthesis_sec=end - start,
            time_to_first_audio_sec=metrics.time_to_first_audio_sec,
            output_wav=output_wav,
            num_characters=len(text),
            sample_rate=metrics.sample_rate,
            notes=notes,
            stage_timings=metrics.stage_timings,
        )
