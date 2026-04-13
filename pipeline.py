import os
import time
from copy import copy

from pipelined_infer import (
    create_pipeline_runtime,
    parse_pipeline_args,
    prepare_segments,
    synthesize_segmented_text,
)


def _configure_thread_env():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


class PipelineTTSRunner:
    def __init__(self):
        _configure_thread_env()
        self.base_args = parse_pipeline_args(
            [
                "--hf-checkpoint",
                "Aratako/Irodori-TTS-500M-v2",
                "--no-ref",
            ]
        )
        print("[pipeline] loading InferenceRuntime once")
        self.runtime = create_pipeline_runtime(self.base_args)
        print("[pipeline] InferenceRuntime ready")

    def run(self, text, output_wav):
        args = copy(self.base_args)
        args.text = text
        args.text_file = None
        args.text_file_lines = None
        args.output_wav = str(output_wav.resolve())

        segments = prepare_segments(args)
        start = time.time()
        synthesize_segmented_text(self.runtime, segments, args)
        end = time.time()
        return end - start


_DEFAULT_RUNNER = None


def run_pipeline_tts(text, output_wav):
    global _DEFAULT_RUNNER
    if _DEFAULT_RUNNER is None:
        _DEFAULT_RUNNER = PipelineTTSRunner()
    return _DEFAULT_RUNNER.run(text, output_wav)
