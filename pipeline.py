import os
import time
from copy import copy

from pipelined_infer import (
    create_pipeline_runtime,
    parse_pipeline_args,
    prepare_segments,
    synthesize_segmented_text_rf_split,
    synthesize_segmented_text_pipelined,
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
        synthesize_segmented_text_pipelined(self.runtime, segments, args)
        end = time.time()
        return end - start


class RFSplitPipelineTTSRunner:
    def __init__(self, split_step=None):
        import torch

        _configure_thread_env()
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError(
                "pipeline3 requires at least 2 CUDA GPUs. "
                f"torch.cuda.device_count()={torch.cuda.device_count()}."
            )
        self.split_step = split_step
        self.base_args = parse_pipeline_args(
            [
                "--hf-checkpoint",
                "Aratako/Irodori-TTS-500M-v2",
                "--no-ref",
            ]
        )

        stage0_args = copy(self.base_args)
        stage0_args.model_device = "cuda:0"
        stage0_args.codec_device = "cpu"
        stage1_args = copy(self.base_args)
        stage1_args.model_device = "cuda:1"
        stage1_args.codec_device = "cuda:1"

        print("[pipeline3] loading stage 0 InferenceRuntime on cuda:0")
        self.runtime_stage0 = create_pipeline_runtime(stage0_args)
        print("[pipeline3] loading stage 1 InferenceRuntime on cuda:1")
        self.runtime_stage1 = create_pipeline_runtime(stage1_args)
        print("[pipeline3] InferenceRuntimes ready")

    def run(self, text, output_wav):
        args = copy(self.base_args)
        args.text = text
        args.text_file = None
        args.text_file_lines = None
        args.output_wav = str(output_wav.resolve())

        segments = prepare_segments(args)
        start = time.time()
        synthesize_segmented_text_rf_split(
            self.runtime_stage0,
            self.runtime_stage1,
            segments,
            args,
            split_step=self.split_step,
        )
        end = time.time()
        return end - start


_DEFAULT_RUNNER = None


def run_pipeline_tts(text, output_wav):
    global _DEFAULT_RUNNER
    if _DEFAULT_RUNNER is None:
        _DEFAULT_RUNNER = PipelineTTSRunner()
    return _DEFAULT_RUNNER.run(text, output_wav)
