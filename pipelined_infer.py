#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from queue import Empty, Queue
import sys
from threading import Thread
import time
from typing import Any
from pathlib import Path

from text_segments import load_text_for_segmentation, split_text_segments

IRO_TTS_DIR = Path(__file__).resolve().parent / "iro_tts"
sys.path.insert(0, str(IRO_TTS_DIR))

FIXED_SECONDS = 30.0


@dataclass
class _DecodeJob:
    index: int
    request: Any
    context: Any
    prepared: Any
    latent: Any
    stage_timings: list[tuple[str, float]]
    stage_start: float
    used_seed: int


@dataclass
class _DecodeResult:
    index: int
    audio: Any
    sample_rate: int
    stage_timings: list[tuple[str, float]]
    total_to_decode: float
    used_seed: int
    completed_at: float


def _parse_optional_float(value: str) -> float | None:
    raw = str(value).strip().lower()
    if raw in {"none", "null", "off", "disable", "disabled"}:
        return None
    try:
        out = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected float or one of [none, null, off, disable, disabled]."
        ) from exc
    if not math.isfinite(out):
        raise argparse.ArgumentTypeError(f"Expected finite float for value={value!r}.")
    return out


def _print_timings(timings: list[tuple[str, float]], total_to_decode: float) -> None:
    print("[timing] ---- post-model-load to decode ----")
    for name, sec in timings:
        print(f"[timing] {name}: {sec * 1000.0:.1f} ms")
    print(f"[timing] total_to_decode: {total_to_decode:.3f} s")


def _concat_audio_segments(
    audios,
    *,
    sample_rate: int,
    silence_ms: float,
):
    import torch

    if not audios:
        raise ValueError("Expected at least one audio segment to concatenate.")
    if len(audios) == 1:
        return audios[0]

    silence_samples = max(0, int(float(sample_rate) * float(silence_ms) / 1000.0))
    if silence_samples == 0:
        return torch.cat(audios, dim=-1)

    parts: list[torch.Tensor] = []
    for audio in audios:
        if parts:
            silence = audio.new_zeros((audio.shape[0], silence_samples))
            parts.append(silence)
        parts.append(audio)
    return torch.cat(parts, dim=-1)


def _resolve_checkpoint_path(args: argparse.Namespace) -> str:
    from huggingface_hub import hf_hub_download

    if args.checkpoint is not None:
        checkpoint_path = Path(str(args.checkpoint)).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"[checkpoint] using local file: {checkpoint_path}", flush=True)
        return str(checkpoint_path)

    repo_id = str(args.hf_checkpoint).strip()
    if repo_id == "":
        raise ValueError("hf_checkpoint must be non-empty.")

    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
    )
    print(
        f"[checkpoint] downloaded model.safetensors from hf://{repo_id} -> {checkpoint_path}",
        flush=True,
    )
    return str(checkpoint_path)


def _load_runtime_api():
    from pipelined_runtime import (
        InferenceRuntime,
        RuntimeKey,
        SamplingRequest,
        default_runtime_device,
        resolve_cfg_scales,
        save_wav,
    )

    return (
        InferenceRuntime,
        RuntimeKey,
        SamplingRequest,
        default_runtime_device,
        resolve_cfg_scales,
        save_wav,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference for Irodori-TTS.")
    checkpoint_group = parser.add_mutually_exclusive_group(required=False)
    checkpoint_group.add_argument(
        "--checkpoint",
        default=None,
        help="Local model checkpoint path (.pt or .safetensors).",
    )
    checkpoint_group.add_argument(
        "--hf-checkpoint",
        default=None,
        help=(
            "Hugging Face model repo id to download model.safetensors from "
            "(e.g. your-org/your-model)."
        ),
    )
    parser.add_argument("--text", default=None)
    parser.add_argument(
        "--text-file",
        default=None,
        help="Optional text file. For transcript rows with pipes, the second field is used.",
    )
    parser.add_argument(
        "--text-file-lines",
        type=int,
        default=None,
        help="Read only the first N non-empty text rows from --text-file.",
    )
    parser.add_argument(
        "--caption",
        default=None,
        help="Optional caption/style-control text for caption-enabled voice-design checkpoints.",
    )
    parser.add_argument("--output-wav", default="output.wav")
    parser.add_argument(
        "--segment-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Split input text into punctuation-delimited segments before synthesis.",
    )
    parser.add_argument(
        "--segment-min-chars",
        type=int,
        default=12,
        help="Merge adjacent punctuation-delimited segments until they reach this length.",
    )
    parser.add_argument(
        "--segment-max-chars",
        type=int,
        default=180,
        help="Hard-split any segment longer than this many characters. Set <=0 to disable.",
    )
    parser.add_argument(
        "--segment-silence-ms",
        type=float,
        default=80.0,
        help="Silence inserted between synthesized segments in the final wav.",
    )
    parser.add_argument(
        "--save-segments",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save each synthesized segment next to the final output wav.",
    )
    parser.add_argument(
        "--pipeline-overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the decode-worker pipeline path instead of segmented serial synthesis.",
    )
    parser.add_argument(
        "--dry-run-segments",
        action="store_true",
        help="Only print text segments and exit without loading the model.",
    )
    parser.add_argument(
        "--model-device",
        default="auto",
        help="Model inference device (e.g. cuda, mps, cpu).",
    )
    parser.add_argument(
        "--model-precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Model precision for weights/compute.",
    )
    parser.add_argument(
        "--codec-device",
        default="auto",
        help="Codec device for reference encode/decode (e.g. cuda, mps, cpu).",
    )
    parser.add_argument(
        "--codec-precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Codec precision for weights/compute.",
    )
    parser.add_argument(
        "--codec-deterministic-encode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic DACVAE encode path (default: enabled).",
    )
    parser.add_argument(
        "--codec-deterministic-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic DACVAE decode watermark-message path (default: enabled).",
    )
    parser.add_argument(
        "--enable-watermark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable DACVAE watermark branch during decode (default: disabled).",
    )
    parser.add_argument(
        "--max-ref-seconds",
        type=float,
        default=30.0,
        help="Maximum reference duration in seconds. Set <=0 to disable the cap.",
    )
    parser.add_argument(
        "--ref-normalize-db",
        type=_parse_optional_float,
        default=-16.0,
        help=(
            "Target loudness (dB/LUFS-like) for reference audio before DACVAE encode "
            "(e.g. -16.0). Set to 'none' to disable. Default: -16."
        ),
    )
    parser.add_argument(
        "--ref-ensure-max",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Scale reference audio down only when peak exceeds 1.0 after optional loudness "
            "normalization. Effective only when --ref-normalize-db is none/null/off "
            "(default: enabled)."
        ),
    )
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=None,
        help=(
            "Maximum token length for text conditioning. "
            "Defaults to checkpoint metadata max_text_len when available, else 256."
        ),
    )
    parser.add_argument(
        "--max-caption-len",
        type=int,
        default=None,
        help=(
            "Maximum token length for caption conditioning. "
            "Defaults to checkpoint metadata max_caption_len when available, else max_text_len."
        ),
    )
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        help="Number of candidates to generate in a single batched sampling pass.",
    )
    parser.add_argument(
        "--decode-mode",
        choices=["sequential", "batch"],
        default="sequential",
        help=(
            "Codec decode mode. "
            "'sequential': decode each candidate one-by-one (lower VRAM), "
            "'batch': decode all candidates at once (faster, higher VRAM)."
        ),
    )
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.compile for core inference methods (default: disabled).",
    )
    parser.add_argument(
        "--compile-dynamic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use dynamic=True for torch.compile (default: disabled).",
    )
    parser.add_argument("--cfg-scale-text", type=float, default=3.0)
    parser.add_argument("--cfg-scale-caption", type=float, default=3.0)
    parser.add_argument("--cfg-scale-speaker", type=float, default=5.0)
    parser.add_argument(
        "--cfg-guidance-mode",
        choices=["independent", "joint", "alternating"],
        default="independent",
        help=(
            "CFG formulation. "
            "'independent': each enabled condition uses its own uncond pass, "
            "'joint': drop all enabled conditions together (2x NFE), "
            "'alternating': alternate enabled condition unconds each step."
        ),
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="Deprecated. If set, overrides --cfg-scale-text/--cfg-scale-caption/--cfg-scale-speaker.",
    )
    parser.add_argument("--cfg-min-t", type=float, default=0.5)
    parser.add_argument("--cfg-max-t", type=float, default=1.0)
    parser.add_argument(
        "--truncation-factor",
        type=float,
        default=None,
        help=(
            "Scale initial Gaussian noise before Euler sampling "
            "(e.g., 0.8 flat / 0.9 sharp). Default: disabled."
        ),
    )
    parser.add_argument(
        "--rescale-k",
        type=float,
        default=None,
        help=(
            "Temporal score rescaling k (Xu et al., 2025). "
            "Set together with --rescale-sigma. Default: disabled."
        ),
    )
    parser.add_argument(
        "--rescale-sigma",
        type=float,
        default=None,
        help=(
            "Temporal score rescaling sigma (Xu et al., 2025). "
            "Set together with --rescale-k. Default: disabled."
        ),
    )
    parser.add_argument(
        "--context-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Precompute per-layer text/speaker context K/V projections for faster sampling "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--speaker-kv-scale",
        type=float,
        default=None,
        help=(
            "Force-speaker mode: scale speaker K/V projections by this factor (>1 strengthens speaker identity). "
            "Default: disabled."
        ),
    )
    parser.add_argument(
        "--speaker-kv-min-t",
        type=float,
        default=0.9,
        help=(
            "Disable speaker KV scaling after crossing this timestep threshold "
            "(applies while t >= value). Default: 0.9."
        ),
    )
    parser.add_argument(
        "--speaker-kv-max-layers",
        type=int,
        default=None,
        help="Apply speaker KV scaling only to first N diffusion layers (default: all layers).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed. If omitted, a random seed is generated per request.",
    )
    parser.add_argument(
        "--trim-tail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Trim trailing near-zero latent region with Echo-style flattening heuristic "
            "(default: enabled)."
        ),
    )
    parser.add_argument("--tail-window-size", type=int, default=20)
    parser.add_argument("--tail-std-threshold", type=float, default=0.05)
    parser.add_argument("--tail-mean-threshold", type=float, default=0.1)
    parser.add_argument(
        "--show-timings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Print per-stage timings from post-model-load through latent decode (default: enabled)."
        ),
    )
    ref_group = parser.add_mutually_exclusive_group(required=False)
    ref_group.add_argument(
        "--ref-wav", default=None, help="Reference waveform path for speaker conditioning."
    )
    ref_group.add_argument(
        "--ref-latent", default=None, help="Reference latent (.pt) path for speaker conditioning."
    )
    ref_group.add_argument(
        "--no-ref",
        action="store_true",
        help="Run without speaker reference conditioning. Use this for voice-design checkpoints.",
    )
    return parser


def _prepare_segments(args: argparse.Namespace) -> list[str]:
    text_input = load_text_for_segmentation(
        text=args.text,
        text_file=args.text_file,
        text_file_lines=args.text_file_lines,
    )

    if bool(args.segment_text):
        segments = split_text_segments(
            text_input,
            min_chars=max(1, int(args.segment_min_chars)),
            max_chars=int(args.segment_max_chars),
        )
    else:
        segments = [text_input]
    if not segments:
        raise ValueError("No non-empty text segments to synthesize.")
    return segments


def parse_pipeline_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def prepare_segments(args: argparse.Namespace) -> list[str]:
    return _prepare_segments(args)


def _print_segments(segments: list[str]) -> None:
    print(f"[pipeline] segment_count: {len(segments)}")
    for index, segment in enumerate(segments, start=1):
        print(f"[pipeline] segment[{index:03d}] chars={len(segment)} text={segment}")


def _validate_runtime_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.checkpoint is None and args.hf_checkpoint is None:
        parser.error(
            "one of --checkpoint or --hf-checkpoint is required unless "
            "--dry-run-segments is set."
        )
    if int(args.num_candidates) != 1:
        raise ValueError("Pipelined inference currently supports --num-candidates 1.")


def create_pipeline_runtime(args: argparse.Namespace):
    InferenceRuntime, RuntimeKey, _, default_runtime_device, _, _ = _load_runtime_api()
    checkpoint_path = _resolve_checkpoint_path(args)
    model_device = (
        default_runtime_device() if str(args.model_device) == "auto" else str(args.model_device)
    )
    codec_device = (
        default_runtime_device() if str(args.codec_device) == "auto" else str(args.codec_device)
    )

    return InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=model_device,
            codec_repo=str(args.codec_repo),
            model_precision=str(args.model_precision),
            codec_device=codec_device,
            codec_precision=str(args.codec_precision),
            codec_deterministic_encode=bool(args.codec_deterministic_encode),
            codec_deterministic_decode=bool(args.codec_deterministic_decode),
            enable_watermark=bool(args.enable_watermark),
            compile_model=bool(args.compile_model),
            compile_dynamic=bool(args.compile_dynamic),
        )
    )


def _resolve_request_cfg_scales(args: argparse.Namespace, runtime):
    _, _, _, _, resolve_cfg_scales, _ = _load_runtime_api()
    if runtime.model_cfg.use_speaker_condition and not (
        args.no_ref or args.ref_wav is not None or args.ref_latent is not None
    ):
        raise ValueError(
            "speaker-conditioned checkpoints require one of --ref-wav, --ref-latent, or --no-ref."
        )
    cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, scale_messages = resolve_cfg_scales(
        cfg_guidance_mode=str(args.cfg_guidance_mode),
        cfg_scale_text=float(args.cfg_scale_text),
        cfg_scale_caption=float(args.cfg_scale_caption),
        cfg_scale_speaker=float(args.cfg_scale_speaker),
        cfg_scale=float(args.cfg_scale) if args.cfg_scale is not None else None,
        use_caption_condition=bool(
            runtime.model_cfg.use_caption_condition
            and args.caption is not None
            and str(args.caption).strip() != ""
        ),
        use_speaker_condition=bool(runtime.model_cfg.use_speaker_condition),
    )
    for msg in scale_messages:
        print(msg)
    return cfg_scale_text, cfg_scale_caption, cfg_scale_speaker


def _build_sampling_request(
    SamplingRequest,
    *,
    args: argparse.Namespace,
    text: str,
    seed: int | None,
    cfg_scale_text: float,
    cfg_scale_caption: float,
    cfg_scale_speaker: float,
):
    return SamplingRequest(
        text=text,
        caption=None if args.caption is None else str(args.caption),
        ref_wav=args.ref_wav,
        ref_latent=args.ref_latent,
        no_ref=bool(args.no_ref),
        ref_normalize_db=args.ref_normalize_db,
        ref_ensure_max=bool(args.ref_ensure_max),
        num_candidates=int(args.num_candidates),
        decode_mode=str(args.decode_mode),
        seconds=FIXED_SECONDS,
        max_ref_seconds=float(args.max_ref_seconds) if args.max_ref_seconds is not None else None,
        max_text_len=None if args.max_text_len is None else int(args.max_text_len),
        max_caption_len=None if args.max_caption_len is None else int(args.max_caption_len),
        num_steps=int(args.num_steps),
        cfg_scale_text=cfg_scale_text,
        cfg_scale_caption=cfg_scale_caption,
        cfg_scale_speaker=cfg_scale_speaker,
        cfg_guidance_mode=str(args.cfg_guidance_mode),
        cfg_scale=None,
        cfg_min_t=float(args.cfg_min_t),
        cfg_max_t=float(args.cfg_max_t),
        truncation_factor=None if args.truncation_factor is None else float(args.truncation_factor),
        rescale_k=None if args.rescale_k is None else float(args.rescale_k),
        rescale_sigma=None if args.rescale_sigma is None else float(args.rescale_sigma),
        context_kv_cache=bool(args.context_kv_cache),
        speaker_kv_scale=None if args.speaker_kv_scale is None else float(args.speaker_kv_scale),
        speaker_kv_min_t=None if args.speaker_kv_scale is None else float(args.speaker_kv_min_t),
        speaker_kv_max_layers=None
        if args.speaker_kv_max_layers is None
        else int(args.speaker_kv_max_layers),
        seed=seed,
        trim_tail=bool(args.trim_tail),
        tail_window_size=int(args.tail_window_size),
        tail_std_threshold=float(args.tail_std_threshold),
        tail_mean_threshold=float(args.tail_mean_threshold),
    )


def synthesize_segmented_text(
    runtime,
    segments: list[str],
    args: argparse.Namespace,
) -> float:
    _, _, SamplingRequest, _, _, save_wav = _load_runtime_api()
    cfg_scale_text, cfg_scale_caption, cfg_scale_speaker = _resolve_request_cfg_scales(
        args,
        runtime,
    )
    output_path = Path(str(args.output_wav))
    suffix = output_path.suffix if output_path.suffix else ".wav"
    segment_audios = []
    segment_timings: list[tuple[int, list[tuple[str, float]], float]] = []
    used_seeds: list[int] = []
    sample_rate: int | None = None
    time_to_first_audio: float | None = None
    pipeline_t0 = time.perf_counter()

    for index, segment in enumerate(segments, start=1):
        segment_seed = None if args.seed is None else int(args.seed) + index - 1
        result = runtime.synthesize(
            _build_sampling_request(
                SamplingRequest,
                args=args,
                text=segment,
                seed=segment_seed,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_caption=cfg_scale_caption,
                cfg_scale_speaker=cfg_scale_speaker,
            ),
            log_fn=None,
        )
        if sample_rate is None:
            sample_rate = int(result.sample_rate)
        elif int(result.sample_rate) != sample_rate:
            raise ValueError(
                f"Segment {index} sample_rate={result.sample_rate} did not match {sample_rate}."
            )

        segment_audios.append(result.audio)
        segment_timings.append((index, result.stage_timings, result.total_to_decode))
        used_seeds.append(int(result.used_seed))
        if time_to_first_audio is None:
            time_to_first_audio = time.perf_counter() - pipeline_t0

        print(f"[seed] segment[{index:03d}] used_seed: {result.used_seed}")
        if bool(args.save_segments):
            segment_path = output_path.with_name(
                f"{output_path.stem}_segment_{index:03d}{suffix}"
            )
            saved = save_wav(segment_path, result.audio, result.sample_rate)
            print(f"Saved segment[{index:03d}]: {saved}")

    assert sample_rate is not None
    final_audio = _concat_audio_segments(
        segment_audios,
        sample_rate=sample_rate,
        silence_ms=float(args.segment_silence_ms),
    )
    out_path = save_wav(args.output_wav, final_audio, sample_rate)
    print(f"Saved: {out_path}")

    total_pipeline_time = time.perf_counter() - pipeline_t0
    print(f"[pipeline] time_to_first_audio: {time_to_first_audio:.3f} s")
    print(f"[pipeline] total_segmented_synthesis: {total_pipeline_time:.3f} s")
    print(f"[pipeline] used_seeds: {used_seeds}")
    if args.show_timings:
        for index, timings, total_to_decode in segment_timings:
            print(f"[timing] ---- segment[{index:03d}] ----")
            _print_timings(timings, total_to_decode)
    return total_pipeline_time


def synthesize_segmented_text_pipelined(
    runtime,
    segments: list[str],
    args: argparse.Namespace,
) -> float:
    import torch

    _, _, SamplingRequest, _, _, save_wav = _load_runtime_api()
    cfg_scale_text, cfg_scale_caption, cfg_scale_speaker = _resolve_request_cfg_scales(
        args,
        runtime,
    )
    output_path = Path(str(args.output_wav))
    suffix = output_path.suffix if output_path.suffix else ".wav"
    decode_queue: Queue[_DecodeJob | None] = Queue()
    result_queue: Queue[_DecodeResult | BaseException] = Queue()
    results_by_index: dict[int, _DecodeResult] = {}
    used_seeds: list[int] = []
    sample_rate: int | None = None
    time_to_first_audio: float | None = None
    pipeline_t0 = time.perf_counter()

    def _ignore_log(_msg: str) -> None:
        return

    def _decode_worker() -> None:
        while True:
            job = decode_queue.get()
            try:
                if job is None:
                    return
                try:
                    with torch.inference_mode():
                        audios = runtime.decode_audio(
                            job.request,
                            job.context,
                            job.prepared,
                            job.latent,
                            stage_timings=job.stage_timings,
                            log_fn=_ignore_log,
                        )
                    if not audios:
                        raise RuntimeError(f"Segment {job.index} produced no decoded audio.")
                    result_queue.put(
                        _DecodeResult(
                            index=job.index,
                            audio=audios[0],
                            sample_rate=int(runtime.codec.sample_rate),
                            stage_timings=job.stage_timings,
                            total_to_decode=time.perf_counter() - job.stage_start,
                            used_seed=job.used_seed,
                            completed_at=time.perf_counter(),
                        )
                    )
                except BaseException as exc:
                    result_queue.put(exc)
                    return
            finally:
                decode_queue.task_done()

    def _record_result(item: _DecodeResult | BaseException) -> None:
        nonlocal sample_rate, time_to_first_audio
        if isinstance(item, BaseException):
            raise item
        if sample_rate is None:
            sample_rate = item.sample_rate
        elif item.sample_rate != sample_rate:
            raise ValueError(
                f"Segment {item.index} sample_rate={item.sample_rate} did not match {sample_rate}."
            )
        results_by_index[item.index] = item
        if time_to_first_audio is None:
            time_to_first_audio = item.completed_at - pipeline_t0

    def _drain_completed_results() -> None:
        while True:
            try:
                item = result_queue.get_nowait()
            except Empty:
                return
            _record_result(item)

    worker = Thread(target=_decode_worker, name="tts-decode-worker", daemon=True)
    worker.start()

    for index, segment in enumerate(segments, start=1):
        segment_seed = None if args.seed is None else int(args.seed) + index - 1
        request = _build_sampling_request(
            SamplingRequest,
            args=args,
            text=segment,
            seed=segment_seed,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_caption=cfg_scale_caption,
            cfg_scale_speaker=cfg_scale_speaker,
        )
        messages: list[str] = []
        stage_timings: list[tuple[str, float]] = []
        stage_start = time.perf_counter()

        with torch.inference_mode():
            context = runtime._build_sampling_context(
                request,
                messages=messages,
                log_fn=_ignore_log,
            )
            prepared = runtime.prepare_sampling_inputs(
                request,
                context,
                messages=messages,
                stage_timings=stage_timings,
                log_fn=_ignore_log,
            )
            z_patched = runtime.generate_latent(
                request,
                context,
                prepared,
                stage_timings=stage_timings,
                log_fn=_ignore_log,
            )
            latent = runtime.unpatchify_sampled_latent(
                z_patched,
                prepared,
                stage_timings=stage_timings,
                log_fn=_ignore_log,
            )

        decode_queue.put(
            _DecodeJob(
                index=index,
                request=request,
                context=context,
                prepared=prepared,
                latent=latent,
                stage_timings=stage_timings,
                stage_start=stage_start,
                used_seed=int(context.used_seed),
            )
        )
        used_seeds.append(int(context.used_seed))
        print(f"[seed] segment[{index:03d}] used_seed: {context.used_seed}")
        _drain_completed_results()

    decode_queue.put(None)
    while len(results_by_index) < len(segments):
        _record_result(result_queue.get())
    decode_queue.join()
    worker.join(timeout=1.0)

    assert sample_rate is not None
    ordered_results = [results_by_index[index] for index in range(1, len(segments) + 1)]
    if bool(args.save_segments):
        for result in ordered_results:
            segment_path = output_path.with_name(
                f"{output_path.stem}_segment_{result.index:03d}{suffix}"
            )
            saved = save_wav(segment_path, result.audio, result.sample_rate)
            print(f"Saved segment[{result.index:03d}]: {saved}")

    final_audio = _concat_audio_segments(
        [result.audio for result in ordered_results],
        sample_rate=sample_rate,
        silence_ms=float(args.segment_silence_ms),
    )
    out_path = save_wav(args.output_wav, final_audio, sample_rate)
    print(f"Saved: {out_path}")

    total_pipeline_time = time.perf_counter() - pipeline_t0
    print(f"[pipeline] time_to_first_audio: {time_to_first_audio:.3f} s")
    print(f"[pipeline] total_segmented_synthesis: {total_pipeline_time:.3f} s")
    print(f"[pipeline] used_seeds: {used_seeds}")
    if args.show_timings:
        for result in ordered_results:
            print(f"[timing] ---- segment[{result.index:03d}] ----")
            _print_timings(result.stage_timings, result.total_to_decode)
    return total_pipeline_time


def run_from_args(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> float | None:
    if parser is None:
        parser = _build_parser()

    segments = prepare_segments(args)
    if not bool(args.dry_run_segments):
        _validate_runtime_args(parser, args)

    _print_segments(segments)
    if bool(args.dry_run_segments):
        return

    runtime = create_pipeline_runtime(args)
    if bool(args.pipeline_overlap):
        return synthesize_segmented_text_pipelined(runtime, segments, args)
    return synthesize_segmented_text(runtime, segments, args)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_from_args(args, parser=parser)


if __name__ == "__main__":
    main()
