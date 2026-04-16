import os
import sys
import subprocess
import time
from pathlib import Path

from benchmarking import make_benchmark_result

REPO_DIR = Path("iro_tts")

def run_serial_tts(text, output_wav):
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["RAYON_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    cmd = [
        sys.executable, "infer.py",
        "--hf-checkpoint", "Aratako/Irodori-TTS-500M-v2",
        "--text", text,
        "--no-ref",
        "--output-wav", str(output_wav.resolve())
    ]

    start = time.time()
    subprocess.run(cmd, check=True, cwd=REPO_DIR, env=env)
    end = time.time()

    runtime = end - start
    notes = [
        "serial baseline runs the upstream infer.py CLI",
        "time_to_first_audio_sec equals total_synthesis_sec because the serial CLI does not expose partial audio emission",
    ]
    return make_benchmark_result(
        method="serial",
        total_synthesis_sec=runtime,
        time_to_first_audio_sec=runtime,
        output_wav=output_wav,
        num_characters=len(text),
        notes=notes,
    )
