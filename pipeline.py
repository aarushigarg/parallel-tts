import os
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(".")

def run_pipeline_tts(text, output_wav):
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["RAYON_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    cmd = [
        sys.executable,
        "pipelined_infer.py",
        "--hf-checkpoint",
        "Aratako/Irodori-TTS-500M-v2",
        "--text",
        text,
        "--no-ref",
        "--output-wav",
        str(output_wav.resolve()),
    ]

    start = time.time()
    subprocess.run(cmd, check=True, cwd=REPO_DIR, env=env)
    end = time.time()

    return end - start
