import os
import sys
import subprocess
import time
from pathlib import Path

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

    return end - start