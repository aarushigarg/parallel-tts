# parallel-tts

## Setup 
* Ensure you have Python 3.11
* Run `bash setup_env.sh` to setup venv and install requirements
* macOS note: if you hit an `llvmlite` build error, install LLVM 20:
  * `brew install llvm@20`
  * Then rerun `bash setup_env.sh` (the script will pick up `llvm-config` automatically if installed)
  * If you have an issue with the python version, create a `.env.local` file with the custom path.
  * Ex: `PYTHON_BIN=python3.11`

* Run `source .venv/bin/activate` to access the venv in terminal
  * Add it to a script to avoid having to access it manually
  * Run `deactivate` to exit venv
  
### Download the dataset
We use the Japanese Single Speaker Speech Dataset:
https://www.kaggle.com/datasets/bryanpark/japanese-single-speaker-speech-dataset

#### Manual Download
1. Go to the link above
2. Click **Download**
3. Move the `.zip` file into this project folder
4. Unzip it:
```bash
unzip -j japanese-single-speaker-speech-dataset.zip "*/transcript.txt"
```

#### With Kaggle
```bash
kaggle datasets download -d bryanpark/japanese-single-speaker-speech-dataset
unzip -j japanese-single-speaker-speech-dataset.zip "*/transcript.txt"
rm japanese-single-speaker-speech-dataset.zip
```

### Prepare benchmark prompts and datasets
The benchmarking setup now includes:
- curated short / medium / long Japanese prompts in `benchmark_prompts.json`
- a dataset prep utility that stages the Kaggle JSSS transcript and filtered CC100 Japanese text

Write prepared assets into `datasets/`:
```bash
python dataset_setup.py \
  --kaggle-zip japanese-single-speaker-speech-dataset.zip \
  --cc100-input path/to/cc100-ja.txt
```

If you already extracted `transcript.txt`:
```bash
python dataset_setup.py \
  --kaggle-transcript transcript.txt \
  --cc100-input path/to/cc100-ja.txt
```
## Run

Ensure you are in the venv:
```bash
source .venv/bin/activate
```

Run one method:
```bash
python main.py serial
python main.py pipeline
python main.py pipeline1
python main.py pipeline2
python main.py chunk
```

Run multiple methods in one command:
```bash
python main.py serial pipeline1 pipeline2 chunk
```

By default, this runs 1 short, 1 medium, and 1 long text from `transcript.txt`.

To run more samples from each length bucket:
```bash
python main.py pipeline1 pipeline2 -n 5
```

To run every valid row in `transcript.txt`:
```bash
python main.py pipeline --all
```

Use `--all` carefully. The dataset has thousands of rows and model inference can take a long time.

If `transcript.txt` is somewhere else:
```bash
python main.py serial --dataset-root path/to/dataset-folder
```

To benchmark with prepared prompts or CC100 samples instead of `transcript.txt`:
```bash
python main.py serial pipeline1 \
  --input-file datasets/prompts/benchmark_prompts.json

python main.py serial pipeline1 \
  --input-file datasets/cc100_ja/cc100_ja_eval_samples.csv \
  -n 3
```

To run the pipeline implementation directly on a few transcript rows:
```bash
python pipelined_infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text-file transcript.txt \
  --text-file-lines 2 \
  --no-ref \
  --pipeline-overlap \
  --output-wav outputs/pipeline_overlap.wav
```

To check only the text splitting without loading the model:
```bash
python pipelined_infer.py --text-file transcript.txt --text-file-lines 2 --dry-run-segments
```

To run chunk parallelism with crossfade and a worker cap (recommended on a single GPU):
```bash
python main.py chunk \
  --input-file benchmark_prompts.json \
  --chunk-crossfade-ms 30 \
  --chunk-max-workers 4
```

`--chunk-crossfade-ms` sets the crossfade overlap in milliseconds at segment boundaries (default: 0, which uses silence instead). `--chunk-max-workers` caps the number of parallel threads (default: one per segment). Start with 4 on a single GPU to avoid memory pressure.

Running pipelined version in Discovery CARC:
```bash
salloc --partition=gpu --constraint=v100 --gres=gpu:1 --cpus-per-task=4 --mem=32GB --time=2:00:00
nvidia-smi
source .venv/bin/activate
python -c "import torch; print('gpu count:', torch.cuda.device_count()); print('cuda available:', torch.cuda.is_available())"
python main.py pipeline1
```
Ensure that a GPU is allocated, and cuda is available. 

`pipeline2` requires two GPUs:
```bash
salloc --partition=gpu --constraint=v100 --gres=gpu:2 --cpus-per-task=4 --mem=64GB --time=1:00:00
nvidia-smi
source .venv/bin/activate
python -c "import torch; print('gpu count:', torch.cuda.device_count()); print('cuda available:', torch.cuda.is_available())"
python main.py pipeline2
```

To leave the salloc:
```bash
exit
```

Outputs are written per method:
```text
outputs/
  buckets.csv
  run_summary.csv
  run_summary.json
  serial/
    serial_results.csv
    serial_summary.json
    short_1.wav
    short_1_gpu.csv
    medium_2.wav
    long_3.wav
  pipeline/
    pipeline_results.csv
    pipeline_summary.json
    short_1.wav
    medium_2.wav
    long_3.wav
  pipeline1/
    pipeline1_results.csv
    pipeline1_summary.json
    short_1.wav
  pipeline2/
    pipeline2_results.csv
    pipeline2_summary.json
    short_1.wav
  chunk/
    chunk_results.csv
    chunk_summary.json
    short_1.wav
    medium_2.wav
    long_3.wav
```

Each method subdirectory is cleared before that method runs. `outputs/buckets.csv` records the exact text inputs used for the run.
Each per-method CSV now includes total synthesis time, time-to-first-audio, throughput, GPU utilization summaries, and speedup versus serial when the serial baseline is part of the same command. Raw GPU traces are saved as `*_gpu.csv` when `nvidia-smi` is available.

## How It Works

### preprocess.py
The `preprocess.py` script reads `transcript.txt` by default, but it can also load prompt `.txt`, `.csv`, and `.json` files. Each transcript row is expected to use this format:

```text
wav_path|japanese_text|romaji_text|duration
```

Only the second field, `japanese_text`, is used as TTS input. The script categorizes phrases into three groups by character length: **short**, **medium**, and **long**. It then selects *n* evenly spaced texts from each category, or all texts when `--all` is used.

### main.py
The `main.py` script runs one or more benchmark methods on the selected texts and records total synthesis time, time-to-first-audio, output audio duration, throughput, GPU utilization summaries, and speedup versus serial when available. Results are saved under `outputs/<method>/`, with cross-method summaries in `outputs/run_summary.csv` and `outputs/run_summary.json`.

### dataset_setup.py
The `dataset_setup.py` script prepares final-evaluation assets. It exports curated short/medium/long Japanese prompts, stages the Kaggle JSSS transcript into `datasets/kaggle_jsss/`, and optionally filters a CC100 Japanese text file into `datasets/cc100_ja/cc100_ja_eval_samples.csv`.

### serial.py
The `serial.py` script implements the baseline serial pipeline. It takes a single input text, runs the TTS model end-to-end using the Irodori-TTS inference script, and measures the total synthesis time. Because the upstream serial CLI does not expose partial audio emission, `time_to_first_audio_sec` is currently recorded as the full synthesis time for this baseline.

### pipeline.py
The `pipeline.py` script uses the refactored pipeline runtime. It loads `InferenceRuntime` once for the whole pipeline benchmark run and reuses it across inputs.

There are two pipeline implementations:

```text
pipeline / pipeline1:
  one-GPU two-stage pipeline
  overlaps text -> latent   with   latent -> audio decoding

pipeline2:
  two-GPU RF-split pipeline
  runs the first half of sample_rf on cuda:0 and the second half plus decode on cuda:1
  CPU waveform decoding is not viable for this model on the tested hardware. DACVAE decode must stay on GPU; otherwise decode dominates runtime.
```

`pipeline` is currently an alias for `pipeline1` so older commands still work.

The `pipeline1` implementation splits each text input into punctuation-based segments, then runs the model in two overlapped stages:

```text
main thread:    segment 1 text -> latent      segment 2 text -> latent
decode worker:                          segment 1 latent -> audio
```

The latent is the model's internal audio representation. It is not a playable WAV file yet. The decode worker converts that latent representation into waveform audio, and the final waveform is saved as a `.wav` file.

This means `pipeline` is not running all chunks independently at the same time. That would be chunk parallelism. The pipeline version overlaps different stages of consecutive segments: while the decode worker turns one segment's latent into audio, the main thread can start generating the next segment's latent.

The `pipeline2` implementation splits the expensive RF sampling stage itself:

```text
GPU 0: segment 1 sample_rf steps 0-20      segment 2 sample_rf steps 0-20
GPU 1:                                      segment 1 sample_rf steps 20-40 + decode
```

This is more experimental and requires an allocation with at least two CUDA GPUs. It loads two full model runtimes, so it uses more GPU memory than `pipeline1`. Decode stays on `cuda:1` because CPU decode was much slower in testing.

### chunk.py
The `chunk.py` script implements chunk parallelism. It splits the input text into punctuation-based segments (reusing `text_segments.py`) and submits all segments to a `ThreadPoolExecutor` simultaneously, so multiple segments can be synthesized in parallel rather than one after another.

```text
thread 1:  segment 1 text -> latent -> audio
thread 2:  segment 2 text -> latent -> audio
thread 3:  segment 3 text -> latent -> audio
...
```

On a single CUDA GPU, PyTorch serializes kernel launches internally, so segments do not truly run at the same time on the hardware. The benefit is that CPU-side work (tokenization, tensor setup) overlaps across threads and the GPU stays fully utilized. On MPS (Apple Silicon), workers are automatically reduced to 1 since Metal does not support concurrent multi-thread GPU access.

The DACVAE decode step is serialized with a threading lock because the codec is not safe for concurrent calls — concurrent calls can return the raw 32-dim latent instead of the decoded 1-channel waveform.

After all segments finish, the audio chunks are concatenated in original text order. Boundaries can be joined with either a silence gap (`--segment-silence-ms`, default 80 ms) or a linear crossfade (`--chunk-crossfade-ms`), which fades out the tail of one segment while fading in the head of the next to reduce audible clicks.

`time_to_first_audio` is recorded as the wall-clock time from synthesis start until the earliest segment completes.

### text_segments.py
The `text_segments.py` script contains shared punctuation-based Japanese text segmentation. Both pipeline and chunk-parallel implementations should use this file so the benchmark compares scheduling strategy rather than different text boundaries.

### pipelined_runtime.py
The `pipelined_runtime.py` file is a local copy of the Irodori runtime with imports adjusted for this repo. Its inference flow has been split into callable stages such as input preparation, latent generation, latent unpatchifying, and waveform decoding. The pipeline code calls these stages directly instead of calling the original end-to-end `runtime.synthesize(...)` function for every segment.

### pipelined_rf.py
The `pipelined_rf.py` file is a local copy of the rectified flow sampler. It adds `sample_euler_rf_cfg_range`, which can run only part of the RF sampling loop. `pipeline2` uses this to run one range of RF steps on `cuda:0`, pass the intermediate latent to `cuda:1`, and finish the remaining RF steps there.

## Warning
The model takes a while so grab a coffee or watch some TV while it runs....
PLEASE DO NOT COMMIT THE REPOSITORIES (iro_tts and dacvae) OR THE DATASET (transcript.txt)
