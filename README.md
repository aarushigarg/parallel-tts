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
## Run

Run one method:

```bash
python main.py serial
python main.py pipeline
```

Run multiple methods in one command:

```bash
python main.py serial pipeline
```

By default, this runs 1 short, 1 medium, and 1 long text from `transcript.txt`.
To run more samples from each length bucket:

```bash
python main.py serial pipeline -n 5
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

Outputs are written per method:

```text
outputs/
  buckets.csv
  serial/
    serial_results.csv
    short_1.wav
    medium_2.wav
    long_3.wav
  pipeline/
    pipeline_results.csv
    short_1.wav
    medium_2.wav
    long_3.wav
```

Each method subdirectory is cleared before that method runs. `outputs/buckets.csv` records the exact text inputs used for the run.

## How It Works

### preprocess.py
The `preprocess.py` script reads `transcript.txt`. Each transcript row is expected to use this format:

```text
wav_path|japanese_text|romaji_text|duration
```

Only the second field, `japanese_text`, is used as TTS input. The script categorizes phrases into three groups by character length: **short**, **medium**, and **long**. It then selects *n* evenly spaced texts from each category, or all texts when `--all` is used.

### main.py
The `main.py` script runs one or more benchmark methods on the selected texts and measures runtime for every input. Results are saved under `outputs/<method>/`.

### serial.py
The `serial.py` script implements the baseline serial pipeline. It takes a single input text, runs the TTS model end-to-end using the Irodori-TTS inference script, and measures the total synthesis time. This serves as the reference point for evaluating performance improvements in later parallel implementations.

### pipeline.py
The `pipeline.py` script uses the refactored pipeline runtime. It loads `InferenceRuntime` once for the whole pipeline benchmark run and reuses it across inputs.

### text_segments.py
The `text_segments.py` script contains shared punctuation-based Japanese text segmentation. Both pipeline and chunk-parallel implementations should use this file so the benchmark compares scheduling strategy rather than different text boundaries.

### pipelined_runtime.py
The `pipelined_runtime.py` file is a local copy of the Irodori runtime with imports adjusted for this repo. Its inference flow has been split into callable stages such as input preparation, latent generation, latent unpatchifying, and waveform decoding. This is the base for implementing pipeline scheduling.

## Warning
The model takes a while so grab a coffee or watch some TV while it runs....
PLEASE DO NOT COMMIT THE REPOSITORIES (iro_tts and dacvae) OR THE DATASET (transcript.txt)
