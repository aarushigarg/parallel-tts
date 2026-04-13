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
```bash
python main.py serial
python main.py pipeline
```
## How It Works

### preprocess.py
The `preprocess.py` script processes the Japanese text dataset by categorizing phrases into three groups based on length: **short**, **medium**, and **long**. It then randomly samples *n* phrases from each category to be used as inputs for benchmarking.

### main.py
The `main.py` script runs the TTS model on each sampled phrase and measures the runtime for every input. The results are recorded and saved as a CSV file in the `outputs/` directory for analysis.

### serial.py
The `serial.py` script implements the baseline serial pipeline. It takes a single input text, runs the TTS model end-to-end using the Irodori-TTS inference script, and measures the total synthesis time. This serves as the reference point for evaluating performance improvements in later parallel implementations.

### pipeline.py
The `pipeline.py` script uses a modified inference script that will use pipelining for efficiency.

## Warning
The model takes a while so grab a coffee or watch some TV while it runs....
