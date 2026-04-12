# parallel-tts

## Setup 
* Ensure you have Python 3.11
* Run `bash setup_env.sh` to setup venv and install requirements

### 1. Create Virtual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```
### 2. Clone Required Repositories
```bash
git clone https://github.com/Aratako/Irodori-TTS.git iro_tts
git clone --depth 1 https://github.com/facebookresearch/dacvae.git
```
### 3. Install DACVAE
```bash
pip install ./dacvae
```
### 4. Download the dataset
We use the Japanese Single Speaker Speech Dataset:
https://www.kaggle.com/datasets/bryanpark/japanese-single-speaker-speech-dataset
#### Manual Download
1. Go to the link above
2. Click **Download**
3. Move the `.zip` file into this project folder
4. Unzip it:
```bash
unzip japanese-single-speaker-speech-dataset.zip -d data/
```
#### With Kaggle
```bash
kaggle datasets download -d bryanpark/japanese-single-speaker-speech-dataset
unzip -j japanese-single-speaker-speech-dataset.zip "*/transcript.txt"
rm japanese-single-speaker-speech-dataset.zip
```
## Run
```bash
python main.py
```
## How It Works

### preprocess.py
The `preprocess.py` script processes the Japanese text dataset by categorizing phrases into three groups based on length: **short**, **medium**, and **long**. It then randomly samples *n* phrases from each category to be used as inputs for benchmarking.

### main.py
The `main.py` script runs the TTS model on each sampled phrase and measures the runtime for every input. The results are recorded and saved as a CSV file in the `outputs/` directory for analysis.

### serial.py
The `serial.py` script implements the baseline serial pipeline. It takes a single input text, runs the TTS model end-to-end using the Irodori-TTS inference script, and measures the total synthesis time. This serves as the reference point for evaluating performance improvements in later parallel implementations.

## Warning
The model takes a while so grab a coffee or watch some TV while it runs....
