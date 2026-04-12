#!/bin/bash
set -e

PY=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/python-3.11.9-x74mtjf/bin/python3

$PY -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

if [ ! -d "iro_tts" ]; then
    git clone https://github.com/Aratako/Irodori-TTS.git iro_tts
fi

if [ ! -d "dacvae" ]; then
    git clone --depth 1 https://github.com/facebookresearch/dacvae.git
fi

pip install ./dacvae