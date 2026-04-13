#!/bin/bash
set -e

if [ -f .env.local ]; then
    source .env.local
fi

# On macOS, llvmlite (via numba) may build from source and needs LLVM 20.
# We do not auto-install brew dependencies; instead we guide users.
if [ "$(uname -s)" = "Darwin" ]; then
    if [ -z "${LLVM_CONFIG:-}" ]; then
        if [ -x /opt/homebrew/opt/llvm@20/bin/llvm-config ]; then
            LLVM_CONFIG=/opt/homebrew/opt/llvm@20/bin/llvm-config
        elif [ -x /usr/local/opt/llvm@20/bin/llvm-config ]; then
            LLVM_CONFIG=/usr/local/opt/llvm@20/bin/llvm-config
        else
            echo "macOS detected. If llvmlite builds from source, install LLVM 20:" >&2
            echo "  brew install llvm@20" >&2
            echo "Then rerun with LLVM_CONFIG set to llvm@20's llvm-config." >&2
        fi
    fi
    if [ -n "${LLVM_CONFIG:-}" ]; then
        export LLVM_CONFIG
        LLVM_PREFIX="$(dirname "$(dirname "$LLVM_CONFIG")")"
        LLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm"
        export LLVM_DIR
        if [ -d "$LLVM_PREFIX" ]; then
            if [ -n "${CMAKE_PREFIX_PATH:-}" ]; then
                CMAKE_PREFIX_PATH="$LLVM_PREFIX:$CMAKE_PREFIX_PATH"
            else
                CMAKE_PREFIX_PATH="$LLVM_PREFIX"
            fi
            export CMAKE_PREFIX_PATH
        fi
    fi
fi

PY=${PYTHON_BIN:-/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/python-3.11.9-x74mtjf/bin/python3}

if ! command -v "$PY" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PY" >&2
    echo "Set PYTHON_BIN to a Python 3.11 executable, then rerun setup_env.sh." >&2
    exit 1
fi

if ! "$PY" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'; then
    echo "Python 3.11 is required. Found: $("$PY" --version)" >&2
    exit 1
fi

"$PY" -m venv .venv
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
