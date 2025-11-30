#!/bin/bash
# Install script for chest X-ray anomaly detection project

TORCH_VERSION=2.7
CUDA_VERSION=cu128

# Redirect all caches to project space
CWD=$(pwd)
CACHE_BASE="${CWD}/cache"

export PIP_CACHE_DIR="${CACHE_BASE}/pip"
export UV_CACHE_DIR="${CACHE_BASE}/uv"
export XDG_CACHE_HOME="${CACHE_BASE}"
export HF_HOME="${CACHE_BASE}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_BASE}/huggingface"
export TORCH_HOME="${CACHE_BASE}/torch"
export TRANSFORMERS_CACHE="${CACHE_BASE}/transformers"
export DATASETS_CACHE="${CACHE_BASE}/datasets"
export MPLCONFIGDIR="${CACHE_BASE}/matplotlib"

# Create cache directories
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" \
         "$TORCH_HOME" "$TRANSFORMERS_CACHE" "$DATASETS_CACHE" "$MPLCONFIGDIR"

# Update uv
uv self update

set -e

# Create virtual environment
echo "Cleaning up previous installation..."
rm -rf .venv uv.lock main.py pyproject.toml

echo "Initializing project..."
uv init --name flare --python=3.11 --no-readme
uv python install

echo "Creating virtual environment..."
uv venv

# Install packages
echo "Installing PyTorch with CUDA support..."
uv pip install torch==${TORCH_VERSION} torchvision --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

echo "Installing core packages..."
uv pip install timm transformers tqdm pandas scipy scikit-learn joblib pyyaml opencv-python matplotlib

echo "Installation completed successfully!"
echo "Activate the environment with: source .venv/bin/activate"
