#!/bin/bash
# Quick Setup Script for HighFold-MeD MCP
# HighFold-MeD: Fine-tuned AlphaFold for cyclic peptide structure prediction
# Supports training and prediction of cyclic peptide structures
# Source: https://github.com/hongliangduan/HighFold-MeD

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up HighFold-MeD MCP ==="

# Step 1: Create Python environment
echo "[1/4] Creating Python 3.10 environment..."
(command -v mamba >/dev/null 2>&1 && mamba create -p ./env python=3.10 pip -y) || \
(command -v conda >/dev/null 2>&1 && conda create -p ./env python=3.10 pip -y) || \
(echo "Warning: Neither mamba nor conda found, creating venv instead" && python3 -m venv ./env)

# Step 2: Install core dependencies
echo "[2/4] Installing core dependencies..."
./env/bin/pip install loguru click pandas numpy tqdm

# Step 3: Install fastmcp
echo "[3/4] Installing fastmcp..."
./env/bin/pip install --force-reinstall --no-cache-dir fastmcp

# Step 4: Install RDKit
echo "[4/4] Installing RDKit..."
(command -v mamba >/dev/null 2>&1 && mamba run -p ./env mamba install -c conda-forge rdkit -y) || \
(command -v conda >/dev/null 2>&1 && conda run -p ./env conda install -c conda-forge rdkit -y) || \
./env/bin/pip install rdkit

echo ""
echo "=== HighFold-MeD MCP Setup Complete ==="
echo "To run the MCP server: ./env/bin/python src/server.py"
