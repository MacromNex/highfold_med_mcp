# HighFold-MeD MCP

> MCP tools for cyclic peptide computational analysis using HighFold-MeD (AlphaFold modified for cyclic peptides with N-methylation and D-amino acids)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The HighFold-MeD MCP provides comprehensive tools for cyclic peptide computational analysis through a Model Context Protocol (MCP) server. This enables AI assistants like Claude Code to perform advanced molecular modeling tasks including structure prediction, energy minimization, and model fine-tuning for cyclic peptides with modified amino acids.

### Key Features
- **3D Structure Prediction**: Template-based AlphaFold inference for cyclic peptides using HighFold-MeD
- **Structure Relaxation**: Energy minimization using OpenMM with backbone restraints
- **Batch Processing**: High-throughput processing of multiple cyclic peptides
- **Model Fine-tuning**: Custom AlphaFold model adaptation for cyclic peptide datasets
- **Sequence Validation**: Parse and validate HighFold notation for modified amino acids
- **Job Management**: Complete async workflow for long-running computations

### Unique Cyclic Peptide Support
- **D-amino acids**: dL, dP, dA notation support
- **N-methylation**: h (methylated histidine), p (methylated proline)
- **Terminal modifications**: . notation for C-terminal modifications
- **Cyclization handling**: / notation for cyclization points
- **Template alignments**: Pre-computed template libraries for structure prediction

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Conda environment (Python 3.10)
├── env_py38/               # Legacy environment for HighFold-MeD
├── src/
│   └── server.py           # MCP server (14 tools)
├── scripts/
│   ├── predict_structure.py   # Single peptide structure prediction
│   ├── batch_predict.py       # Batch processing multiple peptides
│   ├── relax_structure.py     # OpenMM structure relaxation
│   ├── finetune_model.py      # Model fine-tuning (demo mode)
│   └── lib/                   # Shared utilities (32 functions)
│       ├── cyclic_peptides.py # Peptide-specific utilities
│       └── io_utils.py        # I/O and configuration utilities
├── examples/
│   └── data/               # Demo data for testing
│       ├── sequences/      # targets.tsv with cyclic peptide sequences
│       ├── alignments/     # Template alignment files (50+ templates)
│       └── structures/     # Sample PDB structures for relaxation
├── configs/                # JSON configuration files
│   ├── predict_structure_config.json
│   ├── batch_predict_config.json
│   ├── relax_structure_config.json
│   ├── finetune_model_config.json
│   └── default_config.json # Global default settings
└── repo/                   # Original HighFold-MeD repository
```

---

## Installation

### Quick Setup

Run the automated setup script:

```bash
./quick_setup.sh
```

This will create the environment and install all dependencies automatically.

### Manual Setup (Advanced)

For manual installation or customization, follow these steps.

#### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+ (for MCP server)
- Python 3.8 (for HighFold-MeD legacy dependencies, if using full functionality)
- RDKit (installed automatically from conda-forge)
- CUDA toolkit (optional, for GPU acceleration in full mode)

#### Create Environment

**Note**: This project uses a dual environment strategy - Python 3.10 for the MCP server and Python 3.8 for HighFold-MeD legacy dependencies.

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/highfold_med_mcp

# Check if mamba is available (prefer over conda)
if command -v mamba &> /dev/null; then
    PKG_MGR="mamba"
else
    PKG_MGR="conda"
fi
echo "Using package manager: $PKG_MGR"

# Create main MCP environment (Python 3.10)
$PKG_MGR create -p ./env python=3.10 pip -y

# Activate environment
$PKG_MGR activate ./env

# Install core dependencies
$PKG_MGR run -p ./env pip install loguru click pandas numpy tqdm

# Install FastMCP for MCP server
$PKG_MGR run -p ./env pip install --force-reinstall --no-cache-dir fastmcp

# Install RDKit from conda-forge (essential for molecular operations)
$PKG_MGR run -p ./env $PKG_MGR install -c conda-forge rdkit -y

# Verify installation
$PKG_MGR run -p ./env python -c "import pandas; import numpy; import loguru; import fastmcp; from rdkit import Chem; print('Installation successful!')"
```

### Optional: Legacy Environment for Full HighFold Functionality

```bash
# Create Python 3.8 environment for HighFold-MeD dependencies
$PKG_MGR create -p ./env_py38 python=3.8 -y

# Note: Full ML dependencies (JAX, PyTorch, TensorFlow, OpenMM)
# would be installed here for production use
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Runtime | Example |
|--------|-------------|---------|---------|
| `scripts/predict_structure.py` | Predict 3D structure from HighFold notation | ~10 min | See below |
| `scripts/batch_predict.py` | Batch processing multiple peptides | ~N × 10 min | See below |
| `scripts/relax_structure.py` | OpenMM energy minimization | ~5-15 min | See below |
| `scripts/finetune_model.py` | Model fine-tuning (demo mode) | ~30+ min | See below |

### Script Examples

#### Predict 3D Structure

```bash
# Activate environment
mamba activate ./env

# Single peptide prediction
python scripts/predict_structure.py \
  --input examples/data/sequences/targets.tsv \
  --index 0 \
  --output results/prediction.txt \
  --demo_mode

# With specific config file
python scripts/predict_structure.py \
  --input examples/data/sequences/targets.tsv \
  --index 0 \
  --config configs/predict_structure_config.json
```

**Parameters:**
- `--input, -i`: Target file path (TSV format with peptide sequences)
- `--index`: Row index to process (default: 0)
- `--output, -o`: Output file path (default: auto-generated)
- `--demo_mode`: Use demo mode for testing (default: True)
- `--config`: JSON configuration file path

#### Batch Processing

```bash
# Process multiple peptides
python scripts/batch_predict.py \
  --input examples/data/sequences/targets.tsv \
  --max_peptides 5 \
  --output_dir results/batch_predictions/ \
  --demo_mode

# Continue processing even if some peptides fail
python scripts/batch_predict.py \
  --input examples/data/sequences/targets.tsv \
  --max_peptides 10 \
  --continue_on_error
```

#### Structure Relaxation

```bash
# Relax a single PDB structure
python scripts/relax_structure.py \
  --input examples/data/structures/1.pdb \
  --output results/1_relaxed.pdb \
  --demo

# Real OpenMM relaxation (requires OpenMM installation)
python scripts/relax_structure.py \
  --input examples/data/structures/1.pdb \
  --restraint_force 20000.0 \
  --tolerance 2.39
```

#### Model Fine-tuning

```bash
# Demo mode training
python scripts/finetune_model.py \
  --demo \
  --epochs 10 \
  --output_dir results/finetuning/

# Real training (requires JAX/Haiku)
python scripts/finetune_model.py \
  --train_file training_set.tsv \
  --validation_file validation_set.tsv \
  --epochs 20 \
  --batch_size 4
```

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
fastmcp install src/server.py --name highfold-med-tools
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code using absolute paths
claude mcp add highfold-med-tools -- \
  $(pwd)/env/bin/python \
  $(pwd)/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "highfold-med-tools": {
      "command": "/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/highfold_med_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/highfold_med_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What tools are available from highfold-med-tools?
```

#### Environment Verification
```
Use check_dependencies to verify the HighFold-MeD environment setup
```

#### File Validation
```
Use validate_targets_file to check @examples/data/sequences/targets.tsv
```

#### Structure Prediction (Submit API)
```
Submit a 3D structure prediction job for the first peptide in @examples/data/sequences/targets.tsv with demo_mode=True
```

#### Job Status Monitoring
```
Check the status of job abc12345 and show me the latest logs
```

#### Batch Processing
```
Submit a batch prediction job for the first 3 peptides in @examples/data/sequences/targets.tsv
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description | Use Case |
|-----------|-------------|----------|
| `@examples/data/sequences/targets.tsv` | Sample peptide sequences | Structure prediction input |
| `@examples/data/structures/1.pdb` | Sample PDB structure | Structure relaxation input |
| `@configs/predict_structure_config.json` | Prediction config | Custom parameter settings |
| `@results/` | Output directory | View generated results |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "highfold-med-tools": {
      "command": "/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/highfold_med_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/highfold_med_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What tools are available from highfold-med-tools?
> Use validate_targets_file to check the demo data
> Submit structure prediction with demo_mode=True
```

---

## Available Tools

The MCP server provides 14 tools organized into 4 categories:

### Job Management Tools (5 tools)
These tools manage long-running computational jobs:

| Tool | Description | Usage |
|------|-------------|-------|
| `get_job_status` | Check job progress and current status | Monitor running jobs |
| `get_job_result` | Retrieve results from completed jobs | Get final outputs |
| `get_job_log` | View execution logs (default: last 50 lines) | Debug and monitor progress |
| `cancel_job` | Terminate running jobs | Stop long-running tasks |
| `list_jobs` | List all jobs with optional status filter | Job queue management |

### Submit API Tools (4 tools)
Long-running operations that return job_id for tracking:

| Tool | Description | Runtime | GPU Recommended |
|------|-------------|---------|-----------------|
| `submit_structure_prediction` | Predict 3D structure using HighFold-MeD | 10-30 min | No |
| `submit_structure_relaxation` | MD energy minimization with OpenMM | 5-15 min | Optional |
| `submit_model_finetuning` | Fine-tune HighFold models | 30+ min to hours | Yes |
| `submit_batch_prediction` | Process multiple peptides sequentially | N × 10+ min | No |

### Synchronous Tools (4 tools)
Quick operations completing in seconds to minutes:

| Tool | Description | Runtime | Use Case |
|------|-------------|---------|----------|
| `validate_targets_file` | Validate targets.tsv format and sequences | ~10 sec | Pre-processing validation |
| `get_peptide_info` | Extract peptide info by index | ~5 sec | Data exploration |
| `check_dependencies` | Verify environment setup | ~5 sec | System diagnostics |
| `list_example_data` | List available demo files | ~5 sec | Data discovery |

### Utility Tools (1 tool)

| Tool | Description | Runtime |
|------|-------------|---------|
| `create_demo_targets_file` | Generate synthetic targets for testing | ~10 sec |

---

## Examples

### Example 1: Quick Validation and Analysis

**Goal:** Validate and analyze a cyclic peptide dataset before processing

**Using Scripts:**
```bash
# Check file format
python scripts/predict_structure.py --input examples/data/sequences/targets.tsv --help

# Validate specific peptide
python scripts/predict_structure.py \
  --input examples/data/sequences/targets.tsv \
  --index 0 \
  --demo_mode
```

**Using MCP (in Claude Code):**
```
First, use validate_targets_file to check @examples/data/sequences/targets.tsv

Then, use get_peptide_info with index 0 to see details about the first peptide

Finally, use check_dependencies to verify the environment is ready
```

**Expected Output:**
- File validation results with sequence statistics
- Peptide details including HighFold notation analysis
- Environment status with dependency availability

### Example 2: Single Structure Prediction Workflow

**Goal:** Predict 3D structure for a cyclic peptide with D-amino acids

**Using Scripts:**
```bash
# Direct script execution
python scripts/predict_structure.py \
  --input examples/data/sequences/targets.tsv \
  --index 0 \
  --output results/PhdLP_d_prediction.txt \
  --demo_mode
```

**Using MCP (in Claude Code):**
```
Submit a structure prediction job for index 0 in @examples/data/sequences/targets.tsv with demo_mode=True and job_name="PhdLP_d_prediction"

Monitor the job status and show me the logs

When complete, get the results and explain the prediction quality
```

**Expected Output:**
- Job ID for tracking: `"job_12345"`
- Prediction summary with confidence scores
- Template alignment information
- 3D structure coordinates (if available)

### Example 3: Batch Virtual Screening

**Goal:** Process multiple cyclic peptides for drug discovery

**Using MCP (in Claude Code):**
```
I want to screen the first 5 cyclic peptides in @examples/data/sequences/targets.tsv for drug-like properties.

First, validate the targets file to see what peptides we have

Then submit a batch prediction job with max_peptides=5 and demo_mode=True

Monitor the progress and show me the results when complete, including:
- Which peptides completed successfully
- Any confidence score patterns
- Overall batch statistics
```

**Expected Workflow:**
1. File validation shows 5+ valid peptides with HighFold notation
2. Batch job submitted with tracking ID
3. Progress monitoring through logs
4. Final results with individual prediction files
5. Summary statistics and quality metrics

### Example 4: Structure Relaxation Pipeline

**Goal:** Energy minimize a predicted structure

**Using Scripts:**
```bash
# Relax structure with OpenMM
python scripts/relax_structure.py \
  --input examples/data/structures/1.pdb \
  --output results/1_relaxed.pdb \
  --restraint_force 20000.0 \
  --demo
```

**Using MCP (in Claude Code):**
```
Submit a structure relaxation job for @examples/data/structures/1.pdb with:
- restraint_force: 20000.0
- tolerance: 2.39
- demo_mode: True
- job_name: "peptide_relaxation"

Show me the energy changes when complete
```

### Example 5: Model Fine-tuning for Custom Dataset

**Goal:** Adapt HighFold for a specific cyclic peptide family

**Using MCP (in Claude Code):**
```
I want to fine-tune a model for my custom cyclic peptide dataset.

First, create a demo targets file with 10 peptides using create_demo_targets_file

Then submit a model fine-tuning job with:
- num_epochs: 5
- batch_size: 2
- learning_rate: 0.0001
- demo_mode: True

Monitor the training progress and show me the final model performance
```

---

## Demo Data

The `examples/data/` directory contains comprehensive sample data:

### Sequence Data
| File | Description | Format | Use With |
|------|-------------|--------|----------|
| `sequences/targets.tsv` | Sample cyclic peptides with HighFold notation | TSV | All prediction tools |

**Sample peptides from targets.tsv:**
- `PhdLP_d` (D7.6): Proline, histidine, D-leucine, leucine, proline, D-alanine
- `VIhFIh.` (D7.8): Valine, isoleucine, methylated-histidine, phenylalanine, isoleucine, methylated-histidine
- `dLhdL.PL` (D8.1): D-leucine, methylated-histidine, D-leucine, proline, leucine

### Alignment Data
| Directory | Description | Count | Use With |
|-----------|-------------|-------|----------|
| `alignments/` | Template alignment files for structure prediction | 50+ files | `submit_structure_prediction` |

### Structure Data
| File | Description | Format | Use With |
|------|-------------|--------|----------|
| `structures/1.pdb` | Sample cyclic peptide structure (11 residues, 192 atoms) | PDB | `submit_structure_relaxation` |
| `structures/1_relaxed.pdb` | Energy-minimized structure | PDB | Comparison reference |
| `structures/2.pdb` | Additional sample structure | PDB | Testing |
| `structures/PML.pdb` | Pro-Met-Leu cyclic peptide | PDB | Small molecule example |

---

## Configuration Files

The `configs/` directory contains JSON configuration templates:

### Configuration Structure
| Config File | Purpose | Key Sections |
|-------------|---------|--------------|
| `predict_structure_config.json` | Structure prediction parameters | model, processing, output, cyclic_peptide |
| `batch_predict_config.json` | Batch processing settings | processing, output, model, error_handling |
| `relax_structure_config.json` | Molecular dynamics parameters | openmm, minimization, restraints, simulation |
| `finetune_model_config.json` | Model training configuration | model, training, data, optimization, losses |
| `default_config.json` | Global default settings | global, data_paths, cyclic_peptides, models, output |

### Example Configuration

```json
{
  "model": {
    "name": "model_2_ptm",
    "use_templates": true,
    "max_template_date": "2022-01-01"
  },
  "processing": {
    "demo_mode": true,
    "validate_sequences": true,
    "use_gpu": false
  },
  "cyclic_peptide": {
    "handle_d_amino_acids": true,
    "handle_n_methylation": true,
    "cyclization_method": "auto"
  },
  "output": {
    "save_confidence_scores": true,
    "save_template_info": true,
    "output_format": "pdb"
  }
}
```

---

## Cyclic Peptide Notation

HighFold-MeD uses specialized notation for modified amino acids:

### Notation Elements
| Notation | Description | Example |
|----------|-------------|---------|
| **Standard AA** | A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y | `GRGDSP` |
| **D-amino acids** | d prefix (e.g., dL, dP, dA) | `dLPRGH` |
| **N-methylation** | h (methylated histidine), p (methylated proline) | `VIhFIh` |
| **Terminal mods** | . notation for C-terminal modifications | `VIhFIh.` |
| **Cyclization** | / notation (removed during processing) | `G/RGDSP` |

### Real Dataset Examples
From `examples/data/sequences/targets.tsv`:

| Sequence | Target ID | Description |
|----------|-----------|-------------|
| `PhdLP_d` | D7.6 | Mixed D/L amino acids with histidine |
| `VIhFIh.` | D7.8 | N-methylated histidines with C-terminal mod |
| `dLhdL.PL` | D8.1 | D-leucines with methylated histidine |
| `gdhPLOPL` | D8.5 | Complex sequence with unusual residues |

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Check environment exists
ls -la env/

# Recreate if needed
mamba create -p ./env python=3.10 -y
mamba activate ./env
pip install loguru click pandas numpy tqdm fastmcp
mamba install -c conda-forge rdkit -y
```

**Problem:** RDKit import errors
```bash
# Ensure RDKit from conda-forge
mamba activate ./env
mamba install -c conda-forge rdkit -y

# Test import
python -c "from rdkit import Chem; print('RDKit working:', Chem.__version__)"
```

**Problem:** FastMCP installation issues
```bash
# Force reinstall FastMCP
pip install --force-reinstall --no-cache-dir fastmcp

# Verify installation
python -c "import fastmcp; print('FastMCP version:', fastmcp.__version__)"
```

### MCP Server Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Remove and re-add if needed
claude mcp remove highfold-med-tools
claude mcp add highfold-med-tools -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify connection
claude mcp list
# Should show: ✓ Connected
```

**Problem:** Tools not working
```bash
# Test server directly
python -c "
import sys; sys.path.append('src')
from server import mcp
tools = [name for name in dir(mcp) if not name.startswith('_')]
print('Available tools:', len([t for t in tools if hasattr(getattr(mcp, t), '__call__')]))
"
```

**Problem:** Import errors in server
```bash
# Check all imports work
python -c "
import sys; sys.path.append('src')
from jobs.manager import job_manager
import pandas, numpy, loguru
print('All imports successful')
"
```

### File Format Issues

**Problem:** Invalid targets.tsv format
```
Use validate_targets_file to check your file format. Required columns:
- target_chainseq: Peptide sequence in HighFold notation (required)
- targetid: Unique identifier (optional, auto-generated if missing)
```

**Problem:** Invalid HighFold sequence notation
```
Ensure sequences use proper notation:
✓ Correct: "PhdLP_d", "VIhFIh.", "dLhdL.PL"
✗ Incorrect: "PHDLP_D", "vihfih.", "DLHDL.PL"

- Use lowercase 'd' prefix for D-amino acids
- Use lowercase 'h' for methylated histidine
- Standard amino acids in uppercase
```

**Problem:** Missing alignment files
```bash
# Check alignment directory
ls examples/data/alignments/

# Alignments are referenced in targets.tsv but paths may be absolute
# The MCP tools automatically resolve relative paths from examples/data/alignments/
```

### Job Management Issues

**Problem:** Job stuck in pending/running
```
Use get_job_log with job_id "<job_id>" and tail=0 to see all logs
Check for error messages in the log output
Use cancel_job if needed to terminate stuck jobs
```

**Problem:** Job failed with error
```
Use get_job_log to see error details:
get_job_log("<job_id>", tail=100)

Common issues:
- File not found: Check input file paths
- Sequence validation: Use validate_targets_file first
- Dependency missing: Use check_dependencies
```

**Problem:** Demo mode vs real computation
```
All tools default to demo_mode=True for testing
Set demo_mode=False for real computation (requires full dependencies)

Demo mode provides:
- Fast execution (~seconds)
- Realistic output format
- No heavy dependencies required
- Educational value for learning the workflow
```

### Performance Issues

**Problem:** Long job queues
```
Jobs run sequentially to prevent resource conflicts
Split large batches into smaller jobs:
- Single peptides: ~10 minutes each
- Small batch: 5-10 peptides maximum
- Large datasets: submit multiple smaller batches
```

**Problem:** Memory errors
```bash
# Check available memory
free -h

# Reduce batch size or enable demo mode
# Large peptides or batches may require 8-16GB RAM
```

**Problem:** Slow processing
```
Ensure you're using the intended mode:
- demo_mode=True: Fast simulation for testing
- demo_mode=False: Real computation (requires dependencies)

Check system resources:
- CPU: Multi-core beneficial for batch processing
- Memory: 4-8GB for single predictions, 16GB+ for large batches
- GPU: Recommended for model fine-tuning
```

---

## Development and Testing

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Run individual script tests
python scripts/predict_structure.py --input examples/data/sequences/targets.tsv --index 0 --demo_mode
python scripts/batch_predict.py --input examples/data/sequences/targets.tsv --max_peptides 2 --demo_mode
python scripts/relax_structure.py --input examples/data/structures/1.pdb --demo
python scripts/finetune_model.py --demo --epochs 3

# Test MCP server
fastmcp dev src/server.py
```

### Starting Development Server

```bash
# Run MCP server in development mode
fastmcp dev src/server.py
# Access MCP Inspector at: http://localhost:6274
```

### Manual Validation Prompts

For final validation through Claude Code interface:

```
# Quick verification (5 tests, ~5 minutes)
1. "What MCP tools are available from highfold-med-tools?"
2. "Use check_dependencies to verify environment setup"
3. "Use list_example_data to show available files"
4. "Use validate_targets_file to check examples/data/sequences/targets.tsv"
5. "Use list_jobs to show current job queue"

# Comprehensive testing (additional 5 tests)
6. "Create a demo targets file with create_demo_targets_file"
7. "Submit structure prediction with demo_mode=True"
8. "Check the status of the submitted job"
9. "Get logs for the job"
10. "Test error handling with invalid file path"
```

---

## License

This project extends the original [HighFold-MeD](https://github.com/yourusername/HighFold-MeD) repository with MCP integration for AI assistant compatibility.

## Credits

- **HighFold-MeD**: Original AlphaFold modifications for cyclic peptides
- **FastMCP**: Model Context Protocol framework
- **RDKit**: Molecular manipulation and validation
- **OpenMM**: Molecular dynamics for structure relaxation
- **JAX/Haiku**: Deep learning frameworks for model fine-tuning

---

## Quick Reference

### Essential Commands

```bash
# Environment setup
mamba activate ./env

# Register MCP server
claude mcp add highfold-med-tools -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Test installation
claude mcp list

# Development server
fastmcp dev src/server.py
```

### Key File Paths
- MCP Server: `src/server.py`
- Demo data: `examples/data/sequences/targets.tsv`
- Configuration: `configs/predict_structure_config.json`
- Scripts: `scripts/predict_structure.py`

### Important URLs
- MCP Inspector: http://localhost:6274 (when running `fastmcp dev`)
- FastMCP Docs: https://fastmcp.com/docs
- HighFold-MeD Paper: [Add paper link if available]

This README provides comprehensive documentation for using the HighFold-MeD MCP tools for cyclic peptide computational analysis. For additional help, refer to the individual script documentation in the `scripts/` directory or use the `--help` flag with any script.