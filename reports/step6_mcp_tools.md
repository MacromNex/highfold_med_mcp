# Step 6: MCP Tools Documentation

## Server Information
- **Server Name**: highfold-med-tools
- **Version**: 1.0.0
- **Created Date**: 2025-12-31
- **Server Path**: `src/server.py`
- **Framework**: FastMCP 2.14.1

## Overview

This MCP server provides comprehensive tools for cyclic peptide computational analysis using HighFold-MeD (AlphaFold modified for cyclic peptides). The server implements both synchronous and asynchronous APIs to handle operations of varying computational complexity.

### API Design Philosophy

- **Synchronous Tools**: Operations completing in <5 minutes (validation, quick analysis)
- **Submit API Tools**: Long-running operations >10 minutes (structure prediction, MD relaxation, model training)
- **Job Management**: Complete workflow for tracking, monitoring, and retrieving results from background jobs

## Job Management Tools

| Tool | Description | Usage |
|------|-------------|-------|
| `get_job_status` | Check job progress and current status | Monitor running jobs |
| `get_job_result` | Retrieve results from completed jobs | Get final outputs |
| `get_job_log` | View execution logs from jobs | Debug and monitor progress |
| `cancel_job` | Terminate running jobs | Stop long-running tasks |
| `list_jobs` | List all jobs with optional status filter | Job queue management |

### Job Status Flow
```
submit_* → PENDING → RUNNING → COMPLETED/FAILED/CANCELLED
           ↑                     ↓
       get_job_status()    get_job_result()
```

## Synchronous Tools (Quick Operations < 5 min)

| Tool | Description | Source | Est. Runtime | Inputs | Outputs |
|------|-------------|--------|--------------|---------|---------|
| `validate_targets_file` | Validate targets.tsv format and sequences | validate logic | ~10 sec | targets_file path | Validation results, statistics |
| `get_peptide_info` | Extract info for specific peptide by index | predict_structure.py | ~5 sec | targets_file, index | Peptide details, sequence analysis |
| `check_dependencies` | Check system dependencies and setup | system check | ~5 sec | none | Dependency status, recommendations |
| `list_example_data` | List available example/demo files | file system | ~5 sec | none | Available example files |
| `create_demo_targets_file` | Generate synthetic targets for testing | data generation | ~10 sec | output_file, num_peptides | Demo file creation status |

### Example: Quick Validation
```python
# Validate a targets file before processing
result = validate_targets_file("examples/data/sequences/targets.tsv")
print(f"Valid peptides: {result['valid_sequences']}/{result['total_peptides']}")
```

## Submit Tools (Long Operations > 10 min)

| Tool | Description | Source Script | Est. Runtime | Batch Support | GPU Recommended |
|------|-------------|---------------|--------------|---------------|-----------------|
| `submit_structure_prediction` | Predict 3D structure using HighFold-MeD | `scripts/predict_structure.py` | 10-30 min | Via batch tool | No |
| `submit_structure_relaxation` | MD energy minimization with OpenMM | `scripts/relax_structure.py` | 5-15 min | No | Optional |
| `submit_model_finetuning` | Fine-tune HighFold models for cyclic peptides | `scripts/finetune_model.py` | 30+ min to hours | N/A | Yes |
| `submit_batch_prediction` | Process multiple peptides sequentially | `scripts/batch_predict.py` | N × 10+ min | Built-in | No |

### Submit Tool Parameters

#### `submit_structure_prediction`
- **input_file**: Path to targets.tsv file
- **index**: Peptide index in file (default: 0)
- **model_name**: HighFold model variant (default: "model_2_ptm")
- **alignment_dir**: Template alignment directory
- **demo_mode**: Use demo mode for testing (default: True)
- **job_name**: Optional job identifier

#### `submit_structure_relaxation`
- **input_file**: Path to PDB file
- **restraint_force**: Backbone restraint force (default: 20000.0 kJ/mol/nm²)
- **tolerance**: Energy minimization tolerance (default: 2.39 kcal/mol)
- **backbone_restraints**: Apply backbone position restraints (default: True)
- **energy_minimization**: Perform energy minimization (default: True)
- **demo_mode**: Demo mode without OpenMM (default: True)

#### `submit_model_finetuning`
- **train_file**: Training data file (TSV format)
- **validation_file**: Optional validation data file
- **model_name**: Base model to fine-tune (default: "model_2_ptm")
- **num_epochs**: Training epochs (default: 10)
- **batch_size**: Training batch size (default: 1)
- **learning_rate**: Learning rate (default: 1e-4)

#### `submit_batch_prediction`
- **input_file**: Targets file with multiple peptides
- **max_peptides**: Limit number to process (default: all)
- **model_name**: Model for all predictions
- **demo_mode**: Use demo mode for all predictions
- **continue_on_error**: Continue if some predictions fail

## Workflow Examples

### 1. Single Structure Prediction (Submit API)
```bash
# Step 1: Submit job
submit_structure_prediction("examples/data/sequences/targets.tsv", index=0)
# Returns: {"job_id": "abc12345", "status": "submitted"}

# Step 2: Monitor progress
get_job_status("abc12345")
# Returns: {"status": "running", "started_at": "...", ...}

# Step 3: View logs (optional)
get_job_log("abc12345", tail=20)
# Returns: {"log_lines": [...], "total_lines": 150}

# Step 4: Get results when complete
get_job_result("abc12345")
# Returns: {"status": "success", "result_files": {...}}
```

### 2. Quick Validation (Sync API)
```bash
# Immediate validation
validate_targets_file("my_peptides.tsv")
# Returns: {"status": "success", "valid_sequences": 8, "total_peptides": 10}

# Get specific peptide info
get_peptide_info("my_peptides.tsv", index=2)
# Returns: {"target_id": "peptide_3", "sequence": "dLPRGH", ...}
```

### 3. Batch Processing
```bash
# Submit batch job for multiple peptides
submit_batch_prediction("large_peptide_set.tsv", max_peptides=50)
# Returns: {"job_id": "batch_xyz", "status": "submitted"}

# Monitor batch progress
get_job_status("batch_xyz")
# Check periodically until completed

# Get all results
get_job_result("batch_xyz")
# Returns comprehensive results for all processed peptides
```

### 4. Model Fine-tuning Pipeline
```bash
# Step 1: Validate training data
validate_targets_file("training_set.tsv")

# Step 2: Submit fine-tuning (long-running)
submit_model_finetuning(
    train_file="training_set.tsv",
    validation_file="validation_set.tsv",
    num_epochs=20
)

# Step 3: Monitor training progress
get_job_log("train_job_id", tail=10)  # See latest training output

# Step 4: Get trained model when complete
get_job_result("train_job_id")
```

## Error Handling

All tools return structured error responses:
```json
{
  "status": "error",
  "error": "Descriptive error message",
  "details": {...}  // Optional additional context
}
```

Common error scenarios:
- **File not found**: Input files don't exist
- **Invalid format**: Targets file format issues
- **Sequence validation**: Invalid cyclic peptide sequences
- **Dependency missing**: Required packages not installed
- **Job not found**: Invalid job ID provided
- **Job not completed**: Trying to get results from running job

## File Formats

### Input: Targets TSV Format
```tsv
targetid	target_chainseq	templates_alignfile	description
peptide_1	ACDEFGHIK	alignments/peptide_1.tsv	Cyclic peptide 1
peptide_2	dLPRGH_	alignments/peptide_2.tsv	D-amino acid peptide
```

**Required columns:**
- `target_chainseq`: Peptide sequence in HighFold notation
- `targetid`: Unique identifier (optional, auto-generated if missing)

**Sequence notation:**
- Standard amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
- D-amino acids: dA, dL, dP, etc.
- Modifications: _ (N-methylation), h (special residues), . (terminals)
- Cyclization: / (cyclization point, removed during processing)

### Output Files

**Structure prediction outputs:**
- `.txt`: Prediction summary and statistics
- `.pdb`: 3D structure coordinates (when available)
- `job.log`: Execution log

**Relaxation outputs:**
- `_relaxed.pdb`: Energy-minimized structure
- Energy change statistics in result

**Fine-tuning outputs:**
- `training_results.txt`: Training history and configuration
- `finetuned_model_demo.pkl`: Model checkpoint (demo format)

## System Requirements

### Core Dependencies (Required)
- Python 3.8+
- pandas
- numpy
- pathlib

### Optional Dependencies (Full functionality)
- **OpenMM**: For structure relaxation
- **JAX + Haiku**: For model fine-tuning
- **PyTorch/TensorFlow**: Alternative ML frameworks

### Example Data
- `examples/data/sequences/targets.tsv`: Demo peptide sequences
- `examples/data/alignments/`: Template alignment files
- `examples/data/structures/`: Example PDB structures

## Installation and Setup

### 1. Environment Setup
```bash
# Prefer mamba over conda
mamba create -n highfold python=3.9
mamba activate highfold

# Install MCP dependencies
pip install fastmcp loguru

# Optional: Install OpenMM for structure relaxation
mamba install -c conda-forge openmm

# Optional: Install ML dependencies for fine-tuning
pip install jax jaxlib dm-haiku optax
```

### 2. Start MCP Server
```bash
cd src
python server.py
# Or use FastMCP development mode:
fastmcp dev server.py
```

### 3. Verify Installation
```python
# Test dependency check
check_dependencies()

# List available examples
list_example_data()

# Create demo data for testing
create_demo_targets_file("test_peptides.tsv", num_peptides=5)
```

## Demo Mode vs Full Functionality

### Demo Mode (Default)
- **Always available**: No special dependencies
- **Fast execution**: Simulates complex operations
- **Educational**: Shows expected workflow and outputs
- **File processing**: Validates inputs and creates realistic outputs
- **Ideal for**: Testing, development, learning the API

### Full Functionality
- **Dependency requirements**: OpenMM, JAX/Haiku, ML frameworks
- **Real computation**: Actual structure prediction and optimization
- **Long runtime**: Minutes to hours for real tasks
- **Production ready**: Generates research-quality results
- **GPU recommended**: For model training and large-scale prediction

## Performance Guidelines

### Job Sizing Recommendations
- **Single prediction**: 1 peptide, ~10 minutes
- **Small batch**: 5-10 peptides, ~1 hour
- **Medium batch**: 50-100 peptides, ~8 hours
- **Large batch**: 500+ peptides, consider splitting into smaller jobs

### Resource Planning
- **Memory**: 4-8 GB for single predictions, 16+ GB for large batches
- **CPU**: Multi-core beneficial for batch processing
- **GPU**: Recommended for model fine-tuning, optional for prediction
- **Storage**: ~100 MB per peptide (including intermediate files)

## Integration Examples

### Using with Claude Code
```python
# Quick validation before processing
result = validate_targets_file("my_peptides.tsv")
if result['status'] == 'error':
    print(f"File issues: {result['error']}")

# Submit structure prediction
job = submit_structure_prediction("my_peptides.tsv", index=0, demo_mode=False)
print(f"Job submitted: {job['job_id']}")

# Poll for completion
import time
while True:
    status = get_job_status(job['job_id'])
    if status['status'] in ['completed', 'failed']:
        break
    time.sleep(30)

# Get results
if status['status'] == 'completed':
    results = get_job_result(job['job_id'])
    print("Structure prediction completed!")
```

### Batch Processing Pipeline
```python
# 1. Validate input
validation = validate_targets_file("large_dataset.tsv")
print(f"Valid peptides: {validation['valid_sequences']}")

# 2. Submit batch job
batch_job = submit_batch_prediction(
    "large_dataset.tsv",
    max_peptides=100,
    continue_on_error=True
)

# 3. Monitor progress
job_id = batch_job['job_id']
while get_job_status(job_id)['status'] == 'running':
    # Check logs periodically
    logs = get_job_log(job_id, tail=5)
    print("Latest output:", logs['log_lines'][-1] if logs['log_lines'] else "No new output")
    time.sleep(60)

# 4. Process results
final_results = get_job_result(job_id)
print(f"Batch completed: {len(final_results['result_files'])} output files")
```

## Troubleshooting

### Common Issues

**Job stuck in pending/running:**
```python
# Check job logs for issues
logs = get_job_log("job_id", tail=0)  # Get all logs
print('\n'.join(logs['log_lines']))

# Cancel if needed
cancel_job("job_id")
```

**File format errors:**
```python
# Validate before processing
result = validate_targets_file("my_file.tsv")
if result['invalid_sequences'] > 0:
    print("Invalid sequences found:")
    for seq in result['invalid_details']:
        print(f"  {seq['target_id']}: {seq['sequence']}")
```

**Dependency issues:**
```python
# Check what's available
deps = check_dependencies()
if deps['status'] != 'ready':
    print("Missing dependencies:")
    for rec in deps['dependencies']['recommendations']:
        print(f"  - {rec}")
```

### Performance Issues
- **Long job queues**: Jobs run sequentially; split large batches
- **Memory errors**: Reduce batch size or enable demo mode
- **Slow processing**: Check that demo_mode=False for real computation

## Changelog

### v1.0.0 (2025-12-31)
- Initial MCP server implementation
- Complete job management system
- All four HighFold-MeD scripts wrapped as MCP tools
- Demo mode for testing without dependencies
- Comprehensive error handling and validation
- Batch processing support
- Sync/async API design based on computational complexity