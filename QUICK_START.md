# HighFold-MeD MCP Server - Quick Start Guide

## Installation & Setup (5 minutes)

```bash
# 1. Install MCP server in Claude Code
claude mcp add highfold-med-tools -- \
  $(pwd)/env/bin/python \
  $(pwd)/src/server.py

# 2. Verify connection
claude mcp list
# Should show: highfold-med-tools: ... - ✓ Connected
```

## Quick Test (2 minutes)

Open Claude Code and try these prompts:

### 1. Discover Available Tools
```
"What tools are available from highfold-med-tools? List them with descriptions."
```

### 2. Environment Check
```
"Use check_dependencies to verify the HighFold-MeD environment setup."
```

### 3. List Example Data
```
"Use list_example_data to show what example files are available for testing."
```

## Common Workflows

### Basic File Validation
```
"Use validate_targets_file to check the format of 'examples/data/sequences/targets.tsv'."
```

### Submit a Demo Job
```
"Submit a structure prediction using submit_structure_prediction with:
- input_file: 'examples/data/sequences/targets.tsv'
- index: 0
- demo_mode: True
- job_name: 'my_test_job'"
```

### Monitor Jobs
```
"Use list_jobs to show all submitted jobs and their status."
```

### Get Job Details
```
"Use get_job_status with job_id 'your_job_id' to check progress."
"Use get_job_log with job_id 'your_job_id' to see execution logs."
```

## Available Tool Categories

### 🔧 **Job Management (5 tools)**
- `get_job_status` - Check job progress
- `get_job_result` - Get completed job results
- `get_job_log` - View execution logs
- `cancel_job` - Stop running jobs
- `list_jobs` - List all jobs

### 🚀 **Submit API (4 tools)**
- `submit_structure_prediction` - 3D structure prediction
- `submit_structure_relaxation` - OpenMM energy minimization
- `submit_model_finetuning` - Model training
- `submit_batch_prediction` - Process multiple peptides

### ⚡ **Sync Tools (4 tools)**
- `validate_targets_file` - Check file format
- `get_peptide_info` - Examine specific peptides
- `check_dependencies` - Environment verification
- `list_example_data` - Browse example files

### 🛠️ **Utilities (1 tool)**
- `create_demo_targets_file` - Generate test data

## Tips

- **Always use `demo_mode=True`** for testing to prevent long computations
- **Job IDs are returned** by submit tools for tracking
- **Example data** is available in `examples/data/sequences/targets.tsv`
- **Error messages** are informative - read them for troubleshooting
- **File paths** can be relative or absolute

## Troubleshooting

### Server Not Connected
```bash
# Remove and re-add server
claude mcp remove highfold-med-tools
claude mcp add highfold-med-tools -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

### Environment Issues
```bash
# Check Python environment
$(pwd)/env/bin/python -c "import pandas, numpy, fastmcp, loguru; print('Dependencies OK')"
```

### Tool Not Found
Make sure to reference tools exactly as shown, e.g., `check_dependencies` not `check-dependencies`.

## Example Session

```
User: "I want to predict the structure of a cyclic peptide. How do I start?"