# HighFold-MeD MCP Scripts

Clean, self-contained scripts extracted from verified use cases for MCP tool wrapping.

## Design Principles

1. **Minimal Dependencies**: Only essential packages imported (pandas, pathlib, json)
2. **Self-Contained**: Functions inlined where possible, minimal repo dependencies
3. **Configurable**: Parameters externalized to config files, not hardcoded
4. **MCP-Ready**: Each script has a main function ready for MCP wrapping
5. **Demo Mode**: All scripts work in demo mode without heavy dependencies

## Scripts Overview

| Script | Description | Repo Dependent | Config File | MCP Function |
|--------|-------------|----------------|-------------|--------------|
| `predict_structure.py` | Predict 3D structure from sequence | No | `configs/predict_structure_config.json` | `run_predict_structure()` |
| `batch_predict.py` | Batch prediction of multiple peptides | No | `configs/batch_predict_config.json` | `run_batch_predict()` |
| `relax_structure.py` | Structure relaxation with OpenMM | Optional | `configs/relax_structure_config.json` | `run_relax_structure()` |
| `finetune_model.py` | Model fine-tuning demonstration | Optional | `configs/finetune_model_config.json` | `run_finetune_model()` |

## Dependencies

### Essential (Required)
- `pandas` >= 1.3.0 - Data processing for TSV files
- `pathlib` (stdlib) - File path handling
- `json` (stdlib) - Configuration management
- `argparse` (stdlib) - Command line interface

### Optional (For Full Functionality)
- `openmm` - Structure relaxation (falls back to demo mode)
- `ambertools` - Force field preparation (falls back to demo mode)
- `jax`, `haiku`, `optax` - Model training (falls back to demo mode)

## Usage

### Basic Usage

```bash
# Activate environment
mamba activate ./env  # or: conda activate ./env

# Single peptide prediction
python scripts/predict_structure.py \
    --input examples/data/sequences/targets.tsv \
    --index 0 \
    --output results/prediction.txt

# Batch prediction
python scripts/batch_predict.py \
    --input examples/data/sequences/targets.tsv \
    --max_peptides 5 \
    --output results/batch/

# Structure relaxation
python scripts/relax_structure.py \
    --input examples/data/structures/1.pdb \
    --output results/relaxed.pdb \
    --demo

# Model fine-tuning
python scripts/finetune_model.py \
    --demo \
    --epochs 5 \
    --output results/finetuning/
```

### With Custom Configuration

```bash
# Create custom config file
cat > my_config.json << EOF
{
  "model": {"name": "model_2_ptm", "demo_mode": true},
  "processing": {"max_templates": 5},
  "output": {"format": "txt", "include_metadata": true}
}
EOF

# Use custom config
python scripts/predict_structure.py \
    --input examples/data/sequences/targets.tsv \
    --config my_config.json \
    --index 0
```

## Shared Library

Common functions are provided in `scripts/lib/`:

### `cyclic_peptides.py`
- `validate_highfold_sequence(sequence)` - Validate HighFold notation
- `parse_highfold_modifications(sequence)` - Parse D-amino acids, N-methylation
- `load_template_alignment(file)` - Load template alignment data
- `format_prediction_summary(result)` - Format results for output

### `io_utils.py`
- `load_config(file, defaults)` - Configuration management
- `ensure_directory(path)` - Directory creation
- `read_tsv_file(path)` - TSV data loading
- `save_json_result(result, path)` - JSON result serialization

## Input Formats

### Targets File (TSV)
Required columns:
- `target_chainseq` - Cyclic peptide sequence in HighFold notation
- `targetid` (optional) - Peptide identifier
- `templates_alignfile` (optional) - Path to template alignment file

Example:
```tsv
targetid	target_chainseq	templates_alignfile
D7.6	PhdLP_d	datasets_alphafold_finetune_cyclic/template_alignments/Me_20706.tsv
D7.8	VIhFIh.	datasets_alphafold_finetune_cyclic/template_alignments/Me_20708.tsv
```

### HighFold Sequence Notation
- Standard amino acids: `A`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `K`, `L`, `M`, `N`, `P`, `Q`, `R`, `S`, `T`, `V`, `W`, `Y`
- D-amino acids: `dA`, `dL`, `dP`, etc.
- N-methylation: `Ah`, `Lh`, `Ph`, etc.
- Terminal modifications: `.` (C-terminus), `_` (linker/modification)
- Structural separators: `/` (removed during processing)

Examples:
- `PhdLP_d` - Proline, histidine, D-leucine, leucine, proline, terminal modification
- `VIhFIh.` - Valine, isoleucine-N-methyl, phenylalanine, isoleucine-N-methyl, histidine, C-terminus modification

## Output Formats

### Prediction Results (TXT)
```
HighFold-MeD Prediction Summary
========================================

Sequence: PhdLP_d
Target ID: D7.6
Query Length: 6
Templates Used: 1
Model: model_2_ptm
Demo Mode: True
Status: demo_complete

Template Information:
  Template 1:
    PDB: datasets_alphafold_finetune_cyclic/templates/Me_20706_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb
    Identities: 0.000
    Length: 6
    Alignment: 0:0;1:1;2:2;3:3;4:4;5:5
```

### Batch Results Summary
```
HighFold-MeD Batch Prediction Summary
========================================

Input file: examples/data/sequences/targets.tsv
Output directory: results/batch_predictions
Configuration: model_2_ptm

Total peptides in file: 30
Processed: 5
Successful: 5
Errors: 0

Individual Results:
Index	Target_ID	Sequence	Status	Templates	Output_File
0	D7.6	PhdLP_d	demo_complete	1	results/batch_predictions/D7.6_prediction.txt
1	D7.8	VIhFIh.	demo_complete	1	results/batch_predictions/D7.8_prediction.txt
```

## For MCP Wrapping (Step 6)

Each script exports a main function that can be wrapped as MCP tools:

```python
# Example MCP tool wrapper
from scripts.predict_structure import run_predict_structure
from scripts.batch_predict import run_batch_predict
from scripts.relax_structure import run_relax_structure
from scripts.finetune_model import run_finetune_model

@mcp.tool()
def predict_cyclic_peptide_structure(
    input_file: str,
    output_file: str = None,
    index: int = 0,
    config: dict = None
) -> dict:
    """Predict 3D structure of cyclic peptide from HighFold sequence."""
    return run_predict_structure(
        input_file=input_file,
        output_file=output_file,
        index=index,
        config=config
    )

@mcp.tool()
def batch_predict_peptide_structures(
    input_file: str,
    output_dir: str = None,
    max_peptides: int = None,
    config: dict = None
) -> dict:
    """Batch prediction of multiple cyclic peptide structures."""
    return run_batch_predict(
        input_file=input_file,
        output_file=output_dir,
        max_peptides=max_peptides,
        config=config
    )

@mcp.tool()
def relax_peptide_structure(
    input_file: str,
    output_file: str = None,
    demo_mode: bool = True,
    config: dict = None
) -> dict:
    """Relax cyclic peptide structure using OpenMM molecular dynamics."""
    return run_relax_structure(
        input_file=input_file,
        output_file=output_file,
        demo_mode=demo_mode,
        config=config
    )

@mcp.tool()
def finetune_peptide_model(
    train_file: str,
    validation_file: str = None,
    output_dir: str = None,
    epochs: int = 10,
    config: dict = None
) -> dict:
    """Fine-tune AlphaFold model for cyclic peptides."""
    return run_finetune_model(
        input_file=train_file,
        output_file=output_dir,
        validation_file=validation_file,
        epochs=epochs,
        config=config
    )
```

## Testing

All scripts have been tested with the demo data:

```bash
# Test individual script functionality
python scripts/predict_structure.py --input examples/data/sequences/targets.tsv --index 0 --demo_mode
python scripts/batch_predict.py --input examples/data/sequences/targets.tsv --max_peptides 2 --demo_mode
python scripts/relax_structure.py --input examples/data/structures/1.pdb --demo
python scripts/finetune_model.py --demo --epochs 3

# Verify output files are created
ls -la results/
```

## Configuration Files

Each script has a corresponding configuration file in `configs/`:
- `predict_structure_config.json` - Structure prediction settings
- `batch_predict_config.json` - Batch processing options
- `relax_structure_config.json` - Molecular dynamics parameters
- `finetune_model_config.json` - Training configuration
- `default_config.json` - Global default settings

## Error Handling

Scripts include robust error handling:
- **File Validation**: Check input files exist and have correct format
- **Sequence Validation**: Validate HighFold notation sequences
- **Dependency Fallback**: Graceful fallback to demo mode when dependencies missing
- **Configuration Validation**: Merge user config with sensible defaults
- **Progress Tracking**: Clear progress indicators for batch operations

## Limitations

- **Demo Mode Default**: Scripts default to demo mode for MCP compatibility
- **Template Dependencies**: Structure prediction requires template alignment files
- **Force Field Limitations**: Structure relaxation limited by AmberTools force field coverage for modified amino acids
- **Training Dependencies**: Model fine-tuning requires specialized ML libraries (JAX, Haiku)

## Next Steps

These scripts are ready for MCP tool wrapping in Step 6:
1. Import the main functions (`run_*`) into MCP server
2. Create MCP tool decorators with appropriate parameter validation
3. Handle file path resolution for MCP environment
4. Add logging integration for MCP debugging
5. Test MCP tools with Claude Code interface