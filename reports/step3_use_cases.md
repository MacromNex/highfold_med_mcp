# Step 3: Use Cases Report

## Scan Information
- **Scan Date**: 2025-12-31
- **Filter Applied**: cyclic peptide structure prediction with N-methylation and D-amino acids using HighFold-MeD
- **Python Version**: Dual setup (3.10 for MCP, 3.8 for HighFold)
- **Environment Strategy**: dual environment

## Use Cases

### UC-001: Single Cyclic Peptide Structure Prediction
- **Description**: Predict 3D structure of individual cyclic peptides with N-methylation and D-amino acids using template-based AlphaFold inference
- **Script Path**: `examples/use_case_1_single_prediction.py`
- **Complexity**: medium
- **Priority**: high
- **Environment**: `./env` for MCP interface, `./env_py38` for HighFold core
- **Source**: `repo/HighFold-MeD/run_prediction.py`, README.md

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| sequence | string | Cyclic peptide sequence in HighFold notation | --sequence |
| targets_file | file | TSV file with peptide targets | --targets |
| alignment_file | file | Template alignment file | --alignment_file |
| index | integer | Index of peptide in targets file | --index |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| prediction_summary | file | Text summary with structure prediction results |
| pdb_structure | file | Predicted 3D structure in PDB format |
| confidence_scores | data | PLDDT and PAE confidence metrics |

**Example Usage:**
```bash
python examples/use_case_1_single_prediction.py --targets examples/data/sequences/targets.tsv --index 0
python examples/use_case_1_single_prediction.py --sequence "PhdLP_d" --alignment_file examples/data/alignments/Me_20706.tsv
```

**Example Data**: `examples/data/sequences/targets.tsv`, `examples/data/alignments/Me_*.tsv`

---

### UC-002: Batch Cyclic Peptide Structure Prediction
- **Description**: High-throughput structure prediction for multiple cyclic peptides with automated processing and results aggregation
- **Script Path**: `examples/use_case_2_batch_prediction.py`
- **Complexity**: medium
- **Priority**: high
- **Environment**: `./env` for batch processing, `./env_py38` for HighFold predictions
- **Source**: `repo/HighFold-MeD/run_prediction_batch.py`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| targets_file | file | TSV file with multiple peptide targets | --targets |
| model_params | file | Fine-tuned model parameters (.pkl) | --model_params |
| max_peptides | integer | Maximum number to process | --max_peptides |
| dry_run | boolean | Show commands without execution | --dry_run |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| batch_summary | file | Processing summary with success/error counts |
| individual_results | files | Separate prediction results for each peptide |
| aggregated_metrics | file | Combined confidence scores and statistics |

**Example Usage:**
```bash
python examples/use_case_2_batch_prediction.py --targets examples/data/sequences/targets.tsv --dry_run
python examples/use_case_2_batch_prediction.py --targets examples/data/sequences/targets.tsv --max_peptides 5
```

**Example Data**: `examples/data/sequences/targets.tsv`

---

### UC-003: Structure Relaxation with OpenMM
- **Description**: Energy minimization and structure refinement of predicted cyclic peptides using molecular dynamics with backbone restraints
- **Script Path**: `examples/use_case_3_structure_relaxation.py`
- **Complexity**: medium
- **Priority**: medium
- **Environment**: `./env` for general processing, requires OpenMM installation
- **Source**: `repo/HighFold-MeD/relaxed/relaxed.py`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| pdb_file | file | Input PDB structure | --pdb |
| pdb_directory | directory | Batch processing directory | --pdb_dir |
| restraint_force | float | Position restraint force constant | --restraint_force |
| tolerance | float | Energy minimization tolerance | --tolerance |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| relaxed_pdb | file | Energy-minimized structure |
| energy_report | file | Initial/final energies and convergence |
| amber_files | files | Intermediate topology and coordinate files |

**Example Usage:**
```bash
python examples/use_case_3_structure_relaxation.py --pdb_dir examples/data/structures/ --demo
python examples/use_case_3_structure_relaxation.py --pdb examples/data/structures/1.pdb --output relaxed_1.pdb
```

**Example Data**: `examples/data/structures/1.pdb`, `examples/data/structures/2.pdb`

---

### UC-004: Model Fine-tuning for Cyclic Peptides
- **Description**: Fine-tune AlphaFold models specifically for cyclic peptides with N-methylation and D-amino acids using custom training datasets
- **Script Path**: `examples/use_case_4_model_finetuning.py`
- **Complexity**: complex
- **Priority**: medium
- **Environment**: `./env_py38` (requires JAX, TensorFlow, PyTorch)
- **Source**: `repo/HighFold-MeD/run_finetuning.py`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| train_data | file | Training dataset TSV | --train_data |
| valid_data | file | Validation dataset TSV | --valid_data |
| model_name | string | Base AlphaFold model | --model_name |
| epochs | integer | Number of training epochs | --epochs |
| batch_size | integer | Training batch size | --batch_size |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| finetuned_model | file | Fine-tuned model parameters (.pkl) |
| training_log | file | Loss curves and training metrics |
| validation_metrics | file | Performance on validation set |

**Example Usage:**
```bash
python examples/use_case_4_model_finetuning.py --demo
python examples/use_case_4_model_finetuning.py --train_data train.tsv --valid_data valid.tsv --epochs 200
```

**Example Data**: Training data in `repo/HighFold-MeD/datasets_alphafold_finetune_cyclic/`

---

## Summary

| Metric | Count |
|--------|-------|
| Total Use Cases Found | 4 |
| Scripts Created | 4 |
| High Priority | 2 |
| Medium Priority | 2 |
| Low Priority | 0 |
| Demo Data Copied | Yes |

## Demo Data Index

| Source | Destination | Description |
|--------|-------------|-------------|
| `repo/HighFold-MeD/targets/targets.tsv` | `examples/data/sequences/targets.tsv` | Sample cyclic peptide targets with alignments |
| `repo/HighFold-MeD/datasets_alphafold_finetune_cyclic/alignments/alignments/*.tsv` | `examples/data/alignments/` | Template alignment files for structure prediction |
| `repo/HighFold-MeD/relaxed/*.pdb` | `examples/data/structures/` | Sample 3D structures for relaxation testing |

## Cyclic Peptide Notation Examples

The repository uses specialized notation for modified amino acids:

| Notation | Description | Example in Dataset |
|----------|-------------|-------------------|
| `PhdLP_d` | Proline, histidine, D-leucine, leucine, proline, D-alanine | D7.6 |
| `VIhFIh.` | Valine, isoleucine, methylated-histidine, phenylalanine, isoleucine, methylated-histidine | D7.8 |
| `dLhdL.PL` | D-leucine, methylated-histidine, D-leucine, proline, leucine | D8.1 |

### Key Notation Elements:
- **d** prefix: D-amino acids (e.g., `dL` = D-leucine)
- **h**: N-methylated histidine
- **p**: N-methylated proline
- **+**: Modified amino acids with additional functional groups
- **&**: Cross-linked or cyclized residues
- **.**: C-terminal modifications

## MCP Tool Recommendations

Based on the use cases, the following MCP tools would provide maximum value:

### High-Priority Tools
1. **structure_predict**: Single peptide prediction with confidence scoring
2. **batch_predict**: Multiple peptide processing with progress tracking
3. **sequence_analyze**: Parse and validate cyclic peptide notation

### Medium-Priority Tools
4. **structure_relax**: Energy minimization and refinement
5. **template_search**: Find similar template structures
6. **confidence_assess**: Analyze prediction reliability

### Advanced Tools
7. **model_finetune**: Custom model training (requires extensive setup)
8. **property_predict**: ADMET properties from structure
9. **visualization**: 3D structure rendering and analysis

Each tool should accept both individual parameters and batch processing modes for scalability.