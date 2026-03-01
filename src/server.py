"""MCP Server for HighFold-MeD Cyclic Peptide Tools

Provides both synchronous and asynchronous (submit) APIs for cyclic peptide
structure prediction, relaxation, and model fine-tuning.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List
import sys
import os
import tempfile

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager

# Create MCP server
mcp = FastMCP("highfold-med-tools")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted HighFold-MeD computation job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)

@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed HighFold-MeD computation job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)

@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running HighFold-MeD computation job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)

@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted HighFold-MeD computation jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)

# ==============================================================================
# Submit Tools (for long-running operations > 10 min)
# ==============================================================================

@mcp.tool()
def submit_structure_prediction(
    input_file: str,
    index: int = 0,
    model_name: str = "model_2_ptm",
    alignment_dir: str = "examples/data/alignments",
    demo_mode: bool = True,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a 3D structure prediction job for a cyclic peptide using HighFold-MeD.

    This task typically takes 10+ minutes. Use get_job_status() to monitor
    progress and get_job_result() to retrieve results when completed.

    Args:
        input_file: Path to targets.tsv file with peptide sequences
        index: Index of peptide in targets file to predict (default: 0)
        model_name: HighFold model to use (default: model_2_ptm)
        alignment_dir: Directory containing template alignments
        demo_mode: Run in demo mode (default: True for MCP)
        job_name: Optional name for the job (for easier tracking)

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs
    """
    script_path = str(SCRIPTS_DIR / "predict_structure.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": input_file,
            "index": index,
            "model_name": model_name,
            "alignment_dir": alignment_dir,
            "demo_mode": demo_mode
        },
        job_name=job_name or f"predict_{Path(input_file).stem}_idx{index}"
    )

@mcp.tool()
def submit_structure_relaxation(
    input_file: str,
    restraint_force: float = 20000.0,
    tolerance: float = 2.39,
    backbone_restraints: bool = True,
    energy_minimization: bool = True,
    demo_mode: bool = True,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a structure relaxation job for a cyclic peptide PDB using OpenMM.

    Uses molecular dynamics energy minimization with position restraints.
    This task typically takes 5-15 minutes.

    Args:
        input_file: Path to input PDB file
        restraint_force: Position restraint force constant (kJ/mol/nm²)
        tolerance: Energy minimization tolerance (kcal/mol)
        backbone_restraints: Apply restraints to backbone atoms
        energy_minimization: Perform energy minimization
        demo_mode: Run in demo mode (default: True for MCP)
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking
    """
    script_path = str(SCRIPTS_DIR / "relax_structure.py")

    args = {
        "input": input_file,
        "restraint_force": restraint_force,
        "tolerance": tolerance,
        "demo": demo_mode
    }

    if not backbone_restraints:
        args["no_restraints"] = True
    if not energy_minimization:
        args["no_minimization"] = True

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"relax_{Path(input_file).stem}"
    )

@mcp.tool()
def submit_model_finetuning(
    train_file: str,
    validation_file: Optional[str] = None,
    model_name: str = "model_2_ptm",
    num_epochs: int = 10,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    demo_mode: bool = True,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a model fine-tuning job for HighFold-MeD on cyclic peptides.

    Fine-tunes AlphaFold models for better cyclic peptide prediction.
    This is a long-running task (typically 30+ minutes to hours).

    Args:
        train_file: Path to training data file (TSV format)
        validation_file: Path to validation data file (optional)
        model_name: Base HighFold model to fine-tune
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
        demo_mode: Run in demo mode (default: True for MCP)
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking
    """
    script_path = str(SCRIPTS_DIR / "finetune_model.py")

    args = {
        "train": train_file,
        "model_name": model_name,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "demo": demo_mode
    }

    if validation_file:
        args["validation"] = validation_file

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"finetune_{model_name}"
    )

@mcp.tool()
def submit_batch_prediction(
    input_file: str,
    max_peptides: Optional[int] = None,
    model_name: str = "model_2_ptm",
    demo_mode: bool = True,
    continue_on_error: bool = True,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a batch prediction job for multiple cyclic peptides.

    Processes multiple cyclic peptides from a targets.tsv file in sequence.
    Runtime scales with number of peptides (N × 10+ minutes).

    Args:
        input_file: Path to targets.tsv file with multiple peptides
        max_peptides: Maximum number of peptides to process (default: all)
        model_name: HighFold model to use for predictions
        demo_mode: Run in demo mode (default: True for MCP)
        continue_on_error: Continue processing if some peptides fail
        job_name: Optional name for the batch job

    Returns:
        Dictionary with job_id for tracking the batch job
    """
    script_path = str(SCRIPTS_DIR / "batch_predict.py")

    args = {
        "input": input_file,
        "model_name": model_name,
        "demo_mode": demo_mode,
        "continue_on_error": continue_on_error
    }

    if max_peptides:
        args["max_peptides"] = max_peptides

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"batch_{Path(input_file).stem}_{max_peptides or 'all'}"
    )

# ==============================================================================
# Synchronous Tools (for quick operations < 5 min)
# ==============================================================================

@mcp.tool()
def validate_targets_file(targets_file: str) -> dict:
    """
    Validate a targets.tsv file format and content.

    Quick validation of file format and peptide sequences.

    Args:
        targets_file: Path to targets.tsv file

    Returns:
        Dictionary with validation results and file statistics
    """
    try:
        # Import validation logic from predict_structure
        from predict_structure import parse_targets_file, validate_cyclic_peptide_sequence
        import pandas as pd

        file_path = Path(targets_file)
        if not file_path.exists():
            return {"status": "error", "error": f"File not found: {targets_file}"}

        # Read and validate file
        df = pd.read_csv(file_path, sep='\t')

        # Check required columns
        required_cols = ['target_chainseq']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return {
                "status": "error",
                "error": f"Missing required columns: {missing_cols}",
                "available_columns": list(df.columns)
            }

        # Validate sequences
        valid_sequences = 0
        invalid_sequences = []

        for idx, row in df.iterrows():
            sequence = row['target_chainseq']
            if validate_cyclic_peptide_sequence(sequence):
                valid_sequences += 1
            else:
                invalid_sequences.append({
                    "index": idx,
                    "sequence": sequence,
                    "target_id": row.get('targetid', f'peptide_{idx}')
                })

        return {
            "status": "success",
            "total_peptides": len(df),
            "valid_sequences": valid_sequences,
            "invalid_sequences": len(invalid_sequences),
            "invalid_details": invalid_sequences[:5],  # Show first 5 invalid
            "columns": list(df.columns),
            "file_size": file_path.stat().st_size
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def get_peptide_info(targets_file: str, index: int = 0) -> dict:
    """
    Get information about a specific peptide from a targets file.

    Quick lookup of peptide details by index.

    Args:
        targets_file: Path to targets.tsv file
        index: Index of peptide to examine (default: 0)

    Returns:
        Dictionary with peptide information
    """
    try:
        from predict_structure import parse_targets_file

        peptide_info = parse_targets_file(targets_file, index)

        # Add some analysis
        sequence = peptide_info['sequence']
        clean_sequence = sequence.replace('/', '')

        return {
            "status": "success",
            "target_id": peptide_info['target_id'],
            "sequence": sequence,
            "clean_sequence": clean_sequence,
            "sequence_length": len(clean_sequence),
            "has_alignment": bool(peptide_info.get('alignment_file')),
            "alignment_file": peptide_info.get('alignment_file'),
            "contains_modifications": any(c in sequence for c in ['d', 'h', '_', '.']),
            "raw_data": peptide_info['raw_row']
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def check_dependencies() -> dict:
    """
    Check availability of HighFold-MeD dependencies and environment setup.

    Quick system check for required packages and files.

    Returns:
        Dictionary with dependency status and recommendations
    """
    dependencies = {
        "core_packages": {},
        "optional_packages": {},
        "files": {},
        "recommendations": []
    }

    # Check core Python packages
    core_packages = ["pandas", "numpy", "pathlib"]
    for pkg in core_packages:
        try:
            __import__(pkg)
            dependencies["core_packages"][pkg] = "available"
        except ImportError:
            dependencies["core_packages"][pkg] = "missing"

    # Check optional packages for full functionality
    optional_packages = ["openmm", "jax", "haiku", "optax", "torch", "tensorflow"]
    for pkg in optional_packages:
        try:
            __import__(pkg)
            dependencies["optional_packages"][pkg] = "available"
        except ImportError:
            dependencies["optional_packages"][pkg] = "missing"

    # Check for example data files
    example_files = {
        "targets.tsv": SCRIPTS_DIR.parent / "examples" / "data" / "sequences" / "targets.tsv",
        "alignments_dir": SCRIPTS_DIR.parent / "examples" / "data" / "alignments",
        "structures_dir": SCRIPTS_DIR.parent / "examples" / "data" / "structures"
    }

    for name, path in example_files.items():
        dependencies["files"][name] = {
            "path": str(path),
            "exists": path.exists(),
            "type": "directory" if path.is_dir() else "file" if path.exists() else "missing"
        }

    # Generate recommendations
    missing_core = [pkg for pkg, status in dependencies["core_packages"].items() if status == "missing"]
    missing_optional = [pkg for pkg, status in dependencies["optional_packages"].items() if status == "missing"]

    if missing_core:
        dependencies["recommendations"].append(f"Install missing core packages: {', '.join(missing_core)}")

    if missing_optional:
        dependencies["recommendations"].append(f"For full functionality, consider installing: {', '.join(missing_optional)}")

    if not dependencies["files"]["targets.tsv"]["exists"]:
        dependencies["recommendations"].append("Example data files not found. Some tools may not work without proper input files.")

    # Determine overall status
    if missing_core:
        status = "error"
    elif missing_optional:
        status = "partial"
    else:
        status = "ready"

    return {
        "status": status,
        "dependencies": dependencies,
        "demo_mode_available": True,  # Demo mode always works
        "full_functionality": len(missing_optional) == 0
    }

@mcp.tool()
def list_example_data() -> dict:
    """
    List available example data files for testing HighFold-MeD tools.

    Returns:
        Dictionary with available example files and their descriptions
    """
    examples_dir = SCRIPTS_DIR.parent / "examples" / "data"

    if not examples_dir.exists():
        return {
            "status": "error",
            "error": "Examples directory not found",
            "expected_path": str(examples_dir)
        }

    example_files = {}

    # Look for common data patterns
    for subdir in ["sequences", "structures", "alignments"]:
        subdir_path = examples_dir / subdir
        if subdir_path.exists():
            files = []
            for file_path in subdir_path.iterdir():
                if file_path.is_file():
                    files.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "extension": file_path.suffix
                    })
            example_files[subdir] = files

    return {
        "status": "success",
        "examples_directory": str(examples_dir),
        "available_files": example_files,
        "usage_hint": "Use targets.tsv files with structure prediction tools, PDB files with relaxation tools"
    }

# ==============================================================================
# Utility Tools
# ==============================================================================

@mcp.tool()
def create_demo_targets_file(
    output_file: str,
    num_peptides: int = 5,
    sequence_length_range: tuple = (6, 12)
) -> dict:
    """
    Create a demo targets.tsv file for testing HighFold-MeD tools.

    Generates synthetic cyclic peptide sequences for testing purposes.

    Args:
        output_file: Path where to save the demo targets file
        num_peptides: Number of peptides to generate (default: 5)
        sequence_length_range: Min and max sequence lengths (default: 6-12)

    Returns:
        Dictionary with file creation status and details
    """
    try:
        import pandas as pd
        import random

        # Generate synthetic cyclic peptide sequences
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        special_chars = ['d', 'h', '_', '.']  # D-amino acids, N-methylation, modifications

        peptides = []
        for i in range(num_peptides):
            length = random.randint(sequence_length_range[0], sequence_length_range[1])

            sequence = ''
            for j in range(length):
                if random.random() < 0.15:  # 15% chance of modification
                    sequence += random.choice(special_chars)
                sequence += random.choice(amino_acids)

            peptides.append({
                'targetid': f'demo_peptide_{i+1}',
                'target_chainseq': sequence,
                'templates_alignfile': f'alignments/demo_peptide_{i+1}.tsv',
                'description': f'Demo cyclic peptide {i+1}'
            })

        # Create DataFrame and save
        df = pd.DataFrame(peptides)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, sep='\t', index=False)

        return {
            "status": "success",
            "output_file": str(output_path),
            "peptides_created": len(peptides),
            "file_size": output_path.stat().st_size,
            "sample_sequences": [p['target_chainseq'] for p in peptides[:3]]
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()