#!/usr/bin/env python3
"""
Script: predict_structure.py
Description: Predict 3D structure of cyclic peptide from SMILES or sequence

Original Use Case: examples/use_case_1_single_prediction.py
Dependencies Removed: predict_utils (inlined core logic), reduced sys.path manipulation

Usage:
    python scripts/predict_structure.py --input <input_file> --output <output_file>

Example:
    python scripts/predict_structure.py --input examples/data/sequences/targets.tsv --output results/output.txt --index 0
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import os
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json
import pandas as pd

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "model_name": "model_2_ptm",
    "demo_mode": True,  # Default to demo mode for MCP
    "alignment_dir": "examples/data/alignments",
    "output_format": "txt",
    "include_template_info": True,
    "max_templates": 10
}

# ==============================================================================
# Inlined Utility Functions (simplified from repo)
# ==============================================================================
def validate_cyclic_peptide_sequence(sequence: str) -> bool:
    """Validate that sequence is a cyclic peptide in HighFold notation."""
    if not sequence:
        return False

    # Check for typical cyclic peptide notation patterns
    # D-amino acids: dL, dP, etc.
    # N-methylation: special characters like h, _
    # Terminal modifications: dots
    valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._/')
    return all(c in valid_chars for c in sequence)

def parse_targets_file(targets_file: Union[str, Path], index: int = 0) -> Dict[str, Any]:
    """Parse targets TSV file and extract peptide information."""
    targets_file = Path(targets_file)

    if not targets_file.exists():
        raise FileNotFoundError(f"Targets file not found: {targets_file}")

    df = pd.read_csv(targets_file, sep='\t')

    if index >= len(df):
        raise ValueError(f"Index {index} is out of range. File has {len(df)} entries.")

    row = df.iloc[index]

    result = {
        'target_id': row.get('targetid', f'peptide_{index}'),
        'sequence': row['target_chainseq'],
        'alignment_file': row.get('templates_alignfile', ''),
        'raw_row': dict(row)
    }

    return result

def resolve_alignment_file(alignment_file_path: str, alignment_dir: str) -> Optional[Path]:
    """Resolve alignment file path to local examples/data/alignments/ directory."""
    if not alignment_file_path:
        return None

    # Extract filename from path
    alignment_filename = os.path.basename(alignment_file_path)
    local_alignment = Path(alignment_dir) / alignment_filename

    if local_alignment.exists():
        return local_alignment

    # Try the original path as-is
    if Path(alignment_file_path).exists():
        return Path(alignment_file_path)

    return None

def load_template_alignment(alignment_file: Path) -> List[Dict[str, Any]]:
    """Load and parse template alignment file."""
    if not alignment_file.exists():
        return []

    try:
        df = pd.read_csv(alignment_file, sep='\t')

        templates = []
        for _, row in df.iterrows():
            template = {
                'pdb_file': row.get('template_pdbfile', 'N/A'),
                'alignment': row.get('target_to_template_alignstring', 'N/A'),
                'identities': float(row.get('identities', 0.0)),
                'template_len': int(row.get('template_len', 0)),
                'query_start': int(row.get('query_start', 0)),
                'query_end': int(row.get('query_end', 0))
            }
            templates.append(template)

        return templates

    except Exception as e:
        print(f"Warning: Could not parse alignment file {alignment_file}: {e}")
        return []

def save_prediction_output(data: Dict[str, Any], file_path: Path) -> None:
    """Save prediction output in specified format."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        f.write("HighFold-MeD Prediction Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Sequence: {data.get('sequence', 'N/A')}\n")
        f.write(f"Target ID: {data.get('target_id', 'N/A')}\n")
        f.write(f"Query Length: {data.get('query_length', 0)}\n")
        f.write(f"Templates Used: {data.get('templates_used', 0)}\n")
        f.write(f"Model: {data.get('model_name', 'N/A')}\n")
        f.write(f"Demo Mode: {data.get('demo_mode', True)}\n")
        f.write(f"Status: {data.get('status', 'unknown')}\n\n")

        if 'template_info' in data and data['template_info']:
            f.write("Template Information:\n")
            for i, template in enumerate(data['template_info']):
                f.write(f"  Template {i+1}:\n")
                f.write(f"    PDB: {template.get('pdb_file', 'N/A')}\n")
                f.write(f"    Identities: {template.get('identities', 0.0):.3f}\n")
                f.write(f"    Length: {template.get('template_len', 0)}\n")
                f.write(f"    Alignment: {template.get('alignment', 'N/A')[:50]}{'...' if len(str(template.get('alignment', ''))) > 50 else ''}\n\n")
        else:
            f.write("No template information available.\n")

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_predict_structure(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for cyclic peptide structure prediction.

    Args:
        input_file: Path to input file (targets.tsv format or single sequence)
        output_file: Path to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - result: Main computation result
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_predict_structure("examples/data/sequences/targets.tsv", "output.txt", index=0)
        >>> print(result['output_file'])
    """
    # Setup
    input_file = Path(input_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Parse input - either targets file or direct sequence
    if str(input_file).endswith('.tsv'):
        # Targets file format
        index = kwargs.get('index', 0)
        target_info = parse_targets_file(input_file, index)
        sequence = target_info['sequence']
        target_id = target_info['target_id']
        alignment_file_path = target_info['alignment_file']
    else:
        # Direct sequence input (future enhancement)
        with open(input_file, 'r') as f:
            sequence = f.read().strip()
        target_id = input_file.stem
        alignment_file_path = kwargs.get('alignment_file', '')

    # Validate sequence
    if not validate_cyclic_peptide_sequence(sequence):
        raise ValueError(f"Invalid cyclic peptide sequence: {sequence}")

    # Process sequence
    query_sequence = sequence.replace('/', '')
    query_length = len(query_sequence)

    print(f"Processing sequence: {sequence}")
    print(f"Target ID: {target_id}")
    print(f"Query length: {query_length}")

    # Load template alignment if available
    template_info = []
    if alignment_file_path:
        alignment_file = resolve_alignment_file(
            alignment_file_path,
            config.get('alignment_dir', DEFAULT_CONFIG['alignment_dir'])
        )

        if alignment_file:
            print(f"Loading templates from: {alignment_file}")
            template_info = load_template_alignment(alignment_file)

            # Limit templates if configured
            max_templates = config.get('max_templates', DEFAULT_CONFIG['max_templates'])
            if max_templates and len(template_info) > max_templates:
                template_info = template_info[:max_templates]
        else:
            print(f"Warning: Alignment file not found: {alignment_file_path}")

    # Core prediction logic (simplified for MCP)
    if config.get('demo_mode', True):
        print("Running in demo mode - simulating prediction pipeline")
        status = 'demo_complete'
    else:
        print("Note: Full prediction would require AlphaFold parameters and HighFold-MeD setup")
        status = 'prediction_simulated'

    # Create result structure
    result_data = {
        'sequence': sequence,
        'target_id': target_id,
        'query_length': query_length,
        'templates_used': len(template_info),
        'model_name': config.get('model_name', DEFAULT_CONFIG['model_name']),
        'template_info': template_info,
        'demo_mode': config.get('demo_mode', DEFAULT_CONFIG['demo_mode']),
        'status': status
    }

    print(f"Prediction completed: {len(template_info)} templates used")

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        save_prediction_output(result_data, output_path)

    return {
        "result": result_data,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "input_file": str(input_file),
            "config": config,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    }

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input options
    parser.add_argument('--input', '-i', required=True,
                       help='Input file path (targets.tsv format)')
    parser.add_argument('--output', '-o',
                       help='Output file path (default: auto-generated)')
    parser.add_argument('--index', type=int, default=0,
                       help='Index of peptide in targets file (default: 0)')
    parser.add_argument('--config', '-c',
                       help='Config file (JSON format)')

    # Model configuration
    parser.add_argument('--model_name', type=str,
                       default=DEFAULT_CONFIG['model_name'],
                       help=f'Model name (default: {DEFAULT_CONFIG["model_name"]})')
    parser.add_argument('--alignment_dir', type=str,
                       default=DEFAULT_CONFIG['alignment_dir'],
                       help=f'Alignment directory (default: {DEFAULT_CONFIG["alignment_dir"]})')
    parser.add_argument('--demo_mode', action='store_true',
                       help='Force demo mode')

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with command line arguments
    config.update({
        'model_name': args.model_name,
        'alignment_dir': args.alignment_dir,
        'demo_mode': args.demo_mode or config.get('demo_mode', DEFAULT_CONFIG['demo_mode'])
    })

    # Generate output filename if not provided
    if not args.output:
        input_path = Path(args.input)
        if str(input_path).endswith('.tsv'):
            # For targets file, use sequence-based name
            try:
                target_info = parse_targets_file(input_path, args.index)
                sequence = target_info['sequence']
                output_filename = f"{sequence}_prediction.txt"
            except:
                output_filename = f"prediction_{args.index}.txt"
        else:
            output_filename = f"{input_path.stem}_prediction.txt"

        args.output = Path("results") / output_filename

    # Run prediction
    try:
        result = run_predict_structure(
            input_file=args.input,
            output_file=args.output,
            config=config,
            index=args.index
        )

        print(f"\nSuccess! Output saved to: {result['output_file']}")
        print(f"Result summary: {result['result']['status']}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())