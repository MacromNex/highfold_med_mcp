#!/usr/bin/env python3
"""
Script: batch_predict.py
Description: Batch prediction of cyclic peptide structures from targets file

Original Use Case: examples/use_case_2_batch_prediction.py
Dependencies Removed: subprocess calls to separate scripts, simplified for direct function calls

Usage:
    python scripts/batch_predict.py --input <targets_file> --output <output_dir>

Example:
    python scripts/batch_predict.py --input examples/data/sequences/targets.tsv --output results/batch/ --max_peptides 5
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

# Import our predict_structure module
import sys
from pathlib import Path

# Add current directory to path for importing predict_structure
sys.path.insert(0, str(Path(__file__).parent))
import predict_structure

# ==============================================================================
# Configuration
# ==============================================================================
DEFAULT_CONFIG = {
    "max_peptides": None,  # Process all by default
    "continue_on_error": True,
    "output_format": "txt",
    "parallel": False,  # Sequential processing for simplicity
    "save_individual": True,
    "save_summary": True
}

# ==============================================================================
# Core Function
# ==============================================================================
def run_batch_predict(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for batch cyclic peptide structure prediction.

    Args:
        input_file: Path to targets.tsv file
        output_file: Path to output directory (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - result: Batch processing results
            - output_files: List of generated output files
            - metadata: Execution metadata

    Example:
        >>> result = run_batch_predict("targets.tsv", "results/batch/", max_peptides=5)
        >>> print(f"Processed {result['result']['processed']} peptides")
    """
    # Setup
    input_file = Path(input_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Read targets file
    df = pd.read_csv(input_file, sep='\t')
    total_peptides = len(df)

    # Limit peptides if configured
    max_peptides = config.get('max_peptides')
    if max_peptides:
        df = df.head(max_peptides)

    print(f"Processing {len(df)} peptides (total available: {total_peptides})")

    # Setup output directory
    if output_file:
        output_dir = Path(output_file)
    else:
        output_dir = Path("results") / "batch_predictions"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    results = {
        'total_peptides': total_peptides,
        'processed': 0,
        'successful': 0,
        'errors': [],
        'output_files': [],
        'peptide_results': []
    }

    # Process each peptide
    for idx, row in df.iterrows():
        try:
            # Get peptide information
            target_id = row.get('targetid', f'peptide_{idx}')
            sequence = row['target_chainseq']

            print(f"\nProcessing peptide {idx + 1}/{len(df)}: {target_id} ({sequence})")

            # Create individual output file
            if config.get('save_individual', True):
                individual_output = output_dir / f"{target_id}_prediction.txt"
            else:
                individual_output = None

            # Run prediction for this peptide
            prediction_result = predict_structure.run_predict_structure(
                input_file=input_file,
                output_file=individual_output,
                config=config,
                index=idx
            )

            # Track success
            results['processed'] += 1
            if prediction_result['result']['status'] in ['demo_complete', 'prediction_complete']:
                results['successful'] += 1

            # Store detailed result
            peptide_result = {
                'index': idx,
                'target_id': target_id,
                'sequence': sequence,
                'status': prediction_result['result']['status'],
                'templates_used': prediction_result['result']['templates_used'],
                'output_file': str(individual_output) if individual_output else None
            }
            results['peptide_results'].append(peptide_result)

            if individual_output:
                results['output_files'].append(str(individual_output))

            print(f"✓ Completed {target_id}: {prediction_result['result']['status']}")

        except Exception as e:
            error_msg = f"Error processing peptide {idx} ({row.get('targetid', 'unknown')}): {e}"
            results['errors'].append(error_msg)
            print(f"✗ {error_msg}")

            if not config.get('continue_on_error', True):
                break

    # Save batch summary if requested
    if config.get('save_summary', True):
        summary_file = output_dir / "batch_summary.txt"
        save_batch_summary(results, summary_file, input_file, config)
        results['output_files'].append(str(summary_file))

    return {
        "result": results,
        "output_files": results['output_files'],
        "metadata": {
            "input_file": str(input_file),
            "output_dir": str(output_dir),
            "config": config,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    }

def save_batch_summary(results: Dict[str, Any], summary_file: Path, input_file: Path, config: Dict[str, Any]) -> None:
    """Save batch processing summary to file."""
    with open(summary_file, 'w') as f:
        f.write("HighFold-MeD Batch Prediction Summary\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Input file: {input_file}\n")
        f.write(f"Output directory: {summary_file.parent}\n")
        f.write(f"Configuration: {config.get('model_name', 'default')}\n\n")

        f.write(f"Total peptides in file: {results['total_peptides']}\n")
        f.write(f"Processed: {results['processed']}\n")
        f.write(f"Successful: {results['successful']}\n")
        f.write(f"Errors: {len(results['errors'])}\n\n")

        if results['errors']:
            f.write("Errors encountered:\n")
            for error in results['errors']:
                f.write(f"  - {error}\n")
            f.write("\n")

        if results['peptide_results']:
            f.write("Individual Results:\n")
            f.write("Index\tTarget_ID\tSequence\tStatus\tTemplates\tOutput_File\n")
            for result in results['peptide_results']:
                f.write(f"{result['index']}\t")
                f.write(f"{result['target_id']}\t")
                f.write(f"{result['sequence']}\t")
                f.write(f"{result['status']}\t")
                f.write(f"{result['templates_used']}\t")
                f.write(f"{result['output_file'] or 'N/A'}\n")

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/Output
    parser.add_argument('--input', '-i', required=True,
                       help='Input targets.tsv file')
    parser.add_argument('--output', '-o',
                       help='Output directory (default: results/batch_predictions/)')
    parser.add_argument('--config', '-c',
                       help='Config file (JSON format)')

    # Processing options
    parser.add_argument('--max_peptides', type=int,
                       help='Maximum number of peptides to process')
    parser.add_argument('--continue_on_error', action='store_true', default=True,
                       help='Continue processing even if some peptides fail')
    parser.add_argument('--no_individual', action='store_true',
                       help='Skip saving individual prediction files')
    parser.add_argument('--no_summary', action='store_true',
                       help='Skip saving batch summary file')

    # Model configuration (passed to predict_structure)
    parser.add_argument('--model_name', type=str,
                       default=predict_structure.DEFAULT_CONFIG['model_name'],
                       help='Model name for predictions')
    parser.add_argument('--demo_mode', action='store_true',
                       help='Force demo mode for all predictions')

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with command line arguments
    config.update({
        'max_peptides': args.max_peptides,
        'continue_on_error': args.continue_on_error,
        'save_individual': not args.no_individual,
        'save_summary': not args.no_summary,
        'model_name': args.model_name,
        'demo_mode': args.demo_mode or config.get('demo_mode', False)
    })

    # Run batch prediction
    try:
        result = run_batch_predict(
            input_file=args.input,
            output_file=args.output,
            config=config
        )

        # Print summary
        batch_result = result['result']
        print(f"\nBatch processing completed!")
        print(f"Total: {batch_result['total_peptides']} peptides in file")
        print(f"Processed: {batch_result['processed']}")
        print(f"Successful: {batch_result['successful']}")
        print(f"Errors: {len(batch_result['errors'])}")

        if batch_result['errors']:
            print(f"\nFirst few errors:")
            for error in batch_result['errors'][:3]:
                print(f"  - {error}")
            if len(batch_result['errors']) > 3:
                print(f"  ... and {len(batch_result['errors']) - 3} more")

        if result['output_files']:
            print(f"\nOutput files generated:")
            print(f"  Directory: {Path(result['output_files'][0]).parent}")
            print(f"  Files: {len(result['output_files'])} total")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())