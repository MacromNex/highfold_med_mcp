#!/usr/bin/env python3
"""
HighFold-MeD Use Case 2: Batch Cyclic Peptide Structure Prediction

This script performs batch structure prediction for multiple cyclic peptides
with N-methylation and D-amino acids using HighFold-MeD. Based on run_prediction_batch.py.

Usage:
    python use_case_2_batch_prediction.py --targets examples/data/sequences/targets.tsv --output results/
    python use_case_2_batch_prediction.py --targets examples/data/sequences/targets.tsv --max_peptides 5
"""

import argparse
import os
import sys
import pandas as pd
import subprocess
from pathlib import Path

# Add the repo path to sys.path to import HighFold modules
REPO_PATH = Path(__file__).parent.parent / "repo" / "HighFold-MeD"
sys.path.insert(0, str(REPO_PATH))


def process_batch_predictions(targets_file, output_prefix, model_params=None, max_peptides=None, dry_run=False):
    """
    Process batch predictions for multiple cyclic peptides.

    Args:
        targets_file (str): Path to targets.tsv file
        output_prefix (str): Output directory prefix
        model_params (str): Path to fine-tuned model parameters
        max_peptides (int): Maximum number of peptides to process (None for all)
        dry_run (bool): If True, only show what would be done

    Returns:
        dict: Processing results summary
    """

    print(f"Processing batch predictions from: {targets_file}")

    # Read targets file
    if not os.path.exists(targets_file):
        raise FileNotFoundError(f"Targets file not found: {targets_file}")

    df = pd.read_csv(targets_file, sep='\t')
    print(f"Found {len(df)} peptides in targets file")

    if max_peptides:
        df = df.head(max_peptides)
        print(f"Processing first {len(df)} peptides")

    # Create output directory
    os.makedirs(output_prefix, exist_ok=True)
    temp_targets_dir = os.path.join(output_prefix, "temp_targets")
    os.makedirs(temp_targets_dir, exist_ok=True)

    results = {
        'total_peptides': len(df),
        'processed': 0,
        'errors': [],
        'output_files': []
    }

    for idx, row in df.iterrows():
        try:
            # Get peptide information
            peptide_id = row.get('targetid', f'peptide_{idx}')
            target_chainseq = row['target_chainseq']
            alignment_file = row.get('templates_alignfile', '')

            print(f"\nProcessing peptide {idx + 1}/{len(df)}: {peptide_id} ({target_chainseq})")

            # Create individual target file for this peptide
            individual_target_file = os.path.join(temp_targets_dir, f"target_{peptide_id}.tsv")

            # Write individual target file
            with open(individual_target_file, 'w') as f:
                # Write header
                f.write("\t".join(df.columns) + "\n")
                # Write this row
                f.write("\t".join(map(str, row.values)) + "\n")

            print(f"Created target file: {individual_target_file}")

            # Use our working UC-001 script instead of the original run_prediction.py
            # since it has been fixed to work in demo mode
            command = [
                "python", "examples/use_case_1_single_prediction.py",
                "--targets", individual_target_file,
                "--index", "0",  # Only one peptide in the individual file
                "--output", os.path.join(output_prefix, f"pred_{peptide_id}") + "/"
            ]

            print(f"Command: {' '.join(command)}")

            if dry_run:
                print("DRY RUN: Would execute the above command")
                results['processed'] += 1
                results['output_files'].append(f"pred_{peptide_id}/")
                continue

            # Execute prediction
            try:
                print("Executing prediction command...")
                result = subprocess.run(command, capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    print(f"✓ Prediction completed for {peptide_id}")
                    results['processed'] += 1
                    # UC-001 creates summary files, not final.tsv
                    results['output_files'].append(f"pred_{peptide_id}/")
                else:
                    error_msg = f"Prediction failed for {peptide_id}: {result.stderr}"
                    print(f"✗ {error_msg}")
                    results['errors'].append(error_msg)

            except subprocess.TimeoutExpired:
                error_msg = f"Prediction timeout for {peptide_id}"
                print(f"✗ {error_msg}")
                results['errors'].append(error_msg)

            except Exception as e:
                error_msg = f"Error executing prediction for {peptide_id}: {e}"
                print(f"✗ {error_msg}")
                results['errors'].append(error_msg)

        except Exception as e:
            error_msg = f"Error processing peptide {idx}: {e}"
            print(f"✗ {error_msg}")
            results['errors'].append(error_msg)

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch prediction of cyclic peptide structures")

    parser.add_argument('--targets', type=str, required=True,
                       help='Path to targets.tsv file')
    parser.add_argument('--output', type=str, default='./batch_results/',
                       help='Output directory prefix')
    parser.add_argument('--model_params', type=str,
                       help='Path to fine-tuned model parameters (.pkl file)')
    parser.add_argument('--max_peptides', type=int,
                       help='Maximum number of peptides to process')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be done without actually running predictions')
    parser.add_argument('--data_dir', type=str,
                       help='Path to AlphaFold parameters directory')

    args = parser.parse_args()

    print("HighFold-MeD Batch Prediction")
    print("=" * 40)

    # Validate inputs
    if not os.path.exists(args.targets):
        print(f"Error: Targets file not found: {args.targets}")
        return 1

    if args.model_params and not os.path.exists(args.model_params):
        print(f"Warning: Model parameters file not found: {args.model_params}")
        print("Will use default AlphaFold parameters")
        args.model_params = None

    # Run batch processing
    try:
        results = process_batch_predictions(
            targets_file=args.targets,
            output_prefix=args.output,
            model_params=args.model_params,
            max_peptides=args.max_peptides,
            dry_run=args.dry_run
        )

        # Print summary
        print("\n" + "=" * 40)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 40)
        print(f"Total peptides: {results['total_peptides']}")
        print(f"Successfully processed: {results['processed']}")
        print(f"Errors: {len(results['errors'])}")

        if results['errors']:
            print("\nErrors encountered:")
            for error in results['errors']:
                print(f"  - {error}")

        if results['output_files']:
            print(f"\nOutput files created in: {args.output}")
            for output_file in results['output_files'][:5]:  # Show first 5
                print(f"  - {output_file}")
            if len(results['output_files']) > 5:
                print(f"  ... and {len(results['output_files']) - 5} more files")

        # Save summary report
        summary_file = os.path.join(args.output, "batch_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("HighFold-MeD Batch Prediction Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Targets file: {args.targets}\n")
            f.write(f"Output directory: {args.output}\n")
            f.write(f"Model parameters: {args.model_params or 'Default AlphaFold'}\n\n")
            f.write(f"Total peptides: {results['total_peptides']}\n")
            f.write(f"Successfully processed: {results['processed']}\n")
            f.write(f"Errors: {len(results['errors'])}\n\n")

            if results['errors']:
                f.write("Errors encountered:\n")
                for error in results['errors']:
                    f.write(f"  - {error}\n")

            if results['output_files']:
                f.write(f"\nOutput files:\n")
                for output_file in results['output_files']:
                    f.write(f"  - {output_file}\n")

        print(f"\nDetailed summary saved to: {summary_file}")

        if args.dry_run:
            print("\nNote: This was a dry run. Use --dry_run=false to execute actual predictions.")

        return 0

    except Exception as e:
        print(f"Batch processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())