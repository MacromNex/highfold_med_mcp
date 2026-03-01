"""
Shared utilities for cyclic peptide processing.

This module provides common functions for handling cyclic peptides with
HighFold notation, including D-amino acids, N-methylation, and terminal modifications.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd


# ==============================================================================
# Cyclic Peptide Validation and Processing
# ==============================================================================

def validate_highfold_sequence(sequence: str) -> bool:
    """
    Validate that sequence uses valid HighFold notation for cyclic peptides.

    Args:
        sequence: Peptide sequence string

    Returns:
        bool: True if valid HighFold notation

    Examples:
        >>> validate_highfold_sequence("PhdLP_d")
        True
        >>> validate_highfold_sequence("VIhFIh.")
        True
        >>> validate_highfold_sequence("INVALID@SEQ")
        False
    """
    if not sequence or not isinstance(sequence, str):
        return False

    # Check for valid characters in HighFold notation
    valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._/')
    if not all(c in valid_chars for c in sequence):
        return False

    # Basic length check
    clean_seq = sequence.replace('/', '').replace('.', '').replace('_', '')
    if len(clean_seq) < 3 or len(clean_seq) > 50:
        return False

    return True


def parse_highfold_modifications(sequence: str) -> Dict[str, Any]:
    """
    Parse HighFold sequence to identify modifications.

    Args:
        sequence: HighFold notation sequence

    Returns:
        Dict containing modification information

    Examples:
        >>> parse_highfold_modifications("PhdLP_d")
        {'d_amino_acids': ['d'], 'n_methylation': ['h'], 'terminal_mods': ['_'], 'length': 6}
    """
    info = {
        'original_sequence': sequence,
        'clean_sequence': '',
        'length': 0,
        'd_amino_acids': [],
        'n_methylation': [],
        'terminal_modifications': [],
        'special_residues': [],
        'has_modifications': False
    }

    # Remove structural separators
    working_seq = sequence.replace('/', '')

    # Find D-amino acids (lowercase d prefix)
    d_amino_matches = re.findall(r'd[A-Z]', working_seq)
    info['d_amino_acids'] = d_amino_matches

    # Find N-methylation indicators (lowercase h)
    n_methyl_matches = re.findall(r'[A-Z]h', working_seq)
    info['n_methylation'] = n_methyl_matches

    # Find terminal modifications (dots, underscores)
    if '.' in working_seq:
        info['terminal_modifications'].append('.')
    if '_' in working_seq:
        info['terminal_modifications'].append('_')

    # Clean sequence (remove modification indicators)
    clean_seq = working_seq
    for pattern in ['d', 'h', '.', '_']:
        clean_seq = clean_seq.replace(pattern, '')

    info['clean_sequence'] = clean_seq
    info['length'] = len(clean_seq)

    # Check if any modifications present
    info['has_modifications'] = (
        len(info['d_amino_acids']) > 0 or
        len(info['n_methylation']) > 0 or
        len(info['terminal_modifications']) > 0
    )

    return info


def normalize_sequence_for_prediction(sequence: str) -> str:
    """
    Normalize sequence for structure prediction by removing separators.

    Args:
        sequence: Raw HighFold sequence

    Returns:
        str: Normalized sequence for prediction
    """
    # Remove structural separators but keep modification indicators
    return sequence.replace('/', '')


# ==============================================================================
# Template and Alignment Processing
# ==============================================================================

def load_template_alignment(alignment_file: Path) -> List[Dict[str, Any]]:
    """
    Load and parse template alignment file.

    Args:
        alignment_file: Path to alignment TSV file

    Returns:
        List of template dictionaries
    """
    if not alignment_file.exists():
        return []

    try:
        df = pd.read_csv(alignment_file, sep='\t')

        templates = []
        for _, row in df.iterrows():
            template = {
                'pdb_file': row.get('template_pdbfile', 'N/A'),
                'alignment_string': row.get('target_to_template_alignstring', 'N/A'),
                'identities': float(row.get('identities', 0.0)),
                'template_length': int(row.get('template_len', 0)),
                'query_start': int(row.get('query_start', 0)),
                'query_end': int(row.get('query_end', 0)),
                'template_start': int(row.get('template_start', 0)),
                'template_end': int(row.get('template_end', 0)),
                'evalue': float(row.get('evalue', 1e10)),
                'score': float(row.get('score', 0.0))
            }
            templates.append(template)

        # Sort by score (descending) or identities (descending)
        templates.sort(key=lambda x: (x['score'], x['identities']), reverse=True)

        return templates

    except Exception as e:
        print(f"Warning: Could not parse alignment file {alignment_file}: {e}")
        return []


def find_alignment_file(alignment_file_path: str, search_dirs: List[str]) -> Optional[Path]:
    """
    Find alignment file in search directories.

    Args:
        alignment_file_path: Original alignment file path
        search_dirs: List of directories to search

    Returns:
        Path to found alignment file or None
    """
    if not alignment_file_path:
        return None

    alignment_filename = Path(alignment_file_path).name

    # Search in provided directories
    for search_dir in search_dirs:
        candidate = Path(search_dir) / alignment_filename
        if candidate.exists():
            return candidate

    # Try original path as-is
    if Path(alignment_file_path).exists():
        return Path(alignment_file_path)

    return None


# ==============================================================================
# Data File Processing
# ==============================================================================

def load_targets_file(targets_file: Path) -> pd.DataFrame:
    """
    Load and validate targets TSV file.

    Args:
        targets_file: Path to targets TSV file

    Returns:
        pandas.DataFrame: Loaded targets data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not targets_file.exists():
        raise FileNotFoundError(f"Targets file not found: {targets_file}")

    try:
        df = pd.read_csv(targets_file, sep='\t')

        # Validate required columns
        required_columns = ['target_chainseq']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate sequences
        invalid_sequences = []
        for idx, seq in enumerate(df['target_chainseq']):
            if pd.isna(seq) or not validate_highfold_sequence(str(seq)):
                invalid_sequences.append((idx, seq))

        if invalid_sequences:
            print(f"Warning: Found {len(invalid_sequences)} invalid sequences")
            for idx, seq in invalid_sequences[:3]:  # Show first 3
                print(f"  Row {idx}: '{seq}'")

        return df

    except Exception as e:
        raise ValueError(f"Error reading targets file {targets_file}: {e}")


def extract_peptide_info(targets_df: pd.DataFrame, index: int) -> Dict[str, Any]:
    """
    Extract peptide information from targets DataFrame.

    Args:
        targets_df: Loaded targets DataFrame
        index: Row index to extract

    Returns:
        Dict containing peptide information
    """
    if index >= len(targets_df):
        raise ValueError(f"Index {index} is out of range. DataFrame has {len(targets_df)} entries.")

    row = targets_df.iloc[index]

    info = {
        'index': index,
        'target_id': row.get('targetid', f'peptide_{index}'),
        'sequence': row['target_chainseq'],
        'alignment_file': row.get('templates_alignfile', ''),
        'description': row.get('description', ''),
        'raw_row': dict(row)
    }

    # Add sequence analysis
    seq_info = parse_highfold_modifications(info['sequence'])
    info.update({
        'sequence_length': seq_info['length'],
        'clean_sequence': seq_info['clean_sequence'],
        'has_modifications': seq_info['has_modifications'],
        'modifications': {
            'd_amino_acids': seq_info['d_amino_acids'],
            'n_methylation': seq_info['n_methylation'],
            'terminal_modifications': seq_info['terminal_modifications']
        }
    })

    return info


# ==============================================================================
# Output Formatting
# ==============================================================================

def format_prediction_summary(result: Dict[str, Any]) -> str:
    """
    Format prediction results as human-readable summary.

    Args:
        result: Prediction result dictionary

    Returns:
        str: Formatted summary text
    """
    summary = []
    summary.append("HighFold-MeD Prediction Summary")
    summary.append("=" * 40)
    summary.append("")

    # Basic information
    summary.append(f"Target ID: {result.get('target_id', 'N/A')}")
    summary.append(f"Sequence: {result.get('sequence', 'N/A')}")
    summary.append(f"Length: {result.get('sequence_length', result.get('query_length', 0))}")
    summary.append(f"Model: {result.get('model_name', 'N/A')}")
    summary.append(f"Status: {result.get('status', 'unknown')}")
    summary.append("")

    # Template information
    templates = result.get('template_info', [])
    summary.append(f"Templates Used: {len(templates)}")

    if templates:
        summary.append("Template Details:")
        for i, template in enumerate(templates[:5]):  # Show first 5
            summary.append(f"  Template {i+1}:")
            summary.append(f"    PDB: {template.get('pdb_file', 'N/A')}")
            summary.append(f"    Identity: {template.get('identities', 0.0):.3f}")
            summary.append(f"    Length: {template.get('template_length', template.get('template_len', 0))}")

        if len(templates) > 5:
            summary.append(f"  ... and {len(templates) - 5} more templates")

    summary.append("")

    # Modifications
    if 'modifications' in result:
        mods = result['modifications']
        if any(mods.values()):
            summary.append("Sequence Modifications:")
            if mods.get('d_amino_acids'):
                summary.append(f"  D-amino acids: {', '.join(mods['d_amino_acids'])}")
            if mods.get('n_methylation'):
                summary.append(f"  N-methylation: {', '.join(mods['n_methylation'])}")
            if mods.get('terminal_modifications'):
                summary.append(f"  Terminal mods: {', '.join(mods['terminal_modifications'])}")
        else:
            summary.append("No special modifications detected")

    return "\n".join(summary)