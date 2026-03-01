"""
I/O utilities for HighFold-MeD scripts.

This module provides common functions for file handling, data loading, and output formatting.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import pandas as pd
import tempfile
import shutil


# ==============================================================================
# Configuration Management
# ==============================================================================

def load_config(config_file: Optional[Union[str, Path]], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load configuration from JSON file with fallback to defaults.

    Args:
        config_file: Path to config file (optional)
        default_config: Default configuration dictionary

    Returns:
        Dict: Merged configuration
    """
    config = default_config.copy()

    if config_file and Path(config_file).exists():
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                config.update(user_config)
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
            print("Using default configuration")

    return config


def save_config(config: Dict[str, Any], config_file: Union[str, Path]) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        config_file: Path to save config file
    """
    config_file = Path(config_file)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)


# ==============================================================================
# File and Directory Management
# ==============================================================================

def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if needed.

    Args:
        directory: Directory path

    Returns:
        Path: Resolved directory path
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_output_filename(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    suffix: str = "",
    extension: str = ".txt",
    prefix: str = ""
) -> Path:
    """
    Generate output filename based on input file.

    Args:
        input_file: Input file path
        output_dir: Output directory
        suffix: Suffix to add to filename
        extension: File extension
        prefix: Prefix to add to filename

    Returns:
        Path: Generated output file path
    """
    input_path = Path(input_file)
    output_dir = ensure_directory(output_dir)

    base_name = input_path.stem
    filename = f"{prefix}{base_name}{suffix}{extension}"

    return output_dir / filename


def backup_file(file_path: Union[str, Path]) -> Optional[Path]:
    """
    Create a backup copy of a file.

    Args:
        file_path: File to backup

    Returns:
        Path: Backup file path or None if failed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return None

    try:
        backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
        shutil.copy2(file_path, backup_path)
        return backup_path
    except Exception as e:
        print(f"Warning: Could not create backup of {file_path}: {e}")
        return None


# ==============================================================================
# Data File Operations
# ==============================================================================

def read_text_file(file_path: Union[str, Path]) -> str:
    """
    Read text file contents.

    Args:
        file_path: Path to text file

    Returns:
        str: File contents

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        raise IOError(f"Could not read file {file_path}: {e}")


def write_text_file(file_path: Union[str, Path], content: str, backup: bool = False) -> None:
    """
    Write content to text file.

    Args:
        file_path: Path to output file
        content: Content to write
        backup: Create backup if file exists

    Raises:
        IOError: If file cannot be written
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create backup if requested and file exists
    if backup and file_path.exists():
        backup_file(file_path)

    try:
        with open(file_path, 'w') as f:
            f.write(content)
    except Exception as e:
        raise IOError(f"Could not write file {file_path}: {e}")


def read_tsv_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read TSV file into DataFrame.

    Args:
        file_path: Path to TSV file

    Returns:
        pd.DataFrame: Loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be parsed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"TSV file not found: {file_path}")

    try:
        return pd.read_csv(file_path, sep='\t')
    except Exception as e:
        raise ValueError(f"Could not parse TSV file {file_path}: {e}")


def write_tsv_file(df: pd.DataFrame, file_path: Union[str, Path], backup: bool = False) -> None:
    """
    Write DataFrame to TSV file.

    Args:
        df: DataFrame to save
        file_path: Path to output file
        backup: Create backup if file exists

    Raises:
        IOError: If file cannot be written
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create backup if requested and file exists
    if backup and file_path.exists():
        backup_file(file_path)

    try:
        df.to_csv(file_path, sep='\t', index=False)
    except Exception as e:
        raise IOError(f"Could not write TSV file {file_path}: {e}")


# ==============================================================================
# Result Serialization
# ==============================================================================

def save_json_result(result: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save result dictionary as JSON file.

    Args:
        result: Result dictionary
        file_path: Path to output JSON file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert non-serializable types
    serializable_result = make_json_serializable(result)

    with open(file_path, 'w') as f:
        json.dump(serializable_result, f, indent=2)


def load_json_result(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load result dictionary from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dict: Loaded result dictionary
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, 'r') as f:
        return json.load(f)


def make_json_serializable(obj: Any) -> Any:
    """
    Convert object to JSON-serializable format.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable object
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    else:
        return obj


# ==============================================================================
# Temporary File Management
# ==============================================================================

class TemporaryDirectory:
    """
    Context manager for temporary directories with cleanup.
    """

    def __init__(self, prefix: str = "highfold_", cleanup: bool = True):
        self.prefix = prefix
        self.cleanup = cleanup
        self.path = None

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix=self.prefix))
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup and self.path and self.path.exists():
            shutil.rmtree(self.path)


def create_temp_file(suffix: str = ".tmp", content: Optional[str] = None) -> Path:
    """
    Create temporary file with optional content.

    Args:
        suffix: File suffix
        content: Optional content to write

    Returns:
        Path: Path to created temporary file
    """
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    temp_path = Path(temp_path)

    if content:
        with open(fd, 'w') as f:
            f.write(content)
    else:
        os.close(fd)

    return temp_path


# ==============================================================================
# File Validation
# ==============================================================================

def validate_input_file(file_path: Union[str, Path], expected_extensions: List[str] = None) -> None:
    """
    Validate input file exists and has expected extension.

    Args:
        file_path: Path to input file
        expected_extensions: List of expected extensions (e.g., ['.pdb', '.tsv'])

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file has wrong extension
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if expected_extensions:
        if file_path.suffix.lower() not in [ext.lower() for ext in expected_extensions]:
            raise ValueError(f"Invalid file extension. Expected: {expected_extensions}, got: {file_path.suffix}")


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        float: File size in MB
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return 0.0

    return file_path.stat().st_size / (1024 * 1024)


def check_disk_space(directory: Union[str, Path], required_mb: float) -> bool:
    """
    Check if directory has enough free disk space.

    Args:
        directory: Directory to check
        required_mb: Required space in megabytes

    Returns:
        bool: True if enough space available
    """
    directory = Path(directory)

    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    stat = shutil.disk_usage(directory)
    free_mb = stat.free / (1024 * 1024)

    return free_mb >= required_mb