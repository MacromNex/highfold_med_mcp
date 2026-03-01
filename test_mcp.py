#!/usr/bin/env python3
"""Test script for HighFold-MeD MCP tools."""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')

def test_imports():
    """Test that all modules import correctly."""
    print("Testing imports...")

    try:
        from src.jobs.manager import job_manager
        print("✓ Job manager imports successfully")
    except Exception as e:
        print(f"✗ Job manager error: {e}")
        return False

    try:
        import predict_structure
        print("✓ predict_structure imports successfully")
    except Exception as e:
        print(f"✗ predict_structure error: {e}")
        return False

    try:
        from src.server import (
            check_dependencies, validate_targets_file,
            get_peptide_info, list_example_data,
            submit_structure_prediction, get_job_status
        )
        print("✓ MCP tools import successfully")
    except Exception as e:
        print(f"✗ MCP tools error: {e}")
        return False

    return True

def test_sync_tools():
    """Test synchronous tools."""
    print("\nTesting synchronous tools...")

    try:
        from src.server import check_dependencies, list_example_data

        # Test check_dependencies
        print("Testing check_dependencies...")
        result = check_dependencies()
        print(f"  Status: {result['status']}")
        print(f"  Demo mode available: {result['demo_mode_available']}")

        # Test list_example_data
        print("Testing list_example_data...")
        result = list_example_data()
        print(f"  Status: {result['status']}")

        return True
    except Exception as e:
        print(f"✗ Sync tools error: {e}")
        return False

def test_job_manager():
    """Test job manager functionality."""
    print("\nTesting job manager...")

    try:
        from src.jobs.manager import job_manager

        # Test list_jobs (should be empty initially)
        result = job_manager.list_jobs()
        print(f"  List jobs status: {result['status']}")
        print(f"  Total jobs: {result['total']}")

        return True
    except Exception as e:
        print(f"✗ Job manager error: {e}")
        return False

def test_create_demo_file():
    """Test creating a demo targets file."""
    print("\nTesting demo file creation...")

    try:
        from src.server import create_demo_targets_file
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.tsv', delete=False) as f:
            demo_file = f.name

        result = create_demo_targets_file(demo_file, num_peptides=3)
        print(f"  Status: {result['status']}")
        if result['status'] == 'success':
            print(f"  Created file: {result['output_file']}")
            print(f"  Peptides: {result['peptides_created']}")

            # Clean up
            os.unlink(demo_file)

        return True
    except Exception as e:
        print(f"✗ Demo file creation error: {e}")
        return False

def main():
    """Run all tests."""
    print("HighFold-MeD MCP Server Test Suite")
    print("=" * 40)

    success = True

    # Run tests
    success &= test_imports()
    success &= test_sync_tools()
    success &= test_job_manager()
    success &= test_create_demo_file()

    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())