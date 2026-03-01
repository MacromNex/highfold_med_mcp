#!/usr/bin/env python3
"""Test script for job submission."""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')

def test_job_submission():
    """Test job submission with a simple example."""
    print("Testing job submission...")

    try:
        from src.jobs.manager import job_manager
        import tempfile
        import os

        # Create a temporary demo targets file
        demo_content = """targetid	target_chainseq	templates_alignfile	description
demo_peptide_1	ACDEFGHIK	alignments/demo_peptide_1.tsv	Test cyclic peptide
demo_peptide_2	dLPRGH	alignments/demo_peptide_2.tsv	Test with D-amino acid
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write(demo_content)
            targets_file = f.name

        try:
            # Submit a demo structure prediction job
            script_path = str(Path("scripts/predict_structure.py").resolve())

            result = job_manager.submit_job(
                script_path=script_path,
                args={
                    "input": targets_file,
                    "index": 0,
                    "demo_mode": True
                },
                job_name="test_prediction"
            )

            print(f"  Submission status: {result['status']}")
            print(f"  Job ID: {result['job_id']}")

            # Wait a moment and check status
            import time
            time.sleep(2)

            status = job_manager.get_job_status(result['job_id'])
            print(f"  Job status: {status['status']}")

            # Wait for completion (with timeout)
            for i in range(10):  # 10 second timeout
                status = job_manager.get_job_status(result['job_id'])
                if status['status'] in ['completed', 'failed']:
                    break
                time.sleep(1)

            print(f"  Final status: {status['status']}")

            if status['status'] == 'completed':
                result_data = job_manager.get_job_result(result['job_id'])
                print(f"  Result status: {result_data['status']}")
                if 'result_files' in result_data:
                    print(f"  Output files: {len(result_data['result_files'])}")

            return True

        finally:
            # Clean up
            os.unlink(targets_file)

    except Exception as e:
        print(f"✗ Job submission error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run job submission test."""
    print("HighFold-MeD Job Submission Test")
    print("=" * 40)

    success = test_job_submission()

    print("\n" + "=" * 40)
    if success:
        print("✓ Job submission test passed!")
        return 0
    else:
        print("✗ Job submission test failed!")
        return 1

if __name__ == "__main__":
    exit(main())