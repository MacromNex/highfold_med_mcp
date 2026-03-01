#!/usr/bin/env python3
"""Simple test for job manager functionality."""

import sys
from pathlib import Path
import tempfile
import time

# Add paths
sys.path.insert(0, 'src')

def test_basic_job():
    """Test basic job functionality."""
    try:
        from jobs.manager import job_manager

        # Test listing empty jobs
        result = job_manager.list_jobs()
        print(f"✓ Empty job list: {result['total']} jobs")

        # Create demo script content (simulate a simple Python script)
        demo_script = """#!/usr/bin/env python3
import time
import sys
print("Demo job starting...")
time.sleep(2)
print("Demo job completed!")
sys.exit(0)
"""

        # Write demo script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(demo_script)
            script_path = f.name

        # Make it executable
        import os
        os.chmod(script_path, 0o755)

        try:
            # Submit job
            submit_result = job_manager.submit_job(
                script_path=script_path,
                args={},
                job_name="test_demo_job"
            )

            print(f"✓ Job submitted: {submit_result['job_id']}")

            # Wait and check status
            job_id = submit_result['job_id']

            for i in range(10):  # 10 second timeout
                status = job_manager.get_job_status(job_id)
                print(f"  Status: {status['status']}")

                if status['status'] in ['completed', 'failed']:
                    break
                time.sleep(1)

            # Get final result
            if status['status'] == 'completed':
                print("✓ Job completed successfully")
                return True
            else:
                print(f"✗ Job ended with status: {status['status']}")
                return False

        finally:
            # Clean up
            os.unlink(script_path)

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Job Manager")
    print("=" * 20)
    success = test_basic_job()
    print("=" * 20)
    print("✓ Test passed!" if success else "✗ Test failed!")