#!/usr/bin/env python3
"""Manual test script to verify tools work correctly.

This script imports the server module and calls the tool functions directly
to verify they work as expected before testing through MCP.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / "src"))

def test_tool_functions():
    """Test tool functions by importing them directly."""
    print("🧪 Testing HighFold-MeD MCP Tools Manually")
    print(f"📅 Test Date: {datetime.now().isoformat()}")

    results = []

    # Import all the tool functions from server module
    from server import (
        check_dependencies, list_example_data, validate_targets_file,
        get_peptide_info, create_demo_targets_file, get_job_status,
        get_job_result, get_job_log, cancel_job, list_jobs,
        submit_structure_prediction, submit_structure_relaxation,
        submit_model_finetuning, submit_batch_prediction
    )

    def run_test(name, func, *args, **kwargs):
        """Run a test and capture results."""
        try:
            print(f"\n🧪 Testing {name}...")
            result = func(*args, **kwargs)

            # Convert result to string for display
            if isinstance(result, dict):
                status = result.get("status", "unknown")
                if status in ["success", "ready", "partial"]:
                    print(f"  ✓ PASSED - {status}")
                    return {"test": name, "status": "passed", "result": result}
                elif status == "error":
                    print(f"  ✗ FAILED - {result.get('error', 'Unknown error')}")
                    return {"test": name, "status": "failed", "error": result.get('error')}
                else:
                    print(f"  ? UNCLEAR - {status}")
                    return {"test": name, "status": "unclear", "result": result}
            else:
                print(f"  ✓ PASSED - returned {type(result)}")
                return {"test": name, "status": "passed", "result": result}

        except Exception as e:
            print(f"  ✗ ERROR - {str(e)}")
            return {"test": name, "status": "error", "error": str(e)}

    # Test 1: Check Dependencies
    results.append(run_test("check_dependencies", check_dependencies))

    # Test 2: List Example Data
    results.append(run_test("list_example_data", list_example_data))

    # Test 3: Validate targets file (should work)
    targets_file = "examples/data/sequences/targets.tsv"
    results.append(run_test("validate_targets_file_valid",
                          validate_targets_file, targets_file))

    # Test 4: Validate non-existent file (should return error)
    results.append(run_test("validate_targets_file_invalid",
                          validate_targets_file, "does_not_exist.tsv"))

    # Test 5: Get peptide info
    results.append(run_test("get_peptide_info",
                          get_peptide_info, targets_file, 0))

    # Test 6: Create demo targets file
    results.append(run_test("create_demo_targets_file",
                          create_demo_targets_file,
                          "tests/demo_targets.tsv", 3))

    # Test 7: List jobs (should work even if empty)
    results.append(run_test("list_jobs", list_jobs))

    # Test 8: Submit structure prediction
    results.append(run_test("submit_structure_prediction",
                          submit_structure_prediction,
                          targets_file, 0, "model_2_ptm",
                          "examples/data/alignments", True, "manual_test"))

    # Test 9: Get invalid job status (should return error)
    results.append(run_test("get_job_status_invalid",
                          get_job_status, "fake_job_123"))

    # Generate summary
    print(f"\n📊 MANUAL TEST SUMMARY")
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "passed")
    failed = sum(1 for r in results if r["status"] == "failed")
    errors = sum(1 for r in results if r["status"] == "error")

    print(f"   Total: {total}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Errors: {errors}")
    print(f"   Pass Rate: {passed/total*100:.1f}%")

    # Show any issues
    issues = [r for r in results if r["status"] in ["failed", "error"]]
    if issues:
        print(f"\n❌ ISSUES:")
        for issue in issues:
            print(f"   - {issue['test']}: {issue.get('error', 'Unknown issue')}")

    # Save results
    report_file = script_dir.parent / "reports" / "manual_test_results.json"
    report_file.parent.mkdir(exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "summary": {"total": total, "passed": passed, "failed": failed, "errors": errors},
            "results": results
        }, f, indent=2, default=str)

    print(f"\n📝 Results saved to: {report_file}")

    return passed == total

if __name__ == "__main__":
    success = test_tool_functions()
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed.'}")
    sys.exit(0 if success else 1)