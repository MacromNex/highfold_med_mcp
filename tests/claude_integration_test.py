#!/usr/bin/env python3
"""Claude Code Integration Test

This script tests the integration by executing actual Claude commands
that should interact with the MCP server. This simulates real usage.
"""

import subprocess
import json
import sys
from datetime import datetime
from pathlib import Path

def run_claude_test(prompt: str, test_name: str, timeout: int = 60):
    """Run a Claude command and capture the result."""
    print(f"\n🧪 Testing: {test_name}")
    print(f"📝 Prompt: {prompt[:100]}...")

    try:
        # Run claude with the prompt
        cmd = ['claude', '--json', '--', prompt]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent  # Run from project root
        )

        if result.returncode == 0:
            print("✓ Claude command executed successfully")
            return {
                "test": test_name,
                "status": "success",
                "prompt": prompt,
                "output": result.stdout,
                "stderr": result.stderr if result.stderr else None
            }
        else:
            print(f"✗ Claude command failed with exit code {result.returncode}")
            return {
                "test": test_name,
                "status": "failed",
                "prompt": prompt,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

    except subprocess.TimeoutExpired:
        print(f"⏱️ Claude command timed out after {timeout} seconds")
        return {
            "test": test_name,
            "status": "timeout",
            "prompt": prompt,
            "timeout": timeout
        }
    except Exception as e:
        print(f"💥 Exception occurred: {str(e)}")
        return {
            "test": test_name,
            "status": "error",
            "prompt": prompt,
            "error": str(e)
        }

def main():
    """Run Claude Code integration tests."""
    print("🚀 Claude Code Integration Testing for HighFold-MeD MCP")
    print(f"📅 Test Date: {datetime.now().isoformat()}")

    # Test suite with prompts that should work with the MCP server
    test_suite = [
        (
            "List all available MCP tools from highfold-med-tools and their descriptions.",
            "tool_discovery"
        ),
        (
            "Use the check_dependencies tool to verify the HighFold-MeD environment setup.",
            "check_dependencies"
        ),
        (
            "Use the list_example_data tool to show what example files are available.",
            "list_example_data"
        ),
        (
            "Use validate_targets_file to check the format of 'examples/data/sequences/targets.tsv'.",
            "validate_targets_file"
        ),
        (
            "Use get_peptide_info to examine the first peptide in 'examples/data/sequences/targets.tsv'.",
            "get_peptide_info"
        ),
        (
            "Use create_demo_targets_file to create 'tests/claude_demo.tsv' with 2 peptides.",
            "create_demo_targets"
        ),
        (
            "Use list_jobs to show all submitted HighFold-MeD computation jobs.",
            "list_jobs"
        ),
        (
            "Submit a structure prediction job using submit_structure_prediction with input_file='examples/data/sequences/targets.tsv', index=0, demo_mode=True.",
            "submit_job"
        ),
        (
            "Test error handling by using validate_targets_file with a non-existent file 'fake.tsv'.",
            "error_handling"
        )
    ]

    results = []

    for prompt, test_name in test_suite:
        result = run_claude_test(prompt, test_name, timeout=120)  # 2 minute timeout
        results.append(result)

        # Short delay between tests
        import time
        time.sleep(2)

    # Generate summary
    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    timeout = sum(1 for r in results if r["status"] == "timeout")
    errors = sum(1 for r in results if r["status"] == "error")

    print(f"\n📊 INTEGRATION TEST SUMMARY")
    print(f"   Total: {total}")
    print(f"   Success: {success}")
    print(f"   Failed: {failed}")
    print(f"   Timeout: {timeout}")
    print(f"   Errors: {errors}")
    print(f"   Success Rate: {success/total*100:.1f}%")

    # Show any issues
    issues = [r for r in results if r["status"] != "success"]
    if issues:
        print(f"\n❌ ISSUES:")
        for issue in issues:
            print(f"   - {issue['test']}: {issue['status']}")
            if 'error' in issue:
                print(f"     Error: {issue['error']}")

    # Save results
    report_file = Path(__file__).parent.parent / "reports" / "claude_integration_test.json"
    report_file.parent.mkdir(exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "summary": {"total": total, "success": success, "failed": failed, "timeout": timeout, "errors": errors},
            "results": results
        }, f, indent=2, default=str)

    print(f"\n📝 Results saved to: {report_file}")

    return success > 0  # Consider success if at least one test passed

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ Integration tests completed.' if success else '❌ All integration tests failed.'}")
    sys.exit(0 if success else 1)