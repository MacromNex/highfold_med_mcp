#!/usr/bin/env python3
"""Automated integration test runner for HighFold-MeD MCP server.

This script tests the MCP server integration by calling tools directly
and verifying their responses. It simulates what Claude Code would do
when interacting with the server.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add src directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / "src"))

class MCPIntegrationTester:
    def __init__(self, server_path: str):
        self.server_path = Path(server_path)
        self.results = {
            "test_date": datetime.now().isoformat(),
            "server_path": str(server_path),
            "python_path": sys.executable,
            "tests": {},
            "issues": [],
            "summary": {}
        }

        # Import the server
        try:
            from server import mcp
            self.mcp_server = mcp
            print(f"✓ Successfully imported MCP server from {server_path}")
        except Exception as e:
            print(f"✗ Failed to import MCP server: {e}")
            raise

    def run_test(self, test_name: str, test_func) -> Dict[str, Any]:
        """Run a single test and record results."""
        print(f"\n🧪 Running {test_name}...")

        try:
            start_time = datetime.now()
            result = test_func()
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            if result.get("status") in ["success", "ready", "partial"]:
                print(f"✓ {test_name} - PASSED ({duration:.2f}s)")
                status = "passed"
            elif result.get("status") == "error":
                print(f"✗ {test_name} - FAILED: {result.get('error', 'Unknown error')}")
                status = "failed"
                self.issues.append({
                    "test": test_name,
                    "error": result.get("error", "Unknown error"),
                    "details": result
                })
            else:
                print(f"? {test_name} - UNCLEAR: {result}")
                status = "unclear"

            return {
                "status": status,
                "duration": duration,
                "result": result,
                "timestamp": start_time.isoformat()
            }

        except Exception as e:
            print(f"✗ {test_name} - ERROR: {str(e)}")
            self.issues.append({
                "test": test_name,
                "error": str(e),
                "exception": type(e).__name__
            })
            return {
                "status": "error",
                "error": str(e),
                "exception": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            }

    def test_server_import(self) -> Dict[str, Any]:
        """Test that the server imports correctly."""
        try:
            # Already tested in __init__
            return {"status": "success", "message": "Server imported successfully"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def test_check_dependencies(self) -> Dict[str, Any]:
        """Test the check_dependencies tool."""
        return self.mcp_server.call_tool("check_dependencies", {})

    def test_list_example_data(self) -> Dict[str, Any]:
        """Test the list_example_data tool."""
        return self.mcp_server.call_tool("list_example_data", {})

    def test_validate_targets_file(self) -> Dict[str, Any]:
        """Test the validate_targets_file tool."""
        targets_file = "examples/data/sequences/targets.tsv"
        return self.mcp_server.call_tool("validate_targets_file", {"targets_file": targets_file})

    def test_validate_nonexistent_file(self) -> Dict[str, Any]:
        """Test error handling with non-existent file."""
        result = self.mcp_server.call_tool("validate_targets_file", {"targets_file": "does_not_exist.tsv"})
        # This should return an error, which is the expected behavior
        if result.get("status") == "error":
            return {"status": "success", "message": "Correctly handled non-existent file", "original": result}
        else:
            return {"status": "error", "error": "Should have returned error for non-existent file"}

    def test_get_peptide_info(self) -> Dict[str, Any]:
        """Test the get_peptide_info tool."""
        targets_file = "examples/data/sequences/targets.tsv"
        return self.mcp_server.call_tool("get_peptide_info", {"targets_file": targets_file, "index": 0})

    def test_create_demo_targets_file(self) -> Dict[str, Any]:
        """Test creating a demo targets file."""
        output_file = "tests/demo_targets.tsv"
        return self.mcp_server.call_tool("create_demo_targets_file", {
            "output_file": output_file,
            "num_peptides": 3
        })

    def test_submit_structure_prediction(self) -> Dict[str, Any]:
        """Test submitting a structure prediction job."""
        targets_file = "examples/data/sequences/targets.tsv"
        result = self.mcp_server.call_tool("submit_structure_prediction", {
            "input_file": targets_file,
            "index": 0,
            "demo_mode": True,
            "job_name": "integration_test"
        })

        # Store job_id for later tests
        if result.get("status") == "submitted" and "job_id" in result:
            self.test_job_id = result["job_id"]

        return result

    def test_list_jobs(self) -> Dict[str, Any]:
        """Test listing all jobs."""
        return self.mcp_server.call_tool("list_jobs", {})

    def test_get_job_status(self) -> Dict[str, Any]:
        """Test getting job status."""
        if not hasattr(self, 'test_job_id'):
            return {"status": "skipped", "message": "No job_id available from previous tests"}

        return self.mcp_server.call_tool("get_job_status", {"job_id": self.test_job_id})

    def test_get_job_log(self) -> Dict[str, Any]:
        """Test getting job logs."""
        if not hasattr(self, 'test_job_id'):
            return {"status": "skipped", "message": "No job_id available from previous tests"}

        return self.mcp_server.call_tool("get_job_log", {"job_id": self.test_job_id, "tail": 10})

    def test_invalid_job_id(self) -> Dict[str, Any]:
        """Test error handling with invalid job ID."""
        result = self.mcp_server.call_tool("get_job_status", {"job_id": "fake_job_123"})
        # This should return an error, which is the expected behavior
        if result.get("status") == "error":
            return {"status": "success", "message": "Correctly handled invalid job_id", "original": result}
        else:
            return {"status": "error", "error": "Should have returned error for invalid job_id"}

    def run_all_tests(self):
        """Run all integration tests."""
        print("🚀 Starting HighFold-MeD MCP Integration Tests")
        print(f"📅 Test Date: {self.results['test_date']}")
        print(f"🐍 Python: {self.results['python_path']}")
        print(f"🔧 Server: {self.results['server_path']}")

        # Define test suite
        test_suite = [
            ("server_import", self.test_server_import),
            ("check_dependencies", self.test_check_dependencies),
            ("list_example_data", self.test_list_example_data),
            ("validate_targets_file", self.test_validate_targets_file),
            ("validate_nonexistent_file", self.test_validate_nonexistent_file),
            ("get_peptide_info", self.test_get_peptide_info),
            ("create_demo_targets_file", self.test_create_demo_targets_file),
            ("submit_structure_prediction", self.test_submit_structure_prediction),
            ("list_jobs", self.test_list_jobs),
            ("get_job_status", self.test_get_job_status),
            ("get_job_log", self.test_get_job_log),
            ("invalid_job_id", self.test_invalid_job_id),
        ]

        # Run tests
        for test_name, test_func in test_suite:
            test_result = self.run_test(test_name, test_func)
            self.results["tests"][test_name] = test_result

        # Generate summary
        self.generate_summary()

        return self.results

    def generate_summary(self):
        """Generate test summary statistics."""
        total = len(self.results["tests"])
        passed = sum(1 for t in self.results["tests"].values()
                    if t.get("status") == "passed")
        failed = sum(1 for t in self.results["tests"].values()
                    if t.get("status") == "failed")
        errors = sum(1 for t in self.results["tests"].values()
                    if t.get("status") == "error")
        skipped = sum(1 for t in self.results["tests"].values()
                     if t.get("status") == "skipped")

        self.results["summary"] = {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "pass_rate": f"{passed/total*100:.1f}%" if total > 0 else "0%",
            "issues_count": len(self.issues)
        }

        self.results["issues"] = self.issues

        # Print summary
        print(f"\n📊 TEST SUMMARY")
        print(f"   Total: {total}")
        print(f"   Passed: {passed}")
        print(f"   Failed: {failed}")
        print(f"   Errors: {errors}")
        print(f"   Skipped: {skipped}")
        print(f"   Pass Rate: {self.results['summary']['pass_rate']}")

        if self.issues:
            print(f"\n❌ ISSUES FOUND:")
            for issue in self.issues:
                print(f"   - {issue['test']}: {issue['error']}")

    def save_report(self, output_file: str):
        """Save test results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n📝 Test report saved to: {output_file}")

def main():
    """Main test runner."""
    script_dir = Path(__file__).parent
    server_path = script_dir.parent / "src" / "server.py"

    if not server_path.exists():
        print(f"❌ Server not found at: {server_path}")
        sys.exit(1)

    # Run tests
    tester = MCPIntegrationTester(server_path)
    results = tester.run_all_tests()

    # Save report
    report_file = script_dir.parent / "reports" / "step7_integration_test_results.json"
    report_file.parent.mkdir(exist_ok=True)
    tester.save_report(report_file)

    # Exit with appropriate code
    if results["summary"]["failed"] > 0 or results["summary"]["errors"] > 0:
        print("\n❌ Some tests failed. Review the report for details.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()