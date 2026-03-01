# MCP Integration Test Prompts

## Test Information
- **Server Name**: highfold-med-tools
- **Test Date**: 2025-01-01
- **Environment**: /home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/highfold_med_mcp/env

## Tool Discovery Tests

### Prompt 1: List All Tools
**Test**: "What MCP tools are available from highfold-med-tools? Give me a brief description of each tool."

**Expected**: Should list 14 tools including job management, submit tools, sync tools, and utilities.

### Prompt 2: Tool Details
**Test**: "Explain how to use the submit_structure_prediction tool, including all parameters and what it does."

**Expected**: Detailed description of structure prediction submission tool.

## Synchronous Tool Tests

### Prompt 3: Check Dependencies
**Test**: "Use the check_dependencies tool to verify the environment setup."

**Expected**: Should return status report of available packages and dependencies.

### Prompt 4: List Example Data
**Test**: "Use list_example_data to show what test files are available."

**Expected**: Should list files in examples/data with sequences, alignments, structures.

### Prompt 5: Validate Targets File
**Test**: "Use validate_targets_file to check the format of 'examples/data/sequences/targets.tsv'."

**Expected**: Should validate file format and show statistics about peptide sequences.

### Prompt 6: Get Peptide Info
**Test**: "Use get_peptide_info to examine peptide at index 0 in 'examples/data/sequences/targets.tsv'."

**Expected**: Should return detailed info about the first peptide in the targets file.

### Prompt 7: Create Demo File
**Test**: "Use create_demo_targets_file to create a test file 'tests/demo_targets.tsv' with 3 peptides."

**Expected**: Should create a new targets file with synthetic cyclic peptide sequences.

### Prompt 8: Error Handling
**Test**: "Use validate_targets_file with a non-existent file 'does_not_exist.tsv'."

**Expected**: Should return error message about file not found.

## Submit API Tests (Job Management)

### Prompt 9: Submit Structure Prediction
**Test**: "Use submit_structure_prediction with input_file='examples/data/sequences/targets.tsv', index=0, demo_mode=True."

**Expected**: Should return job_id for tracking.

### Prompt 10: Check Job Status
**Test**: "Use get_job_status with the job_id from the previous test."

**Expected**: Should return job status (pending, running, completed, etc.).

### Prompt 11: List All Jobs
**Test**: "Use list_jobs to show all submitted jobs."

**Expected**: Should list jobs with their status and metadata.

### Prompt 12: Get Job Logs
**Test**: "Use get_job_log with a job_id to see the last 20 lines of logs."

**Expected**: Should return log entries from the job execution.

### Prompt 13: Submit Batch Prediction
**Test**: "Use submit_batch_prediction with input_file='examples/data/sequences/targets.tsv', max_peptides=2, demo_mode=True."

**Expected**: Should return job_id for batch processing.

### Prompt 14: Cancel Job (if running)
**Test**: "Use cancel_job with a running job_id."

**Expected**: Should successfully cancel the job or return appropriate error.

## End-to-End Scenarios

### Prompt 15: Full Workflow
**Test**: "I want to:
1. Check if my environment is ready for HighFold-MeD
2. Create a demo targets file with 2 peptides
3. Validate that demo file
4. Submit a structure prediction for the first peptide
5. Check the status of that job

Please execute these steps in sequence."

**Expected**: Should execute all steps and provide a complete workflow demonstration.

### Prompt 16: Error Recovery
**Test**: "Submit a structure prediction job with an invalid targets file path, then show how to properly handle the error and resubmit correctly."

**Expected**: Should demonstrate error handling and recovery workflow.

### Prompt 17: Multiple Job Management
**Test**: "Submit 2 different jobs (structure prediction and batch prediction), then list all jobs and show their current status."

**Expected**: Should demonstrate managing multiple concurrent jobs.

## Performance and Reliability Tests

### Prompt 18: Rapid Fire Tool Calls
**Test**: "Call check_dependencies, list_example_data, and validate_targets_file in rapid succession."

**Expected**: All sync tools should respond quickly without interference.

### Prompt 19: Large Batch Processing
**Test**: "Submit a batch prediction for all peptides in the targets file (demo_mode=True)."

**Expected**: Should handle the job submission regardless of file size.

### Prompt 20: Edge Cases
**Test**: "Try to get status for a non-existent job_id 'fake_job_123'."

**Expected**: Should return appropriate error message about job not found.

## Expected Tool Categories

The server should provide these tool categories:

**Job Management (5 tools)**:
- get_job_status
- get_job_result
- get_job_log
- cancel_job
- list_jobs

**Submit API (4 tools)**:
- submit_structure_prediction
- submit_structure_relaxation
- submit_model_finetuning
- submit_batch_prediction

**Sync Tools (4 tools)**:
- validate_targets_file
- get_peptide_info
- check_dependencies
- list_example_data

**Utilities (1 tool)**:
- create_demo_targets_file

**Total: 14 tools**

## Test Success Criteria

- [ ] All tools are discoverable
- [ ] Sync tools respond in < 30 seconds
- [ ] Submit tools return valid job_ids
- [ ] Job management workflow works
- [ ] Error handling is informative
- [ ] File validation works correctly
- [ ] Demo mode prevents long-running computations
- [ ] Batch processing accepts multiple inputs
- [ ] Environment checks pass
- [ ] Example data is accessible