#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start_job_id> <end_job_id>"
    exit 1
fi

# Assign start and end job IDs
START_JOB_ID=$1
END_JOB_ID=$2

# Validate that the inputs are integers
if ! [[ "$START_JOB_ID" =~ ^[0-9]+$ ]] || ! [[ "$END_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: Job IDs must be integers."
    exit 1
fi

# Cancel jobs within the specified range
for (( job_id=START_JOB_ID; job_id<=END_JOB_ID; job_id++ )); do
    echo "Cancelling job ID: $job_id"
    scancel "$job_id"
done

echo "All jobs between $START_JOB_ID and $END_JOB_ID have been cancelled."
