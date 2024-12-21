#!/bin/bash

# Check if a username was provided as an argument
if [ -z "$1" ]; then
  echo "Usage: ./monitor_progress.sh <username> [keyword]"
  exit 1
fi

# Get the username from the first argument
USERNAME=$1

# Check for an optional second argument (keyword to filter progress)
KEYWORD=$2

# Get the list of running jobs for the user and grab the first job ID
JOBID=$(squeue -u $USERNAME -t RUNNING --noheader | awk 'NR==1{print $1}')

# Check if we found a job ID
if [ -z "$JOBID" ]; then
  echo "No running jobs found for user: $USERNAME"
  exit 1
fi

echo "Found running job ID: $JOBID"

# Use the find command to search for the error file in the user's home or working directory
# Adjust the base path below to match the directory structure where Slurm writes the logs
BASE_PATH="./"  # Adjust this path

ERROR_FILE=$(find $BASE_PATH -type f -name "error_${JOBID}.txt" 2>/dev/null)

# Check if we found the error file
if [ -z "$ERROR_FILE" ]; then
  echo "Error file for job ID ${JOBID} not found in ${BASE_PATH}"
  exit 1
fi

echo "Found error file: $ERROR_FILE"

# If a keyword is provided, tail the error file and filter using grep
if [ -n "$KEYWORD" ]; then
  echo "Tailing the error log with keyword filter: $KEYWORD"
  tail -f $ERROR_FILE | grep --color=always "$KEYWORD"
else
  # Otherwise, just tail the error file normally
  echo "Tailing the error log: $ERROR_FILE"
  tail -f $ERROR_FILE
fi
