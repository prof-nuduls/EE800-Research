#!/bin/bash
#SBATCH --job-name=yolo_to_voc_conversion
#SBATCH --output=%x_%j.out   # Output file name, with job name and job ID
#SBATCH --error=%x_%j.err    # Error file name
#SBATCH --nodes=1           # Number of nodes to use
#SBATCH --ntasks=1           # Number of tasks
#SBATCH --cpus-per-task=8   # Request 40 CPU cores
#SBATCH --mem=64G            # Memory required (adjust as needed)
#SBATCH --time=02:00:00      # Time limit (e.g., 2 hours)
#SBATCH --partition=compute  # Specify the partition (change as needed)


# Run the Python script
python conver_parallel.py
