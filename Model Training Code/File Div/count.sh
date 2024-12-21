#!/bin/bash

# Directory to search (you can specify the directory, or use the current one)
DIR_PATH="/mmfs1/home/dmiller10/EE800 Research/Data/ResData/+25k/Train/images"

# Find all files in the directory and subdirectories that contain an underscore (_) in their names
underscore_file_count=$(find "$DIR_PATH" -type f -name "*_*" | wc -l)

# Find the total number of files in the directory and subdirectories
total_file_count=$(find "$DIR_PATH" -type f | wc -l)

# Calculate the number of files without underscores in their names
non_underscore_file_count=$((total_file_count - underscore_file_count))

# Print the counts
echo "Number of files with an underscore in their name: $underscore_file_count"
echo "Total number of files: $total_file_count"
echo "Number of files without an underscore: $non_underscore_file_count"
