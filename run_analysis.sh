#!/bin/bash

# Create result directories
mkdir -p magagent_result_3
mkdir -p miniswe_result_3
mkdir -p taubench_result_3

# Function to process files in a directory
process_directory() {
    local input_dir=$1
    local output_dir=$2
    
    echo "Processing files in $input_dir..."
    
    # Find all .jsonl files in the directory
    for input_file in "$input_dir"/*.jsonl; do
        # Check if any files exist
        if [ ! -f "$input_file" ]; then
            echo "No .jsonl files found in $input_dir"
            break
        fi
        
        # Get the basename without extension
        filename=$(basename "$input_file" .jsonl)
        
        # Define output path
        output_file="$output_dir/${filename}.png"
        
        echo "  Processing: $input_file -> $output_file"
        
        # Run the analysis script
        python3 prefix_analysis.py -i "$input_file" -o "$output_file"
        
        # Check if the command succeeded
        if [ $? -eq 0 ]; then
            echo "  ✓ Success: $filename"
        else
            echo "  ✗ Failed: $filename"
        fi
        
        echo ""
    done
}

# Process each directory
echo "Starting batch analysis..."
echo "================================"
echo ""

process_directory "magagent" "magagent_result_3"
process_directory "miniswe" "miniswe_result_3"
process_directory "taubench" "taubench_result_3"

echo "================================"
echo "Batch analysis complete!"

