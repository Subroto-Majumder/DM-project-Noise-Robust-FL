#!/bin/bash

# Parse command line arguments
python_args=()
folder=""
num_rounds=""
strategy=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --folder)
            folder="$2"
            shift 2
            ;;
        --num_rounds)
            num_rounds="$2"
            python_args+=("$1" "$2")
            shift 2
            ;;
        --strategy)
            strategy="$2"
            python_args+=("$1" "$2")
            shift 2
            ;;
        *)
            python_args+=("$1")
            shift
            ;;
    esac
done

# Check required arguments
if [ -z "$folder" ]; then
    echo "Error: --folder argument is required"
    exit 1
fi

if [ -z "$num_rounds" ]; then
    echo "Error: --num_rounds argument is required"
    exit 1
fi

# Create output directory
mkdir -p "$folder"

# Determine strategy prefix
if [[ "$strategy" == "basic_strategy" ]]; then
    prefix="BS"
else
    prefix="CS"  # Default to CS for any other strategy
fi

# Build filename
filename="${prefix}_${num_rounds}.txt"

# Run the Python program with all arguments
python Approach2_final.py "${python_args[@]}" > "$folder/$filename" 2>&1

# Process output to keep only SUMMARY section
sed -n '/\[SUMMARY\]/,$p' "$folder/$filename" > "$folder/tmpfile" && mv "$folder/tmpfile" "$folder/$filename"

# Check if file is empty after processing
if [ ! -s "$folder/$filename" ]; then
    echo "Warning: Output file is empty after processing. The Python script might not have produced valid output."
fi

# Run post-processing scripts
(
    cd "$folder" || exit
    python3 csv_creator.py --max 90
    python3 plotter.py
)
