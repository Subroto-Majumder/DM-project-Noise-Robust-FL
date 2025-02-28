#!/bin/bash

# Parse command line arguments
python_args=()
folder=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --folder)
            folder="$2"
            shift 2
            ;;
        *)
            python_args+=("$1")
            shift
            ;;
    esac
done

# Check if folder is provided
if [ -z "$folder" ]; then
    echo "Error: --folder argument is required"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$folder"

# Extract relevant parameters for filename
loss="ce"
symmetric_noise=""
asymmetric_noise=""

for ((i=0; i<${#python_args[@]}; i++)); do
    arg="${python_args[i]}"
    case "$arg" in
        --loss)
            loss="${python_args[i+1]}"
            ;;
        --symmetric_noise)
            symmetric_noise="${python_args[i+1]}"
            ;;
        --asymmetric_noise)
            asymmetric_noise="${python_args[i+1]}"
            ;;
    esac
done

# Build filename
filename="${loss}"

if [ -n "$symmetric_noise" ]; then
    sym_value=$(awk -v sn="$symmetric_noise" 'BEGIN { printf "%d", sn * 100 }')
    filename+="_sym${sym_value}"
fi

if [ -n "$asymmetric_noise" ]; then
    asym_value=$(awk -v an="$asymmetric_noise" 'BEGIN { printf "%d", an * 100 }')
    filename+="_asym${asym_value}"
fi

filename+=".txt"

# Run the Python program and capture output
python approach1_final.py "${python_args[@]}" > "$folder/$filename" 2>&1

# Process the output file to keep only SUMMARY section
sed -n '/\[SUMMARY\]/,$p' "$folder/$filename" > "$folder/tmpfile"
mv "$folder/tmpfile" "$folder/$filename"

# Run csv_creator and plotter in the target folder
(
    cd "$folder" || exit
    python3 csv_creator.py --max 90
    python3 plotter.py
)
