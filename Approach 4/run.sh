#!/bin/bash


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


strategy="reputation" 
malicious_ratio=""



for ((i=0; i<${#python_args[@]}; i++)); do
    arg="${python_args[i]}"
    # Check if the next element exists before accessing it
    next_i=$((i + 1))
    if [[ $next_i -lt ${#python_args[@]} ]]; then
        next_arg="${python_args[next_i]}"
        case "$arg" in
            --strategy)
                strategy="$next_arg"
                ;;
            --malicious_ratio)
                malicious_ratio="$next_arg"
                ;;
        esac
    fi
done

# Build filename
filename="${strategy}" # Start with strategy name

if [ -n "$malicious_ratio" ]; then


    mal_value_percent=$(awk -v mr="$malicious_ratio" 'BEGIN { printf "%d", mr * 100 }')
    filename+="_mal${mal_value_percent}" # e.g., _mal30, _mal60
fi

filename+=".txt" 

echo "Output filename will be: $folder/$filename"

python_script_name="main.py"
echo "Running: python3 $python_script_name ${python_args[@]}"
python3 "$python_script_name" "${python_args[@]}" > "$folder/$filename" 2>&1




echo "Extracting summary from $folder/$filename"
sed -n '/\[SUMMARY\]/,$p' "$folder/$filename" > "$folder/tmpfile"
# Check if tmpfile has content before moving
if [ -s "$folder/tmpfile" ]; then
    mv "$folder/tmpfile" "$folder/$filename"
else
    echo "Warning: Could not find '[SUMMARY]' section in output. Keeping full log."
    rm "$folder/tmpfile" # Remove empty tmpfile
fi


      
# Run csv_creator and plotter in the target folder
echo "Running helper scripts in $folder"
(
    cd "$folder" || { echo "Error: Could not cd into $folder"; exit 1; }

    echo "Running csv_creator.py..."
    python3 ./csv_creator.py
    echo "Running plotter.py..."
    python3 ./plotter.py
)

echo "Workflow finished."

    