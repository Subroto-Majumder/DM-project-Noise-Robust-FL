#!/usr/bin/env python3
import os
import re
import csv
import glob
import argparse

def parse_file(file_path, max_round):
    """
    Parses the given text file (containing SUMMARY output) to extract:
      - Centralized Loss values from "History (loss, centralized):"
      - Centralized Accuracy values from "History (metrics, centralized):"

    Only rounds <= max_round (if provided) are returned.

    Returns two dictionaries:
      loss_data: {round_number: loss_value}
      acc_data: {round_number: accuracy_value}
    """
    loss_data = {}
    acc_data = {}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Warning: File not found {file_path}")
        return loss_data, acc_data # Return empty if file doesn't exist

    # --- MODIFIED: Parse centralized loss values ---
    in_loss_section = False
    for line in lines:
        line_strip = line.strip()
        # Use a more specific marker including the preceding tab/spaces
        if "History (loss, centralized):" in line_strip:
            in_loss_section = True
            continue
        # Once in the loss section, look for lines with "round X:" pattern
        if in_loss_section:
            # Stop if we hit an empty line or a new section marker
            if not line_strip or "History (" in line_strip or "[SUMMARY]" in line_strip:
                # Check if we were actually in the section before breaking
                if "round" not in line_strip and ":" not in line_strip: # Avoid breaking on the section header itself
                     in_loss_section = False # Exit section
                continue # Continue processing lines within section

            # Use regex to capture round number and loss value
            # Allows for potential INFO:flwr: prefix and whitespace variations
            match = re.search(r'round\s+(\d+):\s+([\d\.eE+-]+)', line_strip)
            if match:
                round_num = int(match.group(1))
                # Apply max_round limit if specified
                if max_round is not None and round_num > max_round:
                    continue
                loss_data[round_num] = match.group(2)
            # else: print(f"Debug Loss Skip: {line_strip}") # Uncomment to debug non-matching lines
    # --- END MODIFIED LOSS PARSING ---


    # --- MODIFIED: Parse centralized accuracy values ---
    in_acc_section = False
    acc_dict_str = "" # Accumulate the string representation of the dict
    for line in lines:
        line_strip = line.strip()
        # Use a more specific marker including the preceding tab/spaces
        if "History (metrics, centralized):" in line_strip:
            in_acc_section = True
            continue
        if in_acc_section:
            # Accumulate lines that are part of the dictionary content
            # Stop when we hit a line that doesn't look like dict content or end marker
            if line_strip == "" or "[SUMMARY]" in line_strip or "History (" in line_strip:
                 # Check if we have accumulated anything before breaking
                 if len(acc_dict_str) > 0:
                     in_acc_section = False # Exit section
                 continue # Continue processing lines within section

            # Append line content (remove potential logging prefixes)
            line_content = re.sub(r"^(INFO|ERROR)\s*:flwr:\s*", "", line).strip()
            line_content = re.sub(r"^(INFO|ERROR)\s*:\s*", "", line_content).strip()
            acc_dict_str += line_content


    # Now parse the accumulated string to find accuracy tuples
    # Find the part starting with 'accuracy': [ ... ]
    acc_list_match = re.search(r"'accuracy':\s*\[([^\]]*)\]", acc_dict_str)
    if acc_list_match:
        acc_list_str = acc_list_match.group(1)
        # Find all (round, value) tuples within that list string
        # Handles potential whitespace variations
        acc_tuples = re.findall(r'\(\s*(\d+)\s*,\s*([\d\.eE+-]+)\s*\)', acc_list_str)
        for t in acc_tuples:
            round_num = int(t[0])
            # Apply max_round limit if specified
            if max_round is not None and round_num > max_round:
                continue
            acc_data[round_num] = t[1]
    # else: print(f"Debug Acc Skip: Could not find accuracy list in {acc_dict_str}") # Uncomment for debugging

    # --- END MODIFIED ACCURACY PARSING ---

    return loss_data, acc_data

def main():
    parser = argparse.ArgumentParser(
        description="Convert multiple txt files in current directory into loss.csv and accuracy.csv."
    )
    # Removed --max argument, assuming it's handled externally or not needed for simplicity
    # If needed, add it back: parser.add_argument("--max", type=int, default=None, help="Maximum round number")
    # args = parser.parse_args()
    max_round_limit = None # Set max round limit here if needed, e.g., max_round_limit = 90
    # Or get it from an environment variable if set by run.sh

    # Get all text files in the current directory
    txt_files = glob.glob("*.txt")
    if not txt_files:
        print("No .txt files found in the current directory.")
        return

    print(f"Found files: {txt_files}")

    all_loss = {}  # filename -> {round: loss_value}
    all_acc  = {}  # filename -> {round: accuracy_value}

    for file in txt_files:
        print(f"Parsing {file}...")
        loss_data, acc_data = parse_file(file, max_round_limit) # Pass max_round_limit
        all_loss[file] = loss_data
        all_acc[file]  = acc_data

    # Determine maximum round number actually found across all files
    # Important: Ensure keys are integers before finding max
    max_r_loss = 0
    for data in all_loss.values():
         if data: # Check if dict is not empty
             max_r_loss = max(max_r_loss, max(k for k in data.keys() if isinstance(k, int)))

    max_r_acc = 0
    for data in all_acc.values():
         if data:
             max_r_acc = max(max_r_acc, max(k for k in data.keys() if isinstance(k, int)))

    print(f"Max round found for loss: {max_r_loss}")
    print(f"Max round found for accuracy: {max_r_acc}")

    # Write loss.csv
    if max_r_loss > 0:
        print("Writing loss.csv...")
        with open("loss.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Use filenames (without .txt) as headers
            file_headers = [os.path.splitext(f)[0] for f in all_loss.keys()]
            header = ["Round"] + file_headers
            writer.writerow(header)
            # Iterate from round 1 up to the maximum round found
            for r in range(1, max_r_loss + 1):
                row = [r]
                for file in all_loss.keys(): # Ensure consistent order
                    row.append(all_loss[file].get(r, ""))  # Use .get() for missing rounds
                writer.writerow(row)
        print("loss.csv created.")
    else:
        print("No loss data found, skipping loss.csv.")


    # Write accuracy.csv
    if max_r_acc > 0:
        print("Writing accuracy.csv...")
        with open("accuracy.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            file_headers = [os.path.splitext(f)[0] for f in all_acc.keys()]
            header = ["Round"] + file_headers
            writer.writerow(header)
            for r in range(1, max_r_acc + 1):
                row = [r]
                for file in all_acc.keys(): # Ensure consistent order
                    row.append(all_acc[file].get(r, ""))
                writer.writerow(row)
        print("accuracy.csv created.")
    else:
        print("No accuracy data found, skipping accuracy.csv.")


if __name__ == "__main__":
    main()