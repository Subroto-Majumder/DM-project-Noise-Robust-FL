#!/usr/bin/env python3
import os
import re
import csv
import glob
import argparse

def parse_file(file_path, max_round):
    """
    Parses the given text file (containing SUMMARY output) to extract:
      - Server Eval Loss values from "History (metrics, centralized): {'server_eval_loss': [...]}"
      - Server Eval Accuracy values from "History (metrics, centralized): {'server_eval_accuracy': [...]}"

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

    in_metrics_section = False
    metrics_dict_str = "" # Accumulate the string representation of the dict

    for line in lines:
        line_strip = line.strip()

        # Find the start of the centralized metrics section
        if "History (metrics, centralized):" in line_strip:
            in_metrics_section = True
            # Extract the initial part of the dict string if it's on the same line
            start_brace = line.find('{')
            if start_brace != -1:
                metrics_dict_str += line[start_brace:]
            continue # Move to the next line

        if in_metrics_section:
            # Append line content (remove potential logging prefixes and leading/trailing whitespace)
            # Handles "INFO :", "INFO:flwr:", etc.
            line_content = re.sub(r"^(INFO|ERROR)\s*:\s*flwr\s*:\s*", "", line).strip()
            line_content = re.sub(r"^(INFO|ERROR)\s*:\s*", "", line_content).strip()

            # Stop accumulating if we hit a blank line AFTER starting, or a new major section
            # or the end of the dictionary (heuristically check for '}')
            if (not line_content and metrics_dict_str) or "[SUMMARY]" in line or "History (" in line:
                 # Check if the last added content actually contained the closing brace
                 if '}' in metrics_dict_str:
                     in_metrics_section = False # Exit section
                     # Clean up potential trailing garbage after the last '}'
                     last_brace = metrics_dict_str.rfind('}')
                     if last_brace != -1:
                         metrics_dict_str = metrics_dict_str[:last_brace+1]
                     break # Stop processing lines for this section
                 else:
                     # It might be an empty line within the dict definition, keep going
                     # unless it's clearly a new section marker
                     if "[SUMMARY]" in line or "History (" in line:
                        in_metrics_section = False
                        break
                     else:
                        # Append the (potentially empty) line content if it's just whitespace formatting
                        metrics_dict_str += line_content
                        continue


            # Accumulate the relevant part of the dictionary string
            metrics_dict_str += line_content

            # Check if we've likely accumulated the entire dictionary
            if '}' in line_content:
                # Clean up potential trailing garbage after the last '}'
                last_brace = metrics_dict_str.rfind('}')
                if last_brace != -1:
                     metrics_dict_str = metrics_dict_str[:last_brace+1]
                in_metrics_section = False
                break # Stop processing lines for this section

    # --- PARSE ACCUMULATED METRICS STRING ---
    if not metrics_dict_str:
        # print(f"Debug: No metrics dictionary content found in {file_path}") # Uncomment for debugging
        return loss_data, acc_data

    # Remove newlines and extra spaces within the string for easier regex matching
    metrics_dict_str = re.sub(r'\s+', ' ', metrics_dict_str).strip()
    # print(f"Debug: Accumulated metrics string for {file_path}: {metrics_dict_str}") # Uncomment for debugging


    # --- Extract Server Eval Loss ---
    loss_list_match = re.search(r"'server_eval_loss':\s*\[([^\]]*)\]", metrics_dict_str)
    if loss_list_match:
        loss_list_str = loss_list_match.group(1)
        # Find all (round, value) tuples within that list string
        loss_tuples = re.findall(r'\(\s*(\d+)\s*,\s*([\d\.eE+-]+)\s*\)', loss_list_str)
        for t in loss_tuples:
            round_num = int(t[0])
            # Apply max_round limit if specified
            if max_round is not None and round_num > max_round:
                continue
            loss_data[round_num] = t[1]
    # else: print(f"Debug Loss Skip: Could not find 'server_eval_loss' list in {file_path}") # Uncomment for debugging


    # --- Extract Server Eval Accuracy ---
    acc_list_match = re.search(r"'server_eval_accuracy':\s*\[([^\]]*)\]", metrics_dict_str)
    if acc_list_match:
        acc_list_str = acc_list_match.group(1)
        # Find all (round, value) tuples within that list string
        acc_tuples = re.findall(r'\(\s*(\d+)\s*,\s*([\d\.eE+-]+)\s*\)', acc_list_str)
        for t in acc_tuples:
            round_num = int(t[0])
            # Apply max_round limit if specified
            if max_round is not None and round_num > max_round:
                continue
            acc_data[round_num] = t[1]
    # else: print(f"Debug Acc Skip: Could not find 'server_eval_accuracy' list in {file_path}") # Uncomment for debugging


    return loss_data, acc_data

def main():
    parser = argparse.ArgumentParser(
        description="Convert multiple txt files in current directory into loss.csv and accuracy.csv based on server_eval metrics."
    )
    # Optional: Add max round argument if needed, otherwise set it manually or via env var
    # parser.add_argument("--max", type=int, default=None, help="Maximum round number to include")
    # args = parser.parse_args()
    # max_round_limit = args.max
    max_round_limit = None # Set max round limit here if needed, e.g., max_round_limit = 90

    # Get all text files in the current directory
    txt_files = sorted(glob.glob("*.txt")) # Sort for consistent column order
    if not txt_files:
        print("No .txt files found in the current directory.")
        return

    print(f"Found files: {txt_files}")

    all_loss = {}  # filename -> {round: loss_value}
    all_acc  = {}  # filename -> {round: accuracy_value}
    filenames_in_order = [] # Keep track of file order for CSV headers

    for file in txt_files:
        print(f"Parsing {file}...")
        loss_data, acc_data = parse_file(file, max_round_limit) # Pass max_round_limit
        if loss_data or acc_data: # Only include files that yielded some data
             all_loss[file] = loss_data
             all_acc[file]  = acc_data
             filenames_in_order.append(file)
        else:
             print(f"Warning: No loss or accuracy data extracted from {file}. Skipping.")


    # Determine maximum round number actually found across all included files
    max_r_loss = 0
    for data in all_loss.values():
         if data: # Check if dict is not empty
             # Ensure keys are integers before finding max
             int_keys = [k for k in data.keys() if isinstance(k, int)]
             if int_keys:
                 max_r_loss = max(max_r_loss, max(int_keys))

    max_r_acc = 0
    for data in all_acc.values():
         if data:
             int_keys = [k for k in data.keys() if isinstance(k, int)]
             if int_keys:
                max_r_acc = max(max_r_acc, max(int_keys))

    print(f"Max round found for loss: {max_r_loss}")
    print(f"Max round found for accuracy: {max_r_acc}")

    # Write loss.csv (Server Eval Loss)
    if max_r_loss > 0 and filenames_in_order:
        print("Writing loss.csv (Server Eval Loss)...")
        with open("loss.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Use filenames (without .txt) as headers, based on the order they were processed
            file_headers = [os.path.splitext(f)[0] for f in filenames_in_order]
            header = ["Round"] + file_headers
            writer.writerow(header)
            # Iterate from round 0 up to the maximum round found (data includes round 0)
            for r in range(0, max_r_loss + 1):
                row = [r]
                for file in filenames_in_order: # Use the tracked order
                    # Use .get() for missing rounds, ensure file exists in dict
                    row.append(all_loss.get(file, {}).get(r, ""))
                writer.writerow(row)
        print("loss.csv created.")
    else:
        print("No server eval loss data found, skipping loss.csv.")


    # Write accuracy.csv (Server Eval Accuracy)
    if max_r_acc > 0 and filenames_in_order:
        print("Writing accuracy.csv (Server Eval Accuracy)...")
        with open("accuracy.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            file_headers = [os.path.splitext(f)[0] for f in filenames_in_order]
            header = ["Round"] + file_headers
            writer.writerow(header)
            # Iterate from round 0 up to the maximum round found
            for r in range(0, max_r_acc + 1):
                row = [r]
                for file in filenames_in_order: # Use the tracked order
                    row.append(all_acc.get(file, {}).get(r, ""))
                writer.writerow(row)
        print("accuracy.csv created.")
    else:
        print("No server eval accuracy data found, skipping accuracy.csv.")


if __name__ == "__main__":
    main()
