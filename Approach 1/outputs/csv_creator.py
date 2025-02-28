#!/usr/bin/env python3
import os
import re
import csv
import glob
import argparse

def parse_file(file_path, max_round):
    """
    Parses the given text file to extract:
      - Loss values from the "History (loss, distributed):" section.
      - Accuracy values from the "History (metrics, distributed, evaluate):" section.
      
    Only rounds <= max_round (if provided) are returned.
    
    Returns two dictionaries:
      loss_data: {round_number: loss_value}
      acc_data: {round_number: accuracy_value}
    """
    loss_data = {}
    acc_data = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse loss values
    in_loss_section = False
    for line in lines:
        if "History (loss, distributed):" in line:
            in_loss_section = True
            continue
        # Once in the loss section, look for lines with a "round" value.
        if in_loss_section:
            # Stop if we hit an empty line or a new section marker (optional safeguard)
            if not line.strip() or "History (" in line:
                continue
            # Use regex to capture round number and loss value
            match = re.search(r'round\s+(\d+):\s+([\d\.eE+-]+)', line)
            if match:
                round_num = int(match.group(1))
                if max_round is not None and round_num > max_round:
                    continue
                loss_data[round_num] = match.group(2)
    
    # Parse accuracy values
    in_acc_section = False
    acc_text = ""
    for line in lines:
        if "History (metrics, distributed, evaluate):" in line:
            in_acc_section = True
            continue
        if in_acc_section:
            # Accumulate lines that are part of the dictionary
            # (Assumes the dictionary block is contiguous)
            acc_text += line.strip() + " "
    
    # Use regex to extract tuples of the form: (round, np.float64(value))
    acc_matches = re.findall(r'\((\d+),\s*np\.float64\(([\d\.eE+-]+)\)\)', acc_text)
    for m in acc_matches:
        round_num = int(m[0])
        if max_round is not None and round_num > max_round:
            continue
        acc_data[round_num] = m[1]
    
    return loss_data, acc_data

def main():
    parser = argparse.ArgumentParser(
        description="Convert multiple txt files into loss.csv and accuracy.csv."
    )
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum round number to include (if not set, uses max found).")
    args = parser.parse_args()

    # Get all text files in the current directory
    txt_files = glob.glob("*.txt")
    if not txt_files:
        print("No .txt files found in the current directory.")
        return

    all_loss = {}  # filename -> {round: loss_value}
    all_acc  = {}  # filename -> {round: accuracy_value}

    for file in txt_files:
        loss_data, acc_data = parse_file(file, args.max)
        all_loss[file] = loss_data
        all_acc[file]  = acc_data

    # Determine maximum round number for each CSV.
    # If a user-defined max is provided, use it; otherwise use the max round found.
    if args.max is not None:
        max_round_loss = args.max
        max_round_acc = args.max
    else:
        max_round_loss = max((max(data.keys()) for data in all_loss.values() if data), default=0)
        max_round_acc  = max((max(data.keys()) for data in all_acc.values() if data), default=0)

    # Write loss.csv
    with open("loss.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header: first column is "Round", then one column per filename
        header = ["Round"] + list(all_loss.keys())
        writer.writerow(header)
        for r in range(1, max_round_loss + 1):
            row = [r]
            for file, data in all_loss.items():
                row.append(data.get(r, ""))  # leave empty if that round is missing
            writer.writerow(row)

    # Write accuracy.csv
    with open("accuracy.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["Round"] + list(all_acc.keys())
        writer.writerow(header)
        for r in range(1, max_round_acc + 1):
            row = [r]
            for file, data in all_acc.items():
                row.append(data.get(r, ""))
            writer.writerow(row)

    print("CSV files 'loss.csv' and 'accuracy.csv' have been created.")

if __name__ == "__main__":
    main()
