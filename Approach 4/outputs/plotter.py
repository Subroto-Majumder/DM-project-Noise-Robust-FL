import pandas as pd
import matplotlib.pyplot as plt
import os
import re # Import regex for flexible parsing

# Set global font size (optional)
plt.rcParams.update({'font.size': 14})

ACCURACY_CSV_FILE = "accuracy.csv" # This file now contains Server Eval Accuracy

# Check if the CSV file exists
if not os.path.exists(ACCURACY_CSV_FILE):
    print(f"Error: {ACCURACY_CSV_FILE} not found. Please run csv_creator.py first.")
    exit()

# Read the CSV file
try:
    df = pd.read_csv(ACCURACY_CSV_FILE)
except pd.errors.EmptyDataError:
    print(f"Error: {ACCURACY_CSV_FILE} is empty. No data to plot.")
    exit()
except Exception as e:
    print(f"Error reading {ACCURACY_CSV_FILE}: {e}")
    exit()

if "Round" not in df.columns:
    print("Error: 'Round' column missing in accuracy.csv")
    exit()

# --- MODIFIED: Group columns by experimental condition (e.g., malicious ratio) ---
# This grouping logic should work as before, assuming filenames like reputation_malX.txt
groups = {}
plot_titles = {}

# Sort columns to potentially process them in a more logical order if needed later
# Exclude 'Round' column from sorting for grouping purposes
data_columns = sorted([col for col in df.columns if col != "Round"])

for col in data_columns:
    # Extract the condition part from the column name
    # Assuming format like "strategy_condition1_condition2..." derived from filename
    # Example: "reputation_mal60", "fedavg_mal60" -> condition key "mal60"
    # Example: "reputation_mal30", "fedavg_mal30" -> condition key "mal30"
    # Example: "fedavg" (no condition) -> condition key "fedavg" (or handle as needed)
    match = re.match(r"^(?P<strategy>\w+)_(?P<condition>.+)$", col)
    if match:
        strategy = match.group("strategy")
        # Condition key is everything after the first underscore
        condition_key = match.group("condition") # e.g., "mal60", "mal30"

        # Group by the condition key
        groups.setdefault(condition_key, []).append(col)

        # Create a nice title for the plot based on the condition
        # Example title generation: "mal60" -> "Malicious Ratio 60%"
        try:
            mal_percent = int(re.search(r'mal(\d+)', condition_key).group(1))
            condition_title = f"Malicious Ratio {mal_percent}%"
        except (AttributeError, ValueError):
            # Handle cases where condition_key might not be 'malXX'
            condition_title = condition_key # Fallback to using the key itself

        plot_titles[condition_key] = f"Accuracy Comparison ({condition_title})"

    else:
        # Handle columns that might not have an underscore (e.g., just "fedavg")
        # Group them under their own name or a default key
        condition_key = col # Use the column name itself as the key
        groups.setdefault(condition_key, []).append(col)
        plot_titles[condition_key] = f"Accuracy ({condition_key})"
        print(f"Info: Column '{col}' treated as its own condition group.")


# --- END MODIFIED GROUPING ---


# Plot one graph for each condition group.
if not groups:
    print("No data columns found to plot (excluding 'Round').")
else:
    print(f"Plotting conditions: {list(groups.keys())}")

for condition_key, cols in groups.items():
    plt.figure(figsize=(12, 7))
    plot_successful = False
    # Sort columns within the group for a consistent legend order (e.g., fedavg then reputation)
    cols.sort()
    for col in cols:
        if col in df.columns:
             # Attempt to convert to numeric, coercing errors to NaN, then drop NaN for check
             numeric_col = pd.to_numeric(df[col], errors='coerce').dropna()
             if not numeric_col.empty:
                 # Plot using the full column name as the label (e.g., "fedavg_mal60", "reputation_mal60")
                 plt.plot(df["Round"], pd.to_numeric(df[col], errors='coerce'), label=col, linewidth=2.5) # Coerce errors during plot
                 plot_successful = True
             else:
                 print(f"Warning: Column '{col}' in group '{condition_key}' contains no numeric data after coercion. Skipping plot.")
        else:
             # This case should be rare if df.columns was used initially
             print(f"Warning: Column '{col}' defined in group '{condition_key}' not found in DataFrame. Skipping.")


    if plot_successful:
        plt.xlabel("Round", fontsize=16)
        plt.ylabel("Server Eval Accuracy", fontsize=16) # Updated Y-axis label
        # Use the generated title for the condition
        plt.title(plot_titles.get(condition_key, f"Accuracy ({condition_key})"), fontsize=20)
        plt.legend(fontsize=12, loc='best') # Adjust legend location automatically
        plt.grid(True, linestyle='--', alpha=0.7)
        # Adjust ylim based on data, but ensure it's within [0, 1] for accuracy
        min_val = df[cols].apply(pd.to_numeric, errors='coerce').min().min()
        max_val = df[cols].apply(pd.to_numeric, errors='coerce').max().max()
        plt.ylim(max(0, min_val - 0.05), min(1, max_val + 0.05)) # Dynamic Y limits with buffer, capped at [0,1]
        # If max_val is very low, maybe keep ylim 0-1? Optional.
        # plt.ylim(0, 1) # Keep fixed 0-1 scale if preferred

        plt.tight_layout()
        # Save the plot using the condition key in the filename
        plot_filename = f"accuracy_{condition_key}_comparison.png"
        plt.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}")
        plt.close() # Close figure
    else:
        print(f"Skipped plotting for condition '{condition_key}' as no valid numeric columns were found.")

