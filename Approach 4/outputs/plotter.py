import pandas as pd
import matplotlib.pyplot as plt
import os
import re # Import regex for potentially more flexible parsing

# Set global font size (optional)
plt.rcParams.update({'font.size': 14})

ACCURACY_CSV_FILE = "accuracy.csv"

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
groups = {}
plot_titles = {}

for col in df.columns:
    if col == "Round":
        continue

    # Extract the condition part from the column name
    # Assuming format like "strategy_condition1_condition2..."
    # Example: "reputation_mal60", "fedavg_mal60" -> condition key "mal60"
    # Example: "reputation_mal30", "fedavg_mal30" -> condition key "mal30"
    match = re.match(r"^(?P<strategy>\w+)_(?P<condition>.+)$", col)
    if match:
        strategy = match.group("strategy")
        condition_key = match.group("condition") # e.g., "mal60", "mal30"

        # Group by the condition key
        groups.setdefault(condition_key, []).append(col)

        # Create a nice title for the plot based on the condition
        # You might want to parse the condition_key further for a better title
        condition_title = condition_key.replace("mal", "Malicious Ratio ") + "%" # Example title generation
        plot_titles[condition_key] = f"Accuracy Comparison ({condition_title})"

    else:
        print(f"Warning: Could not parse condition from column '{col}'. Skipping.")

# --- END MODIFIED GROUPING ---


# Plot one graph for each condition group.
if not groups:
    print("No data columns found to plot (excluding 'Round').")
else:
    print(f"Plotting conditions: {list(groups.keys())}")

for condition_key, cols in groups.items():
    plt.figure(figsize=(12, 7))
    plot_successful = False
    # Sort columns to potentially have a consistent legend order (e.g., fedavg then reputation)
    cols.sort()
    for col in cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col].dropna()):
             # Plot using the full column name as the label (e.g., "fedavg_mal60", "reputation_mal60")
             plt.plot(df["Round"], df[col].astype(float), label=col, linewidth=2.5)
             plot_successful = True
        else:
             print(f"Warning: Column '{col}' in group '{condition_key}' skipped (missing or non-numeric).")

    if plot_successful:
        plt.xlabel("Round", fontsize=16)
        plt.ylabel("Accuracy", fontsize=16)
        # Use the generated title for the condition
        plt.title(plot_titles.get(condition_key, f"Accuracy ({condition_key})"), fontsize=20)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1)
        plt.tight_layout()
        # Save the plot using the condition key in the filename
        plot_filename = f"accuracy_{condition_key}_comparison.png"
        plt.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}")
        plt.close() # Close figure
    else:
        print(f"Skipped plotting for condition '{condition_key}' as no valid columns were found.")