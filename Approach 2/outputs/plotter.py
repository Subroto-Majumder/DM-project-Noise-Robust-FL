import pandas as pd
import matplotlib.pyplot as plt

# Set global font size (optional)
plt.rcParams.update({'font.size': 14})

# Read the CSV file
df = pd.read_csv("accuracy.csv")

# Group columns by the "sym" value in their header.
# For example, from "robust_sym20" and "ce_sym20" the key is "sym20".
groups = {}
for col in df.columns:
    if col == "Round":
        continue
    parts = col.split('_')
    if len(parts) >= 2:
        sym_key = parts[-1]  # e.g. sym20, sym60, etc.
        groups.setdefault(sym_key, []).append(col)

# Plot one graph for each sym group.
for sym, cols in groups.items():
    plt.figure(figsize=(10, 6))
    for col in cols:
        plt.plot(df["Round"], df[col], label=col, linewidth=5)  # thicker, bolder lines
    plt.xlabel("Round", fontsize=24)
    plt.ylabel("Accuracy", fontsize=24)
    plt.title(f"Accuracy for {sym}", fontsize=30)
    plt.legend(fontsize=24)
    plt.grid(True)
    plt.ylim(0, 1)  # fixed y-axis from 0 to 1
    plt.show()
