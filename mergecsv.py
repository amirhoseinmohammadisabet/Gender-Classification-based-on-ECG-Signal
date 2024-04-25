
import pandas as pd

# List of CSV files
files = ["ecg-0-300.csv", "ecg-300-500.csv", "ecg-500-700.csv", "ecg-700-900.csv", "ecg-900-1120.csv"]

# List to hold DataFrames
dfs = []

# Read CSV files into pandas DataFrames
for file in files:
    dfs.append(pd.read_csv(file))

# Merge DataFrames side by side
merged_df = pd.concat(dfs, axis=1)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv("merged_file.csv", index=False)

print("Merged file saved successfully.")
