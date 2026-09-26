import pandas as pd
import glob
import os

print("--- Merge process started ---")
print("Current directory:", os.getcwd())

# Get all CSV files
files = glob.glob("data\\raw\\*.csv")

print("Files found:", len(files))

if len(files) == 0:
    print("[-] No CSV files found. Check folder path.")
    exit()

output_file = "data/cicids.csv"

# Remove old file if exists
if os.path.exists(output_file):
    os.remove(output_file)

# Process files one by one (LOW MEMORY SAFE)
for i, file in enumerate(files):
    print(f"Processing {file}")

    try:
        df = pd.read_csv(file, low_memory=False)

        if i == 0:
            # First file -> write with header
            df.to_csv(output_file, index=False)
        else:
            # Append without header
            df.to_csv(output_file, mode='a', header=False, index=False)

    except Exception as e:
        print(f"[-] Error in {file}: {e}")

print("[+] Dataset merged successfully!")