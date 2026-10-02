import glob
import os

import pandas as pd

from config import DATA_PATH, RAW_DIR

print("--- Merge process started ---")

files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
print("Files found:", len(files))

if not files:
    raise SystemExit(f"[-] No CSV files found in {RAW_DIR}.")

if os.path.exists(DATA_PATH):
    os.remove(DATA_PATH)

# Process files one by one to keep memory use low
for i, file in enumerate(files):
    print(f"Processing {file}")
    df = pd.read_csv(file, low_memory=False)
    df.to_csv(DATA_PATH, mode="w" if i == 0 else "a", header=(i == 0), index=False)

print("[+] Dataset merged successfully!")
