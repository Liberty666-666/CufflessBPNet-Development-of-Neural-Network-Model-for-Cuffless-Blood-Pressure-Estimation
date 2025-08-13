import pandas as pd

# === File paths ===
ppg_path = "./data/ppg_train.csv"
bp_path = "./data/bp_labels_train.csv"

# === Load data ===
ppg_df = pd.read_csv(ppg_path)
bp_df = pd.read_csv(bp_path)

# === Step 1: Initial checks ===
print(" Initial Data Check")
print(f"→ Total NaNs in PPG: {ppg_df.isnull().sum().sum()}")
print(f"→ Total NaNs in BP labels: {bp_df.isnull().sum().sum()}")
print("\n PPG Summary:")
print(ppg_df.describe())
print("\n BP Labels Summary:")
print(bp_df.describe())

# === Step 2: Remove rows with NaNs in PPG, and sync delete from BP ===
print("\n Removing rows with NaNs from PPG and syncing BP labels...")

# Identify rows with any NaNs
nan_indices = ppg_df[ppg_df.isnull().any(axis=1)].index
print(f"→ Number of rows with NaNs to remove: {len(nan_indices)}")

# Drop the identified rows in both DataFrames
ppg_df = ppg_df.drop(index=nan_indices).reset_index(drop=True)
bp_df = bp_df.drop(index=nan_indices).reset_index(drop=True)

print(f"→ Remaining rows in cleaned PPG: {ppg_df.shape[0]}")
print(f"→ Remaining rows in cleaned BP: {bp_df.shape[0]}")

# === Step 3: Clean BP Labels (focus on DBP) ===
print("\n Validating BP Labels (DBP)...")

neg_dbp_count = (bp_df['DBP'] < 0).sum()
print(f"→ Negative DBP entries before cleaning: {neg_dbp_count}")

# Replace negative DBP values with median of valid DBP
median_dbp = bp_df.loc[bp_df['DBP'] >= 0, 'DBP'].median()
bp_df.loc[bp_df['DBP'] < 0, 'DBP'] = median_dbp

print(f"→ Negative DBP entries after cleaning: {(bp_df['DBP'] < 0).sum()}")

# === Step 4: Save cleaned files ===
ppg_df.to_csv(ppg_path, index=False)
bp_df.to_csv(bp_path, index=False)
print("\n Cleaned data saved to disk.")
