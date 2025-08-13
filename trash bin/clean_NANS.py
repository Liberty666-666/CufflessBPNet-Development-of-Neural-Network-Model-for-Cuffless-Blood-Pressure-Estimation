import pandas as pd

# Paths to your data files
ppg_path = "./data/ppg_train.csv"
bp_path = "./data/bp_labels_train.csv"

# Load data
ppg_df = pd.read_csv(ppg_path)
bp_df = pd.read_csv(bp_path)

# === Clean PPG data ===
nan_count_before = ppg_df.isna().sum().sum()
print(f"NaNs in PPG before cleaning: {nan_count_before}")

# Fill NaNs by forward fill then backward fill
ppg_df.fillna(method='ffill', inplace=True)
ppg_df.fillna(method='bfill', inplace=True)

nan_count_after = ppg_df.isna().sum().sum()
print(f"NaNs in PPG after cleaning: {nan_count_after}")

# === Clean BP labels ===
neg_dbp_count = (bp_df['DBP'] < 0).sum()
print(f"Negative DBP values before cleaning: {neg_dbp_count}")

# Replace negative DBP with median DBP
median_dbp = bp_df.loc[bp_df['DBP'] >= 0, 'DBP'].median()
bp_df.loc[bp_df['DBP'] < 0, 'DBP'] = median_dbp

neg_dbp_count_after = (bp_df['DBP'] < 0).sum()
print(f"Negative DBP values after cleaning: {neg_dbp_count_after}")

# === Save cleaned data ===
ppg_df.to_csv(ppg_path, index=False)
bp_df.to_csv(bp_path, index=False)

print("✅ Data cleaning complete and files saved.")
