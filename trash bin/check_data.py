import pandas as pd

# Load data
ppg = pd.read_csv("./data/ppg_train.csv")
labels = pd.read_csv("./data/bp_labels_train.csv")

# Count total NaNs
print(f"\n🔍 Total NaNs in PPG: {ppg.isnull().sum().sum()}")
print(f"🔍 Total NaNs in BP Labels: {labels.isnull().sum().sum()}")

# --- Show detailed NaN positions in PPG ---
print("\n📍 NaN Positions in PPG Data (row index and column name):")
ppg_nan_positions = ppg.isnull()
if ppg_nan_positions.values.any():
    for row_idx, row in ppg_nan_positions.iterrows():
        for col_name, is_nan in row.items():
            if is_nan:
                print(f"PPG - Row {row_idx}, Column '{col_name}'")
else:
    print("✅ No NaNs found in PPG data.")

# --- Show detailed NaN positions in BP Labels ---
print("\n📍 NaN Positions in BP Label Data (row index and column name):")
labels_nan_positions = labels.isnull()
if labels_nan_positions.values.any():
    for row_idx, row in labels_nan_positions.iterrows():
        for col_name, is_nan in row.items():
            if is_nan:
                print(f"BP Labels - Row {row_idx}, Column '{col_name}'")
else:
    print("✅ No NaNs found in BP label data.")

# --- Descriptive Stats ---
print("\n📊 PPG Data Description:")
print(ppg.describe())

print("\n📊 BP Labels Description:")
print(labels.describe())
