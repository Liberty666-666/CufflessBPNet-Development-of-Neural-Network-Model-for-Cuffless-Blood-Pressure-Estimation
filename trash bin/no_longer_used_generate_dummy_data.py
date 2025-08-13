import numpy as np
import pandas as pd
import os

# Create the 'data' directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Configuration
num_samples = 100       # number of data samples
seq_len = 500           # length of each PPG sequence

# Generate dummy PPG data: shape (100, 500), values between 0 and 1
ppg_data = np.random.rand(num_samples, seq_len).astype(np.float32)

# Generate dummy blood pressure labels
sbp = np.random.uniform(90, 140, size=(num_samples, 1))   # Systolic
dbp = np.random.uniform(60, 100, size=(num_samples, 1))   # Diastolic

# Combine SBP and DBP into one array: shape (100, 2)
labels = np.concatenate([sbp, dbp], axis=1).astype(np.float32)

# Convert to pandas DataFrame
ppg_df = pd.DataFrame(ppg_data)
labels_df = pd.DataFrame(labels, columns=["SBP", "DBP"])

# Save as CSV (no index column)
ppg_df.to_csv("data/ppg_train.csv", index=False)
labels_df.to_csv("data/bp_labels_train.csv", index=False)

print(" Dummy PPG data and BP labels saved as CSV successfully in /data!")
