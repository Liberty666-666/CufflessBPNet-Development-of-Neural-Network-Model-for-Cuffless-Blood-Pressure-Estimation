import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import find_peaks

# === CONFIGURATION ===
RECORD_NAME = "3000714_0001"
RECORD_PATH = "/home/acky/S-D-Mamba-main/dataset_for_training/"
OUTPUT_DIR = "/home/acky/S-D-Mamba-main/data"
SEGMENT_DURATION_SEC = 8
PEAK_MIN_DISTANCE_SEC = 0.5
SAMPLE_RATE = 125  # MIMIC-III waveform data default

# === LOAD RECORD ===
record = wfdb.rdrecord(os.path.join(RECORD_PATH, RECORD_NAME))
ppg = record.p_signal[:, record.sig_name.index("PLETH")]
abp = record.p_signal[:, record.sig_name.index("ABP")]

print(f"✅ Loaded record: {RECORD_NAME}")
print(f" - PPG length: {len(ppg)}")
print(f" - ABP length: {len(abp)}")
print(f" - Signal names: {record.sig_name}")

# === CALCULATE WINDOW SIZES ===
window_size = int(SEGMENT_DURATION_SEC * SAMPLE_RATE)
peak_distance = int(PEAK_MIN_DISTANCE_SEC * SAMPLE_RATE)

# === PREPARE OUTPUT CONTAINERS ===
ppg_segments = []
sbp_list = []
dbp_list = []

# === SLIDING WINDOW PROCESSING ===
for start in range(0, len(abp) - window_size, window_size):
    end = start + window_size
    abp_window = abp[start:end]
    ppg_window = ppg[start:end]

    # Detect peaks (SBP) and troughs (DBP)
    sbp_peaks, _ = find_peaks(abp_window, distance=peak_distance, prominence=10)
    dbp_troughs, _ = find_peaks(-abp_window, distance=peak_distance, prominence=10)

    if len(sbp_peaks) < 1 or len(dbp_troughs) < 1:
        continue

    sbp = np.median(abp_window[sbp_peaks])
    dbp = np.median(abp_window[dbp_troughs])

    ppg_segments.append(ppg_window)
    sbp_list.append(sbp)
    dbp_list.append(dbp)

print(f"✅ Processed {len(sbp_list)} valid segments.")

# === SAVE TO CSV FILES ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

ppg_df = pd.DataFrame(ppg_segments)
ppg_df.to_csv(os.path.join(OUTPUT_DIR, "ppg_train.csv"), index=False)

bp_df = pd.DataFrame({"SBP": sbp_list, "DBP": dbp_list})
bp_df.to_csv(os.path.join(OUTPUT_DIR, "bp_labels_train.csv"), index=False)

print("✅ Files saved:")
print(" - ppg_train.csv")
print(" - bp_labels_train.csv")
print(bp_df.head())
