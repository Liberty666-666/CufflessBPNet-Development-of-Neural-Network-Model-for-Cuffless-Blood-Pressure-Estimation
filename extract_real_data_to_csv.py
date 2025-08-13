import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import find_peaks

# === CONFIGURATION ===
RECORD_PATH = "/home/acky/S-D-Mamba-main/dataset_for_training/"
OUTPUT_DIR = "/home/acky/S-D-Mamba-main/data"
SEGMENT_DURATION_SEC = 8
PEAK_MIN_DISTANCE_SEC = 0.5
SAMPLE_RATE = 125  # MIMIC waveform default
window_size = int(SEGMENT_DURATION_SEC * SAMPLE_RATE)
peak_distance = int(PEAK_MIN_DISTANCE_SEC * SAMPLE_RATE)

# === USER INPUT: Record Names ===
record_names_input = input("Enter record names separated by commas: ")
record_names = [name.strip() for name in record_names_input.split(",")]

# === OUTPUT CONTAINERS ===
ppg_segments = []
sbp_list = []
dbp_list = []

# === PROCESS EACH RECORD ===
for record_name in record_names:
    print(f"\n Processing record: {record_name}")
    try:
        record = wfdb.rdrecord(os.path.join(RECORD_PATH, record_name))
        ppg = record.p_signal[:, record.sig_name.index("PLETH")]
        abp = record.p_signal[:, record.sig_name.index("ABP")]
    except Exception as e:
        print(f" Failed to load {record_name}: {e}")
        continue

    # Sliding window over the ABP and PPG signals
    for start in range(0, len(abp) - window_size, window_size):
        end = start + window_size
        abp_window = abp[start:end]
        ppg_window = ppg[start:end]

        # Detect peaks (SBP) and troughs (DBP)
        sbp_peaks, _ = find_peaks(abp_window, distance=peak_distance, prominence=10)
        dbp_troughs, _ = find_peaks(-abp_window, distance=peak_distance, prominence=10)

        # Make sure there's at least one valid beat
        if len(sbp_peaks) < 1 or len(dbp_troughs) < 1:
            continue

        # === More accurate SBP/DBP: mean of all valid peaks/troughs ===
        sbp = np.mean(abp_window[sbp_peaks])
        dbp = np.mean(abp_window[dbp_troughs])

        # Append segment and BP values
        ppg_segments.append(ppg_window)
        sbp_list.append(sbp)
        dbp_list.append(dbp)

    print(f" Segments added from {record_name}: {len(sbp_list)} (Total: {len(ppg_segments)})")

# === SAVE COMBINED DATA ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

ppg_df = pd.DataFrame(ppg_segments)
bp_df = pd.DataFrame({"SBP": sbp_list, "DBP": dbp_list})

ppg_df.to_csv(os.path.join(OUTPUT_DIR, "ppg_train.csv"), index=False)
bp_df.to_csv(os.path.join(OUTPUT_DIR, "bp_labels_train.csv"), index=False)

print("\n All valid segments processed and saved.")
print(" -  ppg_train.csv")
print(" -  bp_labels_train.csv")
print(bp_df.head())
