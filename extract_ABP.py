import os
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# === CONFIGURATION ===
RECORD_NAMES = input("Enter record names separated by commas: ").split(",")
RECORD_PATH = "/home/acky/S-D-Mamba-main/dataset_for_training/"
OUTPUT_DIR = "/home/acky/S-D-Mamba-main/data"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
SEGMENT_DURATION_SEC = 8
SAMPLE_RATE = 125  # Hz
window_size = int(SEGMENT_DURATION_SEC * SAMPLE_RATE)
peak_distance = int(0.5 * SAMPLE_RATE)

# === PREPARE OUTPUT CONTAINERS ===
ppg_segments = []
abp_segments = []

# === CREATE PLOT DIRECTORY ===
os.makedirs(PLOT_DIR, exist_ok=True)

# === PROCESS EACH RECORD ===
segment_count = 0
for record_name in RECORD_NAMES:
    record_name = record_name.strip()
    record_file = os.path.join(RECORD_PATH, record_name)

    try:
        record = wfdb.rdrecord(record_file)
    except Exception as e:
        print(f" Could not load record {record_name}: {e}")
        continue

    try:
        ppg = record.p_signal[:, record.sig_name.index("PLETH")]
        abp = record.p_signal[:, record.sig_name.index("ABP")]
    except ValueError as e:
        print(f" Required signals not found in {record_name}: {e}")
        continue

    print(f" Loaded: {record_name} | PPG len: {len(ppg)} | ABP len: {len(abp)}")

    # === SLIDING WINDOW EXTRACTION ===
    for start in range(0, len(abp) - window_size, window_size):
        end = start + window_size
        abp_window = abp[start:end]
        ppg_window = ppg[start:end]

        if np.isnan(abp_window).any() or np.isnan(ppg_window).any():
            continue  # skip if there's missing data

        abp_segments.append(abp_window)
        ppg_segments.append(ppg_window)

        # === DETECT PEAKS AND TROUGHS ===
        sbp_peaks, _ = find_peaks(abp_window, distance=peak_distance, prominence=10)
        dbp_troughs, _ = find_peaks(-abp_window, distance=peak_distance, prominence=10)

        # === TIME AXIS ===
        time_axis = np.linspace(0, SEGMENT_DURATION_SEC, window_size)

        # === PLOT AND SAVE DIAGRAM ===
        fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

        # ABP plot
        axs[0].plot(time_axis, abp_window, color='red', label='ABP')

        # Mark and annotate SBP peaks
        axs[0].plot(time_axis[sbp_peaks], abp_window[sbp_peaks], 'go', label='SBP Peaks')
        for i in sbp_peaks:
            axs[0].annotate(f"{abp_window[i]:.1f}", (time_axis[i], abp_window[i]),
                            textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8, color='green')

        # Mark and annotate DBP troughs
        axs[0].plot(time_axis[dbp_troughs], abp_window[dbp_troughs], 'mo', label='DBP Troughs')
        for i in dbp_troughs:
            axs[0].annotate(f"{abp_window[i]:.1f}", (time_axis[i], abp_window[i]),
                            textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8, color='magenta')

        axs[0].set_title(f"ABP Segment #{segment_count} ({record_name})")
        axs[0].set_ylabel("ABP")
        axs[0].legend()

        # PPG plot
        axs[1].plot(time_axis, ppg_window, color='blue')
        axs[1].set_title("PPG Segment")
        axs[1].set_ylabel("PPG")
        axs[1].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"segment_{segment_count}_{record_name}.png"))
        plt.close()
        segment_count += 1

print(f"\n Total segments extracted: {len(ppg_segments)}")

# === SAVE TO CSV ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
ppg_df = pd.DataFrame(ppg_segments)
abp_df = pd.DataFrame(abp_segments)

ppg_df.to_csv(os.path.join(OUTPUT_DIR, "ppg_train.csv"), index=False)
abp_df.to_csv(os.path.join(OUTPUT_DIR, "abp_labels_train.csv"), index=False)

print(" Files saved:")
print(" - ppg_train.csv")
print(" - abp_labels_train.csv")
print(f" - {segment_count} waveform plots saved in: {PLOT_DIR}")
