import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(results_dir):
    # Paths to CSV files
    true_csv = os.path.join(results_dir, "testset_true.csv")
    pred_csv = os.path.join(results_dir, "predictions.csv")

    # Load data
    true_df = pd.read_csv(true_csv)
    pred_df = pd.read_csv(pred_csv)

    # Merge for alignment
    merged_df = pd.concat([true_df, pred_df], axis=1)

    # Plot SBP
    plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    plt.plot(merged_df["SBP_true"], label="SBP True", color="red")
    plt.plot(merged_df["SBP_pred"], label="SBP Predicted", color="blue", linestyle="--")
    plt.title("Systolic Blood Pressure (SBP): True vs Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("SBP (mmHg)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Plot DBP
    plt.subplot(2, 1, 2)
    plt.plot(merged_df["DBP_true"], label="DBP True", color="green")
    plt.plot(merged_df["DBP_pred"], label="DBP Predicted", color="orange", linestyle="--")
    plt.title("Diastolic Blood Pressure (DBP): True vs Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("DBP (mmHg)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(results_dir, "bp_true_vs_pred.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Plot saved at: {save_path}")

if __name__ == "__main__":
    results_root = "./results"
    # Pick the most recently modified subfolder
    latest_subdir = max(
        [os.path.join(results_root, d) for d in os.listdir(results_root)],
        key=os.path.getmtime
    )
    print(f"📂 Using results from: {latest_subdir}")
    plot_results(latest_subdir)

