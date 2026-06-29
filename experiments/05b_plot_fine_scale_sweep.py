# experiments/05b_plot_fine_scale_sweep.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    dataset_name = "MNIST"
    seed = 42
    calib_size = 1024
    shift_val = 4
    bias_mode = "round"

    out_dir = Path("outputs") / dataset_name.lower() / f"seed{seed}_calib{calib_size}"
    sweep_dir = out_dir / "scale_sweep"

    csv_path = sweep_dir / f"global_fine_sweep_shift{shift_val}_{bias_mode}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"먼저 python experiments/04_sweep_global_scale.py 를 실행하세요."
        )

    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------
    # Plot 1: W_SCALE vs INT8 accuracy, grouped by X_SCALE
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for x_scale, group in df.groupby("x_scale"):
        group = group.sort_values("w_scale")
        plt.plot(
            group["w_scale"],
            group["int8_acc"],
            marker="o",
            label=f"X={int(x_scale)}",
        )

    plt.xlabel("W_SCALE")
    plt.ylabel("INT8 Test Accuracy (%)")
    plt.title("Global Scale Sweep: W_SCALE vs INT8 Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    line_path = sweep_dir / f"plot_fine_wscale_vs_acc_shift{shift_val}_{bias_mode}.png"
    plt.savefig(line_path, dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # Plot 2: X_SCALE-W_SCALE heatmap
    # ------------------------------------------------------------
    pivot = df.pivot(index="x_scale", columns="w_scale", values="int8_acc")
    pivot = pivot.sort_index(ascending=True)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
    )

    plt.xticks(range(len(pivot.columns)), [str(int(v)) for v in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(int(v)) for v in pivot.index])

    plt.xlabel("W_SCALE")
    plt.ylabel("X_SCALE")
    plt.title("Global Scale Sweep Heatmap: INT8 Accuracy (%)")
    plt.colorbar(label="INT8 Test Accuracy (%)")
    plt.tight_layout()

    heatmap_path = sweep_dir / f"fine_heatmap_xscale_wscale_acc_shift{shift_val}_{bias_mode}.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()

    # ------------------------------------------------------------
    # Plot 3: saturation vs accuracy
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df["fc3_score_sat_rate"],
        df["int8_acc"],
    )

    plt.xlabel("FC3 Score Saturation Rate")
    plt.ylabel("INT8 Test Accuracy (%)")
    plt.title("FC3 Saturation vs INT8 Accuracy")
    plt.grid(True)
    plt.tight_layout()

    sat_path = sweep_dir / f"fine_scatter_fc3sat_vs_acc_shift{shift_val}_{bias_mode}.png"
    plt.savefig(sat_path, dpi=150)
    plt.close()

    print("Saved plots")
    print(f"  {line_path}")
    print(f"  {heatmap_path}")
    print(f"  {sat_path}")


if __name__ == "__main__":
    main()