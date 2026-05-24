"""Plot REINFORCE training results.

This script compares:
- REINFORCE without baseline
- REINFORCE with constant baseline = 10
- REINFORCE with constant baseline = 5

It reads CSV files from the results/ folder and saves a comparison plot.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    results_dir = Path("results")

    runs = {
        "No baseline": results_dir / "reinforce_none_raw_seed42.csv",
        "Constant baseline = 10": results_dir / "reinforce_constant10_raw_seed42.csv",
        "Constant baseline = 5": results_dir / "reinforce_constant5_raw_seed42.csv",
    }

    plt.figure(figsize=(10, 6))

    for label, csv_path in runs.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing file: {csv_path}")

        df = pd.read_csv(csv_path)

        plt.plot(
            df["episode"],
            df["moving_avg_return"],
            label=label,
        )

    plt.xlabel("Episode")
    plt.ylabel("Moving Average Return")
    plt.title("REINFORCE on Hopper-v4: Effect of Constant Baseline")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = results_dir / "reinforce_baseline_comparison.png"
    plt.savefig(output_path, dpi=300)

    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()