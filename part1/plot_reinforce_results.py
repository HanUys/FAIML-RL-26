"""Plot Part 1 Hopper-v4 policy-gradient results.

This script creates two plots:

1. REINFORCE baseline comparison over 100 episodes.
2. Long-run comparison between REINFORCE and Actor-Critic over 1000 episodes.

It reads CSV files from:
- results/reinforce/
- results/actor_critic/

and saves plots to:
- results/plots/
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_runs(runs, title, output_path):
    plt.figure(figsize=(11, 6))

    for label, csv_path in runs.items():
        if not csv_path.exists():
            print(f"Skipping missing file: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        plt.plot(
            df["episode"],
            df["moving_avg_return"],
            label=label,
        )

    plt.xlabel("Episode")
    plt.ylabel("Moving Average Return")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    print(f"Plot saved to: {output_path}")


def main():
    results_dir = Path("results")
    reinforce_dir = results_dir / "reinforce"
    actor_critic_dir = results_dir / "actor_critic"
    plots_dir = results_dir / "plots"

    plots_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Plot 1: REINFORCE baseline comparison
    # --------------------------------------------------
    reinforce_runs = {
        "No baseline": reinforce_dir / "reinforce_none_raw_seed42.csv",
        "No baseline + normalized advantages": reinforce_dir / "reinforce_none_normalized_seed42.csv",
        "Constant baseline = 5": reinforce_dir / "reinforce_constant5_raw_seed42.csv",
        "Constant baseline = 10": reinforce_dir / "reinforce_constant10_raw_seed42.csv",
    }

    plot_runs(
        runs=reinforce_runs,
        title="REINFORCE on Hopper-v4: Baseline and Normalization Comparison",
        output_path=plots_dir / "reinforce_baseline_comparison_100eps.png",
    )

    # --------------------------------------------------
    # Plot 2: Long-run REINFORCE vs Actor-Critic
    # --------------------------------------------------
    long_run_runs = {
        "REINFORCE no baseline, 1000 episodes": reinforce_dir / "reinforce_none_1000_seed42.csv",
        "Actor-Critic MC normalized, 1000 episodes": actor_critic_dir / "ac_mc_norm_1000_seed42.csv",
    }

    plot_runs(
        runs=long_run_runs,
        title="Hopper-v4 Long-Run Comparison: REINFORCE vs Actor-Critic",
        output_path=plots_dir / "reinforce_vs_actor_critic_1000eps.png",
    )


if __name__ == "__main__":
    main()