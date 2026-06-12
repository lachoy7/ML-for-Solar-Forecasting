"""Plotting utilities for exploratory data analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ml_for_solar.config import OUTPUTS_DIR, ensure_dirs


def plot_last_year_features(csv_path: Path, output_path: Path | None = None) -> None:
    ensure_dirs()
    data_df = pd.read_csv(csv_path)
    data_last_365 = data_df.tail(365).apply(pd.to_numeric, errors="coerce")

    plt.figure(figsize=(12, 6))
    for column in data_last_365.columns:
        if data_last_365[column].dtype in ["float64", "int64"]:
            plt.plot(data_last_365.index, data_last_365[column], label=column)

    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.title("Last 365 Data Points for Each Column")
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    output_path = output_path or OUTPUTS_DIR / "last_year_features.png"
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exploratory plots.")
    parser.add_argument("csv", type=Path, help="CSV file to visualize")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    plot_last_year_features(args.csv, args.output)
    print(f"Saved plot to {args.output or OUTPUTS_DIR / 'last_year_features.png'}")


if __name__ == "__main__":
    main()
