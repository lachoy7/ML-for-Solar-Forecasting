"""Feature engineering and scaling for TFT training data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ml_for_solar.config import (
    CONTINUOUS_COLS,
    MJO_COLUMNS,
    MJO_URL,
    OUTPUTS_DIR,
    TRAIN_X_PATH,
    ensure_dirs,
)


def load_mjo_index(url: str = MJO_URL) -> pd.DataFrame:
    mjo_df = pd.read_csv(url, delim_whitespace=True, header=None, names=MJO_COLUMNS)
    mjo_df = mjo_df[(mjo_df["Year"] >= 2000) & (mjo_df["Year"] <= 2023)]
    mjo_df = mjo_df[["RMM2", "MJO Amplitude"]]
    mjo_df.replace(999.9, np.nan, inplace=True)
    return mjo_df.dropna().reset_index(drop=True)


def preprocess_location(
    input_csv: Path,
    output_csv: Path = TRAIN_X_PATH,
    mjo_url: str = MJO_URL,
) -> pd.DataFrame:
    """Build the scaled training matrix used by the TFT pipeline."""
    data_df = pd.read_csv(input_csv)
    data_df = data_df.sort_values(["Year", "Month", "Day"]).reset_index(drop=True)
    data_df["time_idx"] = range(len(data_df))
    data_df["Year"] = data_df["Year"].astype("category")
    data_df["Month"] = data_df["Month"].astype("category")
    data_df["Day"] = data_df["Day"].astype("category")

    mjo_df = load_mjo_index(mjo_url)
    data_df = pd.concat([data_df, mjo_df], axis=1)

    grouped_features = {
        "Average Temperature": "avg_temp_by_month",
        "Average Relative Humidity": "avg_humidity_by_month",
    }
    for col, new_col in grouped_features.items():
        data_df[new_col] = data_df.groupby(["time_idx", "Month"])[col].transform("mean")

    data_df["weather"] = data_df["Daylight Weather"].astype("category")
    data_df = data_df.drop(columns=["Daylight Weather"])

    scaler = MinMaxScaler()
    data_df[CONTINUOUS_COLS] = scaler.fit_transform(data_df[CONTINUOUS_COLS])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    data_df.to_csv(output_csv, index=False)
    return data_df


def save_preview_plot(data_df: pd.DataFrame, output_path: Path | None = None) -> None:
    ensure_dirs()
    output_path = output_path or OUTPUTS_DIR / "ghi_preview.png"
    plt.figure(figsize=(15, 5))
    plt.plot(data_df.tail(365)["Total GHI"], label="Forecasted GHI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess daily solar data for TFT training.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a daily aggregated location CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TRAIN_X_PATH,
        help="Path for the processed training CSV",
    )
    args = parser.parse_args()

    data_df = preprocess_location(args.input, args.output)
    save_preview_plot(data_df)
    print(data_df.tail()[["Total GHI"]])
    print(list(data_df.columns))


if __name__ == "__main__":
    main()
