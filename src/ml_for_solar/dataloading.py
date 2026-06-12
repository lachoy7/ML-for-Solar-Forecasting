"""PyTorch Forecasting dataset and dataloader construction."""

from __future__ import annotations

import argparse

import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from ml_for_solar.config import (
    PROCESSED_DIR,
    TEST_DATALOADER_PATH,
    TRAIN_DATALOADER_PATH,
    TRAIN_X_PATH,
    VAL_DATALOADER_PATH,
    ensure_dirs,
)

MAX_PREDICTION_LENGTH = 365
MAX_ENCODER_LENGTH = 365
BATCH_SIZE = 128


def load_dataframe(csv_path=TRAIN_X_PATH) -> pd.DataFrame:
    data_df = pd.read_csv(csv_path)
    data_df["Month"] = data_df["Month"].astype(str).astype("category")
    data_df["Day"] = data_df["Day"].astype(str).astype("category")
    data_df["Location"] = "21.98048, -159.33863"
    data_df["Total GHI"] = data_df["Total GHI"].astype(float)
    data_df = data_df.dropna(subset=["Year"])
    data_df["Year"] = data_df["Year"].astype(int)
    data_df["time_idx"] = data_df["time_idx"].astype(int)
    year_min = data_df["Year"].astype(int).min()
    year_max = data_df["Year"].astype(int).max()
    data_df["Year"] = (data_df["Year"].astype(int) - year_min) / (year_max - year_min)
    return data_df


def build_datasets(data_df: pd.DataFrame):
    train_cutoff = data_df["time_idx"].max() - (4 * 365)
    val_cutoff = data_df["time_idx"].max() - (365 * 2)
    train_df = data_df[data_df["time_idx"] <= train_cutoff]
    val_df = data_df[(data_df["time_idx"] > train_cutoff) & (data_df["time_idx"] <= val_cutoff)]
    test_df = data_df[data_df["time_idx"] > val_cutoff]

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="Total GHI",
        group_ids=["Location"],
        min_encoder_length=MAX_ENCODER_LENGTH // 2,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=["Month", "Day"],
        time_varying_known_reals=["Year", "time_idx"],
        time_varying_unknown_categoricals=["weather"],
        time_varying_unknown_reals=[
            "Average Temperature",
            "Average Ozone",
            "Average AOD",
            "Average Dew Point",
            "Average Relative Humidity",
            "Average Pressure",
            "RMM2",
            "MJO Amplitude",
            "Average Precipitable Water",
            "Max Temperature",
            "Min Temperature",
            "avg_temp_by_month",
            "avg_humidity_by_month",
        ],
        target_normalizer=GroupNormalizer(groups=["Location"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)
    test = TimeSeriesDataSet.from_dataset(training, test_df, predict=True, stop_randomization=True)
    return training, validation, test


def build_dataloaders(training, validation, test, num_workers: int = 0):
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE * 10,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    test_dataloader = test.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE * 10,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_dataloader, val_dataloader, test_dataloader


def save_dataloaders(
    train_dataloader,
    val_dataloader,
    test_dataloader,
    output_dir=PROCESSED_DIR,
) -> None:
    ensure_dirs()
    torch.save(train_dataloader, output_dir / "train_dataloader.pth")
    torch.save(val_dataloader, output_dir / "val_dataloader.pth")
    torch.save(test_dataloader, output_dir / "test_dataloader.pth")


def load_dataloaders(processed_dir=PROCESSED_DIR):
    return (
        torch.load(processed_dir / "train_dataloader.pth", weights_only=False),
        torch.load(processed_dir / "val_dataloader.pth", weights_only=False),
        torch.load(processed_dir / "test_dataloader.pth", weights_only=False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and cache TFT dataloaders.")
    parser.add_argument("--csv", type=str, default=str(TRAIN_X_PATH))
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    data_df = load_dataframe(args.csv)
    training, validation, test = build_datasets(data_df)
    dataloaders = build_dataloaders(training, validation, test, num_workers=args.num_workers)
    save_dataloaders(*dataloaders)
    print(f"Saved dataloaders to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
