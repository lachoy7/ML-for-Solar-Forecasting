"""Evaluation helpers for denormalizing TFT predictions."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from ml_for_solar.baselines import regression_metrics
from ml_for_solar.config import PREDICTIONS_DIR, TRAIN_X_OLD_PATH, ensure_dirs


def denormalize_tft_predictions(
    raw_predictions_path,
    reference_csv=TRAIN_X_OLD_PATH,
    output_path=None,
    n_days: int = 365,
) -> pd.DataFrame:
    """Inverse-transform scaled TFT predictions using the last year of reference GHI."""
    ensure_dirs()
    raw_data = pd.read_csv(reference_csv)
    forecast = pd.read_csv(raw_predictions_path)

    y_true = raw_data["Total GHI"].tail(n_days).values.reshape(-1, 1)
    y_pred_scaled = forecast["predicted"].values.reshape(-1, 1)[-n_days:]

    scaler = MinMaxScaler()
    scaler.fit(y_true)
    y_pred = scaler.inverse_transform(y_pred_scaled)

    result = pd.DataFrame(y_pred, columns=["TFT Predictions--Test"])
    if output_path:
        result.to_csv(output_path, index=False)
    return result


def evaluate_tft_predictions(
    raw_predictions_path,
    reference_csv=TRAIN_X_OLD_PATH,
    n_days: int = 365,
) -> dict[str, float]:
    """Compare denormalized TFT predictions against the last year of reference GHI."""
    raw_data = pd.read_csv(reference_csv)
    y_true = raw_data["Total GHI"].tail(n_days).values
    denorm = denormalize_tft_predictions(raw_predictions_path, reference_csv, n_days=n_days)
    preds = denorm["TFT Predictions--Test"].values
    metrics = regression_metrics(y_true, preds)
    print("TFT (denormalized)")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  MAPE: {metrics['mape']:.4f}")
    print(f"  R^2:  {metrics['r2']:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Denormalize and evaluate TFT predictions.")
    parser.add_argument(
        "--predictions",
        type=str,
        default=str(PREDICTIONS_DIR / "tft_test_predictions.csv"),
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=str(TRAIN_X_OLD_PATH),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PREDICTIONS_DIR / "tft_test_denormalized.csv"),
    )
    args = parser.parse_args()

    denormalize_tft_predictions(args.predictions, args.reference, args.output)
    evaluate_tft_predictions(args.predictions, args.reference)


if __name__ == "__main__":
    main()
