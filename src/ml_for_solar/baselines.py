"""Sklearn baseline models for daily GHI forecasting."""

from __future__ import annotations

import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from ml_for_solar.config import (
    BASELINE_CONFIGS,
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
    TRAIN_X_PATH,
    WEATHER_ENCODING,
    X_FEATURES,
    ensure_dirs,
)


def _encode_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Daylight Weather" in df.columns and df["Daylight Weather"].dtype == object:
        df["Daylight Weather"] = df["Daylight Weather"].map(WEATHER_ENCODING)
    elif "weather" in df.columns:
        df["Daylight Weather"] = df["weather"].astype(str).map(WEATHER_ENCODING)
    return df


def load_time_series_splits(csv_path=TRAIN_X_PATH):
    df = pd.read_csv(csv_path)
    df = _encode_weather(df)

    if "Date" not in df.columns:
        raise ValueError(
            f"{csv_path} must include a Date column for baseline training. "
            "Use a daily CSV from the wrangling step (see scripts/wrangle_data.sh)."
        )

    df_ts = df.copy()
    df_ts["Date"] = pd.to_datetime(df_ts["Date"], errors="coerce")
    df_ts = df_ts.dropna(subset=["Date"])
    df_ts = df_ts.set_index("Date").sort_index().asfreq("D", method="bfill")

    train_x = df_ts.loc["2000-01-01":"2021-12-31", X_FEATURES]
    train_y = df_ts.loc["2000-01-01":"2021-12-31", "Total GHI"]
    val_x = df_ts.loc["2022-01-01":"2022-12-31", X_FEATURES]
    val_y = df_ts.loc["2022-01-01":"2022-12-31", "Total GHI"]
    test_x = df_ts.loc["2023-01-01":"2023-12-31", X_FEATURES]
    test_y = df_ts.loc["2023-01-01":"2023-12-31", "Total GHI"]
    return train_x, train_y, val_x, val_y, test_x, test_y


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _build_model(name: str) -> Any:
    cfg = BASELINE_CONFIGS[name]
    if name == "linear_regression":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=cfg["alpha"]))
    if name == "decision_tree":
        return make_pipeline(
            StandardScaler(),
            DecisionTreeRegressor(
                max_features=cfg["max_features"],
                max_depth=cfg["max_depth"],
                max_leaf_nodes=cfg["max_leaf_nodes"],
            ),
        )
    if name == "random_forest":
        return make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                warm_start=cfg["warm_start"],
                max_features=cfg["max_features"],
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                max_leaf_nodes=cfg["max_leaf_nodes"],
            ),
        )
    if name == "adaboost":
        return make_pipeline(
            StandardScaler(),
            AdaBoostRegressor(
                n_estimators=cfg["n_estimators"],
                learning_rate=cfg["learning_rate"],
            ),
        )
    if name == "hist_gradient_boosting":
        return make_pipeline(
            StandardScaler(),
            HistGradientBoostingRegressor(
                max_depth=cfg["max_depth"],
                max_leaf_nodes=cfg["max_leaf_nodes"],
                learning_rate=cfg["learning_rate"],
                max_iter=cfg["max_iter"],
                min_samples_leaf=cfg["min_samples_leaf"],
                l2_regularization=cfg["l2_regularization"],
            ),
        )
    raise ValueError(f"Unknown baseline model: {name}")


def train_and_evaluate(name: str, save_plot: bool = True) -> dict[str, float]:
    ensure_dirs()
    train_x, train_y, val_x, val_y, test_x, test_y = load_time_series_splits()
    features = BASELINE_CONFIGS[name]["features"]
    model = _build_model(name)
    model.fit(train_x[features], train_y.values)

    val_preds = model.predict(val_x[features])
    test_preds = model.predict(test_x[features])

    val_metrics = regression_metrics(val_y, val_preds)
    test_metrics = regression_metrics(test_y, test_preds)

    pred_path = PREDICTIONS_DIR / f"{name}_test_predictions.csv"
    pd.DataFrame({"predicted": test_preds}, index=test_y.index).to_csv(pred_path)

    print(f"\n{name.replace('_', ' ').title()}")
    print(f"  Validation RMSE: {val_metrics['rmse']:.4f}")
    print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"  Test MAE:  {test_metrics['mae']:.4f}")
    print(f"  Test MAPE: {test_metrics['mape']:.4f}")
    print(f"  Test R^2:  {test_metrics['r2']:.4f}")

    if save_plot:
        fig, ax = plt.subplots(figsize=(16, 6))
        test_y.plot(ax=ax, label="Actual")
        pd.Series(test_preds, index=test_y.index).plot(ax=ax, label="Predictions")
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Total GHI")
        ax.set_title(f"{name.replace('_', ' ').title()} on Test Set")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / f"{name}_test_plot.png")
        plt.close(fig)

    return test_metrics


def run_all_baselines() -> None:
    for name in BASELINE_CONFIGS:
        train_and_evaluate(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sklearn baseline models.")
    parser.add_argument(
        "--model",
        choices=list(BASELINE_CONFIGS.keys()) + ["all"],
        default="all",
        help="Baseline model to train",
    )
    args = parser.parse_args()

    if args.model == "all":
        run_all_baselines()
    else:
        train_and_evaluate(args.model)


if __name__ == "__main__":
    main()
