from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TRAIN_X_PATH = DATA_DIR / "train_x.csv"
TRAIN_X_OLD_PATH = DATA_DIR / "train_x_old.csv"

TRAIN_DATALOADER_PATH = PROCESSED_DIR / "train_dataloader.pth"
VAL_DATALOADER_PATH = PROCESSED_DIR / "val_dataloader.pth"
TEST_DATALOADER_PATH = PROCESSED_DIR / "test_dataloader.pth"

CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
LOGS_DIR = OUTPUTS_DIR / "logs"

MJO_URL = "https://psl.noaa.gov/mjo/mjoindex/vpm.1x.txt"
MJO_COLUMNS = ["Year", "Month", "Day", "Hour", "RMM1", "RMM2", "MJO Amplitude"]

X_FEATURES = [
    "Year",
    "Month",
    "Day",
    "Average Temperature",
    "Average Ozone",
    "Average AOD",
    "Average Dew Point",
    "Average Relative Humidity",
    "Average Pressure",
    "Average Precipitable Water",
    "Max Temperature",
    "Min Temperature",
    "Daylight Weather",
]

WEATHER_ENCODING = {
    "Clear": 1,
    "Partly Cloudy": 2,
    "Cloudy": 3,
    "Fog": 4,
    "Dust": 5,
}

TRAIN_START = "2000-01-01"
TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2023-12-31"

CONTINUOUS_COLS = [
    "Average Temperature",
    "Average Ozone",
    "Average AOD",
    "Total GHI",
    "Average Dew Point",
    "Average Relative Humidity",
    "Average Pressure",
    "Average Precipitable Water",
    "Max Temperature",
    "Min Temperature",
]

TFT_HPARAMS = {
    "learning_rate": 0.004,
    "lstm_layers": 2,
    "hidden_size": 10,
    "attention_head_size": 2,
    "dropout": 0.2328434370945469,
    "hidden_continuous_size": 8,
    "optimizer": "adam",
}

BASELINE_CONFIGS = {
    "linear_regression": {
        "features": [
            "Year",
            "Month",
            "Average Ozone",
            "Average Precipitable Water",
            "Max Temperature",
            "Daylight Weather",
        ],
    },
    "ridge": {
        "features": [
            "Year",
            "Month",
            "Average Ozone",
            "Average Precipitable Water",
            "Max Temperature",
            "Daylight Weather",
        ],
        "alpha": 0,
    },
    "decision_tree": {
        "features": [
            "Year",
            "Month",
            "Average Ozone",
            "Average Dew Point",
            "Average Precipitable Water",
            "Daylight Weather",
        ],
        "max_features": None,
        "max_depth": None,
        "max_leaf_nodes": 42,
    },
    "random_forest": {
        "features": [
            "Month",
            "Day",
            "Average AOD",
            "Average Dew Point",
            "Average Precipitable Water",
            "Max Temperature",
            "Daylight Weather",
        ],
        "warm_start": False,
        "max_features": "log2",
        "n_estimators": 62,
        "max_depth": 100,
        "max_leaf_nodes": None,
    },
    "adaboost": {
        "features": [
            "Month",
            "Day",
            "Average Precipitable Water",
            "Max Temperature",
            "Min Temperature",
            "Daylight Weather",
        ],
        "n_estimators": 39,
        "learning_rate": 2.31,
    },
    "hist_gradient_boosting": {
        "features": X_FEATURES,
        "max_depth": 16,
        "max_leaf_nodes": 31,
        "learning_rate": 0.11,
        "max_iter": 73,
        "min_samples_leaf": 93,
        "l2_regularization": 16,
    },
}


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, CHECKPOINTS_DIR, PREDICTIONS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
