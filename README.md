# Curriculum-Guided Reinforcement Learning with Self-Revision

Class project for Stanford CS 229, Machine Learning, Winter 2025

Forecast daily global horizontal irradiance (GHI) from weather and climate features using classical ML baselines and a Temporal Fusion Transformer (TFT).

## Overview

This repository predicts solar irradiance at a fixed location using:

- **NSRDB** hourly solar/weather data aggregated to daily features
- **MJO index** climate signals merged into the training matrix
- **Sklearn baselines** (linear regression, ridge, tree ensembles, boosting)
- **TFT** via [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) and PyTorch Lightning

## Repository layout

```
ml-for-solar/
├── data/                  # CSV inputs and cached dataloaders
│   ├── raw/               # Raw NSRDB downloads (not committed)
│   └── processed/         # Generated .pth dataloaders
├── outputs/               # Checkpoints, predictions, plots, TensorBoard logs
├── scripts/               # Shell entry points for each pipeline stage
└── src/ml_for_solar/      # Python package
    ├── wrangling.py       # NSRDB hourly → daily aggregation
    ├── preprocess.py      # MJO merge, feature engineering, scaling
    ├── dataloading.py     # TimeSeriesDataSet construction
    ├── baselines.py       # Sklearn models with tuned hyperparameters
    ├── tft.py             # TFT training and evaluation
    ├── evaluation.py      # Denormalize and score TFT predictions
    ├── plots.py           # Exploratory plotting helpers
    └── config.py          # Paths, splits, and model configs
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
./scripts/setup.sh
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
pip install -e .
```

For NSRDB downloads, set your NREL API credentials:

```bash
export NREL_API_KEY="your-key"
export NREL_EMAIL="your-email@example.com"
```

## Usage

Run stages in order:

```bash
# 1. Request raw NSRDB downloads (prints URLs when ready)
export NREL_API_KEY="your-key"
export NREL_EMAIL="your-email@example.com"
./scripts/fetch_nsrdb.sh 25376

# 2. Aggregate raw NSRDB hourly files to daily CSVs (requires data/raw/nsrdb/)
./scripts/wrangle_data.sh

# 3. Merge MJO data, engineer features, and write data/train_x.csv
./scripts/preprocess.sh data/raw/daily/<location>.csv

# 4. Build PyTorch Forecasting dataloaders
./scripts/build_dataloaders.sh

# 5. Train sklearn baselines (all models, or pass a model name)
./scripts/train_baselines.sh
./scripts/train_baselines.sh ridge

# 6. Train TFT (optional: pass max_epochs accelerator num_workers)
./scripts/train_tft.sh 50 auto 0

# 7. Denormalize TFT test predictions
./scripts/evaluate_tft.sh
```

Each step can also be invoked as a Python module, e.g. `python -m ml_for_solar.tft --max-epochs 1`.

## Data


| File                   | Description                                                |
| ---------------------- | ---------------------------------------------------------- |
| `data/train_x.csv`     | Processed, scaled training matrix used by the TFT pipeline |
| `data/train_x_old.csv` | Reference GHI values for denormalizing TFT outputs         |
| `data/raw/nsrdb/`      | Place raw NSRDB CSV folders here before wrangling          |
| `data/processed/`      | Cached train/val/test dataloaders (`.pth`)                 |


Baselines expect a daily CSV with a `Date` column and unscaled features. Use output from the wrangling step, or point `--csv` at your own file when calling `ml_for_solar.baselines`.

## Models

**Baselines** (from hyperparameter tuning in the original analysis):

- Linear Regression, Ridge, Decision Tree, Random Forest, AdaBoost, HistGradientBoosting

**TFT** hyperparameters are stored in `src/ml_for_solar/config.py` (`TFT_HPARAMS`).

## Outputs

Generated artifacts are written to `outputs/`:

- `outputs/checkpoints/` — TFT Lightning checkpoints
- `outputs/predictions/` — Model prediction CSVs and plots
- `outputs/logs/` — TensorBoard logs

