"""NSRDB raw data wrangling: hourly to daily aggregation."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

from ml_for_solar.config import RAW_DIR

CLOUD_TYPE_TO_DESC = {
    0: "Clear",
    1: "Clear",
    2: "Fog",
    3: "Partly Cloudy",
    4: "Partly Cloudy",
    5: "Partly Cloudy",
    6: "Cloudy",
    7: "Cloudy",
    8: "Cloudy",
    9: "Cloudy",
    11: "Dust",
}

NSRDB_ATTRIBUTES = (
    "air_temperature,alpha,aod,asymmetry,clearsky_dhi,clearsky_dni,clearsky_ghi,"
    "cloud_fill_flag,cloud_type,dew_point,dhi,dni,fill_flag,ghi,ozone,relative_humidity,"
    "solar_zenith_angle,ssa,surface_albedo,surface_pressure,total_precipitable_water,"
    "wind_direction,wind_speed"
)
NSRDB_BASE_URL = (
    "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.json?"
)


def day_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly NSRDB records to daily features."""
    day_cols = ["Year", "Month", "Day"]
    avg_cols = [
        "Temperature",
        "Ozone",
        "AOD",
        "Dew Point",
        "Relative Humidity",
        "Pressure",
        "Precipitable Water",
    ]
    sum_cols = ["GHI", "DHI", "DNI", "Clearsky GHI", "Clearsky DHI", "Clearsky DNI"]

    df = df.astype(float)
    df[day_cols + sum_cols] = df[day_cols + sum_cols].astype(int)

    day_df = df.groupby(day_cols)[avg_cols].mean().reset_index().round(3)
    day_df.rename(
        columns={
            "Temperature": "Average Temperature",
            "Ozone": "Average Ozone",
            "AOD": "Average AOD",
            "Dew Point": "Average Dew Point",
            "Relative Humidity": "Average Relative Humidity",
            "Pressure": "Average Pressure",
            "Precipitable Water": "Average Precipitable Water",
        },
        inplace=True,
    )
    day_df[
        [
            "Average Temperature",
            "Average Dew Point",
            "Average Relative Humidity",
            "Average Pressure",
            "Average Precipitable Water",
        ]
    ] = day_df[
        [
            "Average Temperature",
            "Average Dew Point",
            "Average Relative Humidity",
            "Average Pressure",
            "Average Precipitable Water",
        ]
    ].round(1)

    day_df["Max Temperature"] = df.groupby(day_cols)["Temperature"].max().reset_index()["Temperature"]
    day_df["Min Temperature"] = df.groupby(day_cols)["Temperature"].min().reset_index()["Temperature"]
    day_df[
        [
            "Total GHI",
            "Total DHI",
            "Total DNI",
            "Total Clearsky GHI",
            "Total Clearsky DHI",
            "Total Clearsky DNI",
        ]
    ] = df.groupby(day_cols)[sum_cols].sum().reset_index()[sum_cols]

    daylight = (df["Hour"] >= 6) & (df["Hour"] <= 19)
    day_df["Daylight Weather"] = (
        df[daylight]
        .groupby(day_cols)["Cloud Type"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()["Cloud Type"]
    )
    day_df["Daylight Weather"] = day_df["Daylight Weather"].map(CLOUD_TYPE_TO_DESC)

    day_df["Date"] = (
        day_df["Month"].astype(str)
        + "-"
        + day_df["Day"].astype(str)
        + "-"
        + day_df["Year"].astype(str)
    )
    date_col = day_df.pop("Date")
    day_df.insert(0, "Date", date_col)
    return day_df


def _parse_nsrdb_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.iloc[1]
    df.drop([0, 1], inplace=True)
    return df.loc[:, df.columns.notna()]


def clean_nsrdb_location(nsrdb_path: Path, location: str) -> pd.DataFrame:
    """Load and aggregate all yearly NSRDB files for one location."""
    location_dir = nsrdb_path / location
    y_files = sorted(f for f in os.listdir(location_dir) if f != ".DS_Store")
    dfs = [_parse_nsrdb_csv(location_dir / filename) for filename in y_files]
    return day_aggregation(pd.concat(dfs, ignore_index=True))


def wrangle_all_locations(nsrdb_path: Path, output_dir: Path) -> None:
    """Clean every location under `nsrdb_path` and write daily CSVs to `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(output_dir):
        path = output_dir / filename
        if path.is_file():
            path.unlink()

    for location in sorted(os.listdir(nsrdb_path)):
        if location.startswith("."):
            continue
        clean_df = clean_nsrdb_location(nsrdb_path, location)
        clean_df.to_csv(output_dir / f"{location}.csv", index=False)
        print(f"Wrote {output_dir / location}.csv")


def _handle_nsrdb_response(response: requests.Response) -> dict:
    if response.status_code != 200:
        raise RuntimeError(
            f"NSRDB request failed ({response.status_code} {response.reason}): {response.text}"
        )
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError("NSRDB request errors: " + "\n".join(payload["errors"]))
    return payload


def fetch_nsrdb(
    api_key: str,
    email: str,
    location_ids: list[str],
    years: list[str] | None = None,
) -> None:
    """Request NSRDB downloads for the given location IDs (prints download URLs)."""
    years = years or [str(y) for y in range(2015, 2024)]
    input_data = {
        "attributes": NSRDB_ATTRIBUTES,
        "interval": "60",
        "api_key": api_key,
        "email": email,
    }
    headers = {"x-api-key": api_key}

    for year in years:
        for location_id in location_ids:
            input_data["names"] = [year]
            input_data["location_ids"] = location_id
            print(f"Requesting NSRDB data for location {location_id}, year {year}...")
            response = requests.post(NSRDB_BASE_URL, input_data, headers=headers, timeout=120)
            payload = _handle_nsrdb_response(response)
            print(payload["outputs"]["message"])
            print(f"Download URL: {payload['outputs']['downloadUrl']}")
            time.sleep(1)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate raw NSRDB hourly files to daily CSVs.")
    parser.add_argument(
        "--nsrdb-path",
        type=Path,
        default=RAW_DIR / "nsrdb",
        help="Directory containing per-location NSRDB raw CSV folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR / "daily",
        help="Directory to write aggregated daily CSV files",
    )
    args = parser.parse_args()
    wrangle_all_locations(args.nsrdb_path, args.output_dir)


if __name__ == "__main__":
    main()
