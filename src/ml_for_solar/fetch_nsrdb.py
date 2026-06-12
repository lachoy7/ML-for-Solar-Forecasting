"""CLI for requesting NSRDB downloads from NREL."""

from __future__ import annotations

import argparse
import os

from ml_for_solar.wrangling import fetch_nsrdb


def main() -> None:
    parser = argparse.ArgumentParser(description="Request NSRDB data downloads from NREL.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NREL_API_KEY"),
        help="NREL API key (or set NREL_API_KEY env var)",
    )
    parser.add_argument(
        "--email",
        default=os.environ.get("NREL_EMAIL"),
        help="Email registered with NREL (or set NREL_EMAIL env var)",
    )
    parser.add_argument(
        "--location-ids",
        nargs="+",
        required=True,
        help="NSRDB location IDs to download",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=[str(y) for y in range(2015, 2024)],
        help="Years to request",
    )
    args = parser.parse_args()

    if not args.api_key or not args.email:
        raise SystemExit("Set --api-key/--email or NREL_API_KEY/NREL_EMAIL environment variables.")

    fetch_nsrdb(args.api_key, args.email, args.location_ids, args.years)


if __name__ == "__main__":
    main()
