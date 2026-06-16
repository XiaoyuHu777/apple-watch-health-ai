"""Build daily-level health features from parsed Apple Health CSV files."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


INPUT_FILES = {
    "heart_rate": "heart_rate.csv",
    "resting_heart_rate": "resting_heart_rate.csv",
    "hrv": "hrv.csv",
    "sleep": "sleep_analysis.csv",
    "steps": "steps.csv",
    "active_energy": "active_energy.csv",
    "exercise_time": "exercise_time.csv",
}

ASLEEP_VALUE_PREFIX = "HKCategoryValueSleepAnalysisAsleep"


class DailyFeatureBuildError(RuntimeError):
    """Raised when daily health features cannot be built safely."""


def resolve_timezone(timezone_name: str | None) -> tzinfo:
    """Resolve an optional IANA timezone, otherwise use the system timezone."""
    if timezone_name:
        try:
            return ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise DailyFeatureBuildError(
                f"Unknown timezone: {timezone_name}"
            ) from exc

    return datetime.now().astimezone().tzinfo or timezone.utc


def validate_input_files(input_dir: Path) -> dict[str, Path]:
    """Validate that every required processed CSV exists."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise DailyFeatureBuildError(
            f"Input path is not a directory: {input_dir}"
        )

    paths = {
        metric: input_dir / filename
        for metric, filename in INPUT_FILES.items()
    }
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required input CSV files: " + ", ".join(sorted(missing))
        )
    return paths


def read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read selected columns after validating the CSV schema."""
    try:
        available_columns = pd.read_csv(path, nrows=0).columns
    except (OSError, pd.errors.ParserError) as exc:
        raise DailyFeatureBuildError(f"Could not read {path.name}: {exc}") from exc

    missing_columns = [
        column for column in columns if column not in available_columns
    ]
    if missing_columns:
        raise DailyFeatureBuildError(
            f"{path.name} is missing columns: {', '.join(missing_columns)}"
        )

    try:
        return pd.read_csv(path, usecols=columns)
    except (OSError, pd.errors.ParserError) as exc:
        raise DailyFeatureBuildError(f"Could not read {path.name}: {exc}") from exc


def prepare_numeric_records(
    path: Path,
    local_timezone: tzinfo,
) -> pd.DataFrame:
    """Read numeric records and derive their local calendar date."""
    frame = read_columns(path, ["startDate", "value"])
    timestamps = pd.to_datetime(frame["startDate"], errors="coerce", utc=True)
    frame["date"] = timestamps.dt.tz_convert(local_timezone).dt.date
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame[["date", "value"]]


def aggregate_numeric(
    path: Path,
    local_timezone: tzinfo,
    aggregations: dict[str, str],
) -> pd.DataFrame:
    """Aggregate a numeric metric by local calendar date."""
    records = prepare_numeric_records(path, local_timezone)
    records = records.dropna(subset=["date"])
    if records.empty:
        return pd.DataFrame(columns=list(aggregations))

    daily = records.groupby("date")["value"].agg(list(aggregations.values()))
    daily.columns = list(aggregations)
    return daily


def aggregate_total(
    path: Path,
    local_timezone: tzinfo,
    output_column: str,
) -> pd.DataFrame:
    """Sum a numeric metric by local date while preserving missing values."""
    records = prepare_numeric_records(path, local_timezone)
    records = records.dropna(subset=["date"])
    if records.empty:
        return pd.DataFrame(columns=[output_column])

    daily = records.groupby("date")["value"].sum(min_count=1)
    return daily.rename(output_column).to_frame()


def merged_interval_hours(group: pd.DataFrame) -> float:
    """Return the union duration of overlapping sleep intervals in hours."""
    intervals = group.sort_values(["start", "end"])[["start", "end"]]
    merged_seconds = 0.0
    current_start = None
    current_end = None

    for start, end in intervals.itertuples(index=False, name=None):
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            merged_seconds += (current_end - current_start).total_seconds()
            current_start, current_end = start, end

    if current_start is not None:
        merged_seconds += (current_end - current_start).total_seconds()

    return merged_seconds / 3600.0


def aggregate_sleep(
    path: Path,
    local_timezone: tzinfo,
) -> pd.DataFrame:
    """Estimate daily asleep hours, assigning sleep to its local end date."""
    records = read_columns(path, ["startDate", "endDate", "value"])
    records = records[
        records["value"].astype("string").str.startswith(
            ASLEEP_VALUE_PREFIX,
            na=False,
        )
    ].copy()

    records["start"] = pd.to_datetime(
        records["startDate"],
        errors="coerce",
        utc=True,
    )
    records["end"] = pd.to_datetime(
        records["endDate"],
        errors="coerce",
        utc=True,
    )
    records = records.dropna(subset=["start", "end"])
    records = records[records["end"] > records["start"]]
    if records.empty:
        return pd.DataFrame(columns=["sleep_hours"])

    records["date"] = (
        records["end"].dt.tz_convert(local_timezone).dt.date
    )
    daily = records.groupby("date", sort=True)[["start", "end"]].apply(
        merged_interval_hours
    )
    return daily.rename("sleep_hours").to_frame()


def build_daily_features(
    input_dir: Path,
    local_timezone: tzinfo,
) -> pd.DataFrame:
    """Build a continuous daily feature table from processed metric CSVs."""
    paths = validate_input_files(input_dir)
    feature_frames = [
        aggregate_numeric(
            paths["heart_rate"],
            local_timezone,
            {
                "heart_rate_mean": "mean",
                "heart_rate_min": "min",
                "heart_rate_max": "max",
            },
        ),
        aggregate_numeric(
            paths["resting_heart_rate"],
            local_timezone,
            {"resting_heart_rate_mean": "mean"},
        ),
        aggregate_numeric(
            paths["hrv"],
            local_timezone,
            {"hrv_mean": "mean", "hrv_median": "median"},
        ),
        aggregate_total(
            paths["steps"],
            local_timezone,
            "steps_total",
        ),
        aggregate_total(
            paths["active_energy"],
            local_timezone,
            "active_energy_total",
        ),
        aggregate_total(
            paths["exercise_time"],
            local_timezone,
            "exercise_minutes_total",
        ),
        aggregate_sleep(paths["sleep"], local_timezone),
    ]

    non_empty_frames = [frame for frame in feature_frames if not frame.empty]
    if not non_empty_frames:
        raise DailyFeatureBuildError("No valid dated health records were found")

    daily = pd.concat(feature_frames, axis=1)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    full_calendar = pd.date_range(
        start=daily.index.min(),
        end=daily.index.max(),
        freq="D",
    )
    daily = daily.reindex(full_calendar)
    daily.index.name = "date"
    daily = daily.reset_index()
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    return daily


def write_csv_atomically(frame: pd.DataFrame, output_path: Path) -> None:
    """Write a CSV through a temporary file to avoid partial output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise DailyFeatureBuildError(
            f"Could not write output CSV: {exc}"
        ) from exc


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and rates for feature columns."""
    feature_columns = [column for column in frame.columns if column != "date"]
    missing_count = frame[feature_columns].isna().sum()
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "missing_count": missing_count.values,
            "missing_rate": (
                missing_count / len(frame) if len(frame) else float("nan")
            ).values,
        }
    )


def print_summary(frame: pd.DataFrame) -> None:
    """Print aggregate output and missingness statistics only."""
    print("Daily health feature build completed.")
    print(f"Daily rows: {len(frame)}")
    print("Missingness:")
    for row in missingness_summary(frame).itertuples(index=False):
        print(
            f"  {row.feature}: {row.missing_count} "
            f"({row.missing_rate:.2%})"
        )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate processed Apple Health CSV files into daily features. "
            "Only aggregate counts and missingness are printed."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the processed metric CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for daily_health_features.csv.",
    )
    parser.add_argument(
        "--timezone",
        help=(
            "Optional IANA timezone for daily boundaries, such as "
            "Asia/Shanghai. Defaults to the system timezone."
        ),
    )
    return parser


def main() -> int:
    """Run the daily feature builder."""
    args = build_argument_parser().parse_args()

    try:
        local_timezone = resolve_timezone(args.timezone)
        daily = build_daily_features(args.input_dir, local_timezone)
        write_csv_atomically(daily, args.output)
    except (
        DailyFeatureBuildError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(daily)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
