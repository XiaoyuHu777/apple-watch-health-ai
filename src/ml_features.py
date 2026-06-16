"""Build a next-day recovery machine-learning dataset from daily aggregates."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


MODEL_INPUT_COLUMNS = [
    "recovery_score",
    "sleep_hours",
    "hrv_mean",
    "resting_heart_rate_mean",
    "exercise_minutes_total",
    "active_energy_total",
    "steps_total",
]

MISSING_INDICATOR_COLUMNS = {
    "sleep_hours_missing": "sleep_hours",
    "hrv_mean_missing": "hrv_mean",
    "resting_heart_rate_mean_missing": "resting_heart_rate_mean",
    "exercise_minutes_missing": "exercise_minutes_total",
    "active_energy_missing": "active_energy_total",
}

TREND_COLUMNS = [
    "hrv_mean",
    "resting_heart_rate_mean",
    "recovery_score",
]

ROLLING_WINDOWS = (3, 7)


class MLFeatureBuildError(RuntimeError):
    """Raised when the ML dataset cannot be built safely."""


def read_csv(path: Path, label: str) -> pd.DataFrame:
    """Read a CSV after validating its path and date column."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} CSV does not exist: {path}")

    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise MLFeatureBuildError(
            f"Could not read {label} CSV {path}: {exc}"
        ) from exc

    if "date" not in frame.columns:
        raise MLFeatureBuildError(f"{label} CSV is missing required column: date")

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise MLFeatureBuildError(f"{label} CSV contains invalid dates")
    if frame["date"].duplicated().any():
        raise MLFeatureBuildError(f"{label} CSV contains duplicate dates")

    return frame.sort_values("date").reset_index(drop=True)


def load_inputs(
    features_path: Path,
    scores_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the daily feature and recovery score inputs."""
    features = read_csv(features_path, "Features")
    scores = read_csv(scores_path, "Scores")

    missing_feature_columns = [
        column
        for column in MODEL_INPUT_COLUMNS
        if column != "recovery_score" and column not in features.columns
    ]
    if missing_feature_columns:
        raise MLFeatureBuildError(
            "Features CSV is missing required columns: "
            + ", ".join(missing_feature_columns)
        )
    if "recovery_score" not in scores.columns:
        raise MLFeatureBuildError(
            "Scores CSV is missing required column: recovery_score"
        )

    numeric_feature_columns = [
        column for column in features.columns if column != "date"
    ]
    for column in numeric_feature_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    scores["recovery_score"] = pd.to_numeric(
        scores["recovery_score"],
        errors="coerce",
    )
    return features, scores[["date", "recovery_score"]]


def add_lag_features(frame: pd.DataFrame) -> None:
    """Add previous-day values for the requested model inputs."""
    for column in MODEL_INPUT_COLUMNS:
        frame[f"{column}_lag1"] = frame[column].shift(1)


def add_rolling_features(frame: pd.DataFrame) -> None:
    """Add trailing rolling statistics using only current and prior rows."""
    for column in MODEL_INPUT_COLUMNS:
        for window in ROLLING_WINDOWS:
            rolling = frame[column].rolling(window=window, min_periods=1)
            frame[f"{column}_roll{window}_mean"] = rolling.mean()
            frame[f"{column}_roll{window}_std"] = rolling.std(ddof=0)


def add_trend_features(frame: pd.DataFrame) -> None:
    """Add current value minus the value seven days earlier."""
    for column in TREND_COLUMNS:
        frame[f"{column}_trend7"] = frame[column] - frame[column].shift(7)


def add_missing_indicators(frame: pd.DataFrame) -> None:
    """Add binary indicators for missing core daily measurements."""
    for output_column, source_column in MISSING_INDICATOR_COLUMNS.items():
        frame[output_column] = frame[source_column].isna().astype("int8")


def add_calendar_features(frame: pd.DataFrame) -> None:
    """Add deterministic calendar features derived from the daily date."""
    frame["day_of_week"] = frame["date"].dt.dayofweek.astype("int8")
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype("int8")
    frame["month"] = frame["date"].dt.month.astype("int8")


def build_ml_dataset(
    features: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Merge daily inputs and build next-day targets and historical features."""
    frame = features.merge(
        scores,
        on="date",
        how="inner",
        validate="one_to_one",
    ).sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise MLFeatureBuildError(
            "Features and scores CSVs have no overlapping dates"
        )

    add_lag_features(frame)
    add_rolling_features(frame)
    add_trend_features(frame)
    add_missing_indicators(frame)
    add_calendar_features(frame)

    frame["target_recovery_next_day"] = frame["recovery_score"].shift(-1)
    frame["low_recovery_next_day"] = (
        frame["target_recovery_next_day"]
        .lt(40)
        .where(frame["target_recovery_next_day"].notna())
        .astype("Int8")
    )

    frame = frame.iloc[:-1].copy()
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame


def write_csv_atomically(frame: pd.DataFrame, output_path: Path) -> None:
    """Write the dataset through a temporary file to avoid partial output."""
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
        raise MLFeatureBuildError(
            f"Could not write ML dataset CSV: {exc}"
        ) from exc


def print_summary(frame: pd.DataFrame) -> None:
    """Print aggregate dataset dimensions and target coverage only."""
    target_missing = int(frame["target_recovery_next_day"].isna().sum())
    feature_columns = [
        column
        for column in frame.columns
        if column
        not in {"date", "target_recovery_next_day", "low_recovery_next_day"}
    ]
    features_with_missing = int(frame[feature_columns].isna().any().sum())

    print("ML recovery dataset build completed.")
    print(f"Rows: {len(frame)}")
    print(f"Columns: {len(frame.columns)}")
    print(f"Rows with missing next-day target: {target_missing}")
    print(f"Feature columns containing missing values: {features_with_missing}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build historical ML features and next-day recovery targets from "
            "daily aggregate CSV files. No model is trained."
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to daily_health_features.csv.",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        required=True,
        help="Path to daily_recovery_score.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for ml_recovery_dataset.csv.",
    )
    return parser


def main() -> int:
    """Run the ML dataset builder."""
    args = build_argument_parser().parse_args()

    try:
        features, scores = load_inputs(args.features, args.scores)
        dataset = build_ml_dataset(features, scores)
        write_csv_atomically(dataset, args.output)
    except (
        MLFeatureBuildError,
        FileNotFoundError,
        OSError,
        ValueError,
        pd.errors.MergeError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
