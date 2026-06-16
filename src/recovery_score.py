"""Build an explainable personal recovery score for trend exploration."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "apple-watch-health-ai-matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "apple-watch-health-ai-cache"),
)

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .visualize import plot_recovery_trend, save_figure_formats
else:
    from visualize import plot_recovery_trend, save_figure_formats


REQUIRED_COLUMNS = [
    "date",
    "sleep_hours",
    "hrv_mean",
    "resting_heart_rate_mean",
    "exercise_minutes_total",
    "active_energy_total",
]

COMPONENT_WEIGHTS = {
    "sleep_component_score": 0.30,
    "hrv_component_score": 0.30,
    "resting_hr_component_score": 0.20,
    "previous_exercise_load_component_score": 0.10,
    "previous_active_energy_load_component_score": 0.10,
}

CORE_RECOVERY_COMPONENTS = [
    "sleep_component_score",
    "hrv_component_score",
    "resting_hr_component_score",
]

MIN_WEIGHT_COVERAGE = 0.40
Z_SCORE_CLIP = 3.0


class RecoveryScoreError(RuntimeError):
    """Raised when recovery scores cannot be calculated safely."""


def validate_and_read_input(input_path: Path) -> pd.DataFrame:
    """Read and validate a daily health feature CSV."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    try:
        frame = pd.read_csv(input_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise RecoveryScoreError(
            f"Could not read input CSV: {exc}"
        ) from exc

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise RecoveryScoreError(
            "Input CSV is missing columns: " + ", ".join(missing_columns)
        )

    frame = frame[REQUIRED_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise RecoveryScoreError("Input CSV contains invalid dates")
    if frame["date"].duplicated().any():
        raise RecoveryScoreError("Input CSV contains duplicate dates")

    for column in REQUIRED_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.sort_values("date").reset_index(drop=True)


def personalized_z_score(series: pd.Series) -> pd.Series:
    """Calculate a personal z-score while preserving missing observations."""
    mean = series.mean(skipna=True)
    standard_deviation = series.std(skipna=True, ddof=0)

    if pd.isna(mean):
        return pd.Series(np.nan, index=series.index, dtype="float64")
    if pd.isna(standard_deviation) or standard_deviation == 0:
        result = pd.Series(np.nan, index=series.index, dtype="float64")
        result.loc[series.notna()] = 0.0
        return result

    return ((series - mean) / standard_deviation).clip(
        lower=-Z_SCORE_CLIP,
        upper=Z_SCORE_CLIP,
    )


def sigmoid_component(z_score: pd.Series) -> pd.Series:
    """Map a signed z-score to a 0-100 component score."""
    return 100.0 / (1.0 + np.exp(-z_score))


def load_penalty_component(previous_day_z_score: pd.Series) -> pd.Series:
    """Map above-average prior-day load to a penalty; lower load is neutral."""
    excess_load = previous_day_z_score.clip(lower=0.0)
    return 100.0 / (1.0 + np.exp(excess_load))


def calculate_recovery_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate z-scores, component scores, and the weighted recovery score."""
    result = pd.DataFrame({"date": frame["date"]})

    result["sleep_z"] = personalized_z_score(frame["sleep_hours"])
    result["hrv_z"] = personalized_z_score(frame["hrv_mean"])
    result["resting_hr_z"] = personalized_z_score(
        frame["resting_heart_rate_mean"]
    )
    exercise_z = personalized_z_score(frame["exercise_minutes_total"])
    active_energy_z = personalized_z_score(frame["active_energy_total"])
    result["previous_exercise_load_z"] = exercise_z.shift(1)
    result["previous_active_energy_load_z"] = active_energy_z.shift(1)

    result["sleep_component_score"] = sigmoid_component(result["sleep_z"])
    result["hrv_component_score"] = sigmoid_component(result["hrv_z"])
    result["resting_hr_component_score"] = sigmoid_component(
        -result["resting_hr_z"]
    )
    result["previous_exercise_load_component_score"] = load_penalty_component(
        result["previous_exercise_load_z"]
    )
    result["previous_active_energy_load_component_score"] = (
        load_penalty_component(result["previous_active_energy_load_z"])
    )

    component_columns = list(COMPONENT_WEIGHTS)
    available = result[component_columns].notna()
    weights = pd.Series(COMPONENT_WEIGHTS)
    weighted_sum = result[component_columns].mul(weights).sum(
        axis=1,
        min_count=1,
    )
    weight_coverage = available.mul(weights).sum(axis=1)
    component_count = available.sum(axis=1)
    has_core_recovery_signal = result[CORE_RECOVERY_COMPONENTS].notna().any(
        axis=1
    )
    score_is_valid = (
        (weight_coverage >= MIN_WEIGHT_COVERAGE)
        & has_core_recovery_signal
    )

    result.insert(
        1,
        "recovery_score",
        (weighted_sum / weight_coverage).where(score_is_valid).clip(0, 100),
    )
    result.insert(2, "component_count", component_count)
    result.insert(3, "weight_coverage", weight_coverage)
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    return result


def write_csv_atomically(frame: pd.DataFrame, output_path: Path) -> None:
    """Write the recovery CSV through a temporary file."""
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
        raise RecoveryScoreError(
            f"Could not write recovery CSV: {exc}"
        ) from exc


def print_summary(frame: pd.DataFrame) -> None:
    """Print aggregate score coverage without exposing daily values."""
    scored_days = int(frame["recovery_score"].notna().sum())
    total_days = len(frame)
    coverage = scored_days / total_days if total_days else 0.0
    print("Personal recovery score build completed.")
    print(f"Daily rows: {total_days}")
    print(f"Days with sufficient score coverage: {scored_days} ({coverage:.2%})")
    print(
        "Weights: sleep=30%, HRV=30%, resting HR=20%, "
        "previous exercise=10%, previous active energy=10%"
    )
    print("This score is for personal trend exploration, not medical use.")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build an explainable 0-100 personal recovery trend score. "
            "This is not a medical or diagnostic score."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to daily_health_features.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for daily_recovery_score.csv.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directory for clean and portfolio recovery trend figures.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    """Run the recovery score builder."""
    args = build_argument_parser().parse_args()

    try:
        daily_features = validate_and_read_input(args.input)
        recovery_scores = calculate_recovery_scores(daily_features)
        write_csv_atomically(recovery_scores, args.output)
        figure_dir = args.figure.parent if args.figure else args.figure_dir
        for version in ("clean", "portfolio"):
            figure = plot_recovery_trend(
                recovery_scores,
                version=version,
            )
            try:
                save_figure_formats(
                    figure,
                    figure_dir / f"recovery_score_trend_{version}",
                )
            finally:
                plt.close(figure)
    except (
        RecoveryScoreError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(recovery_scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
