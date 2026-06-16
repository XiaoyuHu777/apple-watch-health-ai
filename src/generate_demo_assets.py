"""Generate synthetic public demo artifacts for the project README.

The generated data is artificial and is safe to commit. It is not derived from
Apple Health exports or personal health records.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .ml_features import build_ml_dataset, load_inputs
    from .plot_ml_results import (
        load_plot_inputs,
        plot_confusion_matrix,
        plot_feature_importance,
        plot_predicted_vs_actual,
        plot_prediction_timeline,
        plot_residuals,
    )
    from .recovery_score import calculate_recovery_scores
    from .train_recovery_ml import (
        build_predictions,
        identify_numeric_features,
        make_time_split,
        read_dataset,
        train_classification_models,
        train_regression_models,
    )
    from .visualize import plot_recovery_trend, save_figure_formats
else:
    from ml_features import build_ml_dataset, load_inputs
    from plot_ml_results import (
        load_plot_inputs,
        plot_confusion_matrix,
        plot_feature_importance,
        plot_predicted_vs_actual,
        plot_prediction_timeline,
        plot_residuals,
    )
    from recovery_score import calculate_recovery_scores
    from train_recovery_ml import (
        build_predictions,
        identify_numeric_features,
        make_time_split,
        read_dataset,
        train_classification_models,
        train_regression_models,
    )
    from visualize import plot_recovery_trend, save_figure_formats


RANDOM_SEED = 42


class DemoAssetError(RuntimeError):
    """Raised when synthetic demo assets cannot be generated."""


def write_csv_atomically(frame: pd.DataFrame, output_path: Path) -> None:
    """Write a CSV through a temporary file."""
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
        raise DemoAssetError(f"Could not write {output_path}: {exc}") from exc


def synthetic_stress_blocks(dates: pd.Series) -> np.ndarray:
    """Return a synthetic stress/load pattern for demo data generation."""
    stress = np.zeros(len(dates), dtype="float64")
    blocks = [
        ("2025-07-10", "2025-07-16", 1.0),
        ("2025-09-05", "2025-09-12", 0.9),
        ("2025-11-18", "2025-11-24", 1.2),
        ("2025-12-12", "2025-12-16", 0.8),
    ]
    for start, end, intensity in blocks:
        mask = dates.between(pd.Timestamp(start), pd.Timestamp(end))
        stress[mask.to_numpy()] = intensity
    return stress


def build_synthetic_daily_features() -> pd.DataFrame:
    """Build a deterministic synthetic daily health feature table."""
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range("2025-05-15", "2026-01-01", freq="D")
    day_index = np.arange(len(dates), dtype="float64")
    weekly = np.sin(2 * np.pi * day_index / 7)
    long_cycle = np.sin(2 * np.pi * day_index / 54)
    stress = synthetic_stress_blocks(pd.Series(dates))

    exercise = (
        32
        + 10 * weekly
        + 8 * long_cycle
        + 22 * stress
        + rng.normal(0, 6, len(dates))
    ).clip(0, None)
    active_energy = (
        390
        + 5.5 * exercise
        + 45 * weekly
        + rng.normal(0, 45, len(dates))
    ).clip(80, None)
    steps = (
        7800
        + 85 * exercise
        + 900 * weekly
        + rng.normal(0, 950, len(dates))
    ).clip(1200, None)
    sleep = (
        7.35
        + 0.35 * long_cycle
        - 1.05 * stress
        - 0.006 * exercise
        + rng.normal(0, 0.35, len(dates))
    ).clip(4.1, 9.2)
    hrv = (
        55
        + 4.5 * long_cycle
        + 2.1 * (sleep - 7)
        - 9.5 * stress
        + rng.normal(0, 4.0, len(dates))
    ).clip(20, 95)
    resting_hr = (
        58
        - 0.18 * (hrv - 55)
        + 5.8 * stress
        + 0.02 * exercise
        + rng.normal(0, 1.8, len(dates))
    ).clip(44, 84)
    heart_rate_mean = (
        74
        + 0.11 * exercise
        + 0.35 * (resting_hr - 58)
        + rng.normal(0, 2.5, len(dates))
    )

    frame = pd.DataFrame(
        {
            "date": dates,
            "heart_rate_mean": heart_rate_mean,
            "heart_rate_min": heart_rate_mean - rng.uniform(16, 24, len(dates)),
            "heart_rate_max": heart_rate_mean + rng.uniform(38, 62, len(dates)),
            "resting_heart_rate_mean": resting_hr,
            "hrv_mean": hrv,
            "hrv_median": hrv + rng.normal(0, 1.6, len(dates)),
            "steps_total": steps.round(),
            "active_energy_total": active_energy,
            "exercise_minutes_total": exercise,
            "sleep_hours": sleep,
        }
    )

    missing_columns = [
        "sleep_hours",
        "hrv_mean",
        "hrv_median",
        "resting_heart_rate_mean",
        "exercise_minutes_total",
        "active_energy_total",
    ]
    for column in missing_columns:
        missing_mask = rng.random(len(frame)) < 0.035
        frame.loc[missing_mask, column] = np.nan

    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame


def build_synthetic_recovery_scores(features: pd.DataFrame) -> pd.DataFrame:
    """Build synthetic rule-based recovery scores from synthetic features."""
    recovery_input = features[
        [
            "date",
            "sleep_hours",
            "hrv_mean",
            "resting_heart_rate_mean",
            "exercise_minutes_total",
            "active_energy_total",
        ]
    ].copy()
    recovery_input["date"] = pd.to_datetime(recovery_input["date"])
    return calculate_recovery_scores(recovery_input)


def build_demo_reports(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train demo models on synthetic data and write report CSV files."""
    features_path = data_dir / "synthetic_daily_health_features.csv"
    scores_path = data_dir / "synthetic_daily_recovery_score.csv"
    dataset_path = data_dir / "synthetic_ml_recovery_dataset.csv"

    features = build_synthetic_daily_features()
    recovery_scores = build_synthetic_recovery_scores(features)
    write_csv_atomically(features, features_path)
    write_csv_atomically(recovery_scores, scores_path)

    loaded_features, loaded_scores = load_inputs(features_path, scores_path)
    ml_dataset = build_ml_dataset(loaded_features, loaded_scores)
    write_csv_atomically(ml_dataset, dataset_path)

    dataset = read_dataset(dataset_path)
    feature_columns = identify_numeric_features(dataset)
    split = make_time_split(dataset)
    regression_metrics, _, best_regression = train_regression_models(
        split,
        feature_columns,
    )
    classification_metrics, _, best_classification = train_classification_models(
        split,
        feature_columns,
    )
    predictions = build_predictions(
        split,
        feature_columns,
        best_regression,
        best_classification,
    )

    write_csv_atomically(
        regression_metrics,
        data_dir / "synthetic_ml_regression_metrics.csv",
    )
    write_csv_atomically(
        classification_metrics,
        data_dir / "synthetic_ml_classification_metrics.csv",
    )
    write_csv_atomically(
        predictions,
        data_dir / "synthetic_ml_predictions.csv",
    )
    return recovery_scores, regression_metrics


def build_demo_figures(data_dir: Path, figure_dir: Path) -> list[Path]:
    """Generate synthetic public demo figures."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    recovery_scores = pd.read_csv(
        data_dir / "synthetic_daily_recovery_score.csv",
        parse_dates=["date"],
    )
    recovery_figure = plot_recovery_trend(
        recovery_scores,
        version="portfolio",
        max_gap_days=14,
    )
    add_synthetic_label(recovery_figure)
    recovery_figure.axes[0].set_title(
        "Synthetic Recovery Score Trend",
        loc="left",
        fontsize=20,
        fontweight="bold",
        color="#22313F",
        pad=42,
    )
    try:
        trend_paths = save_figure_formats(
            recovery_figure,
            figure_dir / "demo_recovery_trend",
            dpi=220,
        )
    finally:
        import matplotlib.pyplot as plt

        plt.close(recovery_figure)

    test_predictions, regression_metrics, _, dataset = load_plot_inputs(
        data_dir / "synthetic_ml_predictions.csv",
        data_dir / "synthetic_ml_regression_metrics.csv",
        data_dir / "synthetic_ml_classification_metrics.csv",
        data_dir / "synthetic_ml_recovery_dataset.csv",
    )
    ml_figures = [
        ("ml_predicted_vs_actual", plot_predicted_vs_actual(test_predictions)),
        (
            "ml_recovery_prediction_timeline",
            plot_prediction_timeline(test_predictions),
        ),
        ("ml_residuals", plot_residuals(test_predictions)),
        ("ml_confusion_matrix", plot_confusion_matrix(test_predictions)),
        (
            "ml_feature_importance",
            plot_feature_importance(dataset, regression_metrics)[0],
        ),
    ]

    written_paths = list(trend_paths)
    for filename, figure in ml_figures:
        add_synthetic_label(figure)
        try:
            png_path, svg_path = save_figure_formats(
                figure,
                figure_dir / filename,
                dpi=220,
            )
            written_paths.extend([png_path, svg_path])
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)
    return written_paths


def add_synthetic_label(figure) -> None:
    """Add a visible public-demo label to a figure."""
    figure.text(
        0.99,
        0.008,
        "Synthetic demo data - not personal health records",
        ha="right",
        va="bottom",
        fontsize=8.3,
        color="#657786",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic public demo data and figures. These artifacts "
            "are not derived from personal Apple Health records."
        )
    )
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=Path("examples/synthetic"),
        help="Directory for synthetic demo CSV files.",
    )
    parser.add_argument(
        "--output-figure-dir",
        type=Path,
        default=Path("reports/demo_figures"),
        help="Directory for synthetic public demo figures.",
    )
    return parser


def main() -> int:
    """Generate all synthetic demo assets."""
    args = build_argument_parser().parse_args()

    try:
        build_demo_reports(args.output_data_dir)
        written_paths = build_demo_figures(
            args.output_data_dir,
            args.output_figure_dir,
        )
    except (
        DemoAssetError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Synthetic demo assets generated.")
    print(f"Synthetic CSV directory: {args.output_data_dir}")
    print(f"Synthetic figure directory: {args.output_figure_dir}")
    print(f"Figure files written: {len(written_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
