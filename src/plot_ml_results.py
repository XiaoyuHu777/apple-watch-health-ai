"""Create visual reports for next-day personal recovery prediction results."""

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
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error

if __package__:
    from .train_recovery_ml import (
        REGRESSION_TARGET,
        identify_numeric_features,
        make_time_split,
        read_dataset,
        regression_models,
    )
    from .visualize import save_figure_formats
else:
    from train_recovery_ml import (
        REGRESSION_TARGET,
        identify_numeric_features,
        make_time_split,
        read_dataset,
        regression_models,
    )
    from visualize import save_figure_formats


TEXT_COLOR = "#22313F"
MUTED_TEXT_COLOR = "#657786"
GRID_COLOR = "#DCE3E8"
ACTUAL_COLOR = "#123F5D"
MODEL_COLOR = "#D16B4E"
BASELINE_COLOR = "#8296A3"
POSITIVE_COLOR = "#4D8060"
NEGATIVE_COLOR = "#A85A60"

PREDICTION_COLUMNS = {
    "date",
    "target_recovery_next_day",
    "best_regression_pred",
    "baseline_persistence_pred",
    "low_recovery_next_day",
    "best_classification_pred",
    "split",
}

REGRESSION_METRIC_COLUMNS = {
    "model",
    "model_type",
    "MAE",
    "RMSE",
    "is_best_model",
}

CLASSIFICATION_METRIC_COLUMNS = {
    "model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "is_best_model",
}


class MLPlotError(RuntimeError):
    """Raised when machine-learning result figures cannot be created."""


def read_csv(path: Path, required_columns: set[str], label: str) -> pd.DataFrame:
    """Read a report CSV and validate its required columns."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} CSV does not exist: {path}")

    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise MLPlotError(f"Could not read {label} CSV {path}: {exc}") from exc

    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise MLPlotError(
            f"{label} CSV is missing columns: {', '.join(missing_columns)}"
        )
    return frame


def load_plot_inputs(
    predictions_path: Path,
    regression_metrics_path: Path,
    classification_metrics_path: Path,
    ml_dataset_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate all reports needed for plotting."""
    predictions = read_csv(
        predictions_path,
        PREDICTION_COLUMNS,
        "Predictions",
    )
    regression_metrics = read_csv(
        regression_metrics_path,
        REGRESSION_METRIC_COLUMNS,
        "Regression metrics",
    )
    classification_metrics = read_csv(
        classification_metrics_path,
        CLASSIFICATION_METRIC_COLUMNS,
        "Classification metrics",
    )
    dataset = read_dataset(ml_dataset_path)

    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(
        predictions["date"],
        errors="coerce",
    )
    if predictions["date"].isna().any():
        raise MLPlotError("Predictions CSV contains invalid dates")
    if predictions["date"].duplicated().any():
        raise MLPlotError("Predictions CSV contains duplicate dates")

    numeric_columns = PREDICTION_COLUMNS - {"date", "split"}
    for column in numeric_columns:
        predictions[column] = pd.to_numeric(
            predictions[column],
            errors="coerce",
        )

    test_predictions = (
        predictions[predictions["split"].eq("test")]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if test_predictions.empty:
        raise MLPlotError("Predictions CSV contains no test rows")
    if test_predictions[
        ["target_recovery_next_day", "best_regression_pred"]
    ].isna().any().any():
        raise MLPlotError(
            "Test rows contain missing actual or best regression predictions"
        )

    return (
        test_predictions,
        regression_metrics,
        classification_metrics,
        dataset,
    )


def _style_axis(axis: Axes, *, grid_axis: str = "both") -> None:
    """Apply the project report style to an axis."""
    axis.set_facecolor("white")
    axis.grid(
        axis=grid_axis,
        color=GRID_COLOR,
        linewidth=0.7,
        alpha=0.55,
        zorder=0,
    )
    axis.tick_params(
        axis="both",
        colors=MUTED_TEXT_COLOR,
        labelsize=9.5,
        length=0,
        pad=6,
    )
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(GRID_COLOR)
        axis.spines[spine].set_linewidth(0.8)


def _set_title(axis: Axes, title: str, subtitle: str) -> None:
    """Add a consistent title and subtitle."""
    axis.set_title(
        title,
        loc="left",
        fontsize=17,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=34,
    )
    axis.text(
        0,
        1.025,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=MUTED_TEXT_COLOR,
    )


def _new_figure(figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Create a report-ready figure and axis."""
    figure, axis = plt.subplots(figsize=figsize, facecolor="#FAFBFC")
    return figure, axis


def _placeholder_figure(title: str, message: str) -> Figure:
    """Create an explanatory figure when a requested plot is unavailable."""
    figure, axis = _new_figure((9, 5.5))
    axis.axis("off")
    axis.text(
        0.5,
        0.62,
        title,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    axis.text(
        0.5,
        0.44,
        message,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color=MUTED_TEXT_COLOR,
        linespacing=1.5,
        wrap=True,
    )
    figure.tight_layout()
    return figure


def plot_predicted_vs_actual(test_predictions: pd.DataFrame) -> Figure:
    """Plot test-set predicted scores against actual next-day scores."""
    actual = test_predictions["target_recovery_next_day"]
    predicted = test_predictions["best_regression_pred"]
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    margin = max((upper - lower) * 0.08, 2)

    figure, axis = _new_figure((8, 7))
    axis.scatter(
        actual,
        predicted,
        s=64,
        color=MODEL_COLOR,
        edgecolor="white",
        linewidth=1.0,
        alpha=0.88,
        zorder=3,
    )
    axis.plot(
        [lower - margin, upper + margin],
        [lower - margin, upper + margin],
        color=BASELINE_COLOR,
        linewidth=1.5,
        linestyle=(0, (5, 4)),
        label="Perfect prediction (y = x)",
        zorder=2,
    )
    axis.set_xlim(lower - margin, upper + margin)
    axis.set_ylim(lower - margin, upper + margin)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Actual next-day recovery score", color=TEXT_COLOR, labelpad=10)
    axis.set_ylabel(
        "Predicted next-day recovery score",
        color=TEXT_COLOR,
        labelpad=10,
    )
    axis.text(
        0.04,
        0.95,
        f"MAE  {mae:.2f}\nRMSE {rmse:.2f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontfamily="monospace",
        color=TEXT_COLOR,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": GRID_COLOR,
            "alpha": 0.96,
        },
        zorder=4,
    )
    _set_title(
        axis,
        "Predicted vs Actual Recovery",
        f"Test set only | {len(test_predictions)} target-valid dates",
    )
    _style_axis(axis)
    axis.legend(loc="lower right", frameon=False, fontsize=9)
    figure.tight_layout(rect=(0.03, 0.03, 0.98, 0.94))
    return figure


def plot_prediction_timeline(test_predictions: pd.DataFrame) -> Figure:
    """Plot actual, best-model, and persistence predictions over time."""
    figure, axis = _new_figure((13, 6.5))
    axis.plot(
        test_predictions["date"],
        test_predictions["target_recovery_next_day"],
        color=ACTUAL_COLOR,
        linewidth=2.4,
        marker="o",
        markersize=5.2,
        label="Actual next-day recovery",
        zorder=4,
    )
    axis.plot(
        test_predictions["date"],
        test_predictions["best_regression_pred"],
        color=MODEL_COLOR,
        linewidth=2.1,
        marker="o",
        markersize=4.2,
        alpha=0.92,
        label="Best regression prediction",
        zorder=3,
    )
    axis.plot(
        test_predictions["date"],
        test_predictions["baseline_persistence_pred"],
        color=BASELINE_COLOR,
        linewidth=1.7,
        linestyle=(0, (5, 3)),
        marker=".",
        markersize=5,
        alpha=0.9,
        label="Persistence baseline",
        zorder=2,
    )
    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axis.set_xlim(
        test_predictions["date"].min() - pd.Timedelta(days=1),
        test_predictions["date"].max() + pd.Timedelta(days=1),
    )
    axis.set_ylabel("Next-day recovery score", color=TEXT_COLOR, labelpad=10)
    axis.set_xlabel("")
    axis.set_ylim(0, 100)
    _set_title(
        axis,
        "Next-Day Recovery Prediction Timeline",
        (
            f"Test period: {test_predictions['date'].min():%Y-%m-%d} to "
            f"{test_predictions['date'].max():%Y-%m-%d}"
        ),
    )
    _style_axis(axis, grid_axis="y")
    axis.legend(
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=9.2,
        handlelength=2.8,
    )
    figure.tight_layout(rect=(0.03, 0.03, 0.99, 0.94))
    return figure


def plot_residuals(test_predictions: pd.DataFrame) -> Figure:
    """Plot test residuals and annotate the five largest absolute errors."""
    residuals = (
        test_predictions["target_recovery_next_day"]
        - test_predictions["best_regression_pred"]
    )
    figure, axis = _new_figure((13, 6.5))
    colors = np.where(residuals >= 0, POSITIVE_COLOR, NEGATIVE_COLOR)
    axis.vlines(
        test_predictions["date"],
        0,
        residuals,
        color=colors,
        linewidth=1.2,
        alpha=0.55,
        zorder=2,
    )
    axis.scatter(
        test_predictions["date"],
        residuals,
        c=colors,
        s=45,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    axis.axhline(
        0,
        color=TEXT_COLOR,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        alpha=0.8,
        zorder=1,
    )

    largest_indices = residuals.abs().nlargest(min(5, len(residuals))).index
    for annotation_number, index in enumerate(largest_indices):
        date = test_predictions.loc[index, "date"]
        residual = residuals.loc[index]
        vertical_offset = 14 if residual >= 0 else -24
        horizontal_offset = 8 if annotation_number % 2 == 0 else -42
        axis.annotate(
            f"{date:%b %d}\n{residual:+.1f}",
            xy=(date, residual),
            xytext=(horizontal_offset, vertical_offset),
            textcoords="offset points",
            ha="left",
            va="bottom" if residual >= 0 else "top",
            fontsize=8.5,
            color=TEXT_COLOR,
            arrowprops={
                "arrowstyle": "-",
                "color": MUTED_TEXT_COLOR,
                "linewidth": 0.8,
            },
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": GRID_COLOR,
                "alpha": 0.94,
            },
            zorder=4,
        )

    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axis.set_xlim(
        test_predictions["date"].min() - pd.Timedelta(days=1),
        test_predictions["date"].max() + pd.Timedelta(days=1),
    )
    residual_limit = max(float(residuals.abs().max()) * 1.28, 5)
    axis.set_ylim(-residual_limit, residual_limit)
    axis.set_ylabel("Residual: actual - predicted", color=TEXT_COLOR, labelpad=10)
    axis.set_xlabel("")
    _set_title(
        axis,
        "Regression Residuals",
        "Positive residuals indicate underprediction | Top 5 errors annotated",
    )
    _style_axis(axis, grid_axis="y")
    figure.tight_layout(rect=(0.03, 0.03, 0.99, 0.94))
    return figure


def plot_confusion_matrix(test_predictions: pd.DataFrame) -> Figure:
    """Plot a test-set low-recovery confusion matrix or a placeholder."""
    valid = test_predictions.dropna(
        subset=["low_recovery_next_day", "best_classification_pred"]
    )
    if valid.empty or valid["low_recovery_next_day"].nunique() < 2:
        return _placeholder_figure(
            "Low-Recovery Confusion Matrix",
            (
                "The test set does not contain both target classes.\n"
                "A confusion matrix would not support a meaningful comparison."
            ),
        )

    actual = valid["low_recovery_next_day"].astype("int8")
    predicted = valid["best_classification_pred"].astype("int8")
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])

    figure, axis = _new_figure((7.2, 6.2))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            text_color = "white" if value > matrix.max() * 0.55 else TEXT_COLOR
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color=text_color,
            )

    axis.set_xticks([0, 1], labels=["Not low (0)", "Low (1)"])
    axis.set_yticks([0, 1], labels=["Not low (0)", "Low (1)"])
    axis.set_xlabel("Predicted class", color=TEXT_COLOR, labelpad=10)
    axis.set_ylabel("Actual class", color=TEXT_COLOR, labelpad=10)
    _set_title(
        axis,
        "Low-Recovery Confusion Matrix",
        (
            f"Test set only | Actual low-recovery dates: {int(actual.sum())} "
            f"of {len(actual)}"
        ),
    )
    axis.tick_params(colors=MUTED_TEXT_COLOR, length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(colors=MUTED_TEXT_COLOR, length=0)
    figure.tight_layout(rect=(0.03, 0.03, 0.98, 0.94))
    return figure


def best_regression_name(regression_metrics: pd.DataFrame) -> str:
    """Return the model marked best in the regression metrics report."""
    best_mask = regression_metrics["is_best_model"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    best_rows = regression_metrics[best_mask]
    if len(best_rows) != 1:
        raise MLPlotError(
            "Regression metrics must identify exactly one best model"
        )
    return str(best_rows.iloc[0]["model"])


def extract_feature_importance(
    dataset: pd.DataFrame,
    regression_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame | None, str, str]:
    """Refit the reported best model and extract native feature importance."""
    model_name = best_regression_name(regression_metrics)
    available_models = regression_models()
    if model_name not in available_models:
        return None, model_name, "The reported best model is not available."

    feature_columns = identify_numeric_features(dataset)
    split = make_time_split(dataset)
    pipeline = available_models[model_name]
    pipeline.fit(
        split.train[feature_columns],
        split.train[REGRESSION_TARGET],
    )
    estimator = pipeline.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype="float64")
        measure = "Feature importance"
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype="float64").ravel())
        measure = "Absolute coefficient"
    else:
        return (
            None,
            model_name,
            "The best regression model has no native importance or coefficient.",
        )

    if len(values) != len(feature_columns):
        return (
            None,
            model_name,
            "The transformed feature count does not match the input columns.",
        )

    importance = (
        pd.DataFrame({"feature": feature_columns, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance, model_name, measure


def plot_feature_importance(
    dataset: pd.DataFrame,
    regression_metrics: pd.DataFrame,
) -> tuple[Figure, pd.DataFrame | None]:
    """Plot the top 15 features for the reported best regression model."""
    importance, model_name, context = extract_feature_importance(
        dataset,
        regression_metrics,
    )
    if importance is None:
        return (
            _placeholder_figure(
                "Regression Feature Importance",
                f"{model_name}\n{context}",
            ),
            None,
        )

    top_features = importance.head(15).sort_values(
        "importance",
        ascending=True,
    )
    figure, axis = _new_figure((10.5, 7.5))
    axis.barh(
        top_features["feature"],
        top_features["importance"],
        color=ACTUAL_COLOR,
        alpha=0.88,
        height=0.68,
        zorder=2,
    )
    axis.set_xlabel(context, color=TEXT_COLOR, labelpad=10)
    axis.set_ylabel("")
    _set_title(
        axis,
        "Top 15 Regression Features",
        (
            f"{model_name} | Native model importance describes predictive "
            "signal, not causation"
        ),
    )
    _style_axis(axis, grid_axis="x")
    figure.tight_layout(rect=(0.03, 0.03, 0.99, 0.94))
    return figure, importance


def generate_all_figures(
    test_predictions: pd.DataFrame,
    regression_metrics: pd.DataFrame,
    dataset: pd.DataFrame,
    output_dir: Path,
) -> tuple[list[Path], pd.DataFrame | None]:
    """Generate and save all requested PNG and SVG result figures."""
    figures = [
        (
            "ml_predicted_vs_actual",
            plot_predicted_vs_actual(test_predictions),
        ),
        (
            "ml_recovery_prediction_timeline",
            plot_prediction_timeline(test_predictions),
        ),
        ("ml_residuals", plot_residuals(test_predictions)),
        ("ml_confusion_matrix", plot_confusion_matrix(test_predictions)),
    ]
    importance_figure, importance = plot_feature_importance(
        dataset,
        regression_metrics,
    )
    figures.append(("ml_feature_importance", importance_figure))

    written_paths: list[Path] = []
    for filename, figure in figures:
        try:
            png_path, svg_path = save_figure_formats(
                figure,
                output_dir / filename,
                dpi=240,
            )
            written_paths.extend([png_path, svg_path])
        finally:
            plt.close(figure)
    return written_paths, importance


def print_summary(
    test_predictions: pd.DataFrame,
    regression_metrics: pd.DataFrame,
    classification_metrics: pd.DataFrame,
    written_paths: list[Path],
    importance: pd.DataFrame | None,
) -> None:
    """Print aggregate visualization output without daily health records."""
    best_regression = best_regression_name(regression_metrics)
    best_classification_rows = classification_metrics[
        classification_metrics["is_best_model"]
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    ]
    best_classification = (
        str(best_classification_rows.iloc[0]["model"])
        if len(best_classification_rows) == 1
        else "not uniquely identified"
    )
    print("ML result visualization completed.")
    print(f"Test rows visualized: {len(test_predictions)}")
    print(f"Best regression model: {best_regression}")
    print(f"Best classification model: {best_classification}")
    print(f"Figure files written: {len(written_paths)}")
    if importance is not None:
        print(
            "Top regression features: "
            + ", ".join(importance.head(5)["feature"].tolist())
        )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create test-set visualizations for next-day personal recovery "
            "prediction reports."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to ml_predictions.csv.",
    )
    parser.add_argument(
        "--regression-metrics",
        type=Path,
        required=True,
        help="Path to ml_regression_metrics.csv.",
    )
    parser.add_argument(
        "--classification-metrics",
        type=Path,
        required=True,
        help="Path to ml_classification_metrics.csv.",
    )
    parser.add_argument(
        "--ml-dataset",
        type=Path,
        required=True,
        help="Path to ml_recovery_dataset.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for PNG and SVG figures.",
    )
    return parser


def main() -> int:
    """Run ML result visualization."""
    args = build_argument_parser().parse_args()

    try:
        (
            test_predictions,
            regression_metrics,
            classification_metrics,
            dataset,
        ) = load_plot_inputs(
            args.predictions,
            args.regression_metrics,
            args.classification_metrics,
            args.ml_dataset,
        )
        written_paths, importance = generate_all_figures(
            test_predictions,
            regression_metrics,
            dataset,
            args.output_dir,
        )
    except (
        FileNotFoundError,
        MLPlotError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(
        test_predictions,
        regression_metrics,
        classification_metrics,
        written_paths,
        importance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
