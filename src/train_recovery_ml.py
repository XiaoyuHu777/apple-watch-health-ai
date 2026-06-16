"""Train and evaluate next-day recovery regression and classification models."""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
REGRESSION_TARGET = "target_recovery_next_day"
CLASSIFICATION_TARGET = "low_recovery_next_day"
FIXED_START = pd.Timestamp("2025-06-01")
FIXED_TRAIN_END = pd.Timestamp("2025-10-31")
FIXED_TEST_START = pd.Timestamp("2025-11-01")
FIXED_END = pd.Timestamp("2025-12-31")

REQUIRED_COLUMNS = {
    "date",
    "recovery_score",
    "recovery_score_roll7_mean",
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
}

NON_FEATURE_COLUMNS = {
    "date",
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
    "split",
}


class MLTrainingError(RuntimeError):
    """Raised when recovery model training cannot proceed safely."""


@dataclass(frozen=True)
class TimeSplit:
    """Store target-valid train and test rows plus split metadata."""

    train: pd.DataFrame
    test: pd.DataFrame
    strategy: str


def read_dataset(input_path: Path) -> pd.DataFrame:
    """Read and validate the machine-learning dataset."""
    if not input_path.is_file():
        raise FileNotFoundError(f"ML dataset does not exist: {input_path}")

    try:
        frame = pd.read_csv(input_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise MLTrainingError(
            f"Could not read ML dataset {input_path}: {exc}"
        ) from exc

    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise MLTrainingError(
            "ML dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise MLTrainingError("ML dataset contains invalid dates")
    frame["date"] = frame["date"].dt.normalize()
    if frame["date"].duplicated().any():
        raise MLTrainingError("ML dataset contains duplicate dates")

    for column in frame.columns:
        if column != "date":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.sort_values("date").reset_index(drop=True)


def identify_numeric_features(frame: pd.DataFrame) -> list[str]:
    """Return numeric model inputs while excluding labels and output fields."""
    feature_columns = []
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if column.endswith("_pred") or column.endswith("_proba"):
            continue
        feature_columns.append(column)

    if not feature_columns:
        raise MLTrainingError("No numeric feature columns were found")
    return feature_columns


def covers_fixed_period(frame: pd.DataFrame) -> bool:
    """Return whether every date in the requested fixed period is present."""
    expected_dates = pd.date_range(FIXED_START, FIXED_END, freq="D")
    available_dates = pd.Index(frame["date"].unique())
    return expected_dates.isin(available_dates).all()


def make_time_split(frame: pd.DataFrame) -> TimeSplit:
    """Create the fixed 2025 split or a chronological final-20-percent split."""
    eligible = frame.dropna(
        subset=[REGRESSION_TARGET, CLASSIFICATION_TARGET]
    ).copy()
    if len(eligible) < 2:
        raise MLTrainingError(
            "At least two rows with non-missing targets are required"
        )

    if covers_fixed_period(frame):
        train = eligible[
            eligible["date"].between(FIXED_START, FIXED_TRAIN_END)
        ].copy()
        test = eligible[
            eligible["date"].between(FIXED_TEST_START, FIXED_END)
        ].copy()
        strategy = "fixed_2025_window"
    else:
        eligible = eligible.sort_values("date").reset_index(drop=True)
        test_size = max(1, math.ceil(len(eligible) * 0.20))
        split_index = len(eligible) - test_size
        if split_index < 1:
            raise MLTrainingError(
                "Not enough target-valid rows for an 80/20 time split"
            )
        train = eligible.iloc[:split_index].copy()
        test = eligible.iloc[split_index:].copy()
        strategy = "final_20_percent"

    if train.empty or test.empty:
        raise MLTrainingError(
            "The selected time split has no target-valid train or test rows"
        )
    if train["date"].max() >= test["date"].min():
        raise MLTrainingError(
            "Invalid time split: training dates must precede test dates"
        )

    train["split"] = "train"
    test["split"] = "test"
    return TimeSplit(train=train, test=test, strategy=strategy)


def linear_pipeline(model: BaseEstimator) -> Pipeline:
    """Build a median-imputed and standardized linear-model pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def tree_pipeline(model: BaseEstimator) -> Pipeline:
    """Build a median-imputed tree-model pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def regression_models() -> dict[str, Pipeline]:
    """Create the requested regression pipelines."""
    return {
        "Ridge Regression": linear_pipeline(
            Ridge(alpha=1.0, random_state=RANDOM_STATE)
        ),
        "RandomForestRegressor": tree_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        ),
        "HistGradientBoostingRegressor": tree_pipeline(
            HistGradientBoostingRegressor(
                max_iter=200,
                random_state=RANDOM_STATE,
            )
        ),
    }


def classification_models() -> dict[str, Pipeline]:
    """Create the requested classification pipelines."""
    return {
        "Logistic Regression": linear_pipeline(
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            )
        ),
        "RandomForestClassifier": tree_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        ),
        "HistGradientBoostingClassifier": tree_pipeline(
            HistGradientBoostingClassifier(
                max_iter=200,
                random_state=RANDOM_STATE,
            )
        ),
    }


def regression_metric_row(
    name: str,
    model_type: str,
    actual: pd.Series,
    predicted: pd.Series | np.ndarray,
) -> dict[str, object]:
    """Calculate regression metrics after excluding unavailable predictions."""
    actual_series = pd.Series(actual).reset_index(drop=True)
    predicted_series = pd.Series(predicted).reset_index(drop=True)
    valid = actual_series.notna() & predicted_series.notna()
    actual_valid = actual_series[valid]
    predicted_valid = predicted_series[valid]
    if actual_valid.empty:
        raise MLTrainingError(f"No valid predictions are available for {name}")

    return {
        "model": name,
        "model_type": model_type,
        "MAE": mean_absolute_error(actual_valid, predicted_valid),
        "RMSE": mean_squared_error(
            actual_valid,
            predicted_valid,
        )
        ** 0.5,
        "R2": (
            r2_score(actual_valid, predicted_valid)
            if len(actual_valid) >= 2
            else np.nan
        ),
        "evaluation_samples": len(actual_valid),
    }


def classification_metric_row(
    name: str,
    actual: pd.Series,
    predicted: np.ndarray,
    probability: np.ndarray,
) -> dict[str, object]:
    """Calculate classification metrics, allowing single-class test data."""
    actual_values = actual.astype("int8").to_numpy()
    roc_auc = (
        roc_auc_score(actual_values, probability)
        if np.unique(actual_values).size == 2
        else np.nan
    )
    return {
        "model": name,
        "Accuracy": accuracy_score(actual_values, predicted),
        "Precision": precision_score(
            actual_values,
            predicted,
            zero_division=0,
        ),
        "Recall": recall_score(
            actual_values,
            predicted,
            zero_division=0,
        ),
        "F1": f1_score(actual_values, predicted, zero_division=0),
        "ROC-AUC": roc_auc,
        "evaluation_samples": len(actual_values),
    }


def positive_class_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return predicted probability for class 1 from a fitted pipeline."""
    classes = model.named_steps["model"].classes_
    positive_indices = np.flatnonzero(classes == 1)
    if positive_indices.size != 1:
        raise MLTrainingError("Classification model has no class 1 probability")
    return model.predict_proba(features)[:, positive_indices[0]]


def train_regression_models(
    split: TimeSplit,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, str, Pipeline]:
    """Fit regression models and evaluate them with the two baselines."""
    train_x = split.train[feature_columns]
    train_y = split.train[REGRESSION_TARGET]
    test_x = split.test[feature_columns]
    test_y = split.test[REGRESSION_TARGET]

    prediction_by_name: dict[str, pd.Series | np.ndarray] = {
        "Persistence Baseline": split.test["recovery_score"],
        "7-Day Rolling Mean Baseline": split.test[
            "recovery_score_roll7_mean"
        ],
    }
    rows = [
        regression_metric_row(
            "Persistence Baseline",
            "baseline",
            test_y,
            prediction_by_name["Persistence Baseline"],
        ),
        regression_metric_row(
            "7-Day Rolling Mean Baseline",
            "baseline",
            test_y,
            prediction_by_name["7-Day Rolling Mean Baseline"],
        ),
    ]
    fitted_models: dict[str, Pipeline] = {}
    for name, model in regression_models().items():
        model.fit(train_x, train_y)
        fitted_models[name] = model
        prediction_by_name[name] = model.predict(test_x)
        rows.append(
            regression_metric_row(
                name,
                "model",
                test_y,
                prediction_by_name[name],
            )
        )

    metrics = pd.DataFrame(rows)
    common_mask = (
        test_y.notna()
        & split.test["recovery_score"].notna()
        & split.test["recovery_score_roll7_mean"].notna()
    )
    for row_index, name in metrics["model"].items():
        common_metrics = regression_metric_row(
            str(name),
            str(metrics.loc[row_index, "model_type"]),
            test_y[common_mask],
            pd.Series(
                prediction_by_name[str(name)],
                index=test_y.index,
            )[common_mask],
        )
        metrics.loc[row_index, "common_MAE"] = common_metrics["MAE"]
        metrics.loc[row_index, "common_RMSE"] = common_metrics["RMSE"]
        metrics.loc[row_index, "common_R2"] = common_metrics["R2"]
        metrics.loc[row_index, "common_evaluation_samples"] = common_metrics[
            "evaluation_samples"
        ]

    model_metrics = metrics[metrics["model_type"] == "model"].sort_values(
        ["MAE", "RMSE", "model"],
        ascending=[True, True, True],
    )
    best_name = str(model_metrics.iloc[0]["model"])
    metrics["is_best_model"] = metrics["model"].eq(best_name)
    return metrics, best_name, fitted_models[best_name]


def train_classification_models(
    split: TimeSplit,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, str, Pipeline]:
    """Fit and evaluate the requested classification models."""
    train_x = split.train[feature_columns]
    train_y = split.train[CLASSIFICATION_TARGET].astype("int8")
    test_x = split.test[feature_columns]
    test_y = split.test[CLASSIFICATION_TARGET].astype("int8")
    if train_y.nunique() < 2:
        raise MLTrainingError(
            "Classification training data must contain both target classes"
        )

    rows = []
    fitted_models: dict[str, Pipeline] = {}
    for name, model in classification_models().items():
        model.fit(train_x, train_y)
        fitted_models[name] = model
        prediction = model.predict(test_x).astype("int8")
        probability = positive_class_probability(model, test_x)
        rows.append(
            classification_metric_row(
                name,
                test_y,
                prediction,
                probability,
            )
        )

    metrics = pd.DataFrame(rows)
    ranked = metrics.assign(
        roc_auc_rank=metrics["ROC-AUC"].fillna(float("-inf"))
    ).sort_values(
        ["F1", "roc_auc_rank", "Accuracy", "model"],
        ascending=[False, False, False, True],
    )
    best_name = str(ranked.iloc[0]["model"])
    metrics["is_best_model"] = metrics["model"].eq(best_name)
    return metrics, best_name, fitted_models[best_name]


def build_predictions(
    split: TimeSplit,
    feature_columns: list[str],
    best_regression: Pipeline,
    best_classification: Pipeline,
) -> pd.DataFrame:
    """Build train and test predictions from the selected fitted models."""
    rows = pd.concat([split.train, split.test], ignore_index=True)
    features = rows[feature_columns]
    output = rows[
        [
            "date",
            "recovery_score",
            REGRESSION_TARGET,
            CLASSIFICATION_TARGET,
            "split",
        ]
    ].copy()
    output.insert(
        3,
        "baseline_persistence_pred",
        rows["recovery_score"],
    )
    output.insert(
        4,
        "baseline_roll7_pred",
        rows["recovery_score_roll7_mean"],
    )
    output.insert(
        5,
        "best_regression_pred",
        best_regression.predict(features),
    )
    output.insert(
        7,
        "best_classification_pred",
        best_classification.predict(features).astype("int8"),
    )
    output.insert(
        8,
        "best_classification_proba",
        positive_class_probability(best_classification, features),
    )
    output[CLASSIFICATION_TARGET] = output[CLASSIFICATION_TARGET].astype("int8")
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    return output


def write_csv_atomically(frame: pd.DataFrame, output_path: Path) -> None:
    """Write a report through a temporary file to avoid partial output."""
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
        raise MLTrainingError(f"Could not write report {output_path}: {exc}") from exc


def date_range_text(frame: pd.DataFrame) -> str:
    """Return an ISO date range for a non-empty split."""
    return (
        f"{frame['date'].min().date().isoformat()} to "
        f"{frame['date'].max().date().isoformat()}"
    )


def print_summary(
    split: TimeSplit,
    feature_columns: list[str],
    regression_metrics: pd.DataFrame,
    best_regression_name: str,
    classification_metrics: pd.DataFrame,
    best_classification_name: str,
) -> None:
    """Print aggregate split and model results without daily health records."""
    regression_best = regression_metrics.set_index("model").loc[
        best_regression_name
    ]
    classification_best = classification_metrics.set_index("model").loc[
        best_classification_name
    ]

    print("Recovery ML training completed.")
    print(f"Split strategy: {split.strategy}")
    print(f"Numeric features: {len(feature_columns)}")
    print(f"Train: {date_range_text(split.train)} ({len(split.train)} rows)")
    print(f"Test: {date_range_text(split.test)} ({len(split.test)} rows)")
    print(
        f"Best regression model: {best_regression_name} "
        f"(MAE={regression_best['MAE']:.4f})"
    )
    print(
        f"Best classification model: {best_classification_name} "
        f"(F1={classification_best['F1']:.4f})"
    )
    if split.test[CLASSIFICATION_TARGET].nunique() < 2:
        print(
            "ROC-AUC is NaN because the classification test set contains "
            "only one class."
        )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate next-day recovery regression and "
            "classification models from the aggregate ML dataset."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to ml_recovery_dataset.csv.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="Directory for metric and prediction CSV reports.",
    )
    return parser


def main() -> int:
    """Run recovery model training and report generation."""
    args = build_argument_parser().parse_args()

    try:
        dataset = read_dataset(args.input)
        feature_columns = identify_numeric_features(dataset)
        split = make_time_split(dataset)
        regression_metrics, best_regression_name, best_regression = (
            train_regression_models(split, feature_columns)
        )
        classification_metrics, best_classification_name, best_classification = (
            train_classification_models(split, feature_columns)
        )
        predictions = build_predictions(
            split,
            feature_columns,
            best_regression,
            best_classification,
        )

        write_csv_atomically(
            regression_metrics,
            args.reports_dir / "ml_regression_metrics.csv",
        )
        write_csv_atomically(
            classification_metrics,
            args.reports_dir / "ml_classification_metrics.csv",
        )
        write_csv_atomically(
            predictions,
            args.reports_dir / "ml_predictions.csv",
        )
    except (
        FileNotFoundError,
        MLTrainingError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(
        split,
        feature_columns,
        regression_metrics,
        best_regression_name,
        classification_metrics,
        best_classification_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
