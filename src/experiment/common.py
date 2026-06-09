"""Shared helpers for polynomial regression experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures

DEFAULT_DEGREE_CANDIDATES = tuple(range(1, 9))
DEFAULT_ALPHA_CANDIDATES = (
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
)


def load_csv_dataset(
    csv_path: str | Path,
    target_column: str = "y",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load feature values and target values from a CSV file."""
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV file must include a header row.")
        if target_column not in reader.fieldnames:
            raise ValueError(f"CSV file must include a '{target_column}' column.")

        feature_columns = [
            column for column in reader.fieldnames if column != target_column
        ]
        if not feature_columns:
            raise ValueError("CSV file must include at least one feature column.")

        x_rows: list[list[float]] = []
        y_values: list[float] = []
        for row in reader:
            x_rows.append([float(row[column]) for column in feature_columns])
            y_values.append(float(row[target_column]))

    if not x_rows:
        raise ValueError("CSV file must include at least one data row.")

    return np.array(x_rows, dtype=float), np.array(y_values, dtype=float), feature_columns


def validate_degree_alpha(degree: int, alpha: float) -> None:
    """Validate polynomial degree and regularization strength."""
    if degree <= 0:
        raise ValueError("degree must be positive.")
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")


def format_feature_name(feature_name: str) -> str:
    """Format a scikit-learn polynomial feature name for display."""
    return feature_name.replace(" ", "*")


def train_validation_split(
    x_values: np.ndarray,
    y_values: np.ndarray,
    validation_fraction: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a dataset into deterministic training and validation subsets."""
    if x_values.shape[0] < 2:
        raise ValueError("CSV file must include at least two data rows.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    row_indices = rng.permutation(x_values.shape[0])
    validation_size = max(1, int(round(x_values.shape[0] * validation_fraction)))
    validation_size = min(validation_size, x_values.shape[0] - 1)

    validation_indices = row_indices[:validation_size]
    train_indices = row_indices[validation_size:]

    return (
        x_values[train_indices],
        x_values[validation_indices],
        y_values[train_indices],
        y_values[validation_indices],
    )


def fit_polynomial_model(
    model: Lasso | Ridge,
    x_values: np.ndarray,
    y_values: np.ndarray,
    feature_columns: list[str],
    degree: int,
) -> tuple[Lasso | Ridge, PolynomialFeatures, list[str]]:
    """Fit a model after expanding input values into polynomial features."""
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    x_polynomial = polynomial_features.fit_transform(x_values)
    polynomial_feature_names = polynomial_features.get_feature_names_out(
        feature_columns
    ).tolist()
    model.fit(x_polynomial, y_values)

    return model, polynomial_features, polynomial_feature_names


def mean_squared_error(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """Calculate mean squared error without adding another dependency surface."""
    return float(np.mean((y_true - y_predicted) ** 2))


def is_better_candidate(
    candidate: dict[str, Any],
    best: dict[str, Any] | None,
    tolerance: float = 1e-12,
) -> bool:
    """Return whether a candidate is better than the current best result."""
    if best is None:
        return True

    candidate_mse = float(candidate["validation_mse"])
    best_mse = float(best["validation_mse"])
    if candidate_mse < best_mse - tolerance:
        return True
    if abs(candidate_mse - best_mse) > tolerance:
        return False

    return (
        int(candidate["degree"]),
        float(candidate["alpha"]),
        str(candidate["regularization"]),
    ) < (
        int(best["degree"]),
        float(best["alpha"]),
        str(best["regularization"]),
    )


def format_polynomial(
    intercept: float,
    feature_names: list[str],
    coefficients: np.ndarray,
    zero_tolerance: float = 1e-10,
) -> str:
    """Format fitted polynomial coefficients as a readable equation."""
    intercept = 0.0 if abs(intercept) < zero_tolerance else intercept
    polynomial = f"y = {intercept:.6g}"

    for feature_name, coefficient in zip(feature_names, coefficients):
        if abs(coefficient) < zero_tolerance:
            continue

        sign = "+" if coefficient >= 0 else "-"
        formatted_name = format_feature_name(feature_name)
        polynomial += f" {sign} {abs(coefficient):.6g}*{formatted_name}"

    return polynomial
