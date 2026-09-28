"""ARDRegression polynomial regression experiment helpers."""

from pathlib import Path

import numpy as np
from sklearn.linear_model import ARDRegression
from sklearn.preprocessing import PolynomialFeatures

from src.experiment.common import (
    fit_polynomial_model, format_polynomial, load_csv_dataset, validate_degree_alpha,
)


def fit_ard_polynomial(
    x_values: np.ndarray,
    y_values: np.ndarray,
    feature_columns: list[str],
    degree: int,
    threshold_lambda: float = 10000.0,
) -> tuple[ARDRegression, PolynomialFeatures, list[str]]:
    """Fit ARDRegression on polynomial features in the CSV variable order."""
    validate_degree_alpha(degree, 0.0)
    if not np.isfinite(threshold_lambda) or threshold_lambda <= 0:
        raise ValueError("threshold_lambda must be finite and positive.")
    return fit_polynomial_model(
        ARDRegression(threshold_lambda=threshold_lambda), x_values, y_values, feature_columns, degree,
    )


def fit_ard_polynomial_from_csv(
    csv_path: str | Path,
    degree: int,
    threshold_lambda: float = 10000.0,
    target_column: str = "y",
) -> str:
    """Fit ARDRegression from CSV and return a readable polynomial."""
    x_values, y_values, columns = load_csv_dataset(csv_path, target_column)
    model, _, names = fit_ard_polynomial(
        x_values, y_values, columns, degree, threshold_lambda=threshold_lambda
    )
    return format_polynomial(float(model.intercept_), names, model.coef_)

