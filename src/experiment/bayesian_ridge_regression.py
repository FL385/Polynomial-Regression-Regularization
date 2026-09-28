"""BayesianRidge polynomial regression experiment helpers."""

from pathlib import Path

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures

from src.experiment.common import (
    fit_polynomial_model, format_polynomial, load_csv_dataset, validate_degree_alpha,
)


def fit_bayesian_ridge_polynomial(
    x_values: np.ndarray,
    y_values: np.ndarray,
    feature_columns: list[str],
    degree: int,
) -> tuple[BayesianRidge, PolynomialFeatures, list[str]]:
    """Fit BayesianRidge on polynomial features in the CSV variable order."""
    validate_degree_alpha(degree, 0.0)
    return fit_polynomial_model(
        BayesianRidge(), x_values, y_values, feature_columns, degree,
    )


def fit_bayesian_ridge_polynomial_from_csv(
    csv_path: str | Path,
    degree: int,
    target_column: str = "y",
) -> str:
    """Fit BayesianRidge from CSV and return a readable polynomial."""
    x_values, y_values, columns = load_csv_dataset(csv_path, target_column)
    model, _, names = fit_bayesian_ridge_polynomial(
        x_values, y_values, columns, degree,
    )
    return format_polynomial(float(model.intercept_), names, model.coef_)
