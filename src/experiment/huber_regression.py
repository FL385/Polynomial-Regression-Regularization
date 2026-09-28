"""HuberRegressor polynomial regression experiment helpers."""

from pathlib import Path

import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures

from src.experiment.common import (
    fit_polynomial_model, format_polynomial, load_csv_dataset, validate_degree_alpha,
)


def fit_huber_polynomial(
    x_values: np.ndarray,
    y_values: np.ndarray,
    feature_columns: list[str],
    degree: int,
    alpha: float = 0.0001,
    epsilon: float = 1.35,
) -> tuple[HuberRegressor, PolynomialFeatures, list[str]]:
    """Fit HuberRegressor on polynomial features in the CSV variable order."""
    validate_degree_alpha(degree, alpha)
    if not np.isfinite(epsilon) or epsilon < 1:
        raise ValueError("epsilon must be finite and at least 1.")
    return fit_polynomial_model(
        HuberRegressor(alpha=alpha, epsilon=epsilon, max_iter=3000), x_values, y_values, feature_columns, degree,
    )


def fit_huber_polynomial_from_csv(
    csv_path: str | Path,
    degree: int,
    alpha: float = 0.0001,
    epsilon: float = 1.35,
    target_column: str = "y",
) -> str:
    """Fit HuberRegressor from CSV and return a readable polynomial."""
    x_values, y_values, columns = load_csv_dataset(csv_path, target_column)
    model, _, names = fit_huber_polynomial(
        x_values, y_values, columns, degree, alpha=alpha, epsilon=epsilon
    )
    return format_polynomial(float(model.intercept_), names, model.coef_)

