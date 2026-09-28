"""ElasticNet polynomial regression experiment helpers."""

from pathlib import Path

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures

from src.experiment.common import (
    fit_polynomial_model, format_polynomial, load_csv_dataset, validate_degree_alpha,
)


def fit_elastic_net_polynomial(
    x_values: np.ndarray,
    y_values: np.ndarray,
    feature_columns: list[str],
    degree: int,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
) -> tuple[ElasticNet, PolynomialFeatures, list[str]]:
    """Fit ElasticNet on polynomial features in the CSV variable order."""
    validate_degree_alpha(degree, alpha)
    if not np.isfinite(l1_ratio) or not 0 <= l1_ratio <= 1:
        raise ValueError("l1_ratio must be between 0 and 1.")
    return fit_polynomial_model(
        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50000), x_values, y_values, feature_columns, degree,
    )


def fit_elastic_net_polynomial_from_csv(
    csv_path: str | Path,
    degree: int,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    target_column: str = "y",
) -> str:
    """Fit ElasticNet from CSV and return a readable polynomial."""
    x_values, y_values, columns = load_csv_dataset(csv_path, target_column)
    model, _, names = fit_elastic_net_polynomial(
        x_values, y_values, columns, degree, alpha=alpha, l1_ratio=l1_ratio
    )
    return format_polynomial(float(model.intercept_), names, model.coef_)
