"""L2 regularized polynomial regression experiment helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from src.experiment.common import (
    fit_polynomial_model,
    format_polynomial,
    load_csv_dataset,
    validate_degree_alpha,
)


def fit_l2_polynomial(
    x_values: np.ndarray,
    y_values: np.ndarray,
    feature_columns: list[str],
    degree: int,
    alpha: float = 1.0,
) -> tuple[Ridge, PolynomialFeatures, list[str]]:
    """Fit a Ridge polynomial regression model on loaded data."""
    validate_degree_alpha(degree=degree, alpha=alpha)
    model = Ridge(alpha=alpha)
    fitted_model, polynomial_features, feature_names = fit_polynomial_model(
        model=model,
        x_values=x_values,
        y_values=y_values,
        feature_columns=feature_columns,
        degree=degree,
    )

    return fitted_model, polynomial_features, feature_names


def fit_l2_polynomial_from_csv(
    csv_path: str | Path,
    degree: int,
    alpha: float = 1.0,
    target_column: str = "y",
) -> str:
    """Fit L2 regularized polynomial regression from a CSV dataset."""
    x_values, y_values, feature_columns = load_csv_dataset(
        csv_path=csv_path,
        target_column=target_column,
    )
    model, _, feature_names = fit_l2_polynomial(
        x_values=x_values,
        y_values=y_values,
        feature_columns=feature_columns,
        degree=degree,
        alpha=alpha,
    )

    return format_polynomial(
        intercept=float(model.intercept_),
        feature_names=feature_names,
        coefficients=model.coef_,
    )
