"""Unregularized polynomial regression experiment helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from src.experiment.common import (
    fit_polynomial_model,
    format_polynomial,
    load_csv_dataset,
    validate_degree_alpha,
)


def fit_no_regularization_polynomial(
    x_values: np.ndarray,
    y_values: np.ndarray,
    feature_columns: list[str],
    degree: int,
    alpha: float = 0.0,
) -> tuple[LinearRegression, PolynomialFeatures, list[str]]:
    """Fit an unregularized polynomial regression model on loaded data.

    The ``alpha`` parameter is accepted for API compatibility with the L1 and
    L2 experiment functions, but it is ignored because this model has no
    regularization strength.
    """
    validate_degree_alpha(degree=degree, alpha=alpha)
    model = LinearRegression()
    fitted_model, polynomial_features, feature_names = fit_polynomial_model(
        model=model,
        x_values=x_values,
        y_values=y_values,
        feature_columns=feature_columns,
        degree=degree,
    )

    return fitted_model, polynomial_features, feature_names


def fit_no_regularization_polynomial_from_csv(
    csv_path: str | Path,
    degree: int,
    target_column: str = "y",
) -> str:
    """Fit unregularized polynomial regression from a CSV dataset."""
    x_values, y_values, feature_columns = load_csv_dataset(
        csv_path=csv_path,
        target_column=target_column,
    )
    model, _, feature_names = fit_no_regularization_polynomial(
        x_values=x_values,
        y_values=y_values,
        feature_columns=feature_columns,
        degree=degree,
    )

    return format_polynomial(
        intercept=float(model.intercept_),
        feature_names=feature_names,
        coefficients=model.coef_,
    )
