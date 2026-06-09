"""Automatic experiment selection across L1 and L2 polynomial regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures

from src.experiment.common import (
    DEFAULT_ALPHA_CANDIDATES,
    DEFAULT_DEGREE_CANDIDATES,
    format_polynomial,
    is_better_candidate,
    load_csv_dataset,
    mean_squared_error,
    train_validation_split,
)
from src.experiment.l1_regression import fit_l1_polynomial
from src.experiment.l2_regression import fit_l2_polynomial

ExperimentFitFunction = Callable[
    [Any, Any, list[str], int, float],
    tuple[Lasso | Ridge, PolynomialFeatures, list[str]],
]


def find_best_regularized_polynomial_from_csv(
    csv_path: str | Path,
) -> dict[str, Any]:
    """Choose the best L1/L2 polynomial regression from only a CSV file."""
    x_values, y_values, feature_columns = load_csv_dataset(csv_path=csv_path)
    x_train, x_validation, y_train, y_validation = train_validation_split(
        x_values=x_values,
        y_values=y_values,
    )

    fitters: dict[str, ExperimentFitFunction] = {
        "l1": fit_l1_polynomial,
        "l2": fit_l2_polynomial,
    }
    best_candidate: dict[str, Any] | None = None
    candidate_results: list[dict[str, Any]] = []

    for regularization, fit_function in fitters.items():
        for degree in DEFAULT_DEGREE_CANDIDATES:
            for alpha in DEFAULT_ALPHA_CANDIDATES:
                model, polynomial_features, _ = fit_function(
                    x_train,
                    y_train,
                    feature_columns,
                    degree,
                    alpha,
                )
                validation_predictions = model.predict(
                    polynomial_features.transform(x_validation)
                )
                candidate = {
                    "regularization": regularization,
                    "degree": degree,
                    "alpha": alpha,
                    "validation_mse": mean_squared_error(
                        y_true=y_validation,
                        y_predicted=validation_predictions,
                    ),
                }
                candidate_results.append(candidate)
                if is_better_candidate(candidate, best_candidate):
                    best_candidate = candidate

    if best_candidate is None:
        raise ValueError("No regression candidates were evaluated.")

    final_fit_function = fitters[str(best_candidate["regularization"])]
    final_model, _, final_feature_names = final_fit_function(
        x_values,
        y_values,
        feature_columns,
        int(best_candidate["degree"]),
        float(best_candidate["alpha"]),
    )
    polynomial = format_polynomial(
        intercept=float(final_model.intercept_),
        feature_names=final_feature_names,
        coefficients=final_model.coef_,
    )

    sorted_candidates = sorted(
        candidate_results,
        key=lambda candidate: (
            float(candidate["validation_mse"]),
            int(candidate["degree"]),
            float(candidate["alpha"]),
            str(candidate["regularization"]),
        ),
    )

    return {
        "best": {
            **best_candidate,
            "polynomial": polynomial,
        },
        "candidates": sorted_candidates,
    }
