"""Automatic experiment selection across polynomial regression experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.exceptions import ConvergenceWarning

from src.experiment.common import (
    DEFAULT_DEGREE_CANDIDATES,
    format_polynomial,
    is_better_candidate,
    load_csv_dataset,
    mean_squared_error,
    train_validation_split,
)
from src.experiment.evaluation import extract_polynomial_coefficients
from src.experiment.methods import FITTERS, PARAMETER_GRIDS


def find_best_regularized_polynomial_from_csv(
    csv_path: str | Path,
) -> dict[str, Any]:
    """Choose among seven polynomial regression methods from only a CSV file."""
    x_values, y_values, feature_columns = load_csv_dataset(csv_path=csv_path)
    x_train, x_validation, y_train, y_validation = train_validation_split(
        x_values=x_values,
        y_values=y_values,
    )

    best_candidate: dict[str, Any] | None = None
    candidate_results: list[dict[str, Any]] = []
    failed_candidates: list[dict[str, Any]] = []

    for regularization, fit_function in FITTERS.items():
        for degree in DEFAULT_DEGREE_CANDIDATES:
            for parameters in ParameterGrid(PARAMETER_GRIDS[regularization]):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        model, polynomial_features, _ = fit_function(
                            x_train, y_train, feature_columns, degree, **parameters,
                        )
                except ConvergenceWarning as error:
                    failed_candidates.append({
                        "regularization": regularization, "degree": degree,
                        "parameters": parameters, "reason": str(error),
                    })
                    continue
                validation_predictions = model.predict(
                    polynomial_features.transform(x_validation)
                )
                candidate = {
                    "regularization": regularization,
                    "degree": degree,
                    "alpha": parameters.get("alpha"),
                    "parameters": parameters,
                    "validation_mse": mean_squared_error(
                        y_true=y_validation,
                        y_predicted=validation_predictions,
                    ),
                }
                if not np.isfinite(candidate["validation_mse"]):
                    failed_candidates.append({**candidate, "reason": "Non-finite MSE"})
                    continue
                candidate_results.append(candidate)
                if is_better_candidate(candidate, best_candidate):
                    best_candidate = candidate

    if best_candidate is None:
        raise ValueError("No regression candidates were evaluated.")

    final_fit_function = FITTERS[str(best_candidate["regularization"])]
    final_model, final_features, final_feature_names = final_fit_function(
        x_values,
        y_values,
        feature_columns,
        int(best_candidate["degree"]),
        **best_candidate["parameters"],
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
            float(candidate["alpha"] or 0.0),
            str(candidate["regularization"]),
        ),
    )

    return {
        "best": {
            **best_candidate,
            "polynomial": polynomial,
            "coefficients": extract_polynomial_coefficients(final_model, final_features),
        },
        "candidates": sorted_candidates,
        "failed_candidates": failed_candidates,
    }
