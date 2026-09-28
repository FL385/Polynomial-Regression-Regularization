"""Compare fitted polynomials with known synthetic ground truth."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from src.data_generator import PolynomialCoefficients, evaluate_polynomial
from src.experiment.common import RegressionModel


def extract_polynomial_coefficients(
    model: RegressionModel, polynomial_features: PolynomialFeatures,
) -> PolynomialCoefficients:
    """Extract full-precision coefficients in the original variable order."""
    coefficients = {
        tuple(int(power) for power in powers): float(coefficient)
        for powers, coefficient in zip(polynomial_features.powers_, model.coef_)
    }
    coefficients[(0,) * polynomial_features.n_features_in_] = float(model.intercept_)
    return coefficients


def compare_polynomials(
    fitted_coefficients: PolynomialCoefficients,
    true_coefficients: PolynomialCoefficients,
    x_evaluation: np.ndarray,
) -> dict[str, Any]:
    """Measure coefficient and noiseless output errors on supplied points.

    Missing terms count as zero. Exponent tuples follow the feature column
    order. Output errors describe only the supplied evaluation points; use
    independent points for evaluating generalization. Relative coefficient L2
    error is None when the true coefficient vector is zero.
    """
    x_evaluation = np.asarray(x_evaluation, dtype=float)
    if x_evaluation.ndim != 2 or x_evaluation.shape[0] == 0:
        raise ValueError("x_evaluation must be a non-empty 2D array.")
    if not np.all(np.isfinite(x_evaluation)):
        raise ValueError("x_evaluation must contain only finite values.")
    for coefficients in (fitted_coefficients, true_coefficients):
        if not all(np.isfinite(value) for value in coefficients.values()):
            raise ValueError("coefficients must contain only finite values.")

    fitted_values = evaluate_polynomial(x_evaluation, fitted_coefficients)
    true_values = evaluate_polynomial(x_evaluation, true_coefficients)
    terms = sorted(set(fitted_coefficients) | set(true_coefficients))
    differences = {
        term: fitted_coefficients.get(term, 0.0) - true_coefficients.get(term, 0.0)
        for term in terms
    }
    coefficient_error = np.array(list(differences.values()), dtype=float)
    true_vector = np.array([true_coefficients.get(term, 0.0) for term in terms])
    output_error = fitted_values - true_values
    if not np.all(np.isfinite(output_error)):
        raise ValueError("polynomial evaluation produced non-finite errors.")
    coefficient_l2 = float(np.linalg.norm(coefficient_error))
    true_norm = float(np.linalg.norm(true_vector))
    mse = float(np.mean(output_error ** 2))
    return {
        "coefficient_differences": differences,
        "coefficient_l2": coefficient_l2,
        "relative_coefficient_l2": coefficient_l2 / true_norm if true_norm else None,
        "function_mse": mse,
        "function_rmse": float(np.sqrt(mse)),
        "function_mae": float(np.mean(np.abs(output_error))),
        "function_max_absolute_error": float(np.max(np.abs(output_error))),
        "n_evaluation_samples": int(x_evaluation.shape[0]),
    }
