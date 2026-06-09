"""High-level experiment runners that combine individual experiment files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.experiment.auto_selection import find_best_regularized_polynomial_from_csv
from src.experiment.common import format_polynomial
from src.experiment.l1_regression import fit_l1_polynomial_from_csv
from src.experiment.l2_regression import fit_l2_polynomial_from_csv


def fit_regularized_polynomial_from_csv(
    csv_path: str | Path,
    regularization: str,
    degree: int,
    alpha: float = 1.0,
    target_column: str = "y",
) -> str:
    """Fit an L1 or L2 regularized polynomial model from a CSV dataset."""
    normalized_regularization = regularization.lower()
    if normalized_regularization == "l1":
        return fit_l1_polynomial_from_csv(
            csv_path=csv_path,
            degree=degree,
            alpha=alpha,
            target_column=target_column,
        )
    if normalized_regularization == "l2":
        return fit_l2_polynomial_from_csv(
            csv_path=csv_path,
            degree=degree,
            alpha=alpha,
            target_column=target_column,
        )

    raise ValueError("regularization must be either 'l1' or 'l2'.")


def run_l1_l2_regression_from_csv(
    csv_path: str | Path,
    degree: int,
    l1_alpha: float = 0.01,
    l2_alpha: float = 1.0,
    target_column: str = "y",
) -> dict[str, str]:
    """Fit L1 and L2 polynomial regression models from the same CSV file."""
    return {
        "l1": fit_l1_polynomial_from_csv(
            csv_path=csv_path,
            degree=degree,
            alpha=l1_alpha,
            target_column=target_column,
        ),
        "l2": fit_l2_polynomial_from_csv(
            csv_path=csv_path,
            degree=degree,
            alpha=l2_alpha,
            target_column=target_column,
        ),
    }


def run_experiment(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a polynomial regression regularization experiment."""
    if config is None:
        raise ValueError("config must include csv_path.")
    if "csv_path" not in config:
        raise ValueError("config must include csv_path.")
    if "degree" not in config:
        return find_best_regularized_polynomial_from_csv(config["csv_path"])

    polynomials = run_l1_l2_regression_from_csv(
        csv_path=config["csv_path"],
        degree=int(config["degree"]),
        l1_alpha=float(config.get("l1_alpha", 0.01)),
        l2_alpha=float(config.get("l2_alpha", 1.0)),
        target_column=str(config.get("target_column", "y")),
    )

    return {
        "degree": int(config["degree"]),
        "l1_alpha": float(config.get("l1_alpha", 0.01)),
        "l2_alpha": float(config.get("l2_alpha", 1.0)),
        "polynomials": polynomials,
    }


def summarize_results(results: dict[str, Any]) -> str:
    """Create a short text summary of experiment results."""
    best = results.get("best")
    if isinstance(best, dict):
        return "\n".join(
            [
                f"Best regularization: {best.get('regularization')}",
                f"Best degree: {best.get('degree')}",
                f"Best alpha: {best.get('alpha')}",
                f"Validation MSE: {best.get('validation_mse')}",
                f"Polynomial: {best.get('polynomial')}",
            ]
        )

    polynomials = results.get("polynomials")
    if isinstance(polynomials, dict):
        return "\n".join(
            [
                f"Degree: {results.get('degree')}",
                f"L1 alpha: {results.get('l1_alpha')}",
                f"L1 polynomial: {polynomials.get('l1')}",
                f"L2 alpha: {results.get('l2_alpha')}",
                f"L2 polynomial: {polynomials.get('l2')}",
            ]
        )

    raise ValueError("results must include either best or polynomials.")
