"""High-level experiment runners that combine individual experiment files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.experiment.auto_selection import find_best_regularized_polynomial_from_csv
from src.experiment.l1_regression import fit_l1_polynomial_from_csv
from src.experiment.l2_regression import fit_l2_polynomial_from_csv
from src.experiment.common import load_csv_dataset, format_polynomial
from src.experiment.evaluation import compare_polynomials, extract_polynomial_coefficients
from src.experiment.methods import FITTERS


def fit_regularized_polynomial_from_csv(
    csv_path: str | Path,
    regularization: str,
    degree: int,
    alpha: float = 1.0,
    target_column: str = "y",
    **parameters: Any,
) -> str:
    """Fit any supported polynomial method; Bayesian methods do not use alpha."""
    normalized_regularization = regularization.lower()
    if normalized_regularization == "no_regularization":
        normalized_regularization = "none"
    if normalized_regularization in FITTERS:
        x, y, columns = load_csv_dataset(csv_path, target_column)
        if normalized_regularization in {"none", "l1", "l2", "elastic_net", "huber"}:
            parameters = {"alpha": alpha, **parameters}
        model, _, names = FITTERS[normalized_regularization](
            x, y, columns, degree, **parameters,
        )
        return format_polynomial(float(model.intercept_), names, model.coef_)
    raise ValueError(f"regularization must be one of {', '.join(FITTERS)}.")


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


def run_regression_comparison_from_csv(
    csv_path: str | Path,
    degree: int,
    l1_alpha: float = 0.01,
    l2_alpha: float = 1.0,
    target_column: str = "y",
    method_parameters: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Fit all seven methods on one CSV using optional per-method parameters."""
    return run_experiment({
        "csv_path": csv_path, "degree": degree, "l1_alpha": l1_alpha,
        "l2_alpha": l2_alpha, "target_column": target_column,
        "method_parameters": method_parameters or {},
    })["polynomials"]


def run_experiment(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a polynomial regression regularization experiment."""
    if config is None:
        raise ValueError("config must include csv_path.")
    if "csv_path" not in config:
        raise ValueError("config must include csv_path.")
    if "true_coefficients" in config and "x_evaluation" not in config:
        raise ValueError("x_evaluation is required when true_coefficients is provided.")
    if "degree" not in config:
        results = find_best_regularized_polynomial_from_csv(config["csv_path"])
        if "true_coefficients" in config:
            results["best"]["deviation"] = compare_polynomials(
                results["best"]["coefficients"], config["true_coefficients"],
                config["x_evaluation"],
            )
        return results

    results = {
        "degree": int(config["degree"]),
        "l1_alpha": float(config.get("l1_alpha", 0.01)),
        "l2_alpha": float(config.get("l2_alpha", 1.0)),
        "polynomials": {},
    }
    x_values, y_values, columns = load_csv_dataset(
        config["csv_path"], str(config.get("target_column", "y")),
    )
    if "true_coefficients" in config:
        results["deviations"] = {}
    method_parameters = config.get("method_parameters", {})
    unknown = set(method_parameters) - set(FITTERS)
    if unknown:
        raise ValueError(f"Unknown method_parameters keys: {sorted(unknown)}")
    results["parameters"] = {}
    for name, fitter in FITTERS.items():
        parameters = {}
        if name in {"l1", "l2"}:
            parameters["alpha"] = results[f"{name}_alpha"]
        parameters.update(method_parameters.get(name, {}))
        results["parameters"][name] = parameters
        model, features, names = fitter(
            x_values, y_values, columns, results["degree"], **parameters,
        )
        results["polynomials"][name] = format_polynomial(
            float(model.intercept_), names, model.coef_,
        )
        if "true_coefficients" in config:
            results["deviations"][name] = compare_polynomials(
                extract_polynomial_coefficients(model, features),
                config["true_coefficients"], config["x_evaluation"],
            )
    return results


def summarize_results(results: dict[str, Any]) -> str:
    """Create a short text summary of experiment results."""
    best = results.get("best")
    if isinstance(best, dict):
        return "\n".join(
            [
                f"Best regularization: {best.get('regularization')}",
                f"Best degree: {best.get('degree')}",
                f"Best alpha: {best.get('alpha')}",
                f"Best parameters: {best.get('parameters')}",
                f"Validation MSE: {best.get('validation_mse')}",
                f"Polynomial: {best.get('polynomial')}",
            ]
        )

    polynomials = results.get("polynomials")
    if isinstance(polynomials, dict):
        summary = "\n".join(
            [
                f"Degree: {results.get('degree')}",
                f"No regularization polynomial: {polynomials.get('none')}",
                f"L1 alpha: {results.get('l1_alpha')}",
                f"L1 polynomial: {polynomials.get('l1')}",
                f"L2 alpha: {results.get('l2_alpha')}",
                f"L2 polynomial: {polynomials.get('l2')}",
            ]
        )
        extra = [
            f"{name} polynomial: {polynomial}"
            for name, polynomial in polynomials.items()
            if name not in {"none", "l1", "l2"}
        ]
        return "\n".join([summary, *extra])

    raise ValueError("results must include either best or polynomials.")
