"""Tests for experiment-level polynomial regression helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.data_generator import generate_polynomial_data
from src.experiment.l1_regression import fit_l1_polynomial_from_csv
from src.experiment.l2_regression import fit_l2_polynomial_from_csv
from src.experiments import (
    find_best_regularized_polynomial_from_csv,
    fit_regularized_polynomial_from_csv,
    run_experiment,
    run_l1_l2_regression_from_csv,
    summarize_results,
)


def _write_dataset_csv(csv_path: Path, x_values: np.ndarray, y_values: np.ndarray) -> None:
    """Write generated data using the simple x1, x2, ..., y CSV format."""
    feature_names = [f"x{index + 1}" for index in range(x_values.shape[1])]

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([*feature_names, "y"])
        for x_row, y_value in zip(x_values, y_values):
            writer.writerow([*x_row, y_value])


def test_fit_regularized_polynomial_from_csv_returns_l2_polynomial(
    tmp_path: Path,
) -> None:
    """L2 regression should fit a readable polynomial from a generated CSV."""
    coefficients = {
        (0, 0): 1.0,
        (1, 0): 2.0,
        (0, 2): -0.5,
    }
    x_values, y_values = generate_polynomial_data(
        n_samples=120,
        noise=0.0,
        random_state=42,
        coefficients=coefficients,
    )
    csv_path = tmp_path / "dataset.csv"
    _write_dataset_csv(csv_path, x_values, y_values)

    polynomial = fit_l2_polynomial_from_csv(
        csv_path=csv_path,
        degree=2,
        alpha=0.001,
    )

    assert polynomial.startswith("y =")
    assert "x1" in polynomial
    assert "x2^2" in polynomial


def test_fit_l1_polynomial_from_csv_returns_l1_polynomial(tmp_path: Path) -> None:
    """L1 regression should be available from its own experiment file."""
    coefficients = {
        (0, 0): 1.0,
        (1, 0): 2.0,
        (0, 2): -0.5,
    }
    x_values, y_values = generate_polynomial_data(
        n_samples=120,
        noise=0.0,
        random_state=42,
        coefficients=coefficients,
    )
    csv_path = tmp_path / "dataset.csv"
    _write_dataset_csv(csv_path, x_values, y_values)

    polynomial = fit_l1_polynomial_from_csv(
        csv_path=csv_path,
        degree=2,
        alpha=0.0001,
    )

    assert polynomial.startswith("y =")
    assert "x1" in polynomial


def test_run_l1_l2_regression_from_csv_returns_both_polynomials(
    tmp_path: Path,
) -> None:
    """The experiment helper should return both L1 and L2 polynomial strings."""
    coefficients = {
        (0, 0): 1.0,
        (1, 0): 2.0,
        (0, 2): -0.5,
    }
    x_values, y_values = generate_polynomial_data(
        n_samples=120,
        noise=0.0,
        random_state=42,
        coefficients=coefficients,
    )
    csv_path = tmp_path / "dataset.csv"
    _write_dataset_csv(csv_path, x_values, y_values)

    polynomials = run_l1_l2_regression_from_csv(
        csv_path=csv_path,
        degree=2,
        l1_alpha=0.0001,
        l2_alpha=0.001,
    )

    assert set(polynomials) == {"l1", "l2"}
    assert polynomials["l1"].startswith("y =")
    assert polynomials["l2"].startswith("y =")


def test_run_experiment_and_summarize_results(tmp_path: Path) -> None:
    """run_experiment should support CSV-based L1/L2 comparisons."""
    coefficients = {
        (0, 0): 1.0,
        (1, 0): 2.0,
        (0, 2): -0.5,
    }
    x_values, y_values = generate_polynomial_data(
        n_samples=120,
        noise=0.0,
        random_state=42,
        coefficients=coefficients,
    )
    csv_path = tmp_path / "dataset.csv"
    _write_dataset_csv(csv_path, x_values, y_values)

    results = run_experiment(
        {
            "csv_path": csv_path,
            "degree": 2,
            "l1_alpha": 0.0001,
            "l2_alpha": 0.001,
        }
    )
    summary = summarize_results(results)

    assert results["degree"] == 2
    assert "L1 polynomial:" in summary
    assert "L2 polynomial:" in summary


def test_find_best_regularized_polynomial_from_csv_selects_settings(
    tmp_path: Path,
) -> None:
    """Automatic search should choose degree, alpha, and regularization."""
    coefficients = {
        (0, 0): 1.0,
        (1, 0): 2.0,
        (0, 2): -0.5,
    }
    x_values, y_values = generate_polynomial_data(
        n_samples=160,
        noise=0.05,
        random_state=42,
        coefficients=coefficients,
    )
    csv_path = tmp_path / "dataset.csv"
    _write_dataset_csv(csv_path, x_values, y_values)

    results = find_best_regularized_polynomial_from_csv(csv_path)
    best = results["best"]

    assert best["regularization"] in {"l1", "l2"}
    assert 1 <= best["degree"] <= 8
    assert best["alpha"] > 0
    assert best["validation_mse"] >= 0
    assert best["polynomial"].startswith("y =")
    assert results["candidates"][0]["validation_mse"] <= results["candidates"][-1][
        "validation_mse"
    ]


def test_run_experiment_can_auto_select_from_csv_only(tmp_path: Path) -> None:
    """run_experiment should support a config whose only required input is CSV."""
    coefficients = {
        (0, 0): 1.0,
        (1, 0): 2.0,
        (0, 2): -0.5,
    }
    x_values, y_values = generate_polynomial_data(
        n_samples=160,
        noise=0.05,
        random_state=42,
        coefficients=coefficients,
    )
    csv_path = tmp_path / "dataset.csv"
    _write_dataset_csv(csv_path, x_values, y_values)

    results = run_experiment({"csv_path": csv_path})
    summary = summarize_results(results)

    assert "best" in results
    assert "Best degree:" in summary
    assert "Best alpha:" in summary
    assert "Polynomial:" in summary


def test_fit_regularized_polynomial_from_csv_rejects_invalid_settings(
    tmp_path: Path,
) -> None:
    """Invalid experiment settings should fail with clear errors."""
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("x1,y\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="regularization"):
        fit_regularized_polynomial_from_csv(
            csv_path=csv_path,
            regularization="none",
            degree=2,
        )

    with pytest.raises(ValueError, match="degree"):
        fit_regularized_polynomial_from_csv(
            csv_path=csv_path,
            regularization="l2",
            degree=0,
        )
