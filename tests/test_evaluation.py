"""Tests for comparing polynomials with noiseless ground truth."""

import numpy as np
import pytest

from src.data_generator import generate_polynomial_data, save_dataset_to_csv
from src.experiments import compare_polynomials, run_experiment


def test_known_errors_include_missing_and_extra_terms():
    true = {(0, 0): 1.0, (1, 1): 2.0}
    fitted = {(0, 0): 2.0, (2, 0): 3.0}
    result = compare_polynomials(fitted, true, np.array([[0, 0], [1, 1]]))
    assert result["coefficient_differences"] == {
        (0, 0): 1.0, (1, 1): -2.0, (2, 0): 3.0,
    }
    assert result["coefficient_l2"] == pytest.approx(np.sqrt(14))
    assert result["relative_coefficient_l2"] == pytest.approx(np.sqrt(14 / 5))
    assert result["function_mse"] == pytest.approx(2.5)
    assert result["function_rmse"] == pytest.approx(np.sqrt(2.5))
    assert result["function_mae"] == pytest.approx(1.5)
    assert result["function_max_absolute_error"] == pytest.approx(2)


def test_zero_truth_has_no_relative_coefficient_error():
    result = compare_polynomials({(0,): 2.0}, {(0,): 0.0}, np.zeros((2, 1)))
    assert result["relative_coefficient_l2"] is None
    assert result["function_rmse"] == 2


@pytest.mark.parametrize("points", [np.empty((0, 1)), np.array([[np.nan]])])
def test_invalid_evaluation_points(points):
    with pytest.raises(ValueError):
        compare_polynomials({(0,): 0.0}, {(0,): 0.0}, points)


def test_mismatched_variables_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        compare_polynomials({(0, 0): 1.0}, {(0,): 1.0}, np.zeros((2, 1)))


@pytest.mark.parametrize("automatic", [False, True])
def test_experiment_compares_with_truth_on_independent_points(tmp_path, automatic):
    true = {(0,): 1.0, (1,): 2.0, (2,): -0.5}
    x, y = generate_polynomial_data(80, noise=0, random_state=42, coefficients=true)
    path = tmp_path / "data.csv"
    save_dataset_to_csv(x, y, path)
    points = np.linspace(-3, 3, 51).reshape(-1, 1)
    config = {"csv_path": path, "true_coefficients": true, "x_evaluation": points}
    if not automatic:
        config["degree"] = 2
    result = run_experiment(config)
    if automatic:
        deviation = result["best"]["deviation"]
        assert (2,) in result["best"]["coefficients"]
    else:
        assert set(result["deviations"]) == {
            "none", "l1", "l2", "elastic_net", "bayesian_ridge", "ard", "huber",
        }
        deviation = result["deviations"]["none"]
    tolerance = 1e-7 if automatic else 1e-10
    assert deviation["function_rmse"] < tolerance
    assert deviation["coefficient_l2"] < tolerance
    assert deviation["n_evaluation_samples"] == 51


def test_truth_requires_explicit_evaluation_points():
    with pytest.raises(ValueError, match="x_evaluation"):
        run_experiment({"csv_path": "unused.csv", "true_coefficients": {(0,): 0.0}})
