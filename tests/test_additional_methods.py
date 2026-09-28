"""Regression tests for additional estimators and coefficient restoration."""

import numpy as np
import pytest
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.exceptions import ConvergenceWarning

from src.data_generator import evaluate_polynomial, save_dataset_to_csv
from src.experiment.methods import FITTERS
from src.experiment.evaluation import extract_polynomial_coefficients
from src.experiments import run_experiment, fit_regularized_polynomial_from_csv
import src.experiment.auto_selection as selection


@pytest.mark.parametrize("name,estimator", [
    ("elastic_net", ElasticNet(alpha=0.01, max_iter=50000)),
    ("bayesian_ridge", BayesianRidge()),
    ("ard", ARDRegression()),
    ("huber", HuberRegressor(max_iter=3000)),
])
def test_restored_coefficients_match_scaled_predictions(name, estimator):
    rng = np.random.default_rng(10)
    x = rng.normal(size=(100, 2)) * [0.1, 10]
    y = 1 + 2 * x[:, 0] - 0.5 * x[:, 1] + 0.1 * x[:, 0] * x[:, 1]
    features = PolynomialFeatures(2, include_bias=False)
    expanded = features.fit_transform(x)
    scaler = StandardScaler(with_mean=False).fit(expanded)
    estimator.fit(scaler.transform(expanded), y)
    model, features, _ = FITTERS[name](x, y, ["x1", "x2"], 2)
    points = rng.normal(size=(20, 2)) * [0.1, 10]
    terms = features.transform(points)
    expected = estimator.predict(scaler.transform(terms))
    np.testing.assert_allclose(model.predict(terms), expected, atol=1e-9)
    coefficients = extract_polynomial_coefficients(model, features)
    np.testing.assert_allclose(evaluate_polynomial(points, coefficients), expected, atol=1e-9)
    if name in {"ard", "bayesian_ridge"}:
        _, expected_std = estimator.predict(scaler.transform(terms), return_std=True)
        _, actual_std = model.predict(terms, return_std=True)
        np.testing.assert_allclose(actual_std, expected_std, atol=1e-9)


@pytest.mark.parametrize("name,parameters", [
    ("elastic_net", {"alpha": 0.1, "l1_ratio": 0.9}),
    ("bayesian_ridge", {}),
    ("ard", {"threshold_lambda": 1000.0}),
    ("huber", {"alpha": 0.01, "epsilon": 2.0}),
])
def test_each_added_method_can_be_selected_and_refitted(tmp_path, monkeypatch, name, parameters):
    x = np.linspace(-2, 2, 50).reshape(-1, 1)
    path = tmp_path / "data.csv"
    save_dataset_to_csv(x, 1 + 2 * x[:, 0], path)
    monkeypatch.setattr(selection, "FITTERS", {name: FITTERS[name]})
    monkeypatch.setattr(selection, "DEFAULT_DEGREE_CANDIDATES", (1,))
    monkeypatch.setattr(selection, "PARAMETER_GRIDS", {
        name: {key: [value] for key, value in parameters.items()},
    })
    result = run_experiment({"csv_path": path})
    assert result["best"]["regularization"] == name
    assert result["best"]["parameters"] == parameters
    assert result["best"]["alpha"] == parameters.get("alpha")
    assert result["best"]["polynomial"].startswith("y =")
    assert np.isfinite(result["best"]["validation_mse"])


def test_nonconverged_candidate_is_recorded(tmp_path, monkeypatch):
    x = np.linspace(-2, 2, 30).reshape(-1, 1)
    path = tmp_path / "data.csv"
    save_dataset_to_csv(x, 2 * x[:, 0], path)
    def fail(*args, **kwargs):
        import warnings
        warnings.warn("test convergence failure", ConvergenceWarning)
    monkeypatch.setattr(selection, "FITTERS", {"none": FITTERS["none"], "huber": fail})
    monkeypatch.setattr(selection, "DEFAULT_DEGREE_CANDIDATES", (1,))
    result = selection.find_best_regularized_polynomial_from_csv(path)
    assert result["best"]["regularization"] == "none"
    assert result["failed_candidates"]
    assert all(item["regularization"] == "huber" for item in result["failed_candidates"])


@pytest.mark.parametrize("name,parameters", [
    ("elastic_net", {"l1_ratio": 1.5}),
    ("huber", {"epsilon": 0.5}),
    ("ard", {"threshold_lambda": -1}),
])
def test_invalid_method_parameters(name, parameters):
    with pytest.raises(ValueError):
        FITTERS[name](np.ones((3, 1)), np.ones(3), ["x1"], 1, **parameters)


def test_huber_recovers_line_despite_target_outliers():
    x = np.linspace(-2, 2, 100).reshape(-1, 1)
    truth = 1 + 2 * x[:, 0]
    noisy = truth.copy()
    noisy[-5:] += 40
    huber, features, _ = FITTERS["huber"](x, noisy, ["x1"], 1)
    baseline, _, _ = FITTERS["none"](x, noisy, ["x1"], 1)
    terms = features.transform(x)
    huber_mse = np.mean((huber.predict(terms) - truth) ** 2)
    baseline_mse = np.mean((baseline.predict(terms) - truth) ** 2)
    assert huber_mse < baseline_mse * 0.1


@pytest.mark.parametrize("name", ["elastic_net", "bayesian_ridge", "ard", "huber"])
def test_generic_csv_entry_point_supports_new_methods(tmp_path, name):
    x = np.linspace(-2, 2, 40).reshape(-1, 1)
    path = tmp_path / "data.csv"
    save_dataset_to_csv(x, 1 + 2 * x[:, 0], path)
    polynomial = fit_regularized_polynomial_from_csv(path, name, degree=1)
    assert polynomial.startswith("y =")
    assert "x1" in polynomial


def test_fixed_degree_parameters_reach_estimator_and_summary(tmp_path):
    x = np.linspace(-2, 2, 40).reshape(-1, 1)
    path = tmp_path / "data.csv"
    save_dataset_to_csv(x, 1 + 2 * x[:, 0], path)
    result = run_experiment({
        "csv_path": path, "degree": 1,
        "method_parameters": {"elastic_net": {"alpha": 1e6, "l1_ratio": 0.9}},
    })
    assert result["polynomials"]["elastic_net"] == "y = 1"
    assert result["parameters"]["elastic_net"] == {"alpha": 1e6, "l1_ratio": 0.9}
