"""Tests for plotting helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.plots import (
    plot_best_validation_mse_by_degree,
    plot_metric_comparison,
    plot_model_predictions,
)


def test_plot_model_predictions_returns_figure() -> None:
    """Prediction plots should create a matplotlib figure."""
    x_values = np.array([0.0, 1.0, 2.0])
    y_values = np.array([1.0, 2.0, 5.0])
    predictions = {"model": np.array([1.1, 2.1, 4.9])}

    fig = plot_model_predictions(x_values, y_values, predictions)

    assert fig.axes[0].get_title() == "Model Predictions"
    plt.close(fig)


def test_plot_model_predictions_rejects_multi_feature_input() -> None:
    """Prediction plots should stay explicit about their one-feature limit."""
    x_values = np.array([[0.0, 1.0], [1.0, 2.0]])
    y_values = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="one feature"):
        plot_model_predictions(x_values, y_values, {"model": y_values})


def test_plot_metric_comparison_returns_figure() -> None:
    """Metric comparisons should create a bar chart."""
    fig = plot_metric_comparison({"none": 3.0, "l1": 1.0, "l2": 2.0})

    assert fig.axes[0].get_title() == "Metric Comparison"
    assert len(fig.axes[0].patches) == 3
    plt.close(fig)


def test_plot_best_validation_mse_by_degree_returns_figure() -> None:
    """Candidate search plots should group best validation MSE by degree."""
    candidates = [
        {"regularization": "none", "degree": 1, "validation_mse": 4.0},
        {"regularization": "none", "degree": 2, "validation_mse": 2.0},
        {"regularization": "l1", "degree": 1, "validation_mse": 3.0},
        {"regularization": "l1", "degree": 2, "validation_mse": 1.5},
        {"regularization": "l1", "degree": 2, "validation_mse": 1.0},
    ]

    fig = plot_best_validation_mse_by_degree(candidates)

    assert fig.axes[0].get_title() == "Validation MSE by Degree"
    assert len(fig.axes[0].lines) == 2
    plt.close(fig)


def test_plot_helpers_reject_empty_inputs() -> None:
    """Empty plot inputs should fail clearly."""
    with pytest.raises(ValueError, match="metrics"):
        plot_metric_comparison({})

    with pytest.raises(ValueError, match="candidates"):
        plot_best_validation_mse_by_degree([])
