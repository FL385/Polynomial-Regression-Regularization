"""Visualization helpers for polynomial regression experiments."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _one_dimensional_x(x_values: Any) -> np.ndarray:
    """Return a 1D feature array for simple educational plots."""
    x_array = np.asarray(x_values, dtype=float)
    if x_array.ndim == 1:
        return x_array
    if x_array.ndim == 2 and x_array.shape[1] == 1:
        return x_array[:, 0]

    raise ValueError("plot_model_predictions only supports one feature column.")


def plot_model_predictions(
    x_values: Any,
    y_values: Any,
    predictions: dict[str, Any],
) -> Any:
    """Plot observed data and model predictions for one feature.

    Parameters
    ----------
    x_values:
        Feature values used for the scatter plot. This helper supports a single
        feature because fitted curves are easiest for beginners to read in 2D.
    y_values:
        Target values used for the scatter plot.
    predictions:
        Mapping of model names to predicted values.

    Returns
    -------
    Any
        A matplotlib figure.
    """
    x_array = _one_dimensional_x(x_values)
    y_array = np.asarray(y_values, dtype=float)
    if y_array.shape[0] != x_array.shape[0]:
        raise ValueError("x_values and y_values must have the same row count.")

    sort_order = np.argsort(x_array)
    fig, ax = plt.subplots()
    ax.scatter(x_array, y_array, label="observed", alpha=0.7)

    for model_name, predicted_values in predictions.items():
        prediction_array = np.asarray(predicted_values, dtype=float)
        if prediction_array.shape[0] != x_array.shape[0]:
            raise ValueError("Each prediction array must match x_values length.")
        ax.plot(
            x_array[sort_order],
            prediction_array[sort_order],
            label=model_name,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Model Predictions")
    ax.legend()
    fig.tight_layout()

    return fig


def plot_metric_comparison(metrics: dict[str, Any]) -> Any:
    """Plot model comparison metrics as a bar chart.

    Parameters
    ----------
    metrics:
        Mapping of model names to evaluation metric values.

    Returns
    -------
    Any
        A matplotlib figure.
    """
    if not metrics:
        raise ValueError("metrics must contain at least one value.")

    model_names = list(metrics)
    metric_values = [float(metrics[model_name]) for model_name in model_names]

    fig, ax = plt.subplots()
    ax.bar(model_names, metric_values)
    ax.set_xlabel("model")
    ax.set_ylabel("metric")
    ax.set_title("Metric Comparison")
    fig.tight_layout()

    return fig


def plot_best_validation_mse_by_degree(candidates: list[dict[str, Any]]) -> Any:
    """Plot the best validation MSE at each degree for each regression type.

    Parameters
    ----------
    candidates:
        Candidate dictionaries returned by
        ``find_best_regularized_polynomial_from_csv``.

    Returns
    -------
    Any
        A matplotlib figure.
    """
    if not candidates:
        raise ValueError("candidates must contain at least one result.")

    best_by_group: dict[tuple[str, int], float] = {}
    for candidate in candidates:
        regularization = str(candidate["regularization"])
        degree = int(candidate["degree"])
        validation_mse = float(candidate["validation_mse"])
        key = (regularization, degree)

        if key not in best_by_group or validation_mse < best_by_group[key]:
            best_by_group[key] = validation_mse

    regularization_names = sorted({key[0] for key in best_by_group})

    fig, ax = plt.subplots()
    for regularization in regularization_names:
        points = sorted(
            (degree, mse)
            for (name, degree), mse in best_by_group.items()
            if name == regularization
        )
        degrees = [point[0] for point in points]
        mses = [point[1] for point in points]
        ax.plot(degrees, mses, marker="o", label=regularization)

    ax.set_xlabel("degree")
    ax.set_ylabel("best validation MSE")
    ax.set_title("Validation MSE by Degree")
    ax.legend()
    fig.tight_layout()

    return fig
