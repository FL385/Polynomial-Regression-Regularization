"""Visualization helpers for polynomial regression experiments."""

from __future__ import annotations

from typing import Any


def plot_model_predictions(
    x_values: Any,
    y_values: Any,
    predictions: dict[str, Any],
) -> Any:
    """Plot observed data and model predictions.

    Parameters
    ----------
    x_values:
        Feature values used for the scatter plot.
    y_values:
        Target values used for the scatter plot.
    predictions:
        Mapping of model names to predicted values.

    Returns
    -------
    Any
        A matplotlib figure or axes object.
    """
    raise NotImplementedError("Prediction plots will be implemented later.")


def plot_metric_comparison(metrics: dict[str, Any]) -> Any:
    """Plot model comparison metrics.

    Parameters
    ----------
    metrics:
        Mapping of model names to evaluation metric values.

    Returns
    -------
    Any
        A matplotlib figure or axes object.
    """
    raise NotImplementedError("Metric comparison plots will be implemented later.")
