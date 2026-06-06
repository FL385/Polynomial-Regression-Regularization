"""Utilities for generating synthetic regression datasets.

The default dataset follows the quadratic relationship used in the reference
essay for this project. It has three input features and ten non-zero terms.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

DEFAULT_FEATURE_RANGE = (-3.0, 3.0)


def true_quadratic_function(x_values: np.ndarray) -> np.ndarray:
    """Evaluate the fixed trivariate quadratic relationship.

    Parameters
    ----------
    x_values:
        Feature matrix with shape ``(n_samples, 3)``.

    Returns
    -------
    np.ndarray
        Target values without random noise.
    """
    x_values = np.asarray(x_values, dtype=float)
    if x_values.ndim != 2 or x_values.shape[1] != 3:
        raise ValueError("x_values must have shape (n_samples, 3).")

    x1 = x_values[:, 0]
    x2 = x_values[:, 1]
    x3 = x_values[:, 2]

    return (
        1.5
        - 0.8 * x1
        + 0.6 * x2
        + 1.2 * x3
        - 0.3 * x1 * x2
        + 0.2 * x1 * x3
        + 0.1 * x2 * x3
        + 0.4 * x1**2
        - 0.5 * x2**2
        + 0.3 * x3**2
    )


def generate_polynomial_data(
    n_samples: int = 100,
    noise: float = 0.1,
    random_state: Optional[int] = None,
    feature_range: tuple[float, float] = DEFAULT_FEATURE_RANGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic dataset for polynomial regression experiments.

    Parameters
    ----------
    n_samples:
        Number of data points to generate.
    noise:
        Standard deviation of Gaussian noise added to the target values.
    random_state:
        Optional seed for reproducible data generation.
    feature_range:
        Lower and upper bounds for uniformly sampled feature values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Feature values and target values.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if noise < 0:
        raise ValueError("noise must be non-negative.")

    lower, upper = feature_range
    if lower >= upper:
        raise ValueError("feature_range must be ordered as (lower, upper).")

    rng = np.random.default_rng(random_state)
    x_values = rng.uniform(lower, upper, size=(n_samples, 3))
    y_values = true_quadratic_function(x_values)

    if noise > 0:
        y_values = y_values + rng.normal(loc=0.0, scale=noise, size=n_samples)

    return x_values, y_values
