"""Utilities for generating synthetic regression datasets.

The default dataset follows the quadratic relationship used in the reference
essay for this project. It has three input features and ten non-zero terms.
"""

from __future__ import annotations

from itertools import product
import re
from typing import Optional

import numpy as np

DEFAULT_FEATURE_RANGE = (-3.0, 3.0)
PolynomialCoefficients = dict[tuple[int, ...], float]

_FEATURE_COUNT_PATTERN = re.compile(r"^\d+$")
_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TERM_PATTERN = re.compile(rf"^\(([^()]*)\)\s*=\s*({_NUMBER_PATTERN})$")

DEFAULT_QUADRATIC_COEFFICIENTS: PolynomialCoefficients = {
    (0, 0, 0): 1.5,
    (1, 0, 0): -0.8,
    (0, 1, 0): 0.6,
    (0, 0, 1): 1.2,
    (1, 1, 0): -0.3,
    (1, 0, 1): 0.2,
    (0, 1, 1): 0.1,
    (2, 0, 0): 0.4,
    (0, 2, 0): -0.5,
    (0, 0, 2): 0.3,
}


def _validate_coefficients(coefficients: PolynomialCoefficients) -> int:
    """Validate polynomial coefficients and return the feature count."""
    if not coefficients:
        raise ValueError("coefficients must contain at least one term.")

    exponent_lengths = {len(exponents) for exponents in coefficients}
    if len(exponent_lengths) != 1:
        raise ValueError("all coefficient exponent tuples must have the same length.")

    n_features = exponent_lengths.pop()
    if n_features <= 0:
        raise ValueError("coefficient exponent tuples cannot be empty.")

    for exponents in coefficients:
        if any(not isinstance(power, int) or power < 0 for power in exponents):
            raise ValueError("coefficient exponents must be non-negative integers.")

    return n_features


def _polynomial_exponents(n_features: int, degree: int) -> list[tuple[int, ...]]:
    """Return exponent tuples for all polynomial terms up to a degree."""
    return [
        exponents
        for exponents in product(range(degree + 1), repeat=n_features)
        if sum(exponents) <= degree
    ]


def parse_coefficient_input(input_text: str) -> PolynomialCoefficients:
    """Parse polynomial coefficients from a compact input string.

    The expected format is ``n;(e1,e2,...,en)=a;(e1,e2,...,en)=b``. The first
    field is a single positive integer giving the number of variables. Each
    later field maps an exponent tuple to a coefficient.

    Parameters
    ----------
    input_text:
        Coefficient input string.

    Returns
    -------
    PolynomialCoefficients
        Mapping from exponent tuples to coefficient values.
    """
    parts = [part.strip() for part in input_text.split(";") if part.strip()]
    if len(parts) < 2:
        raise ValueError("input_text must include a feature count and at least one term.")

    if _FEATURE_COUNT_PATTERN.match(parts[0]) is None:
        raise ValueError("first input field must be a positive integer.")

    n_features = int(parts[0])
    if n_features <= 0:
        raise ValueError("number of variables must be positive.")

    coefficients: PolynomialCoefficients = {}
    for term_text in parts[1:]:
        term_match = _TERM_PATTERN.match(term_text)
        if term_match is None:
            raise ValueError(f"invalid coefficient term: {term_text}")

        exponent_text = term_match.group(1)
        exponents = tuple(int(power.strip()) for power in exponent_text.split(","))
        if len(exponents) != n_features:
            raise ValueError(
                "coefficient exponent tuple length must match the number of variables."
            )
        if any(power < 0 for power in exponents):
            raise ValueError("coefficient exponents must be non-negative integers.")

        coefficients[exponents] = float(term_match.group(2))

    _validate_coefficients(coefficients)
    return coefficients


def evaluate_polynomial(
    x_values: np.ndarray,
    coefficients: PolynomialCoefficients,
) -> np.ndarray:
    """Evaluate a polynomial with user-provided coefficients.

    Coefficients are provided as a mapping from exponent tuples to weights. For
    example, ``{(0, 0): 2.0, (1, 0): 3.0, (0, 1): -1.0}`` represents a
    two-feature polynomial with a constant term, an ``x1`` term, and an ``x2``
    term.

    Parameters
    ----------
    x_values:
        Feature matrix with shape ``(n_samples, n_features)``.
    coefficients:
        Mapping from exponent tuples to coefficient values.

    Returns
    -------
    np.ndarray
        Target values without random noise.
    """
    n_features = _validate_coefficients(coefficients)
    x_values = np.asarray(x_values, dtype=float)

    if x_values.ndim != 2 or x_values.shape[1] != n_features:
        raise ValueError(f"x_values must have shape (n_samples, {n_features}).")

    y_values = np.zeros(x_values.shape[0], dtype=float)
    for exponents, coefficient in coefficients.items():
        term_values = np.ones(x_values.shape[0], dtype=float)
        for feature_index, power in enumerate(exponents):
            if power:
                term_values *= x_values[:, feature_index] ** power
        y_values += coefficient * term_values

    return y_values


def generate_random_coefficients(
    n_features: int,
    degree: int,
    random_state: Optional[int] = None,
    coefficient_range: tuple[float, float] = (-1.0, 1.0),
    include_bias: bool = True,
) -> PolynomialCoefficients:
    """Generate random coefficients for a polynomial.

    Parameters
    ----------
    n_features:
        Number of input variables in the polynomial.
    degree:
        Maximum total degree of the polynomial.
    random_state:
        Optional seed for reproducible coefficient generation.
    coefficient_range:
        Lower and upper bounds for uniformly sampled coefficients.
    include_bias:
        Whether to include the constant term.

    Returns
    -------
    PolynomialCoefficients
        Mapping from exponent tuples to coefficient values.
    """
    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    if degree < 0:
        raise ValueError("degree must be non-negative.")

    lower, upper = coefficient_range
    if lower >= upper:
        raise ValueError("coefficient_range must be ordered as (lower, upper).")

    rng = np.random.default_rng(random_state)
    coefficients: PolynomialCoefficients = {}

    for exponents in _polynomial_exponents(n_features, degree):
        if not include_bias and sum(exponents) == 0:
            continue
        coefficients[exponents] = float(rng.uniform(lower, upper))

    return coefficients


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
    return evaluate_polynomial(x_values, DEFAULT_QUADRATIC_COEFFICIENTS)


def generate_polynomial_data(
    n_samples: int = 100,
    noise: float = 0.1,
    random_state: Optional[int] = None,
    feature_range: tuple[float, float] = DEFAULT_FEATURE_RANGE,
    coefficients: PolynomialCoefficients | None = None,
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
    coefficients:
        Optional polynomial coefficients. If omitted, the fixed trivariate
        quadratic relationship from the reference essay is used.

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

    if coefficients is None:
        coefficients = DEFAULT_QUADRATIC_COEFFICIENTS
    n_features = _validate_coefficients(coefficients)

    rng = np.random.default_rng(random_state)
    x_values = rng.uniform(lower, upper, size=(n_samples, n_features))
    y_values = evaluate_polynomial(x_values, coefficients)

    if noise > 0:
        y_values = y_values + rng.normal(loc=0.0, scale=noise, size=n_samples)

    return x_values, y_values


def generate_polynomial_data_from_input(
    input_text: str,
    n_samples: int = 100,
    noise: float = 0.1,
    random_state: Optional[int] = None,
    feature_range: tuple[float, float] = DEFAULT_FEATURE_RANGE,
) -> tuple[np.ndarray, np.ndarray, PolynomialCoefficients]:
    """Generate polynomial data from a coefficient input string.

    Parameters
    ----------
    input_text:
        Coefficient input string parsed by ``parse_coefficient_input``.
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
    tuple[np.ndarray, np.ndarray, PolynomialCoefficients]
        Feature values, target values, and parsed coefficients.
    """
    coefficients = parse_coefficient_input(input_text)
    x_values, y_values = generate_polynomial_data(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
        feature_range=feature_range,
        coefficients=coefficients,
    )

    return x_values, y_values, coefficients


def generate_random_polynomial_data(
    n_samples: int = 100,
    n_features: int = 3,
    degree: int = 2,
    noise: float = 0.1,
    random_state: Optional[int] = None,
    feature_range: tuple[float, float] = DEFAULT_FEATURE_RANGE,
    coefficient_range: tuple[float, float] = (-1.0, 1.0),
    include_bias: bool = True,
) -> tuple[np.ndarray, np.ndarray, PolynomialCoefficients]:
    """Generate a random polynomial dataset and return its coefficients.

    Parameters
    ----------
    n_samples:
        Number of data points to generate.
    n_features:
        Number of input variables.
    degree:
        Maximum total degree of the generated polynomial.
    noise:
        Standard deviation of Gaussian noise added to the target values.
    random_state:
        Optional seed for reproducible data and coefficient generation.
    feature_range:
        Lower and upper bounds for uniformly sampled feature values.
    coefficient_range:
        Lower and upper bounds for uniformly sampled coefficients.
    include_bias:
        Whether to include the constant term.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, PolynomialCoefficients]
        Feature values, target values, and the generated coefficients.
    """
    coefficients = generate_random_coefficients(
        n_features=n_features,
        degree=degree,
        random_state=random_state,
        coefficient_range=coefficient_range,
        include_bias=include_bias,
    )
    x_values, y_values = generate_polynomial_data(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
        feature_range=feature_range,
        coefficients=coefficients,
    )

    return x_values, y_values, coefficients
