"""Tests for synthetic polynomial data generation."""

from __future__ import annotations

import numpy as np
import pytest

from src.data_generator import generate_polynomial_data, true_quadratic_function


def test_generate_polynomial_data_shapes() -> None:
    """Generated features and targets should have predictable shapes."""
    x_values, y_values = generate_polynomial_data(n_samples=25, random_state=42)

    assert x_values.shape == (25, 3)
    assert y_values.shape == (25,)


def test_generate_polynomial_data_is_reproducible() -> None:
    """The same random seed should produce the same dataset."""
    first_x, first_y = generate_polynomial_data(n_samples=10, random_state=42)
    second_x, second_y = generate_polynomial_data(n_samples=10, random_state=42)

    np.testing.assert_allclose(first_x, second_x)
    np.testing.assert_allclose(first_y, second_y)


def test_generate_polynomial_data_without_noise_matches_true_function() -> None:
    """Noiseless targets should match the fixed quadratic function."""
    x_values, y_values = generate_polynomial_data(
        n_samples=10,
        noise=0.0,
        random_state=42,
    )

    np.testing.assert_allclose(y_values, true_quadratic_function(x_values))


def test_true_quadratic_function_known_value() -> None:
    """The fixed equation should be easy to verify at a simple point."""
    x_values = np.array([[1.0, 2.0, 3.0]])

    np.testing.assert_allclose(true_quadratic_function(x_values), np.array([7.2]))


def test_generate_polynomial_data_rejects_invalid_arguments() -> None:
    """Invalid generator settings should fail with clear errors."""
    with pytest.raises(ValueError, match="n_samples"):
        generate_polynomial_data(n_samples=0)

    with pytest.raises(ValueError, match="noise"):
        generate_polynomial_data(noise=-1.0)

    with pytest.raises(ValueError, match="feature_range"):
        generate_polynomial_data(feature_range=(1.0, 1.0))
