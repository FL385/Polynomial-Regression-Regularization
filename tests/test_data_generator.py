"""Tests for synthetic polynomial data generation."""

from __future__ import annotations

import numpy as np
import pytest

from src.data_generator import (
    evaluate_polynomial,
    generate_polynomial_data,
    generate_polynomial_data_from_input,
    generate_random_coefficients,
    generate_random_polynomial_data,
    parse_coefficient_input,
    true_quadratic_function,
)


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


def test_evaluate_polynomial_uses_custom_coefficients() -> None:
    """Custom coefficients should define the generated relationship."""
    coefficients = {
        (0, 0): 2.0,
        (1, 0): 3.0,
        (0, 1): -1.0,
        (1, 1): 0.5,
    }
    x_values = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    np.testing.assert_allclose(
        evaluate_polynomial(x_values, coefficients),
        np.array([4.0, 13.0]),
    )


def test_generate_polynomial_data_accepts_custom_coefficients() -> None:
    """Generated data should support arbitrary feature counts and coefficients."""
    coefficients = {
        (0, 0): 1.0,
        (1, 0): 2.0,
        (0, 2): -0.5,
    }
    x_values, y_values = generate_polynomial_data(
        n_samples=12,
        noise=0.0,
        random_state=42,
        coefficients=coefficients,
    )

    assert x_values.shape == (12, 2)
    np.testing.assert_allclose(y_values, evaluate_polynomial(x_values, coefficients))


def test_parse_coefficient_input_accepts_requested_format() -> None:
    """Coefficient input strings should parse into exponent tuples."""
    input_text = "3;(0,0,0)=1.5;(1,0,0)=-0.8;(0,2,1)=0.25"

    assert parse_coefficient_input(input_text) == {
        (0, 0, 0): 1.5,
        (1, 0, 0): -0.8,
        (0, 2, 1): 0.25,
    }


def test_generate_polynomial_data_from_input_returns_parsed_coefficients() -> None:
    """Input-based generation should reuse the parsed coefficient map."""
    input_text = "2;(0,0)=1;(1,0)=2;(0,2)=-0.5"
    x_values, y_values, coefficients = generate_polynomial_data_from_input(
        input_text,
        n_samples=12,
        noise=0.0,
        random_state=42,
    )

    assert x_values.shape == (12, 2)
    np.testing.assert_allclose(y_values, evaluate_polynomial(x_values, coefficients))


def test_generate_random_coefficients_is_reproducible() -> None:
    """Random coefficient generation should be reproducible with a seed."""
    first = generate_random_coefficients(n_features=2, degree=2, random_state=42)
    second = generate_random_coefficients(n_features=2, degree=2, random_state=42)

    assert first == second
    assert len(first) == 6
    assert all(sum(exponents) <= 2 for exponents in first)


def test_generate_random_polynomial_data_returns_coefficients() -> None:
    """Random polynomial datasets should include the generated coefficients."""
    x_values, y_values, coefficients = generate_random_polynomial_data(
        n_samples=8,
        n_features=4,
        degree=2,
        noise=0.0,
        random_state=42,
    )

    assert x_values.shape == (8, 4)
    np.testing.assert_allclose(y_values, evaluate_polynomial(x_values, coefficients))


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


def test_evaluate_polynomial_rejects_invalid_coefficients() -> None:
    """Invalid custom coefficient maps should fail with clear errors."""
    x_values = np.array([[1.0, 2.0]])

    with pytest.raises(ValueError, match="at least one term"):
        evaluate_polynomial(x_values, {})

    with pytest.raises(ValueError, match="same length"):
        evaluate_polynomial(x_values, {(1,): 1.0, (0, 1): 2.0})

    with pytest.raises(ValueError, match="non-negative integers"):
        evaluate_polynomial(x_values, {(-1, 0): 1.0})


def test_parse_coefficient_input_rejects_invalid_input() -> None:
    """Invalid input strings should fail with clear errors."""
    with pytest.raises(ValueError, match="feature count"):
        parse_coefficient_input("(0,0)=1")

    with pytest.raises(ValueError, match="number of variables"):
        parse_coefficient_input("0;(0)=1")

    with pytest.raises(ValueError, match="positive integer"):
        parse_coefficient_input("n=2;(0,0)=1")

    with pytest.raises(ValueError, match="tuple length"):
        parse_coefficient_input("2;(1)=1")

    with pytest.raises(ValueError, match="invalid coefficient term"):
        parse_coefficient_input("2;(1,0):1")
