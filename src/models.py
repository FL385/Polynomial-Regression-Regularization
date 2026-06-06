"""Model creation and training helpers for polynomial regression experiments."""

from __future__ import annotations

from typing import Any


def create_polynomial_regression(degree: int) -> Any:
    """Create an unregularized polynomial regression model.

    Parameters
    ----------
    degree:
        Polynomial degree used to transform the input feature.

    Returns
    -------
    Any
        A scikit-learn compatible model pipeline.
    """
    raise NotImplementedError("Polynomial regression will be implemented later.")


def create_ridge_regression(degree: int, alpha: float = 1.0) -> Any:
    """Create a polynomial Ridge regression model.

    Parameters
    ----------
    degree:
        Polynomial degree used to transform the input feature.
    alpha:
        Regularization strength for Ridge regression.

    Returns
    -------
    Any
        A scikit-learn compatible model pipeline.
    """
    raise NotImplementedError("Ridge regression will be implemented later.")


def create_lasso_regression(degree: int, alpha: float = 0.01) -> Any:
    """Create a polynomial Lasso regression model.

    Parameters
    ----------
    degree:
        Polynomial degree used to transform the input feature.
    alpha:
        Regularization strength for Lasso regression.

    Returns
    -------
    Any
        A scikit-learn compatible model pipeline.
    """
    raise NotImplementedError("Lasso regression will be implemented later.")


def train_model(model: Any, x_train: Any, y_train: Any) -> Any:
    """Train a regression model on the provided dataset.

    Parameters
    ----------
    model:
        Model object with a ``fit`` method.
    x_train:
        Training feature values.
    y_train:
        Training target values.

    Returns
    -------
    Any
        The trained model.
    """
    raise NotImplementedError("Model training will be implemented later.")
