"""Shared method definitions for comparison and parameter search."""

from src.experiment.ard_regression import fit_ard_polynomial
from src.experiment.bayesian_ridge_regression import fit_bayesian_ridge_polynomial
from src.experiment.elastic_net_regression import fit_elastic_net_polynomial
from src.experiment.huber_regression import fit_huber_polynomial
from src.experiment.l1_regression import fit_l1_polynomial
from src.experiment.l2_regression import fit_l2_polynomial
from src.experiment.no_regularization import fit_no_regularization_polynomial
from src.experiment.common import DEFAULT_ALPHA_CANDIDATES

FITTERS = {
    "none": fit_no_regularization_polynomial,
    "l1": fit_l1_polynomial,
    "l2": fit_l2_polynomial,
    "elastic_net": fit_elastic_net_polynomial,
    "bayesian_ridge": fit_bayesian_ridge_polynomial,
    "ard": fit_ard_polynomial,
    "huber": fit_huber_polynomial,
}

PARAMETER_GRIDS = {
    "none": {"alpha": [0.0]},
    "l1": {"alpha": DEFAULT_ALPHA_CANDIDATES},
    "l2": {"alpha": DEFAULT_ALPHA_CANDIDATES},
    "elastic_net": {"alpha": DEFAULT_ALPHA_CANDIDATES, "l1_ratio": [0.1, 0.5, 0.9]},
    "bayesian_ridge": {},
    "ard": {"threshold_lambda": [1000.0, 10000.0, 100000.0]},
    "huber": {"alpha": [0.0001, 0.01, 1.0], "epsilon": [1.1, 1.35, 2.0]},
}
