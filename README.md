# Polynomial Regression and Regularization

This project is a beginner-friendly demonstration of overfitting and
regularization with polynomial regression.

It will compare:

- Unregularized polynomial regression
- Ridge regression
- Lasso regression

The goal is educational clarity. Each module should stay small, readable, and
easy to review.

## Project Structure

```text
Polynomial-Regression-Regularization/
├── README.md
├── requirements.txt
├── src/
│   ├── data_generator.py
│   ├── models.py
│   ├── experiments.py
│   └── plots.py
└── tests/
```

## Planned Workflow

1. Generate a synthetic regression dataset.
2. Fit polynomial regression models with different degrees.
3. Compare unregularized, Ridge, and Lasso models.
4. Visualize how regularization changes model behavior.

## Setup

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

## Current Status

This repository currently contains the initial project structure and a synthetic
data generator based on the reference essay. The modeling, experiment, and
plotting logic will be added in small follow-up changes.
