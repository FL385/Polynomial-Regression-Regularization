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
│   ├── experiment/
│   │   ├── l1_regression.py
│   │   ├── l2_regression.py
│   │   ├── auto_selection.py
│   │   ├── runner.py
│   │   └── common.py
│   └── plots.py
└── tests/
```

## Planned Workflow

1. Generate a synthetic regression dataset.
2. Fit polynomial regression models with different degrees.
3. Compare unregularized, Ridge, and Lasso models.
4. Visualize how regularization changes model behavior.

## Data Generation

The data generator supports both the reference essay dataset and custom
polynomial datasets.

Use the default reference relationship:

```python
from src.data_generator import generate_polynomial_data

x_values, y_values = generate_polynomial_data(
    n_samples=100,
    noise=0.1,
    random_state=42,
)
```

Use custom coefficients for any number of input variables:

```python
from src.data_generator import generate_polynomial_data

coefficients = {
    (0, 0): 1.0,
    (1, 0): 2.0,
    (0, 2): -0.5,
}

x_values, y_values = generate_polynomial_data(
    n_samples=100,
    noise=0.1,
    random_state=42,
    coefficients=coefficients,
)
```

Generate a random polynomial dataset and keep the coefficients:

```python
from src.data_generator import generate_random_polynomial_data

x_values, y_values, coefficients = generate_random_polynomial_data(
    n_samples=100,
    n_features=4,
    degree=2,
    random_state=42,
)
```

Use a compact coefficient input string:

```python
from src.data_generator import generate_polynomial_data_from_input

input_text = "2;(0,0)=1;(1,0)=2;(0,2)=-0.5"

x_values, y_values, coefficients = generate_polynomial_data_from_input(
    input_text,
    n_samples=100,
    noise=0.1,
    random_state=42,
)
```

The input format is:

```text
n;(e1,e2,...,en)=a;(e1,e2,...,en)=b;...
```

The first field `n` is a single positive integer. Here, `(e1,e2,...,en)` is
the exponent tuple for one polynomial term. For example, `(0,2)` means the term
using `x2^2`, and `(1,0)` means the term using `x1`.

## CSV Regression Experiments

Experiment helpers read generated datasets from CSV files. The expected CSV
format is:

```text
x1,x2,...,y
1.0,2.0,...,3.5
```

The `y` column is the target value, and all other columns are treated as input
features.

Run L1 and L2 polynomial regression from a CSV file:

```python
from src.experiments import run_l1_l2_regression_from_csv

polynomials = run_l1_l2_regression_from_csv(
    csv_path="dataset.csv",
    degree=2,
    l1_alpha=0.01,
    l2_alpha=1.0,
)

print(polynomials["l1"])
print(polynomials["l2"])
```

L1 and L2 are also available as separate experiment files:

```python
from src.experiment.l1_regression import fit_l1_polynomial_from_csv
from src.experiment.l2_regression import fit_l2_polynomial_from_csv
```

The output is a readable fitted polynomial, such as:

```text
y = 1.02 + 1.98*x1 - 0.49*x2^2
```

The regression degree must be provided because polynomial regression first
expands the input features up to a chosen maximum degree. L1 and L2
regularization shrink or remove coefficients after that feature expansion; they
do not decide the maximum degree by themselves.

Automatically choose the regression type, degree, and regularization strength:

```python
from src.experiments import find_best_regularized_polynomial_from_csv

results = find_best_regularized_polynomial_from_csv("dataset.csv")

print(results["best"]["regularization"])
print(results["best"]["degree"])
print(results["best"]["alpha"])
print(results["best"]["polynomial"])
```

This automatic search only requires the CSV file. It uses a deterministic
train/validation split, tests L1 and L2 models, searches degrees from `1` to
`8`, and searches regularization strengths from `1e-6` to `100`. The selected
model is the one with the lowest validation MSE. If two models are effectively
tied, the simpler lower-degree model is preferred.

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
data generator based on the reference essay. It can also generate random
polynomial datasets with custom feature counts and coefficients. CSV-based L1
and L2 polynomial regression helpers are also available, including automatic
degree and regularization-strength selection from a CSV file. The plotting logic
will be added in small follow-up changes.
