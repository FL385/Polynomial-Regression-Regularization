# Polynomial Regression and Regularization

This project is a beginner-friendly demonstration of overfitting and
regularization with polynomial regression.

It will compare:

- Unregularized polynomial regression
- Ridge regression
- Lasso regression
- Elastic Net regression
- Bayesian Ridge regression
- ARD regression
- Huber regression

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
│   │   ├── no_regularization.py
│   │   ├── l1_regression.py
│   │   ├── l2_regression.py
│   │   ├── elastic_net_regression.py
│   │   ├── bayesian_ridge_regression.py
│   │   ├── ard_regression.py
│   │   ├── huber_regression.py
│   │   ├── methods.py
│   │   ├── evaluation.py
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

Save generated data to CSV:

```python
from src.data_generator import generate_polynomial_data, save_dataset_to_csv

x_values, y_values = generate_polynomial_data(
    n_samples=100,
    noise=0.1,
    random_state=42,
)

save_dataset_to_csv(x_values, y_values, "dataset.csv")
```

## CSV Regression Experiments

Experiment helpers read generated datasets from CSV files. The expected CSV
format is:

```text
x1,x2,...,y
1.0,2.0,...,3.5
```

The `y` column is the target value, and all other columns are treated as input
features.

Run all seven polynomial regression methods from a CSV file:

```python
from src.experiments import run_regression_comparison_from_csv

polynomials = run_regression_comparison_from_csv(
    csv_path="dataset.csv",
    degree=2,
    l1_alpha=0.01,
    l2_alpha=1.0,
)

print(polynomials["none"])
print(polynomials["l1"])
print(polynomials["l2"])
print(polynomials["elastic_net"])
print(polynomials["bayesian_ridge"])
print(polynomials["ard"])
print(polynomials["huber"])
```

Each regression type is also available as a separate experiment file:

```python
from src.experiment.no_regularization import fit_no_regularization_polynomial_from_csv
from src.experiment.l1_regression import fit_l1_polynomial_from_csv
from src.experiment.l2_regression import fit_l2_polynomial_from_csv
from src.experiment.elastic_net_regression import fit_elastic_net_polynomial_from_csv
from src.experiment.bayesian_ridge_regression import fit_bayesian_ridge_polynomial_from_csv
from src.experiment.ard_regression import fit_ard_polynomial_from_csv
from src.experiment.huber_regression import fit_huber_polynomial_from_csv
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
train/validation split and searches degrees from `1` to `8` for all seven
methods. Parameter grids are defined in `src/experiment/methods.py`:

| Method | Search parameters |
| --- | --- |
| Unregularized | No penalty; alpha `0` |
| L1 / L2 | Alpha from `1e-6` to `100`, in powers of ten |
| Elastic Net | Same alpha grid; `l1_ratio` of `0.1`, `0.5`, `0.9` |
| Bayesian Ridge | Default priors; noise and coefficient precisions estimated during fitting |
| ARD | `threshold_lambda` of `1000`, `10000`, `100000`; per-term precisions estimated during fitting |
| Huber | Alpha `0.0001`, `0.01`, `1`; epsilon `1.1`, `1.35`, `2` |

`best["parameters"]` contains the selected method's parameters. Bayesian Ridge
and ARD do not use the same alpha parameter as Ridge/Lasso, so their result
`alpha` is `None`. Candidates that emit a convergence warning or produce a
non-finite validation MSE are excluded and recorded in `failed_candidates`.
The selected model
is the one with the lowest validation MSE. If two models are effectively tied,
the simpler lower-degree model is preferred. All methods fit polynomial terms
scaled to unit standard deviation using only their training rows. Coefficients
and Bayesian covariance matrices are then restored to the original variable
units for predictions, equation output, and ground-truth evaluation. This
changes the regularization behavior of the earlier unscaled L1/L2 fits.

For fixed-degree comparisons, customize the new methods using
`method_parameters` (or call their individual CSV helpers):

```python
polynomials = run_regression_comparison_from_csv(
    "dataset.csv", degree=2,
    method_parameters={
        "elastic_net": {"alpha": 0.01, "l1_ratio": 0.5},
        "huber": {"alpha": 0.0001, "epsilon": 1.35},
        "ard": {"threshold_lambda": 10000},
    },
)
```

Fixed-degree `run_experiment` accepts the same `method_parameters` mapping.
The defaults are alpha `0.01` and ratio `0.5` for Elastic Net, alpha `0.0001`
and epsilon `1.35` for Huber, and threshold `10000` for ARD. Degree-free
automatic selection always uses the built-in search grids.

## Comparing With the Original Polynomial

CSV values alone do not identify the original polynomial. Keep the coefficient
mapping returned by the generator and pass it separately for evaluation:

```python
from src.data_generator import generate_random_polynomial_data, save_dataset_to_csv
from src.experiments import run_experiment
import numpy as np

x, y, coefficients = generate_random_polynomial_data(
    n_samples=120, n_features=2, degree=2, noise=0.1, random_state=42,
)
save_dataset_to_csv(x, y, "dataset.csv")
evaluation_points = np.random.default_rng(43).uniform(-3, 3, size=(1000, 2))
results = run_experiment({
    "csv_path": "dataset.csv",
    "true_coefficients": coefficients,
    "x_evaluation": evaluation_points,
})
print(results["best"]["deviation"])
```

With an explicit `degree`, comparisons are returned in `results["deviations"]`
for all seven methods. Without a degree, only the selected final model is
compared. Ground truth is not used to select that model.

Metrics include signed per-term coefficient differences, coefficient L2 error,
relative coefficient L2 error, and function MSE, RMSE, MAE, and maximum absolute
error. Missing terms count as zero; relative coefficient error is `None` for a
zero original polynomial. Comparisons use full-precision fitted coefficients,
including the intercept, rather than parsing rounded equation strings.

Function errors compare predictions with the noiseless original polynomial at
the supplied points. They depend on the chosen input range and sample
distribution; the maximum is a sampled maximum, not a bound over the domain.
Use independent points to assess generalization. Coefficient errors depend on
the variable units and polynomial basis, so assess them alongside function
errors. Exponent tuple order must match the CSV feature column order.

## Plotting

Basic matplotlib plots are available in `src.plots`.

Plot model predictions for a one-feature dataset:

```python
from src.plots import plot_model_predictions

fig = plot_model_predictions(x_values, y_values, predictions)
fig.savefig("predictions.png")
```

Plot a simple model metric comparison:

```python
from src.plots import plot_metric_comparison

fig = plot_metric_comparison({"none": 3.0, "l1": 1.0, "l2": 2.0})
fig.savefig("metrics.png")
```

Plot automatic-search validation MSE by degree:

```python
from src.experiments import find_best_regularized_polynomial_from_csv
from src.plots import plot_best_validation_mse_by_degree

results = find_best_regularized_polynomial_from_csv("dataset.csv")
fig = plot_best_validation_mse_by_degree(results["candidates"])
fig.savefig("validation_mse_by_degree.png")
```

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
and L2 polynomial regression helpers are also available alongside an
unregularized baseline, Elastic Net, Bayesian Ridge, ARD, and Huber, including
automatic degree and method-specific parameter selection from a CSV file.
Ground-truth coefficient and function deviations can also be evaluated.
Basic plotting helpers are available for model
predictions, metric comparisons, and automatic-search validation curves.
