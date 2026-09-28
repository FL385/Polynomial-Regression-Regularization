# Project Instructions

## Goal and Scope

This educational project demonstrates regression simulation and optimization,
including overfitting and regularization. Keep development focused on the chain
from synthetic data generation through model fitting, parameter selection, and
evaluation of regression results.

- Deepen support for multivariate and higher-dimensional polynomials, including
  interaction terms, when relevant to the requested experiment.
- Exponential models and related regression experiments are also within scope
  when explicitly requested.
- Improve degree selection, regularization strength selection, numerical
  stability, and evaluation methods as part of this regression workflow.
- Avoid unrelated features, broad application infrastructure, and unnecessary
  dependencies. Add supporting functionality only when needed for the current
  regression experiment.
- Keep each task small, focused, and independently reviewable. These permitted
  directions are boundaries, not instructions to implement them all at once.

## Coding Style

- Use Python 3 and simple, beginner-readable functions.
- Prefer clarity over advanced optimization; use type hints where reasonable
  and docstrings for public functions.
- Follow the existing project structure and reuse existing helpers.
- Use numpy, scikit-learn, and matplotlib where needed. Do not add seaborn
  unless explicitly requested.
- Add focused tests for changes to regression behavior.

## Communication

Render mathematical formulas with LaTeX unless formula source is explicitly
requested.

## Validation

Run the relevant tests with:

```bash
.venv/bin/python -m pytest
```
