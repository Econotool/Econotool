# econtools

A unified Python interface over the standard applied-econometrics stack —
`statsmodels`, `linearmodels`, `scipy`, and `pandas` — with a consistent
type system for estimates, a common covariance-label vocabulary across
backends, and a LaTeX output layer built on `booktabs` / `threeparttable`.

The goal is modest: make the common cross-section / panel / IV workflows
accessible through one API, and let results flow to publication-quality
tables and figures without bespoke formatting code per project.

## What it provides

- **A single fit interface.** `fit_model(spec, df)` takes a declarative
  `ModelSpec` and dispatches to the appropriate backend — OLS and WLS via
  `statsmodels`; 2SLS, LIML, GMM, and panel estimators (FE, RE, FD, Pooled,
  Between) via `linearmodels`; Probit and Logit via `statsmodels`.
- **Unified result type.** Every estimator returns an `Estimate` with the
  same shape (coefficients, standard errors, fit metrics, raw backend
  result), so downstream code doesn't branch on estimator type.
- **Covariance vocabulary.** A shared label set (`classical`, `HC0`–`HC3`,
  `HAC`, `cluster`, `cluster2`, `driscoll_kraay`) resolves to the right
  keyword arguments for whichever backend is in use.
- **Inference utilities.** Wald, F, t, LR, and score tests; confidence
  intervals; a bootstrap module supporting iid pairs, wild, cluster, panel,
  and wild-cluster resampling.
- **Diagnostics.** Heteroskedasticity, normality, specification,
  multicollinearity, serial correlation, stationarity, IV validity
  (Sargan / Hansen J, Wu–Hausman, weak-instrument tests), panel exogeneity,
  influence measures, and structural-break tests.
- **Tables.** `ResultsTable`, `SummaryTable`, `DiagnosticsTable`, and a
  composite renderer that stacks multiple panels under a shared caption and
  notes block with column widths aligned via least-common-multiple.
- **Plots.** Figure-returning functions for residual diagnostics, QQ plots,
  coefficient forests, correlograms, binscatters, and panel event studies.
- **Specification search.** A sieve module for systematic functional-form
  and instrument exploration with cross-fitting and a Pareto frontier over
  fit vs. instrument strength.

## Installation

```bash
pip install -e .
# optional dev tools
pip install -e ".[dev]"
```

Requires Python 3.11 or later.

## Minimal example

```python
import pandas as pd
from econtools.fit import fit_model
from econtools.model.spec import ModelSpec
from econtools.output.tables import ResultsTable

df = pd.read_parquet("wages_panel.parquet")

specs = [
    ModelSpec(dep_var="log_wage", exog_vars=["educ", "exper", "exper_sq"],
              estimator="ols", cov_type="HC1"),
    ModelSpec(dep_var="log_wage", exog_vars=["educ", "exper", "exper_sq"],
              estimator="fe", entity_col="id", time_col="year",
              cov_type="cluster", cluster_col="id"),
]

results = [fit_model(s, df) for s in specs]

tex = ResultsTable(results, col_labels=["OLS", "FE"]).to_latex()
```

## Package layout

```
econtools/
├── _core/           # shared types (Estimate, FitMetrics, TestResult) and cov-type mapping
├── model/           # ModelSpec declarative fit specification
├── fit/             # fit_model dispatcher + statsmodels / linearmodels adapters
├── evaluation/      # statistical tests and diagnostics
├── uncertainty/     # covariance estimators and bootstrap
├── output/          # tables, figures, LaTeX assembly, knowledge base
├── replication/     # declarative paper-replication helpers (comparison, runner)
├── plots/           # Figure-returning plot functions
├── data/            # pipeline utilities (load, clean, transform, construct, provenance)
└── cli/             # command-line entry points
```

Legacy shim modules (`models/`, `inference/`, `diagnostics/`, `tables/`)
re-export from the new layout for backwards compatibility; new code should
import directly from the new modules.

## Tests

```bash
# fast suite
python -m pytest tests/ -v -m "not phase3 and not slow and not validation"
```

Numerical validation tests under `tests/validation/` cross-check outputs
against reference datasets and require a local data store.

## Status

The core fitting, inference, tables, and diagnostics layers are stable.
System-equation estimators (SUR, 3SLS) and linear factor-model tooling
(Fama–MacBeth) are planned.

## License

MIT. See `LICENSE`.
