"""Bootstrap inference for OLS and 2SLS estimators.

Public API
----------
run_bootstrap(data, y, X, estimator, bootstrap_method, ...) -> dict

Supported estimators
--------------------
- ``"ols"``  — ordinary least squares
- ``"2sls"`` — two-stage least squares / IV (single endogenous variable)

Supported bootstrap methods
----------------------------
- ``"iid_pairs"``       — i.i.d. pairs bootstrap (resample rows)
- ``"wild"``            — wild bootstrap (Rademacher or Mammen weights, iid)
- ``"cluster_pairs"``   — one-way cluster pairs bootstrap
- ``"panel_cluster_id"``— resample entity ids, keep full within-id histories
- ``"wild_cluster"``    — wild cluster bootstrap (WCB): one weight per cluster
                          (Cameron, Gelbach & Miller 2008).  Supports
                          Rademacher, Mammen, and Webb 6-point weights.

Not yet implemented (stub raises ``NotImplementedError`` with guidance):
- ``"block_time_series"``
- ``"stationary_bootstrap"``

Confidence intervals
--------------------
- **percentile**: quantiles of bootstrap coefficient draws
- **basic** (reverse percentile): ``2*beta_hat - quantiles(draws)``

Bootstrap p-values
------------------
Two-sided, based on the centred distribution.  Let d_b = beta*_b - beta_hat.
p = 2 * min((#{d_b <= 0} + 1) / (B + 1), (#{d_b >= 0} + 1) / (B + 1)).
Adding 1 to numerator and denominator avoids zero p-values (Davidson &
MacKinnon 2000).

Reproducibility
---------------
Per-draw RNGs are constructed via ``numpy.random.SeedSequence.spawn(B)``
so results are byte-for-byte identical regardless of execution order
(sequential or parallel via ``ThreadPoolExecutor``).
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from econtools.uncertainty._bootstrap_estimators import ols_fit, twosls_fit
from econtools.uncertainty._bootstrap_manifest import (
    build_manifest,
    write_manifest,
)

logger = logging.getLogger(__name__)

_SUPPORTED_ESTIMATORS = frozenset({"ols", "2sls"})
_SUPPORTED_METHODS = frozenset(
    {"iid_pairs", "wild", "cluster_pairs", "panel_cluster_id", "wild_cluster"}
)
_STUB_METHODS = frozenset({"block_time_series", "stationary_bootstrap"})
# Methods that require a cluster column
_CLUSTER_METHODS = frozenset({"cluster_pairs", "wild_cluster"})


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_bootstrap(
    data: pd.DataFrame,
    y: str,
    X: list[str],
    estimator: str,
    bootstrap_method: str,
    B: int = 1999,
    seed: int = 12345,
    Z: list[str] | None = None,
    endog: list[str] | None = None,
    cluster: str | None = None,
    id_col: str | None = None,
    time_col: str | None = None,
    add_intercept: bool = True,
    ci_level: float = 0.95,
    ci_methods: list[str] | None = None,
    wild_dist: str = "rademacher",
    block_length: int | None = None,
    save_draws_path: str | None = None,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Run bootstrap inference for OLS or 2SLS.

    Parameters
    ----------
    data:
        Input DataFrame.
    y:
        Dependent variable column.
    X:
        Exogenous regressor column names (a constant is added unless
        ``add_intercept=False`` or a constant column is already present).
    estimator:
        ``"ols"`` or ``"2sls"``.
    bootstrap_method:
        One of ``"iid_pairs"``, ``"wild"``, ``"cluster_pairs"``,
        ``"panel_cluster_id"``.
    B:
        Number of bootstrap replications.
    seed:
        Master random seed (integer). Controls all randomness deterministically.
    Z:
        Excluded instrument column names (required for ``estimator="2sls"``).
    endog:
        Endogenous regressor column names (required for ``estimator="2sls"``).
        Currently supports exactly one endogenous variable.
    cluster:
        Cluster identifier column for ``bootstrap_method="cluster_pairs"``.
    id_col:
        Entity identifier column for ``bootstrap_method="panel_cluster_id"``.
        Also used as the cluster column for that method.
    time_col:
        Time identifier column (used for within-id ordering in panel bootstrap).
    add_intercept:
        If ``True`` (default), prepend a constant column unless one already
        exists in *X*.
    ci_level:
        Confidence level, e.g. ``0.95`` for 95% CI.
    ci_methods:
        List of CI methods to compute.  Supported: ``"percentile"``,
        ``"basic"``.  Defaults to ``["percentile", "basic"]``.
    wild_dist:
        Wild bootstrap auxiliary distribution: ``"rademacher"`` (default),
        ``"mammen"``, or ``"webb"`` (6-point; only valid for
        ``bootstrap_method="wild_cluster"``; recommended for G < 10).
    block_length:
        Block length for ``"block_time_series"`` (not yet implemented).
    save_draws_path:
        If given, save the (B × k) draws array as a ``.npy`` file at this path.
    n_jobs:
        Number of parallel threads.  Results are deterministic regardless
        of this value.

    Returns
    -------
    dict with keys:
        ``"point_estimate"`` — original-sample estimates and diagnostics
        ``"bootstrap"``      — draws, SE, CIs, p-values, bagged coefficients
        ``"metadata"``       — reproducibility record (method, seed, manifest)
    """
    if ci_methods is None:
        ci_methods = ["percentile", "basic"]

    # --- validate ---
    estimator = estimator.lower()
    bootstrap_method = bootstrap_method.lower()

    if estimator not in _SUPPORTED_ESTIMATORS:
        raise ValueError(
            f"estimator must be one of {sorted(_SUPPORTED_ESTIMATORS)}, got {estimator!r}"
        )
    if bootstrap_method in _STUB_METHODS:
        raise NotImplementedError(
            f"bootstrap_method={bootstrap_method!r} is not yet implemented. "
            "Use one of: iid_pairs, wild, cluster_pairs, panel_cluster_id."
        )
    if bootstrap_method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"bootstrap_method must be one of "
            f"{sorted(_SUPPORTED_METHODS | _STUB_METHODS)}, "
            f"got {bootstrap_method!r}"
        )
    if estimator == "2sls":
        if endog is None or len(endog) == 0:
            raise ValueError(
                "estimator='2sls' requires `endog` (list of endogenous variable names)."
            )
        if Z is None or len(Z) == 0:
            raise ValueError(
                "estimator='2sls' requires `Z` (list of excluded instrument names)."
            )
    if bootstrap_method in _CLUSTER_METHODS and cluster is None:
        raise ValueError(
            f"bootstrap_method={bootstrap_method!r} requires `cluster` column."
        )
    if bootstrap_method == "panel_cluster_id" and id_col is None:
        raise ValueError(
            "bootstrap_method='panel_cluster_id' requires `id_col` column."
        )
    if not 0 < ci_level < 1:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    _valid_wild_dists = {"rademacher", "mammen", "webb"}
    if wild_dist not in _valid_wild_dists:
        raise ValueError(
            f"wild_dist must be one of {sorted(_valid_wild_dists)}, got {wild_dist!r}"
        )
    if wild_dist == "webb" and bootstrap_method != "wild_cluster":
        raise ValueError(
            "wild_dist='webb' is only supported for bootstrap_method='wild_cluster'. "
            "Use 'rademacher' or 'mammen' for other methods."
        )

    warn_list: list[str] = []

    # --- data preparation ---
    (
        y_arr, X_arr, X_endog_arr, Z_arr,
        coef_names,
        cluster_labels, id_labels, time_labels,
        cluster_col_used, n_obs, n_dropped,
        df_clean,
    ) = _prepare_data(
        data=data,
        y_col=y,
        x_cols=X,
        endog_cols=endog or [],
        z_cols=Z or [],
        cluster_col=cluster,
        id_col=id_col,
        time_col=time_col,
        add_intercept=add_intercept,
        bootstrap_method=bootstrap_method,
    )

    logger.info(
        "Bootstrap: n=%d observations, %d dropped, method=%s, estimator=%s, B=%d",
        n_obs, n_dropped, bootstrap_method, estimator, B,
    )

    # --- cluster info ---
    cluster_count: int | None = None
    if cluster_labels is not None:
        cluster_count = int(np.unique(cluster_labels).size)
        if cluster_count < 30:
            msg = (
                f"Small cluster count G={cluster_count} < 30. "
                "Bootstrap inference may be unreliable."
            )
            warn_list.append(msg)
            warnings.warn(msg, stacklevel=2)

    # --- point estimation ---
    point_est = _point_estimate(
        estimator=estimator,
        y=y_arr,
        X=X_arr,
        X_endog=X_endog_arr,
        Z=Z_arr,
        coef_names=coef_names,
    )
    beta_hat: np.ndarray = point_est["beta_hat"]
    resid_hat: np.ndarray = point_est["residuals"]

    # --- RNG setup: one independent Generator per draw ---
    ss = np.random.SeedSequence(seed)
    child_sequences = ss.spawn(B)
    rngs = [np.random.default_rng(s) for s in child_sequences]

    # --- bootstrap replications ---
    draws_list = _run_replications(
        B=B,
        rngs=rngs,
        method=bootstrap_method,
        estimator=estimator,
        y=y_arr,
        X_exog=X_arr,
        X_endog=X_endog_arr,
        Z=Z_arr,
        beta_hat=beta_hat,
        resid_hat=resid_hat,
        cluster_labels=cluster_labels,
        id_labels=id_labels,
        time_labels=time_labels,
        wild_dist=wild_dist,
        n_jobs=n_jobs,
    )
    draws = np.array(draws_list)  # (B, k)

    # --- inference ---
    alpha = 1.0 - ci_level
    q_lo = alpha / 2
    q_hi = 1.0 - alpha / 2

    bootstrap_se = draws.std(axis=0, ddof=1)

    # Centered draws for p-values: d_b = beta*_b - beta_hat
    centered = draws - beta_hat[np.newaxis, :]  # (B, k)

    # CIs
    ci_out: dict[str, dict] = {}
    if "percentile" in ci_methods:
        lo = np.quantile(draws, q_lo, axis=0)
        hi = np.quantile(draws, q_hi, axis=0)
        ci_out["percentile"] = {
            "lower": dict(zip(coef_names, lo.tolist())),
            "upper": dict(zip(coef_names, hi.tolist())),
        }
    if "basic" in ci_methods:
        lo_b = 2.0 * beta_hat - np.quantile(draws, q_hi, axis=0)
        hi_b = 2.0 * beta_hat - np.quantile(draws, q_lo, axis=0)
        ci_out["basic"] = {
            "lower": dict(zip(coef_names, lo_b.tolist())),
            "upper": dict(zip(coef_names, hi_b.tolist())),
        }

    # p-values: two-sided, centred (Davidson–MacKinnon finite-sample adj)
    pvalues: dict[str, float] = {}
    for j, name in enumerate(coef_names):
        col = centered[:, j]
        p_lo = (float(np.sum(col <= 0)) + 1.0) / (B + 1.0)
        p_hi = (float(np.sum(col >= 0)) + 1.0) / (B + 1.0)
        pvalues[name] = float(2.0 * min(p_lo, p_hi))

    # --- build output ---
    config_record: dict[str, Any] = {
        "estimator": estimator,
        "bootstrap_method": bootstrap_method,
        "B": B,
        "seed": seed,
        "add_intercept": add_intercept,
        "ci_level": ci_level,
        "ci_methods": sorted(ci_methods),
        "wild_dist": wild_dist,
        "y": y,
        "X": sorted(X),
        "endog": sorted(endog) if endog else [],
        "Z": sorted(Z) if Z else [],
        "cluster": cluster,
        "id_col": id_col,
        "time_col": time_col,
        "n_jobs": n_jobs,
    }

    manifest = build_manifest(
        config=config_record,
        n_obs=n_obs,
        n_dropped=n_dropped,
        y_col=y,
        x_cols=X,
        endog_cols=endog or [],
        z_cols=Z or [],
        cluster_col=cluster,
        id_col=id_col,
        cluster_count=cluster_count,
        warnings_list=warn_list,
    )

    result: dict[str, Any] = {
        "point_estimate": {
            "params": dict(zip(coef_names, beta_hat.tolist())),
            "coef_names": coef_names,
            "fitted": point_est["fitted"].tolist(),
            "residuals": resid_hat.tolist(),
            "first_stage_diagnostics": point_est.get("first_stage_diagnostics"),
        },
        "bootstrap": {
            "draws": draws,          # (B, k) ndarray — raw draws
            "se": dict(zip(coef_names, bootstrap_se.tolist())),
            "ci": ci_out,
            "pvalues": pvalues,
            "bagged_mean": dict(zip(coef_names, draws.mean(axis=0).tolist())),
            "bagged_median": dict(zip(coef_names, np.median(draws, axis=0).tolist())),
        },
        "metadata": {
            "method": bootstrap_method,
            "estimator": estimator,
            "B": B,
            "seed": seed,
            "n_obs": n_obs,
            "n_dropped": n_dropped,
            "cluster_count": cluster_count,
            "config_hash": manifest["config_hash"],
            "timestamp": manifest["timestamp"],
            "git_commit": manifest["git_commit"],
            "package_versions": manifest["package_versions"],
            "warnings": warn_list,
        },
        "_manifest": manifest,
    }

    # --- optional saves ---
    if save_draws_path is not None:
        p = Path(save_draws_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), draws)
        logger.info("Draws saved to %s", p)

    return result


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _prepare_data(
    data: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    endog_cols: list[str],
    z_cols: list[str],
    cluster_col: str | None,
    id_col: str | None,
    time_col: str | None,
    add_intercept: bool,
    bootstrap_method: str,
) -> tuple:
    """Select columns, perform listwise deletion, build numpy arrays.

    Returns
    -------
    y_arr, X_arr, X_endog_arr, Z_arr,
    coef_names,
    cluster_labels, id_labels, time_labels,
    cluster_col_used,
    n_obs, n_dropped,
    df_clean
    """
    # columns needed for estimation (listwise deletion)
    analysis_cols = list(dict.fromkeys([y_col] + x_cols + endog_cols + z_cols))
    # auxiliary columns
    aux_cols = [c for c in [cluster_col, id_col, time_col] if c is not None]
    all_needed = list(dict.fromkeys(analysis_cols + aux_cols))

    missing = [c for c in all_needed if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    df = data[all_needed].copy()
    n_before = len(df)
    df = df.dropna(subset=analysis_cols).reset_index(drop=True)
    n_dropped = n_before - len(df)
    n_obs = len(df)

    if n_obs == 0:
        raise ValueError("No observations remain after listwise deletion.")

    logger.info("Listwise deletion: %d → %d obs (%d dropped)", n_before, n_obs, n_dropped)

    # --- y ---
    y_arr = df[y_col].to_numpy(dtype=float)

    # --- X (exogenous) with optional constant ---
    X_raw = df[x_cols].to_numpy(dtype=float) if x_cols else np.empty((n_obs, 0))
    coef_names_x: list[str] = list(x_cols)

    if add_intercept:
        ones = np.ones(n_obs)
        has_const = X_raw.shape[1] > 0 and any(
            np.allclose(X_raw[:, j], ones) for j in range(X_raw.shape[1])
        )
        if not has_const:
            X_arr = np.column_stack([ones, X_raw]) if X_raw.shape[1] > 0 else ones[:, np.newaxis]
            coef_names = ["const"] + coef_names_x
        else:
            X_arr = X_raw
            coef_names = coef_names_x
    else:
        X_arr = X_raw
        coef_names = coef_names_x

    # --- endogenous regressors ---
    if endog_cols:
        X_endog_arr = df[endog_cols].to_numpy(dtype=float)
        coef_names = coef_names + list(endog_cols)
    else:
        X_endog_arr = None

    # --- instruments ---
    Z_arr = df[z_cols].to_numpy(dtype=float) if z_cols else None

    # --- cluster / id / time ---
    cluster_labels: np.ndarray | None = None
    id_labels: np.ndarray | None = None
    time_labels: np.ndarray | None = None
    cluster_col_used: str | None = None

    if cluster_col is not None:
        cluster_labels = df[cluster_col].to_numpy()
        cluster_col_used = cluster_col
    if id_col is not None:
        id_labels = df[id_col].to_numpy()
        if bootstrap_method == "panel_cluster_id":
            cluster_labels = id_labels  # panel bootstrap uses id as cluster
            cluster_col_used = id_col
    if time_col is not None:
        time_labels = df[time_col].to_numpy()

    return (
        y_arr, X_arr, X_endog_arr, Z_arr,
        coef_names,
        cluster_labels, id_labels, time_labels,
        cluster_col_used,
        n_obs, n_dropped,
        df,
    )


# ---------------------------------------------------------------------------
# Point estimation
# ---------------------------------------------------------------------------


def _point_estimate(
    estimator: str,
    y: np.ndarray,
    X: np.ndarray,
    X_endog: np.ndarray | None,
    Z: np.ndarray | None,
    coef_names: list[str],
) -> dict[str, Any]:
    """Compute point estimates on the original sample."""
    if estimator == "ols":
        coefs, resid, fitted = ols_fit(X, y)
        return {
            "beta_hat": coefs,
            "residuals": resid,
            "fitted": fitted,
        }
    else:  # 2sls
        coefs, resid, fitted, f_stat = twosls_fit(y, X, X_endog, Z)
        return {
            "beta_hat": coefs,
            "residuals": resid,
            "fitted": fitted,
            "first_stage_diagnostics": {
                "first_stage_F": f_stat,
                "weak_iv_flag": (not np.isnan(f_stat)) and (f_stat < 10.0),
            },
        }


# ---------------------------------------------------------------------------
# Bootstrap replications
# ---------------------------------------------------------------------------


def _run_replications(
    *,
    B: int,
    rngs: list[np.random.Generator],
    method: str,
    estimator: str,
    y: np.ndarray,
    X_exog: np.ndarray,
    X_endog: np.ndarray | None,
    Z: np.ndarray | None,
    beta_hat: np.ndarray,
    resid_hat: np.ndarray,
    cluster_labels: np.ndarray | None,
    id_labels: np.ndarray | None,
    time_labels: np.ndarray | None,
    wild_dist: str,
    n_jobs: int,
) -> list[np.ndarray]:
    """Run all B bootstrap replications; return list of coefficient arrays."""

    # Pre-build cluster mapping for cluster-based methods
    cluster_to_indices: dict[Any, np.ndarray] | None = None
    unique_clusters: np.ndarray | None = None
    if cluster_labels is not None and method in (
        "cluster_pairs", "panel_cluster_id", "wild_cluster"
    ):
        unique_clusters = np.unique(cluster_labels)
        cluster_to_indices = {
            c: np.where(cluster_labels == c)[0] for c in unique_clusters
        }

    def _single_draw(b: int) -> np.ndarray:
        rng = rngs[b]
        return _one_draw(
            rng=rng,
            method=method,
            estimator=estimator,
            y=y,
            X_exog=X_exog,
            X_endog=X_endog,
            Z=Z,
            beta_hat=beta_hat,
            resid_hat=resid_hat,
            cluster_labels=cluster_labels,
            unique_clusters=unique_clusters,
            cluster_to_indices=cluster_to_indices,
            time_labels=time_labels,
            wild_dist=wild_dist,
        )

    if n_jobs == 1:
        return [_single_draw(b) for b in range(B)]

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(_single_draw, range(B)))


def _one_draw(
    *,
    rng: np.random.Generator,
    method: str,
    estimator: str,
    y: np.ndarray,
    X_exog: np.ndarray,
    X_endog: np.ndarray | None,
    Z: np.ndarray | None,
    beta_hat: np.ndarray,
    resid_hat: np.ndarray,
    cluster_labels: np.ndarray | None,
    unique_clusters: np.ndarray | None,
    cluster_to_indices: dict | None,
    time_labels: np.ndarray | None,
    wild_dist: str,
) -> np.ndarray:
    """Execute a single bootstrap replication; return coefficient vector."""
    n = len(y)

    if method == "iid_pairs":
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        X_b = X_exog[idx]
        Xe_b = X_endog[idx] if X_endog is not None else None
        Z_b = Z[idx] if Z is not None else None

    elif method == "wild":
        v = _wild_weights(rng, n, wild_dist)
        if estimator == "ols":
            y_b = X_exog @ beta_hat + resid_hat * v
            X_b, Xe_b, Z_b = X_exog, None, None
        else:
            # For 2SLS: y* = X_structural @ beta_hat + u_structural * v
            X_structural = np.column_stack([X_exog, X_endog])
            y_b = X_structural @ beta_hat + resid_hat * v
            X_b = X_exog
            Xe_b = X_endog
            Z_b = Z

    elif method == "wild_cluster":
        # Wild cluster bootstrap (Cameron, Gelbach & Miller 2008).
        # Draw one scalar weight per cluster; apply to all obs in that cluster.
        # X, Z are kept fixed (no row resampling).
        G = len(unique_clusters)
        cluster_weights = _wild_weights(rng, G, wild_dist)
        # Map per-observation weight: v_i = cluster_weights[g(i)]
        v = np.empty(n)
        for g_idx, g in enumerate(unique_clusters):
            obs_idx = cluster_to_indices[g]
            v[obs_idx] = cluster_weights[g_idx]

        if estimator == "ols":
            y_b = X_exog @ beta_hat + resid_hat * v
            X_b, Xe_b, Z_b = X_exog, None, None
        else:
            X_structural = np.column_stack([X_exog, X_endog])
            y_b = X_structural @ beta_hat + resid_hat * v
            X_b = X_exog
            Xe_b = X_endog
            Z_b = Z

    elif method in ("cluster_pairs", "panel_cluster_id"):
        G = len(unique_clusters)
        sampled_clusters = unique_clusters[rng.integers(0, G, size=G)]
        # Stack rows for each sampled cluster (with replacement)
        idx_parts = [cluster_to_indices[c] for c in sampled_clusters]
        idx = np.concatenate(idx_parts)

        # For panel_cluster_id with time_col: sort by time within each cluster
        # (preserves within-id ordering in the stacked dataset)
        if time_labels is not None and method == "panel_cluster_id":
            # We can sort within each block, but idx is already contiguous by cluster
            # Sort within each chunk of sampled cluster indices
            sorted_parts = []
            offset = 0
            for c in sampled_clusters:
                chunk = cluster_to_indices[c]
                if time_labels is not None:
                    order = np.argsort(time_labels[chunk])
                    sorted_parts.append(chunk[order])
                else:
                    sorted_parts.append(chunk)
                offset += len(chunk)
            idx = np.concatenate(sorted_parts)

        y_b = y[idx]
        X_b = X_exog[idx]
        Xe_b = X_endog[idx] if X_endog is not None else None
        Z_b = Z[idx] if Z is not None else None

    else:
        raise ValueError(f"Unknown bootstrap method: {method!r}")

    # Fit estimator on bootstrap sample
    return _fit_draw(estimator, y_b, X_b, Xe_b, Z_b)


def _fit_draw(
    estimator: str,
    y: np.ndarray,
    X_exog: np.ndarray,
    X_endog: np.ndarray | None,
    Z: np.ndarray | None,
) -> np.ndarray:
    """Fit the estimator on bootstrap data and return coefficients."""
    if estimator == "ols":
        coefs, _, _ = ols_fit(X_exog, y)
        return coefs
    else:  # 2sls
        coefs, _, _, _ = twosls_fit(y, X_exog, X_endog, Z)
        return coefs


def _wild_weights(rng: np.random.Generator, n: int, dist: str) -> np.ndarray:
    """Sample wild bootstrap auxiliary weights.

    Parameters
    ----------
    dist:
        ``"rademacher"`` — ±1 with probability 0.5 each.
        ``"mammen"``     — two-point distribution matching first three moments
                           of N(0,1).
        ``"webb"``       — Webb (2023) 6-point distribution:
                           {±1, ±sqrt(3/2), ±sqrt(1/2)}, each with prob 1/6.
                           Recommended for wild cluster bootstrap with G < 10;
                           avoids the discrete structure of Rademacher with
                           odd numbers of clusters.
    """
    if dist == "rademacher":
        return rng.choice(np.array([-1.0, 1.0]), size=n)
    elif dist == "mammen":
        # Mammen (1993): two-point distribution
        # v = -(sqrt(5)-1)/2 with prob (sqrt(5)+1)/(2*sqrt(5))
        # v = +(sqrt(5)+1)/2 with prob (sqrt(5)-1)/(2*sqrt(5))
        sqrt5 = np.sqrt(5.0)
        v_lo = -(sqrt5 - 1.0) / 2.0      # ≈ -0.618
        v_hi = (sqrt5 + 1.0) / 2.0       # ≈ +1.618
        p_lo = (sqrt5 + 1.0) / (2.0 * sqrt5)  # ≈ 0.724
        vals = np.where(rng.random(size=n) < p_lo, v_lo, v_hi)
        return vals
    elif dist == "webb":
        # Webb (2023) six-point distribution: uniform on
        # {-sqrt(3/2), -1, -sqrt(1/2), +sqrt(1/2), +1, +sqrt(3/2)}
        webb_vals = np.array([
            -np.sqrt(1.5),
            -1.0,
            -np.sqrt(0.5),
            np.sqrt(0.5),
            1.0,
            np.sqrt(1.5),
        ])
        indices = rng.integers(0, 6, size=n)
        return webb_vals[indices]
    else:
        raise ValueError(
            f"wild_dist must be 'rademacher', 'mammen', or 'webb', got {dist!r}"
        )
