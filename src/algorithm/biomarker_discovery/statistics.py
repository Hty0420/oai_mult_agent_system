"""Dependency-light statistics for ordinal imaging-biomarker discovery.

The proportional-odds likelihood follows McCullagh (1980).  The implementation
is intentionally small and deterministic so the MCP service does not depend on
an LLM or an optional statistics package at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, norm


@dataclass(frozen=True)
class OrderedLogitFit:
    converged: bool
    log_likelihood: float
    coefficients: np.ndarray
    standard_errors: np.ndarray
    thresholds: np.ndarray
    message: str
    iterations: int


def _ordered_thresholds(raw: np.ndarray) -> np.ndarray:
    if raw.size == 0:
        return raw
    thresholds = np.empty_like(raw, dtype=np.float64)
    thresholds[0] = raw[0]
    if raw.size > 1:
        increments = np.exp(np.clip(raw[1:], -20.0, 20.0))
        thresholds[1:] = raw[0] + np.cumsum(increments)
    return thresholds


def _initial_threshold_parameters(y: np.ndarray, category_count: int) -> np.ndarray:
    cumulative = np.asarray(
        [np.mean(y <= index) for index in range(category_count - 1)],
        dtype=np.float64,
    )
    cumulative = np.clip(cumulative, 1e-4, 1.0 - 1e-4)
    thresholds = np.log(cumulative / (1.0 - cumulative))
    # Empty/interleaved categories can make adjacent empirical thresholds
    # equal.  Enforce a small, deterministic separation before transforming.
    for index in range(1, thresholds.size):
        thresholds[index] = max(thresholds[index], thresholds[index - 1] + 0.1)
    raw = np.empty_like(thresholds)
    raw[0] = thresholds[0]
    if thresholds.size > 1:
        raw[1:] = np.log(np.maximum(np.diff(thresholds), 1e-4))
    return raw


def _finite_difference_hessian(function, point: np.ndarray) -> np.ndarray:
    """Central finite-difference Hessian for the small ordinal model."""
    point = np.asarray(point, dtype=np.float64)
    count = int(point.size)
    hessian = np.empty((count, count), dtype=np.float64)
    steps = np.power(np.finfo(np.float64).eps, 0.25) * (1.0 + np.abs(point))
    center = float(function(point))
    for row in range(count):
        row_step = np.zeros(count, dtype=np.float64)
        row_step[row] = steps[row]
        hessian[row, row] = (
            float(function(point + row_step))
            - 2.0 * center
            + float(function(point - row_step))
        ) / (steps[row] ** 2)
        for column in range(row + 1, count):
            column_step = np.zeros(count, dtype=np.float64)
            column_step[column] = steps[column]
            value = (
                float(function(point + row_step + column_step))
                - float(function(point + row_step - column_step))
                - float(function(point - row_step + column_step))
                + float(function(point - row_step - column_step))
            ) / (4.0 * steps[row] * steps[column])
            hessian[row, column] = value
            hessian[column, row] = value
    return hessian


def fit_ordered_logit(
    y: np.ndarray,
    x: np.ndarray,
    *,
    max_iterations: int = 600,
) -> OrderedLogitFit:
    """Fit a proportional-odds cumulative-logit model with BFGS.

    The model is ``logit(P(Y <= j)) = threshold_j - X beta``.  A positive
    coefficient therefore indicates higher odds of a higher KL grade.
    """
    y = np.asarray(y, dtype=np.int64)
    x = np.asarray(x, dtype=np.float64)
    if y.ndim != 1 or x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("ordered-logit inputs have incompatible shapes")
    if y.size < 20:
        raise ValueError("ordered-logit analysis requires at least 20 observations")
    if not np.all(np.isfinite(x)):
        raise ValueError("ordered-logit covariates contain non-finite values")

    levels = np.unique(y)
    if levels.size < 3:
        raise ValueError("ordinal KL analysis requires at least three observed grades")
    level_map = {int(value): index for index, value in enumerate(levels.tolist())}
    encoded = np.asarray([level_map[int(value)] for value in y], dtype=np.int64)
    category_count = int(levels.size)
    coefficient_count = int(x.shape[1])
    initial_thresholds = _initial_threshold_parameters(encoded, category_count)
    initial = np.concatenate(
        [np.zeros(coefficient_count, dtype=np.float64), initial_thresholds]
    )

    def negative_log_likelihood(params: np.ndarray) -> float:
        beta = params[:coefficient_count]
        thresholds = _ordered_thresholds(params[coefficient_count:])
        eta = x @ beta if coefficient_count else np.zeros(encoded.size)
        upper = np.ones(encoded.size, dtype=np.float64)
        lower = np.zeros(encoded.size, dtype=np.float64)
        not_last = encoded < category_count - 1
        not_first = encoded > 0
        upper[not_last] = expit(thresholds[encoded[not_last]] - eta[not_last])
        lower[not_first] = expit(thresholds[encoded[not_first] - 1] - eta[not_first])
        probability = np.clip(upper - lower, 1e-12, 1.0)
        value = -float(np.sum(np.log(probability)))
        if not math.isfinite(value):
            return 1e100
        return value

    result = minimize(
        negative_log_likelihood,
        initial,
        method="L-BFGS-B",
        options={
            "maxiter": int(max_iterations),
            "ftol": 1e-10,
            "gtol": 1e-6,
            "maxls": 50,
        },
    )
    coefficients = np.asarray(result.x[:coefficient_count], dtype=np.float64)
    standard_errors = np.full(coefficient_count, np.nan, dtype=np.float64)
    try:
        observed_information = _finite_difference_hessian(
            negative_log_likelihood, np.asarray(result.x, dtype=np.float64)
        )
        covariance = np.linalg.pinv(observed_information, hermitian=True)
        diagonal = np.diag(covariance)[:coefficient_count]
        standard_errors = np.sqrt(np.where(diagonal > 0, diagonal, np.nan))
    except (TypeError, ValueError, np.linalg.LinAlgError):
        pass

    return OrderedLogitFit(
        converged=bool(result.success and math.isfinite(float(result.fun))),
        log_likelihood=-float(result.fun),
        coefficients=coefficients,
        standard_errors=standard_errors,
        thresholds=_ordered_thresholds(result.x[coefficient_count:]),
        message=str(result.message),
        iterations=int(getattr(result, "nit", 0) or 0),
    )


def feature_association(
    y: np.ndarray,
    covariates: np.ndarray,
    feature: np.ndarray,
    *,
    reduced_fit: OrderedLogitFit | None = None,
) -> dict[str, float | bool | str | int]:
    """Fit one adjusted ordinal association and return effect evidence."""
    covariates = np.asarray(covariates, dtype=np.float64)
    feature = np.asarray(feature, dtype=np.float64).reshape(-1, 1)
    full_x = np.column_stack([covariates, feature])
    reduced = reduced_fit or fit_ordered_logit(y, covariates)
    full = fit_ordered_logit(y, full_x)
    if not full.converged:
        return {
            "converged": False,
            "error": full.message,
            "iterations": full.iterations,
        }

    beta = float(full.coefficients[-1])
    standard_error = float(full.standard_errors[-1])
    likelihood_ratio = max(0.0, 2.0 * (full.log_likelihood - reduced.log_likelihood))
    p_value = float(chi2.sf(likelihood_ratio, 1))
    if not math.isfinite(p_value):
        if math.isfinite(standard_error) and standard_error > 0:
            p_value = float(2.0 * norm.sf(abs(beta / standard_error)))
        else:
            p_value = 1.0
    ci_low = beta - 1.959963984540054 * standard_error
    ci_high = beta + 1.959963984540054 * standard_error
    return {
        "converged": True,
        "beta_per_sd": beta,
        "standard_error": standard_error,
        "odds_ratio_per_sd": float(math.exp(np.clip(beta, -50.0, 50.0))),
        "odds_ratio_ci_low": float(math.exp(np.clip(ci_low, -50.0, 50.0))),
        "odds_ratio_ci_high": float(math.exp(np.clip(ci_high, -50.0, 50.0))),
        "likelihood_ratio": likelihood_ratio,
        "p_value": min(max(p_value, 0.0), 1.0),
        "log_likelihood": float(full.log_likelihood),
        "iterations": full.iterations,
    }


def adjust_bh(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values in original order."""
    values = np.clip(np.asarray(list(p_values), dtype=np.float64), 0.0, 1.0)
    count = int(values.size)
    if count == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranked = values[order] * count / np.arange(1, count + 1, dtype=np.float64)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def adjust_holm(p_values: Iterable[float]) -> np.ndarray:
    """Holm family-wise adjusted p-values in original order."""
    values = np.clip(np.asarray(list(p_values), dtype=np.float64), 0.0, 1.0)
    count = int(values.size)
    if count == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranked = values[order] * (count - np.arange(count, dtype=np.float64))
    ranked = np.maximum.accumulate(ranked)
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted
