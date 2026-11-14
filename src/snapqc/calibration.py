"""Flag precision and miss rate estimated from random category checks, with shrinkage toward the pooled rate.

A rate here is the share of fact groups that hold an error: among flagged groups it is the flag precision,
among unflagged groups the miss rate. A category check on a group that holds an error finds it with
probability `success` (0.5 in the review model), so from n checks that find x errors the estimate is
x / (success * n).
"""
from __future__ import annotations

import numpy as np

from .review_sim import CAT_SUCCESS


def rate_se(rate, n, success=CAT_SUCCESS):
    """Standard error of the estimate x / (success * n), where x ~ Binomial(n, success * rate)."""
    q = success * np.asarray(rate, float)
    return np.sqrt(q * (1 - q) / np.asarray(n, float)) / success


def n_for_half_width(rate, half_width, success=CAT_SUCCESS, z=1.96):
    """Checks needed for a 95% interval of +/- half_width around the rate."""
    q = success * np.asarray(rate, float)
    return z ** 2 * q * (1 - q) / (success * half_width) ** 2


def estimate(found, n, success=CAT_SUCCESS):
    """Filer-specific, pooled and shrunk rate estimates.

    `found` and `n` are (filers, fact groups) arrays: errors found, and checks made, in each cell.
    - specific: found / (success * n) in the cell (the pooled value where the cell has no check);
    - pooled: the same from the checks of all filers in the fact group;
    - shrunk: pooled + w * (specific - pooled), with w = t2 / (t2 + v). v is the cell's sampling variance and
      t2 the variance of true filer rates around the pooled rate, estimated once across all cells by the
      method of moments (squared gaps minus the sampling variance they would have with no true difference,
      floored at 0).
    All estimates are clipped to [0, 1].
    """
    found, n = np.asarray(found, float), np.asarray(n, float)
    N = n.sum(0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.clip(np.where(N > 0, found.sum(0, keepdims=True) / (success * N), 0.0), 0, 1)
        specific = np.where(n > 0, np.clip(found / (success * n), 0, 1), pooled)
        q = success * pooled
        v = np.where(n > 0, q * (1 - q) / (n * success ** 2), np.inf)
        excess = (specific - pooled) ** 2 - v * (1 - n / np.where(N > 0, N, 1))
    has = n > 0
    t2 = max(0.0, float(excess[has].mean())) if has.any() else 0.0
    w = np.where(np.isfinite(v), t2 / (t2 + np.where(np.isfinite(v), v, 1.0) + 1e-300), 0.0)
    shrunk = pooled + w * (specific - pooled)
    return {"specific": specific, "pooled": np.broadcast_to(pooled, specific.shape).copy(), "shrunk": shrunk,
            "weight": w, "between_filer_variance": t2}


def simulate_checks(rng, n_total, precision, miss_rate, flag_share, success=CAT_SUCCESS, reps=2000):
    """Draw `reps` calibration samples of `n_total` random category checks, spread evenly over the cells.

    `precision`, `miss_rate` and `flag_share` are (filers, fact groups) arrays of true values. Returns, for
    precision and for miss rate, the estimates of `estimate` stacked over draws: dict name -> (reps, F, G).
    """
    precision, miss_rate, flag_share = (np.asarray(x, float) for x in (precision, miss_rate, flag_share))
    per_cell = rng.multinomial(n_total, np.full(precision.size, 1 / precision.size), size=reps).reshape(reps, *precision.shape)
    n_flag = rng.binomial(per_cell, flag_share)
    n_clear = per_cell - n_flag
    found_flag = rng.binomial(n_flag, success * precision)
    found_clear = rng.binomial(n_clear, success * miss_rate)
    out = {"precision": {}, "miss_rate": {}}
    for name, found, n in [("precision", found_flag, n_flag), ("miss_rate", found_clear, n_clear)]:
        est = [estimate(found[r], n[r], success) for r in range(reps)]
        for k in ("specific", "pooled", "shrunk"):
            out[name][k] = np.stack([e[k] for e in est])
    return out
