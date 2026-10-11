import numpy as np
from scipy import stats

SPEARMAN_VARIANCE_FACTOR = 1.06


def fisher_z(r):
    return np.arctanh(np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999))


def normal_scores(x):
    x = np.asarray(x, dtype=float)
    return stats.norm.ppf(stats.rankdata(x) / (len(x) + 1))


def cliffs_delta(y1, y0):
    test = stats.mannwhitneyu(y1, y0, method="asymptotic")
    return float(2 * test.statistic / (len(y1) * len(y0)) - 1), float(test.pvalue)


def cliffs_delta_se(y1, y0):
    n1, n0 = len(y1), len(y0)
    if n1 < 2 or n0 < 2:
        return np.nan
    s1, s0 = np.sort(y1), np.sort(y0)
    below_0 = np.searchsorted(s1, y0, side="left")
    at_or_below_0 = np.searchsorted(s1, y0, side="right")
    d_row = (n1 - at_or_below_0 - below_0) / n1
    below_1 = np.searchsorted(s0, y1, side="left")
    at_or_below_1 = np.searchsorted(s0, y1, side="right")
    d_col = (below_1 - (n0 - at_or_below_1)) / n0
    delta = d_row.mean()
    n_tied = np.sum(at_or_below_0 - below_0)
    ss_pairs = (n0 * n1 - n_tied) - n0 * n1 * delta**2
    ss_row = np.sum((d_row - delta) ** 2) * n1**2
    ss_col = np.sum((d_col - delta) ** 2) * n0**2
    variance = (ss_col + ss_row - ss_pairs) / (n0 * n1 * (n0 - 1) * (n1 - 1))
    return float(np.sqrt(variance)) if variance > 0 else np.nan


def rank_association(y, a):
    y, a = np.asarray(y, dtype=float), np.asarray(a, dtype=float)
    if set(np.unique(a)) <= {0.0, 1.0}:
        y1, y0 = y[a == 1], y[a == 0]
        delta, p = cliffs_delta(y1, y0)
        se = cliffs_delta_se(y1, y0)
        return {"effect_type": "cliffs_delta", "effect": delta, "se_z": se / (1 - min(delta**2, 0.999999)), "p_value": p}
    test = stats.spearmanr(a, y)
    n = len(y)
    return {
        "effect_type": "spearman_rho",
        "effect": float(test.statistic),
        "se_z": float(np.sqrt(SPEARMAN_VARIANCE_FACTOR / (n - 3))) if n > 3 else np.nan,
        "p_value": float(test.pvalue),
    }


def partial_correlation(Y, X, column=1):
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    df = n - p
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    residual = Y - X @ beta
    sigma2 = (residual**2).sum(axis=0) / df
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * xtx_inv[column, column])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta[column] / se, np.nan)
    return t / np.sqrt(t**2 + df), 2 * stats.t.sf(np.abs(t), df), df
