import warnings

import numpy as np
import statsmodels.api as sm

MAX_STANDARD_ERROR = 1e3


def _cluster_robust_ols(y, X, groups, column, status):
    fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return {"beta": float(fit.params[column]), "se": float(fit.bse[column]), "p_value": float(fit.pvalues[column]), "method": "ols_cluster_robust", "status": status}


def random_intercept_effect(y, X, groups, column=1):
    y, X = np.asarray(y, dtype=float), np.asarray(X, dtype=float)
    groups = np.unique(np.asarray(groups).astype(str), return_inverse=True)[1]
    order = np.lexsort((*X.T[::-1], y, groups))
    y, X, groups = y[order], X[order], groups[order]
    if np.bincount(groups).max() <= 1:
        return _cluster_robust_ols(y, X, groups, column, "no_within_patient_replication")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.MixedLM(y, X, groups=groups).fit(reml=False, maxiter=200)
    except (np.linalg.LinAlgError, ValueError):
        return _cluster_robust_ols(y, X, groups, column, "mixed_model_failed")
    beta, se, p = float(fit.fe_params[column]), float(fit.bse[column]), float(fit.pvalues[column])
    if not fit.converged:
        return _cluster_robust_ols(y, X, groups, column, "mixed_model_not_converged")
    if not (np.isfinite(beta) and np.isfinite(se) and np.isfinite(p)) or se > MAX_STANDARD_ERROR:
        return _cluster_robust_ols(y, X, groups, column, "mixed_model_unidentified")
    return {"beta": beta, "se": se, "p_value": p, "method": "mixed_random_intercept", "status": "converged"}
