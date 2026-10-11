from dataclasses import dataclass

import numpy as np
from scipy import stats

from .design import MISSING_LEVEL, covariate_frame, independent_columns

EFFECTS = ("IIE", "IDE", "Total", "IIE_diff", "CDE", "INTref", "INTmed", "PIE")
CDE_MEDIATOR_REFERENCE = 0.0
LAMBDA_MAX = 1e8


@dataclass
class Design:
    m: np.ndarray
    Y: np.ndarray
    X_mediator: np.ndarray
    X_outcome: np.ndarray
    penalty_mediator: np.ndarray
    penalty_outcome: list
    a_lo: float
    a_hi: float
    label: np.ndarray
    lambda_mediator: float
    lambda_outcome: list
    n_random_levels: int
    adjusted_for: list
    mediator_a: int = 1
    outcome_a: int = 1
    outcome_m: int = 2
    outcome_am: int = 3
    outcome_lm: int = None

    @property
    def delta(self):
        return self.a_hi - self.a_lo

    @property
    def mediator_nuisance(self):
        return [j for j in range(self.X_mediator.shape[1]) if j != self.mediator_a]


def random_intercept_lambda(y, X_fixed, groups):
    beta, *_ = np.linalg.lstsq(X_fixed, y, rcond=None)
    residual = y - X_fixed @ beta
    codes = np.unique(groups, return_inverse=True)[1]
    n, J = len(residual), codes.max() + 1
    if J < 2 or n - J < 1:
        return LAMBDA_MAX
    group_mean = (np.bincount(codes, weights=residual) / np.bincount(codes))[codes]
    ms_within = np.sum((residual - group_mean) ** 2) / (n - J)
    ms_between = np.sum((group_mean - residual.mean()) ** 2) / (J - 1)
    tau2 = (ms_between - ms_within) / (n / J)
    if not np.isfinite(tau2) or tau2 <= 0 or ms_within <= 0:
        return LAMBDA_MAX
    return min(ms_within / tau2, LAMBDA_MAX)


def penalty_rows(n_columns, columns, lam):
    rows = np.zeros((len(columns), n_columns))
    rows[np.arange(len(columns)), columns] = np.sqrt(lam)
    return rows


def build_design(frame, a, m, Y, label, covariates, spec, a_lo, a_hi):
    random_columns = [c for c in spec.random_intercept if c in covariates and frame[c].nunique(dropna=False) > 1]
    C = covariate_frame(frame, [c for c in covariates if c not in random_columns], spec)
    has_label = label is not None and np.unique(label).size > 1
    ones = np.ones(len(m))
    mediator_names = ["const", "exposure"] + list(C.columns)
    X_mediator = np.column_stack([ones, a, C.to_numpy()])
    outcome_names = ["const", "exposure", "mediator", "exposure:mediator"] + (["label", "label:mediator"] if has_label else []) + list(C.columns)
    X_outcome = np.column_stack([ones, a, m, a * m] + ([label, label * m] if has_label else []) + [C.to_numpy()])
    keep_mediator = independent_columns(X_mediator, protect=(0, 1))
    keep_outcome = independent_columns(X_outcome, protect=(0, 1, 2, 3))
    if keep_mediator is None or keep_outcome is None:
        return None
    X_mediator, X_outcome = X_mediator[:, keep_mediator], X_outcome[:, keep_outcome]
    outcome_names = [outcome_names[j] for j in keep_outcome]

    n_levels, lambda_mediator, lambda_outcome = 0, np.nan, [np.nan] * Y.shape[1]
    penalty_mediator = np.zeros((0, X_mediator.shape[1]))
    penalty_outcome = [np.zeros((0, X_outcome.shape[1]))] * Y.shape[1]
    if random_columns:
        groups = frame[random_columns].astype("object").where(frame[random_columns].notna(), MISSING_LEVEL).astype(str).agg("|".join, axis=1).to_numpy()
        levels, codes = np.unique(groups, return_inverse=True)
        if len(levels) > 1:
            n_levels = len(levels)
            Z = np.eye(n_levels)[codes]
            lambda_mediator = random_intercept_lambda(m, X_mediator, groups)
            lambda_outcome = [random_intercept_lambda(Y[:, k], X_outcome, groups) for k in range(Y.shape[1])]
            p_mediator, p_outcome = X_mediator.shape[1], X_outcome.shape[1]
            X_mediator, X_outcome = np.column_stack([X_mediator, Z]), np.column_stack([X_outcome, Z])
            penalty_mediator = penalty_rows(X_mediator.shape[1], np.arange(p_mediator, X_mediator.shape[1]), lambda_mediator)
            penalty_outcome = [penalty_rows(X_outcome.shape[1], np.arange(p_outcome, X_outcome.shape[1]), lam) for lam in lambda_outcome]

    return Design(
        m=m, Y=Y, X_mediator=X_mediator, X_outcome=X_outcome,
        penalty_mediator=penalty_mediator, penalty_outcome=penalty_outcome,
        a_lo=a_lo, a_hi=a_hi, label=label if has_label else None,
        lambda_mediator=lambda_mediator, lambda_outcome=lambda_outcome, n_random_levels=n_levels,
        adjusted_for=[mediator_names[j] for j in keep_mediator][2:] + random_columns,
        outcome_lm=outcome_names.index("label:mediator") if "label:mediator" in outcome_names else None,
    )


def penalized_fit(X, y, penalty):
    X_aug = np.vstack([X, penalty])
    y_aug = np.concatenate([y, np.zeros(len(penalty))])
    beta, _, rank, _ = np.linalg.lstsq(X_aug, y_aug, rcond=None)
    return beta if rank == X.shape[1] else None


def fit_once(design, rows):
    d = design
    out = np.full((d.Y.shape[1], len(EFFECTS)), np.nan)
    X_mediator = d.X_mediator[rows]
    coef_m = penalized_fit(X_mediator, d.m[rows], d.penalty_mediator)
    if coef_m is None:
        return out
    a_effect = coef_m[d.mediator_a] * d.delta
    nuisance = d.mediator_nuisance
    mediator_lo = np.mean(X_mediator[:, nuisance] @ coef_m[nuisance]) + coef_m[d.mediator_a] * d.a_lo
    X_outcome = d.X_outcome[rows]
    prevalence = d.label[rows].mean() if d.outcome_lm is not None else 0.0
    for k in range(d.Y.shape[1]):
        coef = penalized_fit(X_outcome, d.Y[rows, k], d.penalty_outcome[k])
        if coef is None:
            continue
        label_shift = prevalence * coef[d.outcome_lm] if d.outcome_lm is not None else 0.0
        slope_hi = coef[d.outcome_m] + coef[d.outcome_am] * d.a_hi + label_shift
        slope_lo = coef[d.outcome_m] + coef[d.outcome_am] * d.a_lo + label_shift
        iie = slope_hi * a_effect
        iie_diff = coef[d.outcome_lm] * a_effect if d.outcome_lm is not None else np.nan
        direct = coef[d.outcome_a] * d.delta
        interaction = coef[d.outcome_am] * d.delta
        ide = direct + interaction * mediator_lo
        cde = direct + interaction * CDE_MEDIATOR_REFERENCE
        int_med = (slope_hi - slope_lo) * a_effect
        out[k] = [iie, ide, iie + ide, iie_diff, cde, ide - cde, int_med, iie - int_med]
    return out


def _penalized_inference(X, y, penalty):
    X_aug = np.vstack([X, penalty])
    beta, _, rank, _ = np.linalg.lstsq(X_aug, np.concatenate([y, np.zeros(len(penalty))]), rcond=None)
    if rank < X.shape[1]:
        return None
    xtx_inverse = np.linalg.inv(X_aug.T @ X_aug)
    df = len(y) - np.sum(xtx_inverse * (X.T @ X))
    if df <= 0:
        return None
    sigma2 = np.sum((y - X @ beta) ** 2) / df
    return beta, sigma2 * xtx_inverse, df


def _linear_combination(beta, covariance, df, weights):
    estimate = float(weights @ beta)
    variance = float(weights @ covariance @ weights)
    se = float(np.sqrt(variance)) if variance > 0 else np.nan
    return estimate, se, float(2 * stats.t.sf(abs(estimate / se), df)) if se > 0 else np.nan


def path_coefficients(design, k):
    d = design
    out = {}
    fit = _penalized_inference(d.X_mediator, d.m, d.penalty_mediator)
    if fit is not None:
        weights = np.zeros(d.X_mediator.shape[1])
        weights[d.mediator_a] = d.delta
        out["a_path"], out["a_path_se"], out["a_path_p"] = _linear_combination(*fit, weights)
    fit = _penalized_inference(d.X_outcome, d.Y[:, k], d.penalty_outcome[k])
    if fit is not None:
        base = np.zeros(d.X_outcome.shape[1])
        base[d.outcome_m], base[d.outcome_am] = 1.0, d.a_hi
        if d.outcome_lm is None:
            out["b_path"], out["b_path_se"], out["b_path_p"] = _linear_combination(*fit, base)
        else:
            average, interaction, positive = base.copy(), np.zeros_like(base), base.copy()
            average[d.outcome_lm] = d.label.mean()
            interaction[d.outcome_lm] = 1.0
            positive[d.outcome_lm] = 1.0
            out["b_path"], out["b_path_se"], out["b_path_p"] = _linear_combination(*fit, average)
            out["b_label_interaction"], out["b_label_interaction_se"], out["b_label_interaction_p"] = _linear_combination(*fit, interaction)
            out["b_negative"], out["b_negative_se"], _ = _linear_combination(*fit, base)
            out["b_positive"], out["b_positive_se"], _ = _linear_combination(*fit, positive)
    return out


def decompose(design, n_boot, rng):
    n = len(design.m)
    point = fit_once(design, np.arange(n))
    boot = np.stack([fit_once(design, rng.integers(0, n, n)) for _ in range(n_boot)])
    rows = []
    for k in range(design.Y.shape[1]):
        row = {"n_boot_valid": int(np.isfinite(boot[:, k, 0]).sum())}
        for j, effect in enumerate(EFFECTS):
            draws = boot[:, k, j][np.isfinite(boot[:, k, j])]
            estimate = point[k, j]
            se = draws.std(ddof=1) if len(draws) > 1 else np.nan
            row[effect] = estimate
            row[f"{effect}_se"] = se
            row[f"{effect}_ci_low"], row[f"{effect}_ci_high"] = np.quantile(draws, [0.025, 0.975]) if len(draws) > 1 else (np.nan, np.nan)
            row[f"{effect}_p_value"] = float(2 * stats.norm.sf(abs(estimate / se))) if np.isfinite(estimate) and se > 0 else np.nan
        row.update(path_coefficients(design, k))
        rows.append(row)
    return rows
