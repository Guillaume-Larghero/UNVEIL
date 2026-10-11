import numpy as np
import pandas as pd
from scipy import stats

Z_975 = stats.norm.ppf(0.975)


def random_effects(estimate, se):
    estimate, se = np.asarray(estimate, dtype=float), np.asarray(se, dtype=float)
    k = len(estimate)
    w = 1 / se**2
    fixed = np.sum(w * estimate) / np.sum(w)
    q = float(np.sum(w * (estimate - fixed) ** 2))
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    w_re = 1 / (se**2 + tau2)
    pooled = float(np.sum(w_re * estimate) / np.sum(w_re))
    se_re = float(np.sqrt(1 / np.sum(w_re)))
    out = {
        "k": k,
        "estimate": pooled,
        "se": se_re,
        "ci_low": pooled - Z_975 * se_re,
        "ci_high": pooled + Z_975 * se_re,
        "p_value": float(2 * stats.norm.sf(abs(pooled / se_re))),
        "tau2": tau2,
        "cochran_q": q,
        "i2": max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0,
        "same_sign": bool(len(set(np.sign(estimate[estimate != 0]))) <= 1),
    }
    if k > 1:
        se_hk = max(float(np.sqrt(np.sum(w_re * (estimate - pooled) ** 2) / ((k - 1) * np.sum(w_re)))), se_re)
        t_crit = stats.t.ppf(0.975, k - 1)
        out.update(
            se_hk=se_hk,
            p_value_hk=float(2 * stats.t.sf(abs(pooled / se_hk), k - 1)),
            ci_low_hk=pooled - t_crit * se_hk,
            ci_high_hk=pooled + t_crit * se_hk,
        )
    if k > 2:
        half = stats.t.ppf(0.975, k - 2) * np.sqrt(tau2 + se_re**2)
        out.update(pi_low=pooled - half, pi_high=pooled + half)
    return out


def leave_one_out(estimate, se):
    estimate, se = np.asarray(estimate, dtype=float), np.asarray(se, dtype=float)
    worst_p, signs = 0.0, set()
    for i in range(len(estimate)):
        keep = np.arange(len(estimate)) != i
        if keep.sum() > 1:
            fit = random_effects(estimate[keep], se[keep])
            pooled, p = fit["estimate"], fit["p_value"]
        else:
            pooled = float(estimate[keep][0])
            p = float(2 * stats.norm.sf(abs(pooled / se[keep][0])))
        worst_p = max(worst_p, p)
        signs.add(np.sign(pooled))
    return {"loo_worst_p": worst_p, "loo_same_sign": len(signs) == 1}


BACK_TRANSFORMED = {"estimate", "ci_low", "ci_high", "ci_low_hk", "ci_high_hk", "pi_low", "pi_high"}


def pool_table(df, keys, estimate, se, unit, fisher=False, alpha=0.05):
    keys = list(keys)
    usable = df[np.isfinite(df[estimate]) & np.isfinite(df[se]) & (df[se] > 0)]
    rows = []
    for key, group in usable.groupby(keys, sort=True, dropna=False):
        if group[unit].nunique() != len(group):
            raise ValueError(f"{dict(zip(keys, key))}: more than one row per {unit}")
        if len(group) < 2:
            continue
        values = group[estimate].to_numpy(dtype=float)
        if fisher:
            values = np.arctanh(np.clip(values, -0.999999, 0.999999))
        errors = group[se].to_numpy(dtype=float)
        fit = random_effects(values, errors)
        robustness = leave_one_out(values, errors)
        row = dict(zip(keys, key))
        row[f"{unit}s"] = "+".join(sorted(group[unit].astype(str)))
        for name, value in fit.items():
            row[name] = float(np.tanh(value)) if fisher and name in BACK_TRANSFORMED else value
        row.update(robustness)
        row["loo_robust"] = bool(robustness["loo_same_sign"] and robustness["loo_worst_p"] < alpha)
        rows.append(row)
    return pd.DataFrame(rows)
