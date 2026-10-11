import numpy as np


def bh(p):
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m == 0:
        return q
    values = p[finite]
    order = np.argsort(values, kind="stable")
    adjusted = values[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    restored = np.empty(m)
    restored[order] = adjusted
    q[finite] = restored
    return q


def add_fdr(df, family, p_column="p_value", q_column="q_value"):
    out = df.copy()
    out[q_column] = np.nan
    out[f"{q_column}_n_tests"] = 0
    family = [column for column in family if column in out.columns]
    groups = out.groupby(family, dropna=False, sort=False).groups if family else {None: out.index}
    for index in groups.values():
        p = out.loc[index, p_column].to_numpy(dtype=float)
        out.loc[index, q_column] = bh(p)
        out.loc[index, f"{q_column}_n_tests"] = int(np.isfinite(p).sum())
    out[f"{q_column}_family"] = ";".join(family) if family else "all"
    return out


def storey_pi0(p, lam=0.5):
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return 1.0
    return float(min(1.0, np.mean(p > lam) / (1 - lam)))


def composite_null_p(p_a, p_b, lam=0.5):
    p_a, p_b = np.asarray(p_a, dtype=float), np.asarray(p_b, dtype=float)
    pi0_a, pi0_b = storey_pi0(p_a, lam), storey_pi0(p_b, lam)
    pi_null_a = pi0_a * (1 - pi0_b)
    pi_null_b = (1 - pi0_a) * pi0_b
    pi_null_both = pi0_a * pi0_b
    total = pi_null_a + pi_null_b + pi_null_both
    t = np.where(np.isfinite(p_a) & np.isfinite(p_b), np.fmax(p_a, p_b), np.nan)
    if total <= 0:
        return t, pi_null_both
    return np.minimum(1.0, ((pi_null_a + pi_null_b) * t + pi_null_both * t**2) / total), pi_null_both
