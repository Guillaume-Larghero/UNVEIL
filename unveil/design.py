import numpy as np
import pandas as pd

MISSING_LEVEL = "missing"
OTHER_LEVEL = "other"


def lump_rare(series, max_levels, min_count):
    labels = series.astype("object").where(series.notna(), MISSING_LEVEL).astype(str)
    counts = labels.value_counts()
    ranked = sorted(counts.index, key=lambda level: (-counts[level], level))
    keep = [level for level in ranked[:max_levels] if counts[level] >= min_count]
    return labels.where(labels.isin(keep), OTHER_LEVEL)


def covariate_frame(df, columns, spec):
    parts = []
    for column in columns:
        series = df[column]
        if spec.is_categorical(column, series):
            lumped = lump_rare(series, spec.max_levels, spec.min_level_count)
            counts = lumped.value_counts()
            levels = sorted(counts.index, key=lambda level: (-counts[level], level))
            for level in levels[1:]:
                parts.append((lumped == level).astype(float).rename(f"{column}={level}"))
        else:
            values = pd.to_numeric(series, errors="coerce").astype(float)
            sd = values.std()
            parts.append(((values - values.mean()) / (sd if sd > 0 else 1.0)).rename(column))
    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def independent_columns(X, protect=(), tol=1e-8):
    X = np.asarray(X, dtype=float)
    basis = np.zeros((X.shape[0], 0))
    keep = []
    for j in range(X.shape[1]):
        v = X[:, j]
        r = v - basis @ (basis.T @ v)
        r = r - basis @ (basis.T @ r)
        norm = np.linalg.norm(r)
        if norm > tol * max(np.linalg.norm(v), 1.0):
            basis = np.column_stack([basis, r / norm])
            keep.append(j)
        elif j in protect:
            return None
    return keep


def adjustment_design(df, a, covariates, spec):
    C = covariate_frame(df, covariates, spec)
    complete = np.isfinite(a) & C.notna().all(axis=1).to_numpy()
    C = C.loc[complete]
    X = pd.concat(
        [pd.Series(1.0, index=C.index, name="const"), pd.Series(a[complete], index=C.index, name="exposure"), C],
        axis=1,
    )
    X = X.loc[:, (X.std(ddof=0) > 0) | (X.columns == "const")]
    if "exposure" not in X.columns:
        return None, complete
    keep = independent_columns(X.to_numpy(), protect=(0, 1))
    if keep is None:
        return None, complete
    return X.iloc[:, keep], complete


def residualize(Y, X):
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Y - X @ beta
