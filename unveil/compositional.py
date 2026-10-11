import numpy as np
import pandas as pd

DETECTION_FRACTION = 0.65
ZERO_REPLACEMENT_CAP = 1e-4


def clr(composition, n_total=None):
    X = np.clip(np.asarray(composition, dtype=float), 0, None)
    totals = X.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        X = np.where(totals > 0, X / totals, np.nan)
    if n_total is None:
        delta = np.full(len(X), ZERO_REPLACEMENT_CAP)
    else:
        with np.errstate(divide="ignore"):
            delta = np.minimum(DETECTION_FRACTION / np.asarray(n_total, dtype=float), ZERO_REPLACEMENT_CAP)
        delta = np.where(np.isfinite(delta), delta, ZERO_REPLACEMENT_CAP)
    zero = X == 0
    scale = 1 - delta * zero.sum(axis=1)
    X = np.where(zero, delta[:, None], X * scale[:, None])
    log_x = np.log(X)
    return log_x - log_x.mean(axis=1, keepdims=True)


def attention_weighted_composition(tiles, keys, count_columns, attention_column):
    rows = []
    for key, group in tiles.groupby(list(keys), sort=True):
        counts = group[count_columns].to_numpy(dtype=float)
        attention = group[attention_column].to_numpy(dtype=float)
        weight = attention / attention.sum() if attention.sum() > 0 else np.full(len(group), 1 / len(group))
        row = {**dict(zip(keys, key)), "n_tiles": len(group), "n_nuclei": counts.sum()}
        row.update({f"weighted_{c}": value for c, value in zip(count_columns, weight @ counts)})
        row.update({f"unweighted_{c}": value for c, value in zip(count_columns, counts.sum(axis=0))})
        rows.append(row)
    return pd.DataFrame(rows)
