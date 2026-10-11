from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t" if path.suffix in (".tsv", ".txt") else ",")


def write_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, sep="\t" if path.suffix in (".tsv", ".txt") else ",", index=False)


def require_unique(df, keys, what):
    duplicated = df.duplicated(list(keys), keep=False)
    if duplicated.any():
        example = df.loc[duplicated, list(keys)].head(3).to_dict("records")
        raise ValueError(f"{what}: {int(duplicated.sum())} rows share {list(keys)}, e.g. {example}")


def attach_clinical(table, clinical, spec):
    keys = [k for k in (spec.cohort, spec.cancer, spec.patient_id) if k in table.columns and k in clinical.columns]
    if spec.patient_id not in keys:
        raise KeyError(f"both tables need a {spec.patient_id} column")
    require_unique(clinical, keys, "clinical table")
    overlap = (set(table.columns) & set(clinical.columns)) - set(keys)
    if overlap:
        raise ValueError(f"columns present in both tables: {sorted(overlap)}")
    return table.merge(clinical, on=keys, how="inner", validate="many_to_one")


def _mode(series):
    counts = series.dropna().astype(str).value_counts()
    if counts.empty:
        return np.nan
    return sorted(counts.index, key=lambda level: (-counts[level], level))[0]


def to_patient_level(df, keys, sum_columns=()):
    keys = list(keys)
    grouped = df.groupby(keys, sort=True, dropna=False)
    numeric = [c for c in df.columns if c not in keys and df[c].dtype.kind in "biuf"]
    summed = [c for c in numeric if c in set(sum_columns)]
    averaged = [c for c in numeric if c not in set(sum_columns)]
    other = [c for c in df.columns if c not in keys and c not in numeric]
    parts = []
    if averaged:
        parts.append(grouped[averaged].mean())
    if summed:
        parts.append(grouped[summed].sum(min_count=1))
    if other:
        parts.append(grouped[other].agg(_mode))
    out = pd.concat(parts, axis=1).reset_index()
    out["n_slides"] = grouped.size().to_numpy()
    return out[keys + [c for c in df.columns if c not in keys] + ["n_slides"]]


def exposure_contrasts(df, exposure, min_group_size):
    values = df[exposure.name]
    if exposure.kind == "continuous":
        a = pd.to_numeric(values, errors="coerce")
        keep = a.notna().to_numpy()
        if keep.sum() >= 2 * min_group_size and a[keep].nunique() > 1:
            yield "continuous", df.loc[keep], a[keep].to_numpy(dtype=float)
        return
    labels = values.astype("string")
    for level in exposure.levels:
        keep = labels.isin([exposure.reference, level]).fillna(False).to_numpy()
        a = (labels[keep] == level).to_numpy(dtype=float)
        if min(a.sum(), (1 - a).sum()) >= min_group_size:
            yield f"{level}_vs_{exposure.reference}", df.loc[keep], a
