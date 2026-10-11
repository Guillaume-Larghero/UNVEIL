import argparse

import numpy as np
import pandas as pd

from unveil.cli import add_analysis_arguments, resolve
from unveil.data import attach_clinical, exposure_contrasts, read_table, to_patient_level, write_table
from unveil.design import adjustment_design
from unveil.effects import normal_scores, partial_correlation, rank_association
from unveil.multiplicity import add_fdr

FDR_FAMILY = ("cohort", "exposure", "contrast", "adjustment")


def feature_columns(table, spec):
    ids = {spec.patient_id, spec.slide_id, spec.cohort, spec.cancer}
    return [c for c in table.columns if c not in ids and table[c].dtype.kind in "biuf"]


def enough_patients(a, min_group_size, binary):
    if binary:
        return min(a.sum(), len(a) - a.sum()) >= min_group_size
    return len(a) >= 2 * min_group_size


def unadjusted_associations(frame, a, features, spec, binary):
    rows = []
    for feature in features:
        y = frame[feature].to_numpy(dtype=float)
        finite = np.isfinite(y)
        if not enough_patients(a[finite], spec.min_group_size, binary) or np.ptp(y[finite]) == 0:
            continue
        rows.append({"feature": feature, "n": int(finite.sum()), **rank_association(y[finite], a[finite])})
    return rows


def adjusted_associations(frame, a, features, covariates, spec, binary):
    Y = frame[features].to_numpy(dtype=float)
    finite = np.isfinite(Y)
    patterns = {}
    for j in range(len(features)):
        patterns.setdefault(finite[:, j].tobytes(), []).append(j)
    rows = []
    for columns in patterns.values():
        observed = np.flatnonzero(finite[:, columns[0]])
        X, complete = adjustment_design(frame.iloc[observed], a[observed], covariates, spec)
        if X is None:
            continue
        used = observed[complete]
        df = X.shape[0] - X.shape[1]
        if df < spec.min_residual_df or not enough_patients(a[used], spec.min_group_size, binary):
            continue
        X = X.assign(exposure=normal_scores(X["exposure"]))
        Y_scores = np.apply_along_axis(normal_scores, 0, Y[np.ix_(used, columns)])
        r, p, df = partial_correlation(Y_scores, X.to_numpy())
        adjusted_for = ";".join(X.columns[2:])
        for j, effect, p_value in zip(columns, r, p):
            rows.append({
                "feature": features[j],
                "n": len(used),
                "effect_type": "partial_correlation",
                "effect": effect,
                "se_z": 1 / np.sqrt(df - 1),
                "p_value": p_value,
                "residual_df": df,
                "adjusted_for": adjusted_for,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Patient-level association of each exposure with every nuclear morphology feature.")
    parser.add_argument("--features", required=True, help="feature table keyed by cohort, cancer and patient (slide rows are averaged per patient)")
    parser.add_argument("--clinical", required=True, help="patient table with exposures and covariates")
    parser.add_argument("--output", required=True)
    parser.add_argument("--unadjusted", action="store_true", help="also report the unadjusted rank-based arm")
    add_analysis_arguments(parser, FDR_FAMILY)
    args = parser.parse_args()
    spec, exposures, blocks = resolve(args)

    features = read_table(args.features)
    keys = [spec.cohort, spec.cancer, spec.patient_id]
    names = feature_columns(features, spec)
    if features.duplicated(keys).any():
        features = to_patient_level(features[keys + names], keys).drop(columns="n_slides")
    data = attach_clinical(features, read_table(args.clinical), spec)

    results = []
    for (cohort, cancer), cell in data.groupby([spec.cohort, spec.cancer], sort=True):
        for exposure in exposures:
            covariates = spec.covariates_for(exposure.name, blocks)
            binary = exposure.kind == "categorical"
            for contrast, frame, a in exposure_contrasts(cell, exposure, spec.min_group_size):
                arms = []
                if blocks:
                    arms.append(("adjusted", adjusted_associations(frame, a, names, covariates, spec, binary)))
                if args.unadjusted or not blocks:
                    arms.append(("unadjusted", unadjusted_associations(frame, a, names, spec, binary)))
                label = {"cohort": cohort, "cancer": cancer, "exposure": exposure.name, "contrast": contrast}
                results.extend({**label, "adjustment": arm, **row} for arm, rows in arms for row in rows)
    if not results:
        raise SystemExit("no association could be estimated; check group sizes and covariate completeness")
    write_table(add_fdr(pd.DataFrame(results), args.fdr_family), args.output)


if __name__ == "__main__":
    main()
