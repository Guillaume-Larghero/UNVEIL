import argparse
import zlib

import numpy as np
import pandas as pd

from unveil.cli import add_analysis_arguments, resolve
from unveil.data import attach_clinical, exposure_contrasts, read_table, require_unique, to_patient_level, write_table
from unveil.decomposition import build_design, decompose
from unveil.design import covariate_frame

OUTCOMES = ("score_logit", "brier_logit_error")
CONTINUOUS_CONTRAST = (0.10, 0.90)
TASK_KEYS = ["cohort", "cancer", "task"]


def load(args, spec):
    outcomes = read_table(args.outcomes_table)
    mediators = read_table(args.mediators)
    keys = [spec.cohort, spec.cancer, spec.patient_id]
    names = args.mediator_columns or [c for c in mediators.columns if c not in keys + [spec.slide_id] and mediators[c].dtype.kind in "biuf"]
    if mediators.duplicated(keys).any():
        mediators = to_patient_level(mediators[keys + names], keys).drop(columns="n_slides")
    data = outcomes.merge(mediators[keys + names], on=keys, how="inner", validate="many_to_one")
    data = attach_clinical(data, read_table(args.clinical), spec)
    require_unique(data, TASK_KEYS + [spec.patient_id], "decomposition input")
    return data, names


def analysis_set(frame, a, m, args, covariates, spec):
    keep = np.isfinite(a) & np.isfinite(m) & np.isfinite(frame[list(args.outcomes)].to_numpy(dtype=float)).all(axis=1)
    if args.label_column:
        keep &= frame[args.label_column].notna().to_numpy()
    if covariates:
        keep &= covariate_frame(frame, covariates, spec).notna().all(axis=1).to_numpy()
    return keep


def run_test(frame, a, mediator, covariates, args, spec, rng):
    m = frame[mediator].to_numpy(dtype=float)
    keep = analysis_set(frame, a, m, args, covariates, spec)
    frame, a, m = frame.loc[keep], a[keep], m[keep]
    binary = set(np.unique(a)) <= {0.0, 1.0}
    too_small = min(a.sum(), len(a) - a.sum()) < spec.min_group_size if binary else len(a) < 2 * spec.min_group_size
    if too_small or m.std() == 0:
        return []
    a_lo, a_hi = (0.0, 1.0) if binary else tuple(np.quantile(a, CONTINUOUS_CONTRAST))
    label = frame[args.label_column].to_numpy(dtype=float) if args.label_column else None
    Y = frame[list(args.outcomes)].to_numpy(dtype=float)
    design = build_design(frame, a, (m - m.mean()) / m.std(), Y, label, covariates, spec, a_lo, a_hi)
    if design is None or len(a) - (design.X_outcome.shape[1] - design.n_random_levels) < spec.min_residual_df:
        return []
    rows = decompose(design, args.n_boot, rng)
    shared = {
        "n": len(a),
        "a_lo": a_lo,
        "a_hi": a_hi,
        "label_prevalence": label.mean() if design.label is not None else np.nan,
        "lambda_mediator": design.lambda_mediator,
        "n_random_intercept_levels": design.n_random_levels,
        "adjusted_for": ";".join(design.adjusted_for),
    }
    return [{"outcome": outcome, "lambda_outcome": design.lambda_outcome[k], **shared, **rows[k]} for k, outcome in enumerate(args.outcomes)]


def main():
    parser = argparse.ArgumentParser(description="Interventional decomposition of the prediction disparity through each morphology feature.")
    parser.add_argument("--outcomes-table", required=True, help="output of 1_build_patient_outcomes.py")
    parser.add_argument("--mediators", required=True, help="patient morphology features keyed by cohort, cancer and patient")
    parser.add_argument("--clinical", required=True, help="patient table with exposures and covariates")
    parser.add_argument("--mediator-columns", nargs="+", default=None)
    parser.add_argument("--outcomes", nargs="+", default=list(OUTCOMES))
    parser.add_argument("--label-column", default="label", help="adds the label x mediator term; pass an empty string to omit")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=0, help="index of this chunk, for job arrays")
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument("--output", required=True)
    add_analysis_arguments(parser)
    args = parser.parse_args()
    spec, exposures, blocks = resolve(args)

    data, mediators = load(args, spec)
    results, test_index = [], 0
    for key, cell in data.groupby(TASK_KEYS, sort=True):
        for exposure in exposures:
            covariates = spec.covariates_for(exposure.name, blocks)
            for contrast, frame, a in exposure_contrasts(cell, exposure, spec.min_group_size):
                for mediator in mediators:
                    test_index += 1
                    if (test_index - 1) % args.n_chunks != args.chunk:
                        continue
                    label = {**dict(zip(TASK_KEYS, key)), "exposure": exposure.name, "contrast": contrast, "mediator": mediator}
                    rng = np.random.default_rng([args.seed, zlib.crc32("|".join(map(str, label.values())).encode())])
                    results.extend({**label, **row} for row in run_test(frame, a, mediator, covariates, args, spec, rng))
    write_table(pd.DataFrame(results), args.output)


if __name__ == "__main__":
    main()
