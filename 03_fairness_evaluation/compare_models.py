import argparse

import numpy as np
import pandas as pd

from unveil.cli import DEFAULT_SPEC
from unveil.config import load_spec
from unveil.data import attach_clinical, read_table, write_table
from unveil.fairness import (DERIVED_METRICS, METRICS, OVERALL_METRICS, PATIENT_KEYS, bootstrap_metrics, fairness_groups,
                             fairness_metrics, paired_swap_test, patient_predictions)
from unveil.multiplicity import add_fdr

FDR_FAMILY = ("cohort", "exposure", "contrast", "metric")


def paired_patients(baseline, mitigated, threshold):
    baseline = patient_predictions(baseline, threshold)
    mitigated = patient_predictions(mitigated, threshold)
    paired = baseline.merge(mitigated, on=PATIENT_KEYS + ["label"], suffixes=("_baseline", "_mitigated"), how="outer", validate="one_to_one", indicator=True)
    if (paired.pop("_merge") != "both").any():
        raise ValueError("baseline and mitigated predictions do not cover the same patients and labels")
    return paired


def compare_cell(frame, group, privileged, args, rng):
    y = frame["label"].to_numpy()
    patient = frame["patient_id"].to_numpy()
    models = [(frame[f"probability_{m}"].to_numpy(), frame[f"prediction_{m}"].to_numpy()) for m in ("baseline", "mitigated")]
    observed = [fairness_metrics(y, score, pred, group, privileged) for score, pred in models]
    change, p_values = paired_swap_test(y, group, patient, models, args.n_boot, rng, privileged)
    draws, _ = bootstrap_metrics(y, group, patient, models, args.n_boot, rng, privileged)
    low, high = np.nanquantile(draws[:, 1] - draws[:, 0], [0.025, 0.975], axis=0)
    return [
        {"metric": m, "derived": m in DERIVED_METRICS, "baseline": observed[0][m], "mitigated": observed[1][m],
         "change": change[j], "ci_low": low[j], "ci_high": high[j], "p_value": p_values[j]}
        for j, m in enumerate(METRICS + OVERALL_METRICS)
    ]


def main():
    parser = argparse.ArgumentParser(description="Paired comparison of fairness disparities and performance between a baseline and a mitigated model.")
    parser.add_argument("--baseline", required=True, help="baseline predictions: cohort, cancer, task, patient_id, probability, label")
    parser.add_argument("--mitigated", required=True, help="mitigated predictions on the same patients")
    parser.add_argument("--clinical", required=True)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--exposures", nargs="+", default=None, help="defaults to the demographic covariate block")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--privileged-reference", action="store_true")
    parser.add_argument("--pairwise", action="store_true")
    parser.add_argument("--min-group-size", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=1000, help="swap permutations and bootstrap replicates")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fdr-family", nargs="+", default=list(FDR_FAMILY))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = load_spec(args.spec)
    min_group_size = spec.min_group_size if args.min_group_size is None else args.min_group_size

    paired = paired_patients(read_table(args.baseline), read_table(args.mitigated), args.threshold)
    patients = attach_clinical(paired, read_table(args.clinical), spec)
    rng = np.random.default_rng(args.seed)
    results = []
    for key, cell in patients.groupby(PATIENT_KEYS[:3], sort=True):
        for exposure in map(spec.exposure, args.exposures or spec.covariates["demographic"]):
            for contrast, frame, group, reference in fairness_groups(cell, exposure, min_group_size, args.pairwise):
                privileged = reference if args.privileged_reference else None
                label = {**dict(zip(PATIENT_KEYS[:3], key)), "exposure": exposure.name, "contrast": contrast,
                         "groups": ";".join(np.unique(group)), "privileged": privileged or "best_group"}
                results.extend({**label, **row} for row in compare_cell(frame, group, privileged, args, rng))
    if not results:
        raise SystemExit("no task had at least two groups of the minimum size")
    write_table(add_fdr(pd.DataFrame(results), args.fdr_family), args.output)


if __name__ == "__main__":
    main()
