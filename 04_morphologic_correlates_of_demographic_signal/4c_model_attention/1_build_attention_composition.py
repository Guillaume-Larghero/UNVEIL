import argparse

import numpy as np

from unveil.compositional import attention_weighted_composition, clr
from unveil.data import read_table, require_unique, write_table
from unveil.fairness import PATIENT_KEYS, patient_predictions

SLIDE_KEYS = PATIENT_KEYS + ["slide_id"]


def attention_metric(tiles, keys, components, metric):
    composition = attention_weighted_composition(tiles, keys, components, "attention")
    n_nuclei = composition["n_nuclei"].to_numpy()
    values = clr(composition[[f"weighted_{c}" for c in components]].to_numpy(), n_nuclei)
    if metric == "perturbation":
        values = values - clr(composition[[f"unweighted_{c}" for c in components]].to_numpy(), n_nuclei)
    out = composition[keys + ["n_tiles", "n_nuclei"]].copy()
    out[[f"clr_{c}" for c in components]] = values
    return out


def prediction_correctness(predictions, threshold):
    patients = patient_predictions(predictions, threshold)
    require_unique(patients, PATIENT_KEYS, "patient predictions")
    correct = patients["prediction"] == patients["label"]
    return patients[PATIENT_KEYS].assign(prediction_correct=np.where(correct, "correct", "incorrect"))


def main():
    parser = argparse.ArgumentParser(description="Slide-level attention-weighted cell composition, the input of 2_test_attention_composition.py.")
    parser.add_argument("--tiles", required=True, help="tile table: cohort, cancer, task, patient_id, slide_id, attention, component counts, optional encoder")
    parser.add_argument("--components", nargs="+", required=True, help="per-tile cell-type count columns")
    parser.add_argument("--metric", choices=["weighted", "perturbation"], default="perturbation", help="clr of the attention-weighted composition, or its shift from the unweighted composition")
    parser.add_argument("--average-over", default=None, help="column (e.g. encoder) whose per-slide metrics are averaged rather than stacked")
    parser.add_argument("--predictions", default=None, help="slide or patient predictions (cohort, cancer, task, patient_id, probability, label) that define prediction_correct")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    keys = SLIDE_KEYS + ([args.average_over] if args.average_over else [])
    slides = attention_metric(read_table(args.tiles), keys, args.components, args.metric)
    if args.average_over:
        slides = slides.drop(columns=args.average_over).groupby(SLIDE_KEYS, sort=True).mean().reset_index()
    if args.predictions:
        slides = slides.merge(prediction_correctness(read_table(args.predictions), args.threshold), on=PATIENT_KEYS, how="inner", validate="many_to_one")
    write_table(slides, args.output)


if __name__ == "__main__":
    main()
