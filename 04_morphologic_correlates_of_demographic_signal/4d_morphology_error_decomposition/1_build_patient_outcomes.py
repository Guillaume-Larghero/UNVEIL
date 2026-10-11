import argparse

from scipy.special import logit

from unveil.data import read_table, write_table

PATIENT_KEYS = ["cohort", "cancer", "task", "patient_id"]
SCORE_CLIP = 1e-4
ERROR_CLIP = 1e-3


def patient_outcomes(predictions):
    patients = predictions.groupby(PATIENT_KEYS, sort=True).agg(probability=("probability", "mean"), label=("label", "first"), n_slides=("probability", "size")).reset_index()
    inconsistent = predictions.groupby(PATIENT_KEYS)["label"].nunique() > 1
    if inconsistent.any():
        raise ValueError(f"{int(inconsistent.sum())} patients have conflicting labels across slides")
    patients["score_logit"] = logit(patients["probability"].clip(SCORE_CLIP, 1 - SCORE_CLIP))
    brier = (patients["probability"] - patients["label"]) ** 2
    patients["brier_logit_error"] = logit(brier.clip(ERROR_CLIP, 1 - ERROR_CLIP))
    return patients


def main():
    parser = argparse.ArgumentParser(description="Patient-level prediction outcomes for the morphology-error decomposition.")
    parser.add_argument("--predictions", required=True, help="slide or patient predictions: cohort, cancer, task, patient_id, probability, label")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    patients = patient_outcomes(read_table(args.predictions))
    degenerate = patients.groupby(["cohort", "cancer", "task"])["label"].transform("nunique") < 2
    if degenerate.any():
        print(f"dropping {int(degenerate.sum())} patients from tasks with a single label class")
    write_table(patients.loc[~degenerate].assign(label=lambda d: d["label"].astype(int)), args.output)


if __name__ == "__main__":
    main()
