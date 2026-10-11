import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from mil import MANIFEST_COLUMNS

PATIENT_COLUMNS = ("patient_id", "case_submitter_id", "Patient ID")
TASK_SPEC = Path(__file__).resolve().parent / "configs" / "tasks.yaml"


def read(path):
    return pd.read_csv(path, sep="\t" if str(path).endswith((".tsv", ".txt")) else ",")


def patient_table(path):
    table = read(path)
    column = next((c for c in PATIENT_COLUMNS if c in table.columns), None)
    if column is None:
        raise ValueError(f"{path} has none of the patient id columns {PATIENT_COLUMNS}")
    return table.rename(columns={column: "patient_id"}).astype({"patient_id": str})


def discover_slides(args):
    if args.slides:
        slides = read(args.slides).astype({"slide_id": str, "patient_id": str})
    else:
        stems = sorted(p.stem for p in Path(args.features_dir).glob("*.pt"))
        slides = pd.DataFrame({"slide_id": stems, "patient_id": [s[: args.patient_id_length] for s in stems]})
    if "feature_path" not in slides.columns:
        slides["feature_path"] = [Path(args.features_dir) / f"{s}.pt" for s in slides["slide_id"]]
    slides["feature_path"] = [str(Path(p).resolve()) for p in slides["feature_path"]]
    exists = slides["feature_path"].map(lambda p: Path(p).is_file())
    if not exists.all():
        print(f"dropping {int((~exists).sum())} slides without a feature file")
    slides = slides[exists]
    fields = barcode_fields(slides)
    if args.slide_type != "all":
        ffpe = fields[2].str[:2] == "DX"
        slides = slides[ffpe if args.slide_type == "ffpe" else ~ffpe]
    if not (args.sample_type_labels or args.include_normal_slides):
        slides = slides[~(sample_type(slides) >= 10)]
    return slides.reset_index(drop=True)


def barcode_fields(slides):
    rest = pd.Series([s[len(p):].lstrip("-") for s, p in zip(slides["slide_id"], slides["patient_id"])], index=slides.index)
    return rest.str.split("-", expand=True).reindex(columns=range(3))


def sample_type(slides):
    return pd.to_numeric(barcode_fields(slides)[0].str[:2], errors="coerce")


def patient_labels(table, column, mapping=None, positive=None):
    values = table[column]
    if mapping is not None:
        values = values.astype(str).str.strip().map(mapping)
        values = (values == positive).astype(int).where(values.notna())
    labelled = pd.DataFrame({"patient_id": table["patient_id"], "label": values}).dropna().drop_duplicates()
    conflicting = labelled["patient_id"][labelled["patient_id"].duplicated()].unique()
    if len(conflicting):
        raise ValueError(f"{len(conflicting)} patients have conflicting labels, e.g. {list(conflicting[:3])}")
    return labelled.astype({"label": int})


def attach_labels(slides, args):
    if args.sample_type_labels:
        code = sample_type(slides)
        return slides.assign(label=(code < 10).astype(int))[code < 20]
    if args.mutation_table:
        labels = patient_labels(patient_table(args.mutation_table), args.mutation_column)
        if not set(labels["label"]) <= {0, 1}:
            raise ValueError(f"{args.mutation_column} must be coded 0/1")
    else:
        spec = yaml.safe_load(Path(args.task_spec).read_text())[args.task]
        labels = patient_labels(patient_table(args.clinical), spec["column"], spec["labels"], spec["positive"])
    return slides.merge(labels, on="patient_id", how="inner")


def attach_folds(slides, args):
    if args.splits or args.fold_column:
        table = patient_table(args.splits) if args.splits else patient_table(args.clinical)
        folds = table[["patient_id", "test_fold" if args.splits else args.fold_column]].dropna().drop_duplicates()
        folds.columns = ["patient_id", "fold"]
        return slides.merge(folds.astype({"fold": int}), on="patient_id", how="inner", validate="many_to_one")
    strata = slides["label"].astype(str)
    if args.stratify_by:
        clinical = patient_table(args.clinical).drop_duplicates("patient_id").set_index("patient_id")
        for column in args.stratify_by:
            strata = strata + "|" + slides["patient_id"].map(clinical[column]).astype(str)
    splitter = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    slides["fold"] = -1
    for fold, (_, test) in enumerate(splitter.split(slides, strata, groups=slides["patient_id"])):
        slides.loc[slides.index[test], "fold"] = fold
    return slides


def main():
    parser = argparse.ArgumentParser(description="Build the slide manifest for one binary pathology task.")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--cancer", required=True)
    parser.add_argument("--task", required=True, help="task name; for clinical labels, a key of --task-spec")
    parser.add_argument("--features-dir", required=True, help="directory of <slide_id>.pt tile-feature tensors")
    parser.add_argument("--slides", default=None, help="optional slide table: slide_id, patient_id[, feature_path]")
    parser.add_argument("--patient-id-length", type=int, default=12, help="patient id prefix length of the slide id when --slides is not given")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mutation-table", help="patient table with a binary mutation column")
    source.add_argument("--clinical-labels", action="store_true", help="label from --clinical using the --task entry of --task-spec")
    source.add_argument("--sample-type-labels", action="store_true", help="tumor detection: TCGA sample type 01-09 tumor, 10-19 normal")
    parser.add_argument("--mutation-column", default="Altered")
    parser.add_argument("--clinical", default=None, help="patient clinical table")
    parser.add_argument("--task-spec", default=str(TASK_SPEC))
    parser.add_argument("--slide-type", choices=["all", "ffpe", "frozen"], default="all")
    parser.add_argument("--include-normal-slides", action="store_true", help="keep TCGA normal-tissue slides (sample type 10-19) for patient-level labels")
    parser.add_argument("--splits", default=None, help="patient table with test_fold, e.g. a released split")
    parser.add_argument("--fold-column", default=None, help="column of --clinical holding a fixed patient fold")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--stratify-by", nargs="+", default=[], help="clinical columns to balance across generated folds, in addition to the label")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (args.clinical_labels or args.fold_column or args.stratify_by) and not args.clinical:
        parser.error("--clinical is required for --clinical-labels, --fold-column and --stratify-by")

    slides = attach_labels(discover_slides(args), args)
    if slides.empty:
        raise SystemExit("no labelled slides remain after filtering")
    slides = attach_folds(slides, args)
    if slides["label"].nunique() < 2:
        raise SystemExit("the task has a single label class after filtering")
    slides = slides.assign(cohort=args.cohort, cancer=args.cancer, task=args.task)[MANIFEST_COLUMNS]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    slides.sort_values(["fold", "patient_id", "slide_id"]).to_csv(args.output, index=False)
    print(slides.groupby(["fold", "label"]).agg(slides=("slide_id", "size"), patients=("patient_id", "nunique")).to_string())


if __name__ == "__main__":
    main()
