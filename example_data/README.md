# Example data

Mock data in the formats the pipeline expects. Identifiers, labels and features are
synthetic. The five patients are enough to run every step of stages 1 and 2 end to end, but
not to produce meaningful metrics.

## Contents

| Path | Format |
| --- | --- |
| `features/<slide_id>.pt` | float tensor `(n_tiles, 768)`, standing in for CHIEF tile features |
| `demographic_classifier/labels/TCGA-BRCA-{Age,Race,Sex}.json` | slide id to one-hot demographic label |
| `pathology_task_prediction/clinical.csv` | one row per patient: `case_submitter_id`, `age`, `race`, `sex`, `primary_diagnosis`, `kfold_partition` |
| `pathology_task_prediction/mutations/{PIK3CA,TP53}.csv` | `Patient ID`, `Altered` (0 wild type, 1 mutated) |
| `pathology_task_prediction/models/TCGA_BRCA_PIK3CA_CHIEF_fold0.pt` | attention-MIL state dict of a PIK3CA model trained on TCGA-BRCA CHIEF features |

Slide ids follow the TCGA barcode layout, so the patient id is the first 12 characters and
the sample type follows it.

## Running the examples

From the repository root:

```bash
cd 01_demographic_classifier && ./run.sh configs/example_train.json && cd ..

python 02_pathology_task_prediction/build_task_manifest.py \
    --cohort TCGA --cancer BRCA --task PIK3CA --features-dir example_data/features \
    --mutation-table example_data/pathology_task_prediction/mutations/PIK3CA.csv \
    --clinical example_data/pathology_task_prediction/clinical.csv --fold-column kfold_partition \
    --output output/example/manifest.csv
python 02_pathology_task_prediction/train.py --manifest output/example/manifest.csv \
    --output-dir output/example/baseline --epochs 2 --num-workers 0
python 02_pathology_task_prediction/predict.py --manifest output/example/manifest.csv \
    --checkpoint example_data/pathology_task_prediction/models/TCGA_BRCA_PIK3CA_CHIEF_fold0.pt \
    --num-workers 0 --output output/example/released_model_predictions.csv
```

`--task subtype_brca --clinical-labels` in place of `--mutation-table` builds the ductal
versus lobular subtype task from `primary_diagnosis`.
