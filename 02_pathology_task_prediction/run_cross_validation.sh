#!/bin/bash
#SBATCH --job-name=unveil_task
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail
[ -f build_task_manifest.py ] || cd "$(dirname "${BASH_SOURCE[0]}")"

COHORT=${COHORT:-TCGA}
CANCER=${CANCER:-BRCA}
TASK=${TASK:-PIK3CA}
FEATURES=${FEATURES:-../example_data/features}
CLINICAL=${CLINICAL:-../example_data/pathology_task_prediction/clinical.csv}
LABELS=${LABELS:---mutation-table ../example_data/pathology_task_prediction/mutations/PIK3CA.csv}
FOLDS=${FOLDS:---fold-column kfold_partition}
TILE_MASKS=${TILE_MASKS:-}
EPOCHS=${EPOCHS:-100}
SEED=${SEED:-0}
OUT=${OUT:-output/${COHORT}_${CANCER}_${TASK}}

mkdir -p "$OUT"
python build_task_manifest.py --cohort "$COHORT" --cancer "$CANCER" --task "$TASK" \
    --features-dir "$FEATURES" --clinical "$CLINICAL" $LABELS $FOLDS --output "$OUT/manifest.csv"
python train.py --manifest "$OUT/manifest.csv" --output-dir "$OUT/baseline" --epochs "$EPOCHS" --seed "$SEED"
if [ -n "$TILE_MASKS" ]; then
    python train.py --manifest "$OUT/manifest.csv" --output-dir "$OUT/signal_aware" --tile-masks "$TILE_MASKS" --epochs "$EPOCHS" --seed "$SEED"
fi
