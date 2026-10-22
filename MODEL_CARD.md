---
license: cc-by-nc-4.0
language: en
tags:
  - computational-pathology
  - whole-slide-images
  - multiple-instance-learning
  - fairness
  - foundation-model-embeddings
---

# UNVEIL model weights

Attention-based multiple instance learning (ABMIL) models from Kao TW, Vremenko D, et al.,
*Demographic Differences in Cancer Tissue Architecture Drive Disparities in Pathology AI
Performance* (Cancer Cell, in revision). Code and usage: https://github.com/hms-dbmi/UNVEIL.
Weights: **[HF_REPO_URL]**.

The models take tile embeddings of H&E whole-slide images (WSIs) from one of four public
pathology foundation models (FMs), not pixels: CHIEF (768-dimensional), UNI (1024), GigaPath
(1536) or Virchow2 (2560), from 224 × 224-pixel tiles at ×20. No FM weights are included or
modified; obtain the FMs from their authors under their own licenses.

## Contents

| Folder | Models | Files |
| --- | --- | --- |
| `demographic_classifiers/` | 31 gated ABMIL classifiers of age group, self-reported race or sex from slide embeddings, trained on TCGA; one per cancer type × attribute × slide type (frozen section FS, permanent FFPE section PM), 38 cells | `<cancer>_<attribute>_<FM>_fold<k>.pt` (state_dict), `.json` (model class, arguments, training settings), `.split.json` (TCGA patient barcodes of the fold's train/validation/test sets); `cells.csv` maps the 38 cells to the files |
| `task_models/` | 544 ABMIL task models (136 tasks × 4 folds) for tumour detection, histologic subtyping, grade, stage and mutation prediction, trained on TCGA with the UNVEIL demographic-signal-aware tile filter | `<task>/fold<k>/model.pt` (state_dict), `args.json` (architecture and training arguments), `thresholds.csv` (decision thresholds); `splits/<task>.csv` (TCGA patients with `test_fold`, `label`, `sens_attr`; slide-level for tumour detection); `mapping.csv` (task, task type, cancer, target, attribute, FM, slide type, train and evaluation cohort, fold, file, split file) |

All files are plain PyTorch state_dicts loadable with `weights_only=True`.

### Demographic classifiers

| Cancer | Attribute | Slide type | FM | Fold | Selection metric |
| --- | --- | --- | --- | --- | --- |
| BRCA | Age | FS, PM | Virchow2 | 5 | balanced accuracy |
| BRCA | Race | FS, PM | Virchow2 | 2 | AUROC |
| COAD/READ | Age | FS | UNI | 6 | AUROC |
| COAD/READ | Age | PM | CHIEF | 6 | cross-entropy |
| COAD/READ | Race | FS | CHIEF | 3 | cross-entropy |
| COAD/READ | Race | PM | UNI | 2 | balanced accuracy |
| GBM | Age | FS | UNI | 10 | AUROC |
| GBM | Age | PM | Virchow2 | 5 | balanced accuracy |
| GBM | Race | FS | UNI | 9 | AUROC |
| GBM | Race | PM | GigaPath | 7 | AUROC |
| GBM | Sex | FS | GigaPath | 2 | AUROC |
| GBM | Sex | PM | UNI | 9 | AUROC |
| KIRC | Age | FS | GigaPath | 6 | AUROC |
| KIRC | Age | PM | CHIEF | 3 | balanced accuracy |
| KIRC | Race | FS | Virchow2 | 8 | AUROC |
| KIRC | Race | PM | UNI | 9 | AUROC |
| KIRC | Sex | FS, PM | Virchow2 | 9 | balanced accuracy |
| LGG | Age | FS, PM | UNI | 3 | AUROC |
| LGG | Race | FS, PM | GigaPath | 4 | cross-entropy |
| LGG | Sex | FS | UNI | 4 | balanced accuracy |
| LGG | Sex | PM | GigaPath | 3 | balanced accuracy |
| LUAD | Age | FS | GigaPath | 3 | cross-entropy |
| LUAD | Age | PM | Virchow2 | 6 | cross-entropy |
| LUAD | Race | FS | CHIEF | 3 | AUROC |
| LUAD | Race | PM | Virchow2 | 1 | cross-entropy |
| LUAD | Sex | FS | Virchow2 | 10 | AUROC |
| LUAD | Sex | PM | UNI | 4 | AUROC |
| LUSC | Age | FS, PM | Virchow2 | 7 | cross-entropy |
| LUSC | Race | FS | UNI | 9 | AUROC |
| LUSC | Race | PM | GigaPath | 5 | AUROC |
| LUSC | Sex | FS, PM | UNI | 8 | balanced accuracy |

Each is the fold model of a ten-fold TCGA cross-validation selected for that cell. A cell's
classifier provides the per-tile demographic signal for the task models of all four FMs.

## Architecture and training

- **Architecture:** gated attention MIL (Ilse et al., 2018): tile embeddings → gated
  attention (tanh and sigmoid branches, one logit per tile, softmax over the slide) →
  attention-weighted slide embedding → MLP → class probability.
- **Demographic classifiers:** TCGA, stratified ten-fold cross-validation, Adam (1e-4), 30
  epochs, binary cross-entropy. Labels: age dichotomized at the cancer-specific TCGA median;
  sex female/male; race White versus non-White (self-reported or recorded).
- **Task models:** TCGA, stratified four-fold cross-validation with all slides of a patient in
  one fold, Adam (1e-4), class-weighted cross-entropy, early stopping after ten epochs without
  improvement, one model per FM. Training slides had the 33% of tiles scoring highest on the
  cell's demographic classifier (evidence weighted by attention rank) withheld, at least 64
  tiles kept; validation, test and external slides were whole. Of the 136 tasks, 93 are
  evaluated on the TCGA test folds and 43 on DFCI or CPTAC, where the four TCGA-trained fold
  models score every external patient.
- **Environment:** Python 3.8, PyTorch 2.0.0.

## Data

TCGA primary-tumour H&E slides of seven cancer types (BRCA, COAD/READ, GBM, KIRC, LGG, LUAD,
LUSC), frozen and FFPE sections, from the NCI Genomic Data Commons. External evaluation
cohorts (CPTAC, PLCO, DFCI, HKU, UPenn, TVGH, Mayo Clinic, MUV) were not used for training;
institutional cohorts are under data use agreements and are not released. Cohort
demographics: Table S1 of the supplement.

## Evaluation

Demographic classification reached mean AUROCs across FMs of 0.57 ± 0.05 (age), 0.60 ± 0.09
(race) and 0.55 ± 0.04 (sex) on the held-out TCGA test set and 0.63 ± 0.09, 0.64 ± 0.12 and
0.61 ± 0.09 on eight external cohorts (Tables S2 to S4). Task-model AUROC and the six
fairness metrics (EOpp0, EOpp1, EOddAbs, EBAcc, ACCDiff, AUCDiff) with permutation p-values:
Tables S12 to S14. The demographic signal is modest: the classifiers show that FM embeddings
carry demographic information; they are not tools for predicting demographics.

## Intended use

Reproducing the study and applying its framework to other cohorts (repository README);
auditing a pathology AI pipeline built on one of the four FMs for demographic signal and
subgroup disparities; fairness research on pathology foundation models.

**Not intended for:** clinical use of any kind (the task models are research models trained on
TCGA and not validated for deployment); inferring the age, race or sex of individuals; use with
embeddings from another FM, magnification or tile size; commercial use (CC BY-NC 4.0).

## Limitations

Race is self-reported or recorded and binarized to White versus non-White: a social category
confounded with institution, geography and case mix, not a biological phenotype. TCGA is
multi-institutional, and demographic groups are unevenly distributed across tissue source
sites, scanners and preparation protocols. FS and PM models are separate and not
interchangeable. Mitigation was externally evaluated on DFCI and CPTAC only.

## How to use

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('[HF_REPO_ID]', local_dir='weights')"

# score slides with a task model (repository module 2)
python 02_pathology_task_prediction/predict.py --manifest manifest.csv \
    --checkpoint weights/task_models/detection_COAD_READ_Age_GIGAPATH_FS/fold0/model.pt --output predictions.csv

# per-tile demographic signal from a classifier (repository module 5)
unveil-tilefilter signal --features-dir features/ --model-code 01_demographic_classifier \
    --checkpoint weights/demographic_classifiers/BRCA_Race_VIRCHOW2_fold2.pt \
    --model-config weights/demographic_classifiers/BRCA_Race_VIRCHOW2_fold2.json --out signals/
```

Feature files are `<slide_id>.pt` tensors of shape `(n_tiles, feature_dim)`; the repository
README gives the manifest and label formats.

## License and citation

Weights: CC BY-NC 4.0. Code: GNU AGPL v3.0. Non-commercial academic use only.

```
Kao TW, Vremenko D, Lin SY, Su FY, Li B, Tsai PC, Lee TH, Wang X, Yang S, Li JJX, Lo YC,
Lin SC, Sholl L, Meredith DM, Ligon KL, Lin NU, Nasrallah MP, Golden JA, Meyerhardt J,
Ogino S, Roetzer-Pejrimovsky T, Woehrer A, Signoretti S, Schnitt S, Yu KH.
Demographic Differences in Cancer Tissue Architecture Drive Disparities in Pathology AI
Performance. Cancer Cell (in revision). https://github.com/hms-dbmi/UNVEIL
```

Contact: Kun-Hsing Yu, kun-hsing_yu@hms.harvard.edu.
