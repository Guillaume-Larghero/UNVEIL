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
Performance* (Cancer Cell, in revision). Code: https://github.com/hms-dbmi/UNVEIL. This
file is the model card for the weights released at **[HF_REPO_URL]**; the same text is
kept as `MODEL_CARD.md` in the code repository.

The models take **tile embeddings of H&E whole-slide images (WSIs)** produced by one of four
public pathology foundation models (FMs), not pixels: CHIEF (768-dimensional), UNI (1024),
GigaPath (1536) or Virchow2 (2560), from 224 × 224-pixel tiles at ×20. No FM weights are
included or modified; obtain the FMs from their authors under their own licenses.

## What is released

| Group | Contents | Count | Purpose in the study |
| --- | --- | --- | --- |
| Demographic classifiers | one gated ABMIL classifier per cancer type × attribute (age group, self-reported race, sex) × slide type (frozen section FS, permanent FFPE section PM); the FM and the cross-validation fold selected in the paper | 38 | quantify demographic signal in FM representations (module 1); provide the per-tile demographic signal that the tile filter of module 5 uses |
| Baseline task models | ABMIL checkpoints per task × FM × fold, trained on TCGA with patient-level four-fold cross-validation, with the fold assignments | [N] | downstream pathology prediction (module 2); the baselines of the fairness, morphology and mitigation analyses (modules 3 to 5) |

**Not released:** the signal-aware (mitigated) task models, which module 5 retrains from the
baseline recipe and the released classifiers; the comparator fairness methods, which are
existing published methods; the foundation models.

### Demographic classifiers

| Cancer | Attribute | Slide type | FM | Fold | Selection metric |
| --- | --- | --- | --- | --- | --- |
| BRCA | Age | FS | Virchow2 | 5 | balanced accuracy |
| BRCA | Age | PM | Virchow2 | 5 | balanced accuracy |
| BRCA | Race | FS | Virchow2 | 2 | AUROC |
| BRCA | Race | PM | Virchow2 | 2 | AUROC |
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
| KIRC | Sex | FS | Virchow2 | 9 | balanced accuracy |
| KIRC | Sex | PM | Virchow2 | 9 | balanced accuracy |
| LGG | Age | FS | UNI | 3 | AUROC |
| LGG | Age | PM | UNI | 3 | AUROC |
| LGG | Race | FS | GigaPath | 4 | cross-entropy |
| LGG | Race | PM | GigaPath | 4 | cross-entropy |
| LGG | Sex | FS | UNI | 4 | balanced accuracy |
| LGG | Sex | PM | GigaPath | 3 | balanced accuracy |
| LUAD | Age | FS | GigaPath | 3 | cross-entropy |
| LUAD | Age | PM | Virchow2 | 6 | cross-entropy |
| LUAD | Race | FS | CHIEF | 3 | AUROC |
| LUAD | Race | PM | Virchow2 | 1 | cross-entropy |
| LUAD | Sex | FS | Virchow2 | 10 | AUROC |
| LUAD | Sex | PM | UNI | 4 | AUROC |
| LUSC | Age | FS | Virchow2 | 7 | cross-entropy |
| LUSC | Age | PM | Virchow2 | 7 | cross-entropy |
| LUSC | Race | FS | UNI | 9 | AUROC |
| LUSC | Race | PM | GigaPath | 5 | AUROC |
| LUSC | Sex | FS | UNI | 8 | balanced accuracy |
| LUSC | Sex | PM | UNI | 8 | balanced accuracy |

Files are named `demographic_classifiers/<cancer>_<attribute>_<slide type>_<FM>.pt`. Each is
the fold model of the paper's ten-fold TCGA cross-validation that was selected for that
cell (fold and selection metric above); these are the classifiers whose per-tile signal the
tile filter of module 5 used. The classifier of one cell was applied to the task models of
all four FMs, so it need not share the task model's FM. **[Wan: are the classifiers of the
other cells of Tables S2 to S4, e.g. COAD/READ sex and the other FMs, released too?]**

### Baseline task models

`task_models/<cancer>_<task>_<FM>_fold<k>.pt` with `task_models/folds.csv` (cohort, cancer,
patient_id, fold). Tasks are listed in Table S5 of the supplement: tumor detection,
histologic subtyping, grade, stage and mutation prediction across BRCA, COAD/READ, GBM,
KIRC, LGG, LUAD and LUSC. **[Shih-Yen: confirm naming and list.]**

## Architecture and training

- **Architecture:** gated attention MIL (Ilse et al., 2018): tile embeddings → gated attention
  (tanh and sigmoid branches, one attention logit per tile, softmax over the slide) →
  attention-weighted slide embedding → three-layer MLP → class probability.
- **Demographic classifiers:** TCGA development set, stratified ten-fold cross-validation,
  Adam (learning rate 1e-4), 30 epochs, binary cross-entropy; fold models selected by
  cross-entropy, balanced accuracy or AUROC. Labels: age dichotomized at the cancer-specific
  TCGA median; sex female/male; race White versus non-White (self-reported or recorded),
  with White/Black/Asian analyses reported separately in the paper.
- **Task models:** stratified four-fold cross-validation with all slides of a patient in one
  fold, Adam (1e-4), class-weighted cross-entropy, early stopping after ten epochs without
  improvement; one model per FM.
- **Environment:** Python 3.8, PyTorch 2.0.0 (see the repository's `requirements.txt`).

## Training data

TCGA (The Cancer Genome Atlas) primary-tumour H&E slides of seven cancer types, frozen and
FFPE sections, accessed through the NCI Genomic Data Commons. The study as a whole covers
17,402 WSIs from 10,853 patients in nine cohorts (TCGA, CPTAC, PLCO, DFCI, HKU, UPenn, TVGH,
Mayo Clinic, MUV); the external cohorts were used for evaluation only. Cohort demographics
are in Table S1 of the supplement. Institutional cohorts are under data use agreements and
are not released.

## Evaluation

Demographic classification on the held-out TCGA test set reached mean AUROCs across FMs of
0.57 ± 0.05 (age), 0.60 ± 0.09 (race) and 0.55 ± 0.04 (sex); on eight external cohorts
0.63 ± 0.09, 0.64 ± 0.12 and 0.61 ± 0.09. Per cancer, FM and cohort values with sensitivity,
specificity, PPV, NPV and balanced accuracy: Tables S2 to S4. Baseline task-model AUROCs
with 95% confidence intervals and the six fairness metrics per demographic contrast: Tables
S6 to S8; performance stratified by demographic group with subgroup counts: Tables S9 to S11.

The demographic signal is modest: these classifiers are evidence that FM embeddings carry
demographic information, not tools for predicting demographics.

## Intended use

- Reproducing the study, and applying its five modules to other cohorts (see the repository
  README).
- Auditing a pathology AI pipeline built on one of the four FMs for demographic signal and
  subgroup performance disparities.
- Research on fairness of pathology foundation models.

## Out-of-scope use

- Any clinical use, including diagnosis, grading, staging or molecular-status prediction for
  patient care. The task models are research baselines trained on TCGA and were not
  validated for clinical deployment.
- Inferring the age, race or sex of individuals from their tissue.
- Use with tile embeddings from a different FM, magnification or tile size than the model
  was trained with, or commercial use (CC BY-NC 4.0).

## Limitations

- Race is self-reported or recorded, binarized to White versus non-White for the primary
  analyses; it is a social category confounded with institution, geography and case mix,
  not a biological phenotype.
- TCGA is multi-institutional; demographic groups are unevenly distributed across tissue
  source sites, scanners and preparation protocols. Technical confounding is adjusted for in
  the paper's analyses but cannot be excluded from the models themselves.
- Frozen-section (FS) and FFPE (PM) models are separate and not interchangeable.
- Mitigation was externally validated on DFCI and CPTAC only.

## How to use

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('[HF_REPO_ID]', local_dir='weights')"

# score slides with a baseline task model (module 2)
python 02_pathology_task_prediction/predict.py --manifest manifest.csv \
    --checkpoint weights/task_models/BRCA_PIK3CA_CHIEF_fold0.pt --output predictions.csv

# per-tile demographic signal from a demographic classifier (module 5)
unveil-tilefilter signal --features-dir features/ --model-code 01_demographic_classifier \
    --checkpoint weights/demographic_classifiers/BRCA_Race_FS_VIRCHOW2.pt --out signals/
```

Feature files are `<slide_id>.pt` tensors of shape `(n_tiles, feature_dim)`; see the
repository README for the manifest and label formats.

## License and citation

Weights: CC BY-NC 4.0. Code: GNU AGPL v3.0. For academic, non-commercial use only; data
access enquiries go to the lead contact.

```
Kao TW, Vremenko D, Lin SY, Su FY, Li B, Tsai PC, Lee TH, Wang X, Yang S, Li JJX, Lo YC,
Lin SC, Sholl L, Meredith DM, Ligon KL, Lin NU, Nasrallah MP, Golden JA, Meyerhardt J,
Ogino S, Roetzer-Pejrimovsky T, Woehrer A, Signoretti S, Schnitt S, Yu KH.
Demographic Differences in Cancer Tissue Architecture Drive Disparities in Pathology AI
Performance. Cancer Cell (in revision). https://github.com/hms-dbmi/UNVEIL
```

Contact: Kun-Hsing Yu, kun-hsing_yu@hms.harvard.edu.
