# UNVEIL: Uncovering Non-apparent Visual Encodings in Latent Space

Kao TW et al. _Demographic Differences in Tissue Architecture Drive Disparities in Pathology AI Performance._

UNVEIL identifies, attributes and mitigates demographic signal in pathology foundation-model
representations. It detects the signal with demographic classifiers, measures the disparities
it produces in pathology tasks, attributes those disparities to nuclear morphology, cell
composition and model attention, and removes the signal from training bags with an
agentically discovered tile filter.

## Stages

| Stage | Folder | Purpose |
| --- | --- | --- |
| 1 | `01_demographic_classifier` | Predict age, sex and race from slide tile features |
| 2 | `02_pathology_task_prediction` | Attention-MIL models for mutation, subtype, grade, stage and tumor detection |
| 3 | `03_fairness_evaluation` | Six fairness metrics with bias tests across demographic groups, and the paired baseline-versus-mitigated comparison |
| 4a | `04_morphologic_correlates_of_demographic_signal/4a_nuclear_morphology` | Exposure associations with nuclear morphology features, pooled across cohorts |
| 4b | `04_morphologic_correlates_of_demographic_signal/4b_cell_composition` | Exposure effects on cell-type composition and nuclear density |
| 4c | `04_morphologic_correlates_of_demographic_signal/4c_model_attention` | Exposure effects on the attention-weighted cell composition of task models |
| 4d | `04_morphologic_correlates_of_demographic_signal/4d_morphology_error_decomposition` | Decomposition of prediction disparities through morphology features |
| 5 | `05_demographic_signal_agentic_modulation` | `unveil_tilefilter`: the agentically discovered tile filter, its matched controls, training-bag masking and the resolution rule |

Shared statistics live in the `unveil` package: covariate design, effect sizes, mixed and
penalized models, permutation tests, meta-analysis, multiplicity control and fairness metrics.
`configs/analysis.yaml` declares the exposures and covariates used by stages 3 and 4.
`tests/` checks the statistics against brute-force or reference implementations.

## Installation

```bash
conda create -n unveil python=3.8
conda activate unveil
pip install -r requirements.txt
pip install -e .
pytest tests
```

## Data conventions

Every table is keyed by `cohort`, `cancer` and `patient_id`. Slide tables add `slide_id` and
prediction tables add `task`. Tile features are one tensor per slide, `<slide_id>.pt`, of
shape `(n_tiles, feature_dim)`.

The clinical table holds one row per patient with the exposures and covariates named in
`configs/analysis.yaml`. Covariates are grouped into three blocks, published here under proxy
names:

| Block | Columns | Examples of what they stand for |
| --- | --- | --- |
| technical | `technical_cov1` to `technical_cov4` | tissue source site, scanner, stain batch, slide type |
| biological | `biological_cov1`, `biological_cov2` | histologic subtype, tumor purity |
| demographic | `age`, `sex`, `race` | |

Rename your columns or edit the spec to match. Columns listed under `categorical_covariates`
are encoded as indicators; rare levels are pooled into `other` and missing values become
their own level. Columns listed under `random_intercept` are modelled as random intercepts
where a script supports it.

Three defaults apply to every analysis in stage 4:

- Analyses are adjusted for all three blocks. The exposure being tested is always removed
  from its own adjustment set. `--adjust technical biological` restricts the blocks and
  `--adjust none` gives the unadjusted analysis.
- Slide-level inputs are averaged to the patient before testing, except where a patient
  random intercept models the slides directly (4b).
- Benjamini-Hochberg q-values are computed within broad families, such as all features of
  one exposure contrast in one cohort. The family is printed in every output and can be
  changed with `--fdr-family`.

Categorical exposures are tested as pairwise contrasts against the reference level, for
example `male_vs_female` or `black_vs_white`. Continuous exposures enter linearly. In 4d
they are contrasted between their 10th and 90th percentiles, and in 3 they are split at the
patient median.

## Usage

The commands below run from the repository root.

### 1. Demographic classifier

```bash
cd 01_demographic_classifier && ./run.sh configs/example_train.json
```

### 2. Pathology task prediction

`build_task_manifest.py` turns one task into a slide manifest with a patient-level fold.
Labels come from a 0/1 mutation table, a clinical column mapped through
`configs/tasks.yaml` (grade, stage, subtype, receptor status), or the TCGA sample type
(tumor detection). Folds come from a released split, a fixed clinical column, or a
patient-grouped stratified split.

```bash
python 02_pathology_task_prediction/build_task_manifest.py \
    --cohort TCGA --cancer BRCA --task PIK3CA \
    --features-dir example_data/features \
    --mutation-table example_data/pathology_task_prediction/mutations/PIK3CA.csv \
    --clinical example_data/pathology_task_prediction/clinical.csv --fold-column kfold_partition \
    --output output/manifest.csv

python 02_pathology_task_prediction/train.py --manifest output/manifest.csv --output-dir output/baseline
```

For each test fold k, the patients of fold k+1 form the validation set and the remaining
folds form the training set. Training follows the study recipe:

- class-weighted cross-entropy, Adam at 1e-4 with step decay 0.955, batch 16 with two
  accumulation steps;
- at most 2000 randomly drawn tiles per training bag;
- up to 100 epochs, stopping after 10 epochs without improvement in patient-level
  validation AUC.

`predictions.csv` holds the out-of-fold slide probabilities and is the input of stages 3
and 4. `predict.py` scores an external cohort or a released checkpoint.
`run_cross_validation.sh` wraps the manifest, baseline and masked training steps for a job
scheduler.

### 3. Fairness evaluation

```bash
python 03_fairness_evaluation/evaluate_fairness.py --predictions output/baseline/predictions.csv \
    --clinical clinical.csv --output fairness.csv --group-output group_performance.csv

python 03_fairness_evaluation/compare_models.py --baseline output/baseline/predictions.csv \
    --mitigated output/signal_aware/predictions.csv --clinical clinical.csv --output comparison.csv
```

AUC, TPR, TNR and balanced accuracy are computed in every demographic group. Each disparity
is the best group minus the worst group, or with `--privileged-reference` the reference level
minus the worst other group:

| Metric | Disparity in |
| --- | --- |
| `EOpp0` | TNR |
| `EOpp1` | TPR |
| `EBAcc` | balanced accuracy |
| `EOddAbs` | mean of the absolute TPR and TNR disparities (derived) |
| `EOddMax` | larger of the absolute TPR and TNR disparities (derived) |
| `AUCDiff` | AUC |

Continuous exposures are split at the patient median. Categorical exposures compare all
levels with at least `min_group_size` patients, and `--pairwise` adds each level against the
reference. Units are patients, or patient and label pairs when a patient contributes both
tumor and normal slides.

- The bias test keeps each patient's group and redraws outcome and prediction pairs into the
  group slots, with replacement (`--null bootstrap`, the study default) or without
  (`--null permutation`), within `--stratify-by` levels if given. The one-sided p-value is the
  share of null disparities at least as large as the observed one.
- Intervals come from a patient bootstrap within groups.
- `--group-output` lists per-group performance with intervals, and the overall AUC with a
  bootstrap test against 0.5.
- The comparison reports each disparity under both models, the paired bootstrap interval of
  the change, and a p-value from swapping the two models' predictions within patients.
- Benjamini-Hochberg q-values are computed per cohort, exposure, contrast and metric, across
  cancers and tasks.

### 4a. Nuclear morphology

```bash
python 04_morphologic_correlates_of_demographic_signal/4a_nuclear_morphology/1_test_associations.py \
    --features morphology.csv --clinical clinical.csv --output associations.csv
python 04_morphologic_correlates_of_demographic_signal/4a_nuclear_morphology/2_pool_cohorts.py \
    --associations associations.csv --output pooled.csv
python 04_morphologic_correlates_of_demographic_signal/4a_nuclear_morphology/3_test_enrichment.py \
    --results pooled.csv --annotations feature_annotations.csv --output enrichment.csv
```

Every numeric column of the morphology table that is not an identifier is treated as a
feature.

- The adjusted effect is a partial correlation on normal scores, with complete-case
  covariates per feature.
- `--unadjusted` adds Cliff's delta (binary exposures) or Spearman's rho (continuous
  exposures).
- Cohorts are pooled on the Fisher z scale with DerSimonian-Laird random effects. The
  pooling also reports Hartung-Knapp intervals, prediction intervals and leave-one-cohort-out
  robustness.
- Enrichment of significant features in cell type by feature group categories is tested by
  permuting the category labels.

### 4b. Cell composition

```bash
python 04_morphologic_correlates_of_demographic_signal/4b_cell_composition/test_composition.py \
    --slides composition.csv --clinical clinical.csv \
    --components tumor lymphocyte stroma necrosis --n-tiles-column n_tiles --output composition_effects.csv
```

- Counts are closed and centred log-ratio transformed with multiplicative zero replacement.
- Each component is modelled at slide level with a patient random intercept, and checked at
  patient level with a Freedman-Lane permutation test.
- `--omnibus-output` adds a multivariate PERMANOVA and a dispersion test.

### 4c. Model attention

```bash
python 04_morphologic_correlates_of_demographic_signal/4c_model_attention/1_build_attention_composition.py \
    --tiles attention_tiles.csv --components tumor lymphocyte stroma necrosis \
    --predictions output/baseline/predictions.csv --output attention_composition.csv
python 04_morphologic_correlates_of_demographic_signal/4c_model_attention/2_test_attention_composition.py \
    --attention attention_composition.csv --clinical clinical.csv \
    --components tumor lymphocyte stroma necrosis --output attention_effects.csv --omnibus-output attention_omnibus.csv
```

The attention-weighted composition is tested with the 4b model, per task by default
(`--stratify-by`), on the centred log-ratio scale. Passing `--predictions` adds
`prediction_correct`, so attention can also be compared between correct and incorrect
predictions with `--exposures prediction_correct`.

### 4d. Morphology-error decomposition

```bash
python 04_morphologic_correlates_of_demographic_signal/4d_morphology_error_decomposition/1_build_patient_outcomes.py \
    --predictions output/baseline/predictions.csv --output outcomes.csv
python 04_morphologic_correlates_of_demographic_signal/4d_morphology_error_decomposition/2_decompose_disparity.py \
    --outcomes-table outcomes.csv --mediators morphology.csv --clinical clinical.csv --output decomposition.csv
python 04_morphologic_correlates_of_demographic_signal/4d_morphology_error_decomposition/3_test_and_pool.py \
    --decomposition decomposition.csv --output decomposition_tested.csv --pooled-output decomposition_pooled.csv
```

The outcomes are the logit of the patient score and the logit of the Brier error. For each
morphology feature, the exposure effect on an outcome is split by interventional
g-computation into an indirect effect through the feature and a direct effect. The
decomposition also reports the controlled direct effect, the interaction components and the
label-dependent indirect effect.

- The models include a label by mediator term and a random intercept for the
  `random_intercept` covariates.
- Inference uses a patient bootstrap.
- Indirect effects are tested against the composite null with the HDMT procedure before
  Benjamini-Hochberg correction.
- `--chunk` and `--n-chunks` split the features across a job array.

### 5. Demographic-signal agentic modulation

For each training slide, the demographic classifier scores which tiles carry the most
evidence for the slide's demographic group, and the 33% of tiles that score highest are
removed before the task model is trained. Validation, test and external slides are never
filtered. The scoring rule was found by an LLM-driven evolutionary search (OpenEvolve) on
TCGA development tasks and then locked.

```bash
pip install -e 05_demographic_signal_agentic_modulation
unveil-tilefilter signal --features-dir example_data/features --checkpoint demographic_classifier.pt \
    --model-code 01_demographic_classifier --out signals/
unveil-tilefilter masks --signals signals/ --out masks/adaptive.npz
python 02_pathology_task_prediction/train.py --manifest output/manifest.csv \
    --output-dir output/signal_aware --tile-masks masks/adaptive.npz
```

`--filter random_matched` and `--filter attention_matched` build the two controls, which
remove the same number of tiles from each slide. `unveil-tilefilter resolve` applies the
paper's rule for whether a run resolves a biased task. See
`05_demographic_signal_agentic_modulation/README.md` for the filter, the `fairtune_pipeline`
adapter and the verification against the paper's files.

## Model weights

The trained baseline and signal-aware task models are released separately because of their
size.

## License

GNU Affero General Public License v3.0. See `LICENSE`.
