# Demographic-signal agentic modulation

The mitigation removes, from each training bag, the tiles that carry the most demographic
signal, then retrains the task model with the unmitigated recipe. Validation and test bags
are never filtered, so any change in fairness is attributable to which training tiles were
seen and to nothing else.

The filtering rule was discovered by an LLM-guided program search and then frozen. This
folder contains the search, the frozen rule, its two dose-matched controls, and the scripts
that apply it.

| File | Role |
| --- | --- |
| `programs/seed_program.py` | program the search starts from |
| `programs/frozen_program.py` | program the search returned, frozen |
| `agentic_search.py` | candidate masks, re-scoring of masked bags, G, H and fitness; the search's evaluator |
| `run_search.py` | dry run, search driver, candidate summary, freeze check |
| `configs/search.yaml` | search settings, LLM and `search:` block |
| `configs/cells.example.csv` | development and held-out cells |
| `tile_filter.py` | frozen rule and its controls, used to build training masks |
| `export_demographic_signal.py` | per-tile attention and demographic score from a 01 classifier |
| `build_tile_masks.py` | training masks for `02_pathology_task_prediction/train.py --tile-masks` |

## Program search

A program defines `drop_scores(attention, tile_score, slide_pred)` for one slide and returns
a drop score per tile and a drop fraction. The LLM may rewrite only the code between the
`EVOLVE-BLOCK` markers; a program that changes anything else, raises, or returns
non-finite or misshaped output scores -10.

```
for each candidate program P:
    for each slide:
        n_drop   <- min(floor(min(dose_P, 0.33) * n), n - max(64, ceil(0.05 * n)));  0 if n <= 64
        M_P      <- drop the n_drop tiles with the highest score
        M_random <- drop n_drop tiles at random (seeded by slide id and control seed)
    for each cell, on its out-of-fold slides, rerun the frozen models on the masked bags:
        excess(P, cell) <- [AUC(none) - AUC(M_P)] - [AUC(none) - AUC(M_random)]
    G <- median over development demographic cells of excess        (slide level)
    H <- median over development task cells of excess               (patient level)
    fitness(P) <- G - lam * max(0, H - eps)
the LLM proposes edits of the fittest programs; the best is frozen and never re-tuned
```

Demographic cells re-score the 01 classifier and task cells the 02 baseline model, fold by
fold, so every slide is scored by a model that did not train on it. Both terms subtract the
random control, so removing more tissue is not rewarded by itself. `eps` and `lam` have no
defaults and must be fixed in the config before the search starts.

The cell table has one row per cell: `cell`, `kind` (`demographic` or `task`), `role` (`dev`
or `held_out`), `manifest` (a 02 slide manifest whose `label` is the exposure or the task
label), `checkpoint` (with `{fold}`), and `signal_dir` (output of `export_demographic_signal.py`
for those slides).

```bash
python run_search.py --config configs/search.yaml --dry-run

source ~/.config/unveil/llm.env
python run_search.py --config configs/search.yaml --out search/

python run_search.py --score search/best_program.py --role held_out --control-seeds 0 1 2 \
    --output search/freeze_check.csv
```

The search writes `manifest.json`, OpenEvolve checkpoints, `candidates.csv` and
`best_program.py` under `--out`; `--summarise <out>` rebuilds the last two from an
interrupted run. Each candidate re-scores every slide of every development cell, so run the
search as a batch job. The search itself additionally needs `pip install openevolve`; the dry
run and the freeze check do not. The API key is read from the environment only.

The study search ran 200 iterations with a population of 60 and returned
`programs/frozen_program.py` (development G 0.12, H 0.005, mean dose 0.32). It was exploratory
and not pre-registered.

## Frozen rule, applied per training slide (`tile_filter.signal_aware_mask`)

```
input: attention a_i and demographic score s_i for each tile i, slide prediction p
n <- number of tiles
if n <= 64: keep every tile
evidence_i <- s_i            if p >= 0.5
              1 - s_i        otherwise
r_i        <- dense rank of a_i, scaled to [0, 1]
score_i    <- evidence_i * (0.25 + 0.75 * r_i) + 0.02 * r_i
n_drop     <- min(floor(0.33 * n), n - max(64, ceil(0.05 * n)))
drop the n_drop tiles with the highest score
```

### Controls, at the same per-slide dose

```
random_matched:    drop n_drop tiles uniformly at random (seeded by slide id)
attention_matched: drop the n_drop tiles with the highest attention
```

A mitigation effect should be reported against both controls, not only against the
unfiltered baseline.

## Training masks

```bash
python export_demographic_signal.py --checkpoint <01 model checkpoint> \
    --features-dir <tile features> --output-dir signal/

for method in signal_aware random_matched attention_matched; do
    python build_tile_masks.py --signal-dir signal/ --method $method \
        --output masks/$method.npz --summary masks/$method.csv
done

python ../02_pathology_task_prediction/train.py --manifest manifest.csv \
    --output-dir runs/signal_aware --tile-masks masks/signal_aware.npz
```

The masks must come from a demographic classifier that was not trained on the evaluation
slides of the task model.
