"""Fitness of a tile-filter program, and the evaluator entry point the program search calls.

For a program P and a dose-matched random control R, on held-out slides of each cell:

    excess(P, cell) = [AUC(no mask) - AUC(P)] - [AUC(no mask) - AUC(R)]

    G = median over development demographic cells of excess(P, cell)   (signal removed)
    H = median over development task cells of excess(P, cell)          (task information removed)
    fitness = G - lam * max(0, H - eps)

Both terms subtract the random control, so removing more tissue earns nothing by itself.
"""
import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "02_pathology_task_prediction"))
sys.path.insert(0, str(HERE))

from export_demographic_signal import attention_logits, bag_prediction, load_classifier, load_tiles  # noqa: E402
from mil import load_model, read_manifest  # noqa: E402
from tile_filter import MIN_TILES, drop_top, load_signal, max_drop, random_matched_mask  # noqa: E402

from unveil.fairness import auc  # noqa: E402

SEED_PROGRAM = HERE / "programs" / "seed_program.py"
DEFAULT_CONFIG = HERE / "configs" / "search.yaml"
BLOCK = ("# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END")
INVALID_SCORE = -10.0
CELL_COLUMNS = ["cell", "kind", "role", "manifest", "checkpoint", "signal_dir"]


class CandidateError(ValueError):
    """The program is not a valid filter; it scores INVALID_SCORE."""


class MeasurementError(RuntimeError):
    """The data cannot be measured; this stops the search rather than penalising a program."""


@dataclass(frozen=True)
class Settings:
    cells: pd.DataFrame
    eps: float
    lam: float
    demographic_level: str = "slide"
    task_level: str = "patient"
    max_dose: float = 0.33
    min_cells: int = 3
    min_tasks: int = 1
    control_seed: int = 0


def load_settings(path=None):
    path = Path(path or os.environ.get("UNVEIL_SEARCH_CONFIG") or DEFAULT_CONFIG)
    block = dict((yaml.safe_load(path.read_text()) or {}).get("search") or {})
    if block.get("eps") is None or block.get("lam") is None:
        raise SystemExit(f"{path}: search.eps and search.lam must be fixed before the search starts")
    cells_path = Path(block.pop("cells"))
    cells = pd.read_csv(cells_path if cells_path.is_absolute() else path.parent / cells_path)
    missing = set(CELL_COLUMNS) - set(cells.columns)
    if missing:
        raise SystemExit(f"cell table lacks {sorted(missing)}")
    if not set(cells["kind"]) <= {"demographic", "task"} or not set(cells["role"]) <= {"dev", "held_out"}:
        raise SystemExit("cell kind must be demographic or task, and role dev or held_out")
    return Settings(cells=cells, **block)


def block_parts(text):
    start, end = (text.count(marker) for marker in BLOCK)
    if start != 1 or end != 1:
        raise CandidateError(f"expected one EVOLVE-BLOCK, found {start} start and {end} end markers")
    head, rest = text.split(BLOCK[0])
    body, tail = rest.split(BLOCK[1])
    return head + tail, body


def load_program(path, template=SEED_PROGRAM):
    text = Path(path).read_text()
    outside, body = block_parts(text)
    if outside != block_parts(Path(template).read_text())[0]:
        raise CandidateError("the code outside the EVOLVE-BLOCK was modified")
    name = "candidate_" + hashlib.sha256(body.encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise CandidateError(f"the program does not import: {type(error).__name__}: {error}") from error
    if not callable(getattr(module, "drop_scores", None)):
        raise CandidateError("the program defines no drop_scores")
    return module.drop_scores


def candidate_masks(drop_scores, attention, tile_score, slide_pred, slide_key, max_dose, control_seed=0):
    """The program's keep-mask, the random control at the same dose, and that dose in tiles."""
    n = len(attention)
    if n <= MIN_TILES:
        keep = np.ones(n, dtype=bool)
        return keep, keep, 0
    try:
        scores, dose = drop_scores(attention.astype(np.float64), tile_score.astype(np.float64), float(slide_pred))
        scores, dose = np.asarray(scores, dtype=np.float64), float(dose)
    except Exception as error:
        raise CandidateError(f"{slide_key}: drop_scores raised {type(error).__name__}: {error}") from error
    if scores.shape != (n,) or not np.isfinite(scores).all():
        raise CandidateError(f"{slide_key}: drop_scores must return {n} finite scores")
    if not 0.0 <= dose <= 1.0:
        raise CandidateError(f"{slide_key}: drop fraction {dose} outside [0, 1]")
    n_drop = min(int(np.floor(min(dose, max_dose) * n)), max_drop(n))
    return drop_top(scores, n_drop), random_matched_mask(n, n_drop, slide_key, control_seed), n_drop


class DemographicScorer:
    def __init__(self, checkpoint):
        self.model = load_classifier(checkpoint)

    def prepare(self, x):
        return attention_logits(self.model, x)

    def predict(self, prepared, keep=None):
        return bag_prediction(self.model, *prepared, keep)


class TaskScorer:
    def __init__(self, checkpoint):
        self.model = load_model(checkpoint).eval()

    def prepare(self, x):
        return x[None]

    @torch.inference_mode()
    def predict(self, prepared, keep=None):
        mask = torch.ones(prepared.shape[:2], dtype=torch.bool) if keep is None else torch.as_tensor(keep, dtype=torch.bool)[None]
        return float(torch.softmax(self.model(prepared, mask), dim=1)[0, 1])


SCORERS = {"demographic": DemographicScorer, "task": TaskScorer}


def cell_predictions(drop_scores, cell, settings, control_seed):
    """Predictions without a mask, under the program and under the control, for every slide of a cell."""
    manifest = read_manifest(cell.manifest)
    rows = []
    for fold, slides in manifest.groupby("fold", sort=True):
        checkpoint = cell.checkpoint.format(fold=fold)
        if not os.path.exists(checkpoint):
            raise MeasurementError(f"{cell.cell}: no checkpoint for fold {fold} at {checkpoint}")
        scorer = SCORERS[cell.kind](checkpoint)
        for slide in slides.itertuples():
            x = load_tiles(slide.feature_path)
            attention, tile_score, slide_pred = load_signal(Path(cell.signal_dir) / f"{slide.slide_id}.npz")
            if len(attention) != len(x):
                raise MeasurementError(f"{cell.cell}/{slide.slide_id}: {len(attention)} signal values for {len(x)} tiles")
            keep, control, n_drop = candidate_masks(drop_scores, attention, tile_score, slide_pred, slide.slide_id, settings.max_dose, control_seed)
            prepared = scorer.prepare(x)
            rows.append({"slide_id": slide.slide_id, "patient_id": slide.patient_id, "label": int(slide.label), "dose": n_drop / len(x),
                         "none": scorer.predict(prepared), "program": scorer.predict(prepared, keep), "control": scorer.predict(prepared, control)})
    return pd.DataFrame(rows)


def excess_auc_drop(predictions, level):
    """AUC removed by the program beyond the dose-matched control; positive means signal removed."""
    if level == "patient":
        predictions = predictions.sort_values("slide_id", kind="stable").drop_duplicates("patient_id")
    y = predictions["label"].to_numpy()
    aucs = {arm: auc(y, predictions[arm].to_numpy()) for arm in ("none", "program", "control")}
    return {"excess": aucs["control"] - aucs["program"], **{f"auc_{arm}": value for arm, value in aucs.items()},
            "n": len(predictions), "mean_dose": float(predictions["dose"].mean())}


def median_of(values, min_n, what):
    finite = [v for v in values.values() if np.isfinite(v)]
    if len(finite) < min_n:
        raise MeasurementError(f"only {len(finite)} of {len(values)} {what} could be scored; at least {min_n} are needed")
    return float(np.median(finite))


def fitness(G, H, eps, lam):
    return G - lam * max(0.0, H - eps)


def score_program(drop_scores, settings, role="dev", control_seed=None):
    control_seed = settings.control_seed if control_seed is None else control_seed
    levels = {"demographic": settings.demographic_level, "task": settings.task_level}
    rows = []
    for cell in settings.cells[settings.cells["role"] == role].itertuples():
        result = excess_auc_drop(cell_predictions(drop_scores, cell, settings, control_seed), levels[cell.kind])
        rows.append({"cell": cell.cell, "kind": cell.kind, "role": role, "control_seed": control_seed, **result})
    table = pd.DataFrame(rows)
    min_cells, min_tasks = (settings.min_cells, settings.min_tasks) if role == "dev" else (1, 1)
    G = median_of(dict(table.loc[table["kind"] == "demographic", ["cell", "excess"]].values), min_cells, "demographic cells")
    H = median_of(dict(table.loc[table["kind"] == "task", ["cell", "excess"]].values), min_tasks, "task cells")
    return {"G": G, "H": H, "fitness": fitness(G, H, settings.eps, settings.lam), "mean_dose": float(table["mean_dose"].mean())}, table


def evaluate(program_path):
    """Entry point called by the program search for every candidate."""
    settings = load_settings()
    try:
        summary, _ = score_program(load_program(program_path), settings)
    except CandidateError as error:
        return {"combined_score": INVALID_SCORE, "valid": 0.0, "error": str(error)}
    return {"combined_score": summary["fitness"], "valid": 1.0, **summary}
