import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

pytest.importorskip("yaml")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_demographic_signal_agentic_modulation"))
import agentic_search as search
import export_demographic_signal as export
import tile_filter as tf

PROGRAMS = search.HERE / "programs"


def signals(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.gamma(1.0, size=n), rng.uniform(size=n), float(rng.uniform())


def edited(tmp_path, old, new):
    path = tmp_path / "candidate.py"
    path.write_text(search.SEED_PROGRAM.read_text().replace(old, new))
    return path


def test_frozen_program_reproduces_the_released_filter():
    drop_scores = search.load_program(PROGRAMS / "frozen_program.py")
    for n in (65, 300, 2000):
        attention, tile_score, slide_pred = signals(n, seed=n)
        keep, control, n_drop = search.candidate_masks(drop_scores, attention, tile_score, slide_pred, "slide", max_dose=0.33)
        assert np.array_equal(keep, tf.tile_mask("signal_aware", attention, tile_score, slide_pred, "slide"))
        assert np.array_equal(control, tf.tile_mask("random_matched", attention, tile_score, slide_pred, "slide"))
        assert n - keep.sum() == n - control.sum() == n_drop


def test_dose_is_capped_and_small_slides_are_untouched(tmp_path):
    greedy = search.load_program(edited(tmp_path, "0.2 * rank, 0.30", "0.2 * rank, 1.0"))
    attention, tile_score, slide_pred = signals(1000)
    keep, _, n_drop = search.candidate_masks(greedy, attention, tile_score, slide_pred, "slide", max_dose=0.33)
    assert n_drop == 330 and keep.sum() >= tf.keep_floor(1000)
    keep, control, n_drop = search.candidate_masks(greedy, *signals(tf.MIN_TILES), "slide", max_dose=0.33)
    assert n_drop == 0 and keep.all() and control.all()


def test_edits_outside_the_block_are_rejected(tmp_path):
    with pytest.raises(search.CandidateError):
        search.load_program(edited(tmp_path, "import numpy as np", "import numpy as np\nimport os"))
    with pytest.raises(search.CandidateError):
        search.load_program(edited(tmp_path, "# EVOLVE-BLOCK-END", ""))


@pytest.mark.parametrize("body", ["np.full(len(attention), np.nan), 0.3", "np.zeros(3), 0.3", "np.zeros(len(attention)), 1.5"])
def test_malformed_output_is_a_candidate_error(tmp_path, body):
    drop_scores = search.load_program(edited(tmp_path, "0.8 * evidence + 0.2 * rank, 0.30", body))
    with pytest.raises(search.CandidateError):
        search.candidate_masks(drop_scores, *signals(500), "slide", max_dose=0.33)


def test_excess_auc_drop_and_fitness():
    y = np.array([0, 0, 1, 1])
    predictions = pd.DataFrame({"slide_id": list("abcd"), "patient_id": list("abcd"), "label": y, "dose": 0.3,
                                "none": [0.1, 0.2, 0.8, 0.9], "program": [0.9, 0.2, 0.8, 0.1], "control": [0.1, 0.2, 0.8, 0.9]})
    result = search.excess_auc_drop(predictions, "slide")
    assert result["auc_program"] == pytest.approx(0.25)
    assert result["excess"] == pytest.approx(1.0 - 0.25)
    assert search.fitness(0.2, 0.01, eps=0.02, lam=1.0) == 0.2
    assert search.fitness(0.2, 0.05, eps=0.02, lam=2.0) == pytest.approx(0.2 - 0.06)


def test_patient_level_keeps_one_slide_per_patient():
    predictions = pd.DataFrame({"slide_id": ["a2", "a1", "b1"], "patient_id": ["a", "a", "b"], "label": [0, 0, 1], "dose": 0.3,
                                "none": [0.9, 0.1, 0.5], "program": [0.9, 0.1, 0.5], "control": [0.9, 0.1, 0.5]})
    assert search.excess_auc_drop(predictions, "patient")["n"] == 2


class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W, self.theta1, self.theta2 = nn.Linear(8, 1), nn.Linear(d, 8), nn.Linear(d, 8)
        self.temperature = 2.0


class Classifier(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.deep_attention, self.mlp = Head(d), nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())
        self.input_batch_norm_features = None


def test_masked_bag_prediction_equals_prediction_on_kept_tiles():
    torch.manual_seed(0)
    model, x = Classifier().eval(), torch.randn(200, 16)
    keep = np.random.default_rng(0).uniform(size=200) > 0.3
    prepared = export.attention_logits(model, x)
    _, _, full = export.demographic_signal(model, x)
    assert export.bag_prediction(model, *prepared) == pytest.approx(full)
    _, _, kept = export.demographic_signal(model, x[torch.from_numpy(keep)])
    assert export.bag_prediction(model, *prepared, keep) == pytest.approx(kept, abs=1e-6)


def test_task_scorer_mask_matches_dropping_tiles():
    sys.path.insert(0, str(search.HERE.parent / "02_pathology_task_prediction"))
    from mil import AttentionMIL

    torch.manual_seed(0)
    scorer = search.TaskScorer.__new__(search.TaskScorer)
    scorer.model = AttentionMIL(16, dropout=0.0).eval()
    x = torch.randn(150, 16)
    keep = np.random.default_rng(1).uniform(size=150) > 0.4
    masked = scorer.predict(scorer.prepare(x), keep)
    assert masked == pytest.approx(scorer.predict(scorer.prepare(x[torch.from_numpy(keep)])), abs=1e-6)


def test_settings_refuse_unset_penalty(tmp_path):
    config = tmp_path / "search.yaml"
    config.write_text("search:\n  cells: cells.csv\n  eps: null\n  lam: 1.0\n")
    with pytest.raises(SystemExit):
        search.load_settings(config)
