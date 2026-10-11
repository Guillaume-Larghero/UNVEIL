import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_demographic_signal_agentic_modulation"))
import tile_filter as tf


def signals(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.gamma(1.0, size=n), rng.uniform(size=n), float(rng.uniform())


@pytest.mark.parametrize("n", [64, 65, 100, 194, 1000, 5000])
def test_dose_and_floor(n):
    attention, tile_score, slide_pred = signals(n)
    masks = {m: tf.tile_mask(m, attention, tile_score, slide_pred, "slide") for m in tf.METHODS}
    kept = {m: int(mask.sum()) for m, mask in masks.items()}
    assert len(set(kept.values())) == 1
    if n <= tf.MIN_TILES:
        assert kept["signal_aware"] == n
    else:
        assert kept["signal_aware"] == n - min(int(np.floor(tf.DROP_FRACTION * n)), tf.max_drop(n))
        assert kept["signal_aware"] >= tf.keep_floor(n)


def test_signal_aware_drops_high_evidence_attended_tiles():
    n = 500
    attention = np.linspace(0, 1, n)
    tile_score = np.linspace(0, 1, n)
    mask = tf.signal_aware_mask(attention, tile_score, slide_pred=0.9)
    assert not mask[-1] and mask[0]
    flipped = tf.signal_aware_mask(attention, tile_score, slide_pred=0.1)
    assert flipped[-1]


def test_random_control_is_reproducible_per_slide():
    attention, tile_score, slide_pred = signals(800)
    first = tf.tile_mask("random_matched", attention, tile_score, slide_pred, "slide_a", seed=3)
    again = tf.tile_mask("random_matched", attention, tile_score, slide_pred, "slide_a", seed=3)
    other = tf.tile_mask("random_matched", attention, tile_score, slide_pred, "slide_b", seed=3)
    assert np.array_equal(first, again) and not np.array_equal(first, other)
