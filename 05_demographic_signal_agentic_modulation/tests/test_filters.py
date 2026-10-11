"""The filters: invariants, parameters, and equality with the paper's locked filter.

The locked filter (reference/filters_locked_c844fcf6.py, the file every training run of
the paper used) is imported as it is and run on the same synthetic signals: 96 slides,
576 masks. With the default parameters every mask must be identical. Under 400 random
non-default parameter sets, `adaptive` must equal the README's formula written out
independently here, and both controls must copy its dose.
"""

import importlib.util
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import ROOT, raises, run_module_tests  # noqa: E402

from unveil_tilefilter.filters import (  # noqa: E402
    FILTERS, FilterParams, FilterViolation, SignalBundle, keep_floor, make_mask, mask_id,
    max_drop)

LOCKED = os.path.join(ROOT, "reference", "filters_locked_c844fcf6.py")
LOCKED_NAME = {"adaptive": "evolved_v0", "random_matched": "random_matched",
               "attention_matched": "attention_matched", "identity": "identity"}
SIZES = (0, 1, 2, 10, 63, 64, 65, 66, 67, 100, 128, 129, 1000, 1345, 4000, 20000)


def _locked():
    spec = importlib.util.spec_from_file_location("filters_locked", LOCKED)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _signals(seed=0):
    """Synthetic signals stored and read back the way the paper's were (float16 ->
    float32): peaked softmax attention with float16 zeros, tied scores, and slide
    predictions on and around the 0.5 threshold."""
    rng = np.random.default_rng(seed)
    out = []
    preds = (0.5, np.float32(0.49999997), 0.0, 1.0, 0.73, 0.21)
    k = 0
    for n in SIZES:
        for scale in (0.5, 4.0, 12.0):
            logits = rng.standard_normal(n) * scale
            att = np.exp(logits - logits.max()) if n else logits
            att = (att / att.sum()) if n else att
            tile = 1.0 / (1.0 + np.exp(-rng.standard_normal(n) * 2.0))
            if n and scale == 12.0:
                tile[: n // 3] = tile[0]                      # many exact ties
            pred = float(np.float32(preds[k % len(preds)]))
            k += 1
            att16 = np.asarray(att, dtype=np.float16).astype(np.float32)
            tile16 = np.asarray(tile, dtype=np.float16).astype(np.float32)
            out.append((f"S{k:03d}-{n}", att16, tile16, pred))
    return out


def test_defaults_are_the_papers():
    p = FilterParams()
    assert (p.drop_frac, p.evidence_base, p.attention_gain, p.attention_bonus,
            p.decision_threshold, p.min_tiles, p.min_keep_frac) == \
        (0.33, 0.25, 0.75, 0.02, 0.5, 64, 0.05)


def test_equal_to_locked_filter():
    L = _locked()
    compared = 0
    for data_seed in (0, 1):
        for fid, att, tile, pred in _signals(data_seed):
            n = att.shape[0]
            new_sig = SignalBundle(att, tile, pred, n, fid)
            old_sig = SimpleNamespace(attention=att, tile_score=tile, slide_pred=pred,
                                      n_tiles=n, folder_id=fid)
            for name in FILTERS:
                for seed in ((0, 7, 12345) if name == "random_matched" else (0,)):
                    ctx = L.TileContext("CANCER", "Race", "FS", "FM", fid, n, seed)
                    old = L.apply_filter(LOCKED_NAME[name], None, old_sig, ctx)
                    new = make_mask(new_sig, name, FilterParams(), seed)
                    assert new.dtype == np.bool_ and np.array_equal(old, new), \
                        (name, fid, seed, int(old.sum()), int(new.sum()))
                    compared += 1
    assert compared == 2 * len(SIZES) * 3 * 6, compared


def test_invariants():
    p = FilterParams()
    for fid, att, tile, pred in _signals(3):
        n = att.shape[0]
        sig = SignalBundle(att, tile, pred, n, fid)
        drops = set()
        for name in FILTERS:
            m = make_mask(sig, name)
            assert m.shape == (n,) and m.dtype == np.bool_
            if n <= p.min_tiles:
                assert m.all(), (name, n)
            else:
                assert int(m.sum()) >= keep_floor(n, p)
            if name != "identity":
                drops.add(int(n - m.sum()))
        assert len(drops) == 1, drops                # matched controls copy the dose


def test_dose_is_floor_of_drop_frac():
    sig = SignalBundle(np.full(1000, 0.001, np.float32), np.linspace(0, 1, 1000), 0.9, 1000,
                       "X")
    assert int((~make_mask(sig, "adaptive")).sum()) == 330
    p = FilterParams(drop_frac=0.1)
    assert int((~make_mask(sig, "adaptive", p)).sum()) == 100
    # a dose above the budget is capped so the keep floor holds
    p = FilterParams(drop_frac=0.99)
    assert int(make_mask(sig, "adaptive", p).sum()) == keep_floor(1000, p) == 64
    assert max_drop(1000, p) == 936


def test_adaptive_drops_evidence_with_attention():
    n = 200
    att = np.full(n, 1.0 / n, np.float32)
    att[:20] = 0.02                                         # attended tiles
    tile = np.full(n, 0.5, np.float32)
    tile[:20] = 0.99                                        # strong group-1 evidence
    tile[20:40] = 0.99                                      # evidence, little attention
    m = make_mask(SignalBundle(att, tile, 0.9, n, "X"), "adaptive",
                  FilterParams(drop_frac=0.1))
    assert not m[:20].any() and m[20:].all()
    # the same tiles are evidence for group 0 when the slide is predicted group 0
    m0 = make_mask(SignalBundle(att, 1.0 - tile, 0.1, n, "X"), "adaptive",
                   FilterParams(drop_frac=0.1))
    assert np.array_equal(m, m0)


def test_random_matched_seed_and_slide():
    n = 500
    sig = SignalBundle(np.full(n, 1 / n, np.float32), np.random.default_rng(0).random(n), 0.6,
                       n, "A")
    a = make_mask(sig, "random_matched", seed=0)
    assert np.array_equal(a, make_mask(sig, "random_matched", seed=0))
    assert not np.array_equal(a, make_mask(sig, "random_matched", seed=1))
    sig_b = SignalBundle(sig.attention, sig.tile_score, 0.6, n, "B")
    assert not np.array_equal(a, make_mask(sig_b, "random_matched", seed=0))


def test_validation():
    assert raises(ValueError, FilterParams(drop_frac=1.0).validate)
    assert raises(ValueError, FilterParams(drop_frac=-0.1).validate)
    assert raises(ValueError, FilterParams(min_tiles=-1).validate)
    assert raises(ValueError, FilterParams(min_tiles=2.5).validate)
    assert raises(ValueError, FilterParams(decision_threshold=1.5).validate)
    assert raises(ValueError, FilterParams(evidence_base=float("nan")).validate)
    assert raises(ValueError, FilterParams.from_dict, {"drop_fraction": 0.3})
    sig = SignalBundle(np.ones(100, np.float32) / 100, np.zeros(100), 0.5, 100, "X")
    assert raises(ValueError, make_mask, sig, "no_such_filter")
    assert raises(FilterViolation, SignalBundle, np.ones(3), np.ones(4), 0.5)
    assert raises(FilterViolation, SignalBundle, np.ones(3), np.ones(3), 0.5, 4)


def _formula(att, tile, pred, p):
    """The README's formula, written out independently of the package."""
    att = np.asarray(att, dtype=np.float64)
    tile = np.asarray(tile, dtype=np.float64)
    n = att.shape[0]
    if n <= p.min_tiles:
        return np.ones(n, dtype=bool)
    evidence = tile if pred >= p.decision_threshold else 1.0 - tile
    values = np.unique(att)                                  # sorted distinct values
    rank = np.searchsorted(values, att) / float(len(values) - 1 if len(values) > 1 else 1)
    score = evidence * (p.evidence_base + p.attention_gain * rank) + p.attention_bonus * rank
    drop = min(int(np.floor(p.drop_frac * n)),
               n - max(p.min_tiles, int(np.ceil(p.min_keep_frac * n))))
    order = sorted(range(n), key=lambda i: (-score[i], i))
    mask = np.ones(n, dtype=bool)
    mask[order[:max(drop, 0)]] = False
    return mask


def test_non_default_parameters_follow_the_formula():
    rng = np.random.default_rng(11)
    sigs = _signals(5)
    checked = 0
    for k in range(400):
        p = FilterParams(drop_frac=float(rng.uniform(0, 0.9)),
                         evidence_base=float(rng.uniform(-1, 1)),
                         attention_gain=float(rng.uniform(-1, 1)),
                         attention_bonus=float(rng.uniform(-1, 1)),
                         decision_threshold=float(rng.uniform(0, 1)),
                         min_tiles=int(rng.integers(0, 200)),
                         min_keep_frac=float(rng.uniform(0, 0.5)))
        fid, att, tile, pred = sigs[k % len(sigs)]
        if att.shape[0] > 4000:
            continue
        sig = SignalBundle(att, tile, pred, att.shape[0], fid)
        new = make_mask(sig, "adaptive", p)
        assert np.array_equal(new, _formula(att, tile, pred, p)), (k, p)
        dose = int((~new).sum())
        for ctrl in ("random_matched", "attention_matched"):
            assert int((~make_mask(sig, ctrl, p, seed=k)).sum()) == dose
        checked += 1
    assert checked > 300, checked


def test_mask_id():
    a = mask_id("adaptive")
    assert len(a) == 16 and a == mask_id("adaptive", FilterParams(), seed=99)
    assert a != mask_id("adaptive", FilterParams(drop_frac=0.3))
    assert a != mask_id("attention_matched")
    assert mask_id("random_matched", seed=0) != mask_id("random_matched", seed=1)


if __name__ == "__main__":
    sys.exit(run_module_tests(globals()))
