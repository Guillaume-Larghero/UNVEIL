"""Mask building, the mask file and its sidecar, and the read-side MaskSet."""

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import raises, run_module_tests  # noqa: E402

from unveil_tilefilter.filters import FilterParams, SignalBundle, make_mask, mask_id  # noqa: E402
from unveil_tilefilter.masks import MaskError, MaskSet, build_masks, write_mask_file  # noqa: E402
from unveil_tilefilter.signal import SignalError, save_signal  # noqa: E402


def _signal_dir(tmp, sizes=(10, 70, 300, 1000), prefix="SLIDE"):
    rng = np.random.default_rng(5)
    d = os.path.join(tmp, "signals")
    for i, n in enumerate(sizes):
        att = rng.random(n)
        save_signal(os.path.join(d, f"{prefix}-{i}.npz"), att / att.sum(), rng.random(n),
                    float(rng.random()))
    return d


def test_build_and_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        d = _signal_dir(tmp)
        masks = build_masks(d, "adaptive")
        assert sorted(masks) == [f"SLIDE-{i}" for i in range(4)]
        assert masks["SLIDE-0"].all()                          # 10 tiles: exempt
        assert int((~masks["SLIDE-3"]).sum()) == 330
        out = os.path.join(tmp, "m", "cell.npz")
        meta = write_mask_file(out, masks, "adaptive", FilterParams(), 0, d)
        assert meta["mask_id"] == mask_id("adaptive") and meta["seed"] is None
        assert meta["stats"]["slides"] == 4 and meta["stats"]["slides_filtered"] == 3
        side = json.load(open(out + ".json"))
        assert side["params"] == FilterParams().to_dict() and side["filter"] == "adaptive"
        ms = MaskSet([out])
        assert len(ms) == 4 and ms.has("SLIDE-2")
        for k, m in masks.items():
            assert np.array_equal(ms.get(k), m)
        assert ms.meta[0]["mask_id"] == meta["mask_id"]


def test_masks_equal_make_mask_on_reloaded_signal():
    with tempfile.TemporaryDirectory() as tmp:
        d = _signal_dir(tmp)
        for name in ("random_matched", "attention_matched"):
            masks = build_masks(d, name, seed=3)
            for fid, m in masks.items():
                with np.load(os.path.join(d, fid + ".npz")) as z:
                    sig = SignalBundle(z["attention"], z["tile_score"], float(z["slide_pred"]),
                                       int(z["n_tiles"]), fid)
                assert np.array_equal(m, make_mask(sig, name, seed=3))


def test_slide_list_and_feature_check():
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        d = _signal_dir(tmp, sizes=(80, 90))
        assert sorted(build_masks(d, slides=["SLIDE-1"])) == ["SLIDE-1"]
        assert raises(MaskError, build_masks, d, slides=["NOPE"])
        feats = os.path.join(tmp, "feats")
        os.makedirs(feats)
        torch.save(torch.zeros(80, 4), os.path.join(feats, "SLIDE-0.pt"))
        torch.save(torch.zeros(91, 4), os.path.join(feats, "SLIDE-1.pt"))    # wrong length
        assert sorted(build_masks(d, slides=["SLIDE-0"], features_dir=feats)) == ["SLIDE-0"]
        assert raises(MaskError, build_masks, d, features_dir=feats)


def test_maskset_refuses_overlap_and_bad_masks():
    with tempfile.TemporaryDirectory() as tmp:
        d = _signal_dir(tmp)
        masks = build_masks(d)
        a = os.path.join(tmp, "a.npz")
        write_mask_file(a, masks, "adaptive", None, 0, d)
        assert raises(MaskError, MaskSet, [a, a])
        bad = os.path.join(tmp, "bad.npz")
        np.savez(bad, X=np.ones(5, dtype=np.int8))
        assert raises(MaskError, MaskSet, [bad])
        assert raises(MaskError, MaskSet([a]).get, "NOPE")
        assert raises(MaskError, write_mask_file, os.path.join(tmp, "x.zip"), masks,
                      "adaptive", None, 0, d)


def test_signal_validation():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "s.npz")
        assert raises(SignalError, save_signal, p, [0.5, 0.5], [0.2, 1.3], 0.5)   # > 1
        assert raises(SignalError, save_signal, p, [0.5, 0.5], [0.2, 0.3], 2.0)   # logit
        assert raises(SignalError, save_signal, p, [-0.1, 1.1], [0.2, 0.3], 0.5)
        assert raises(SignalError, save_signal, p, [0.5, 0.5], [0.2], 0.5)
        assert raises(SignalError, save_signal, p, [0.5, np.nan], [0.2, 0.3], 0.5)


if __name__ == "__main__":
    sys.exit(run_module_tests(globals()))
