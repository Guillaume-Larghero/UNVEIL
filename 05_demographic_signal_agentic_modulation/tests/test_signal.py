"""The signal step: the chunked gated-ABMIL computation, feature formats, the manifest
guard, custom adapters."""

import json
import os
import sys
import tempfile
import textwrap

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import raises, run_module_tests  # noqa: E402

import torch  # noqa: E402

from unveil_tilefilter.signal import (  # noqa: E402
    SignalError, compute_signals, gated_abmil_signal, load_features, load_signal)

# A small model with the attributes the default adapter reads. It is written to a file so
# the pickled checkpoint can be loaded through --model-code, as a user's would be.
MODEL_CODE = textwrap.dedent('''
    import torch
    from torch import nn

    class DeepAttention(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.theta1 = nn.Linear(d, h)
            self.theta2 = nn.Linear(d, h)
            self.W = nn.Linear(h, 1)
            self.temperature = 1.7

    class TinyABMIL(nn.Module):
        def __init__(self, d=6, h=5):
            super().__init__()
            self.input_dim = d
            self.deep_attention = DeepAttention(d, h)
            self.mlp = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())

    def logit_adapter(model, x):
        da = model.deep_attention
        a = torch.softmax(da.W(torch.tanh(da.theta1(x)) * torch.sigmoid(da.theta2(x))).squeeze(-1), 0)
        lin = model.mlp[0]
        return a.detach(), lin(x).squeeze(-1).detach(), float(lin((a[:, None] * x).sum(0)))
''')


def _setup(tmp, sizes=(5, 33, 150)):
    code = os.path.join(tmp, "code")
    os.makedirs(code)
    with open(os.path.join(code, "tinymodel.py"), "w") as fh:
        fh.write(MODEL_CODE)
    sys.path.insert(0, code)
    import tinymodel

    torch.manual_seed(0)
    model = tinymodel.TinyABMIL()
    ckpt = os.path.join(tmp, "clf.pt")
    torch.save({"model": model}, ckpt)
    feats = os.path.join(tmp, "feats")
    os.makedirs(feats)
    for i, n in enumerate(sizes):
        torch.save(torch.randn(1, n, 6), os.path.join(feats, f"S{i}.pt"))   # (1, n, d) too
    return code, model, ckpt, feats


def _naive(model, x):
    da = model.deep_attention
    a = torch.softmax(da.W(torch.tanh(da.theta1(x)) * torch.sigmoid(da.theta2(x))).squeeze(-1)
                      / da.temperature, 0)
    return (a.detach().numpy(), model.mlp(x).squeeze(-1).detach().numpy(),
            float(model.mlp((a[:, None] * x).sum(0, keepdim=True)).ravel()[0]))


def test_gated_abmil_matches_unchunked():
    with tempfile.TemporaryDirectory() as tmp:
        _code, model, _ckpt, _feats = _setup(tmp)
        x = torch.randn(1000, 6)
        for chunk in (7, 64, 2048):
            att, tile, pred = gated_abmil_signal(model, x, chunk)
            a0, t0, p0 = _naive(model, x)
            assert np.allclose(att, a0, atol=1e-7) and np.allclose(tile, t0, atol=1e-7)
            assert abs(pred - p0) < 1e-6
            assert abs(float(att.sum()) - 1.0) < 1e-5


def test_compute_signals_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        code, model, ckpt, feats = _setup(tmp)
        out = os.path.join(tmp, "sig")
        m = compute_signals(feats, ckpt, out, code_dirs=[code], log=lambda *a: None)
        assert m["slides_computed"] == 3 and m["slides_failed"] == 0
        for i, n in enumerate((5, 33, 150)):
            s = load_signal(out, f"S{i}")
            assert s.n_tiles == n and s.attention.dtype == np.float32
            x = load_features(os.path.join(feats, f"S{i}.pt"))
            a0, t0, p0 = gated_abmil_signal(model, x)
            assert np.array_equal(s.attention, a0.astype(np.float16).astype(np.float32))
            assert np.array_equal(s.tile_score, t0.astype(np.float16).astype(np.float32))
            assert s.slide_pred == float(np.float32(p0))
        # a second run keeps the existing slides; other settings are refused
        m2 = compute_signals(feats, ckpt, out, code_dirs=[code], log=lambda *a: None)
        assert m2["slides_kept"] == 3 and m2["slides_computed"] == 0
        assert raises(SignalError, compute_signals, feats, ckpt, out, code_dirs=[code],
                      chunk=64, log=lambda *a: None)
        assert json.load(open(os.path.join(out, "signal_manifest.json")))["chunk"] == 2048


def test_one_directory_never_mixes_two_classifiers():
    with tempfile.TemporaryDirectory() as tmp:
        code, model, ckpt, feats = _setup(tmp, sizes=(5, 33, 150, 80))
        import tinymodel

        torch.manual_seed(1)
        ckpt_b = os.path.join(tmp, "clf_b.pt")
        torch.save({"model": tinymodel.TinyABMIL()}, ckpt_b)
        out = os.path.join(tmp, "sig")
        quiet = dict(code_dirs=[code], log=lambda *a: None)
        # classifier A stops after two slides (as a killed job would) ...
        compute_signals(feats, ckpt, out, slides=["S0", "S1"], **quiet)
        m = json.load(open(os.path.join(out, "signal_manifest.json")))
        assert m["status"] == "finished" and m["slides_computed"] == 2
        # ... and B may not finish the directory, with or without a partial --overwrite
        assert raises(SignalError, compute_signals, feats, ckpt_b, out, **quiet)
        assert raises(SignalError, compute_signals, feats, ckpt_b, out, slides=["S2", "S3"],
                      overwrite=True, **quiet)
        # A continues where it stopped; B may only recompute the whole directory
        assert compute_signals(feats, ckpt, out, **quiet)["slides_computed"] == 2
        mb = compute_signals(feats, ckpt_b, out, overwrite=True, **quiet)
        assert mb["slides_computed"] == 4 and mb["checkpoint_md5"] != m["checkpoint_md5"]
        model_b = torch.load(ckpt_b, weights_only=False)["model"]
        for i in range(4):
            x = load_features(os.path.join(feats, f"S{i}.pt"))
            att = gated_abmil_signal(model_b, x)[0].astype(np.float16).astype(np.float32)
            assert np.array_equal(load_signal(out, f"S{i}").attention, att)
        # signal files of unknown origin (no manifest) are refused too
        os.remove(os.path.join(out, "signal_manifest.json"))
        assert raises(SignalError, compute_signals, feats, ckpt, out, **quiet)


def test_manifest_is_written_before_the_first_slide():
    with tempfile.TemporaryDirectory() as tmp:
        code, model, ckpt, feats = _setup(tmp)
        out = os.path.join(tmp, "sig")
        seen = []

        def log(msg):
            seen.append(os.path.exists(os.path.join(out, "signal_manifest.json")))

        # "0bad" sorts first, so it is the first slide processed; it fails (wrong feature
        # dimension), and the manifest must already exist when that failure is logged
        torch.save(torch.randn(10, 7), os.path.join(feats, "0bad.pt"))
        m = compute_signals(feats, ckpt, out, code_dirs=[code], log=log)
        assert m["slides_failed"] == 1 and m["failures"][0]["folder_id"] == "0bad"
        assert seen and all(seen)


def test_load_refuses_hand_written_logits():
    with tempfile.TemporaryDirectory() as tmp:
        np.savez(os.path.join(tmp, "X.npz"), attention=np.full(4, 0.25, np.float16),
                 tile_score=np.array([-2.0, 0.5, 3.0, 1.0], np.float16),
                 slide_pred=np.float32(0.4), n_tiles=np.int32(4))
        assert raises(SignalError, load_signal, tmp, "X")


def test_adapter_and_activation():
    with tempfile.TemporaryDirectory() as tmp:
        code, model, ckpt, feats = _setup(tmp)
        out = os.path.join(tmp, "sig")
        # the adapter returns logits: refused without --activation sigmoid
        m = compute_signals(feats, ckpt, out, code_dirs=[code],
                            adapter="tinymodel:logit_adapter", log=lambda *a: None)
        assert m["slides_failed"] == 3
        m = compute_signals(feats, ckpt, out, code_dirs=[code], adapter="tinymodel:logit_adapter",
                            activation="sigmoid", overwrite=True, log=lambda *a: None)
        assert m["slides_computed"] == 3 and m["slides_failed"] == 0
        assert raises(SignalError, compute_signals, feats, ckpt, out, code_dirs=[code],
                      adapter="tinymodel", log=lambda *a: None)


def test_feature_formats():
    import h5py

    with tempfile.TemporaryDirectory() as tmp:
        x = np.random.default_rng(0).random((12, 3)).astype(np.float32)
        np.save(os.path.join(tmp, "a.npy"), x)
        with h5py.File(os.path.join(tmp, "a.h5"), "w") as fh:
            fh["features"] = x
            fh["other"] = x[:2]
        torch.save(torch.from_numpy(x).double(), os.path.join(tmp, "a.pt"))
        for ext in (".npy", ".h5", ".pt"):
            y = load_features(os.path.join(tmp, "a" + ext))
            assert y.dtype == torch.float32 and np.allclose(y.numpy(), x)
        assert load_features(os.path.join(tmp, "a.h5"), h5_key="other").shape == (2, 3)
        assert raises(SignalError, load_features, os.path.join(tmp, "a.h5"), "nope")


if __name__ == "__main__":
    sys.exit(run_module_tests(globals()))
