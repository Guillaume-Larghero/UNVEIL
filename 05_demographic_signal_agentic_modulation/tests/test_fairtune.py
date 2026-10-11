"""The fairtune_pipeline adapter, against a small stand-in for the pipeline.

The stand-in has the same shape as the real one where the adapter touches it:
CancerDataset(df, task, fold_idx, ..., max_train_tiles) whose __getitem__ subsamples
max_train_tiles and returns (bag, len, label), a ReweightDataset subclass, a
get_datasets factory, and should_emit_reweight_suffix. Its source differs from the
tested pipeline, so runs need --allow-upstream-drift -- which is itself tested. Runs go
through the command line in a subprocess (the adapter imports the pipeline's modules
and changes directory, which must not leak between tests).

Covered: the overlay (rebuilding it under a reader, and two jobs making the same link), a
filtered run, the ReweightDataset path, a baseline run, the drift stop, --inference with
the pipeline's --inference_only or --sig_test_only, the census failure when nothing was
filtered, a failing run's record, a missing mask, and the refusals.

Tests against the real pipeline: verification/eq_fairtune.py.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import SRC, raises, run_module_tests  # noqa: E402

import torch  # noqa: E402

from unveil_tilefilter import fairtune as FT  # noqa: E402
from unveil_tilefilter.masks import build_masks, write_mask_file  # noqa: E402
from unveil_tilefilter.signal import save_signal  # noqa: E402

DATASET = textwrap.dedent('''
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import Dataset

    PARTITION = {0: "training", 1: "validation", 2: "testing", None: "all"}

    class CancerDataset(Dataset):
        def __init__(self, df, task, fold_idx=None, feature_type="tile", split_type="kfold",
                     exp_idx=0, add_sensitive=False, max_train_tiles=None,
                     custom_output_col=None, preload=False):
            self.df = df.reset_index(drop=True)
            self.partition_type = PARTITION[fold_idx]
            self.max_train_tiles = max_train_tiles
            self.add_sensitive = add_sensitive

        def __getitem__(self, idx):
            row = self.df.loc[idx]
            sample = torch.load(row.path, weights_only=True)
            if self.max_train_tiles and len(sample) > self.max_train_tiles:
                sel = np.random.choice(len(sample), self.max_train_tiles, replace=False)
                sample = sample[sel, :]
            return sample, len(sample), int(row.label)

        def __len__(self):
            return len(self.df)

    class ReweightDataset(CancerDataset):
        pass

    def get_datasets(df, task, split_type, exp_idx, feature_type="tile",
                     reweight_method="none", max_train_tiles=None, preload=False):
        train_cls = ReweightDataset if reweight_method != "none" else CancerDataset
        return [train_cls(df, task, fold_idx=0, max_train_tiles=max_train_tiles),
                CancerDataset(df, task, fold_idx=1, max_train_tiles=max_train_tiles),
                CancerDataset(df, task, fold_idx=2, max_train_tiles=max_train_tiles)]
''')

FRAMEWORK = textwrap.dedent('''
    def should_emit_reweight_suffix(args):
        return bool(getattr(args, "pretrained", False))
''')

MAIN = textwrap.dedent('''
    import argparse
    import json
    import os
    import yaml
    from dataset import *
    from framework import *

    p = argparse.ArgumentParser()
    p.add_argument("--cancer", nargs="+")
    p.add_argument("--train_method")
    p.add_argument("--model_path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_train_tiles", type=int, default=None)
    p.add_argument("--reweight_method", default="none")
    p.add_argument("--fail", action="store_true")
    p.add_argument("--inference_only", action="store_true")
    p.add_argument("--sig_test_only", action="store_true")
    p.add_argument("--no_train_partition", action="store_true")
    args = p.parse_args()
    cfg = yaml.load(open("configs/train_configs.yaml"), Loader=yaml.FullLoader)
    args.pretrained = cfg[args.train_method]["pretrained"]
    df = pd.read_csv(os.path.join(args.model_path, "df.csv"))
    if args.sig_test_only:                      # like the pipeline: reads saved results only
        out = {"suffix": should_emit_reweight_suffix(args), "sig_test_only": True}
    else:
        if args.no_train_partition:             # a stand-in that builds no training data
            tr, va, te = (CancerDataset(df, 4, fold_idx=k) for k in (None, 1, 2))
        else:
            tr, va, te = get_datasets(df, 4, "kfold", 0, reweight_method=args.reweight_method,
                                      max_train_tiles=args.max_train_tiles)
        if args.fail:
            raise RuntimeError("training failed")
        out = {"suffix": should_emit_reweight_suffix(args), "cfg": cfg[args.train_method],
               "classes": [type(d).__name__ for d in (tr, va, te)],
               "train": [tr[i][1] for i in range(len(tr))],
               "valid": [va[i][1] for i in range(len(va))],
               "test": [te[i][1] for i in range(len(te))]}
    json.dump(out, open(os.path.join(args.model_path, "out.json"), "w"))
''')

CONFIG = textwrap.dedent('''
    baseline:
      reweight: False # whether to reweight
      selection: AUROC
      pretrained: False
      finetune_layer_names:
        - fc_target.8
    reweightingClassifier:
      reweight: True
      selection: AUROC
      pretrained: False
''')

SIZES = (30, 120, 400, 900)


def _world(tmp):
    up = os.path.join(tmp, "fairtune_pipeline")
    os.makedirs(os.path.join(up, "configs"))
    os.makedirs(os.path.join(up, "mutation_models"))           # a results tree: not linked
    for name, text in (("dataset.py", DATASET), ("framework.py", FRAMEWORK),
                       ("main_genetic.py", MAIN), ("main.py", MAIN),
                       ("configs/train_configs.yaml", CONFIG)):
        with open(os.path.join(up, name), "w") as fh:
            fh.write(text)
    feats = os.path.join(tmp, "feats")
    sig = os.path.join(tmp, "signals")
    os.makedirs(feats)
    rng = np.random.default_rng(0)
    rows = []
    for i, n in enumerate(SIZES):
        fid = f"TCGA-{i:02d}"
        torch.save(torch.randn(n, 4), os.path.join(feats, fid + ".pt"))
        att = rng.random(n)
        save_signal(os.path.join(sig, fid + ".npz"), att / att.sum(), rng.random(n),
                    float(rng.random()))
        rows.append(f"{fid},{os.path.join(feats, fid + '.pt')},{i % 2}")
    mp = os.path.join(tmp, "results", "run1")
    os.makedirs(mp)
    with open(os.path.join(mp, "df.csv"), "w") as fh:
        fh.write("folder_id,path,label\n" + "\n".join(rows) + "\n")
    masks = {}
    for name in ("adaptive", "random_matched"):
        m = build_masks(sig, name)
        masks[name] = os.path.join(tmp, f"{name}.npz")
        write_mask_file(masks[name], m, name, None, 0, sig)
    return up, mp, masks


def _cli(tmp, args, passthrough):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(PYTHONPATH=SRC, HOME=tmp, PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "unveil_tilefilter", "fairtune"] + args + ["--"] + passthrough
    return subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=300)


def _record(mp):
    recs = [f for f in os.listdir(mp) if f.startswith("unveil_tilefilter_run_")]
    assert len(recs) == 1, recs
    return os.path.join(mp, recs[0])


def _expected(masks_path, cap):
    with np.load(masks_path) as z:
        return [min(int(z[f"TCGA-{i:02d}"].sum()), cap) for i in range(len(SIZES))]


def test_overlay():
    with tempfile.TemporaryDirectory() as tmp:
        up, _mp, _m = _world(tmp)
        ov = os.path.join(tmp, "ov")
        for _ in range(2):                                     # rebuilding is safe
            FT.build_overlay(up, ov, log=lambda *a: None)
        assert os.path.islink(os.path.join(ov, "dataset.py"))
        assert not os.path.exists(os.path.join(ov, "mutation_models"))
        link = os.path.join(ov, "configs")
        target = os.readlink(link)
        assert os.path.islink(link) and target.startswith(".configs-") and \
            os.path.isdir(os.path.join(ov, target)) and not os.path.islink(os.path.join(ov, target))
        import yaml

        cfg = yaml.load(open(os.path.join(ov, "configs", "train_configs.yaml")),
                        Loader=yaml.FullLoader)
        for m in FT.METHODS:
            assert cfg[m] == cfg["baseline"]
        assert "reweightingClassifier" in cfg
        # the pipeline's own configs are not touched
        assert "agenticFilter" not in open(os.path.join(up, "configs", "train_configs.yaml")).read()
        foreign = os.path.join(tmp, "foreign")
        os.makedirs(foreign)
        open(os.path.join(foreign, "x"), "w").close()
        assert raises(FT.AdapterError, FT.build_overlay, up, foreign, lambda *a: None)
        other = os.path.join(tmp, "other_pipeline")
        os.makedirs(os.path.join(other, "configs"))
        for name in FT.REQUIRED:
            open(os.path.join(other, name), "w").write(CONFIG if name.endswith("yaml") else "")
        assert raises(FT.AdapterError, FT.build_overlay, other, ov, lambda *a: None)


def test_overlay_rebuilds_without_disturbing_readers():
    """Concurrent runs share an overlay: an unchanged config copy is left alone, a changed
    one is swapped in atomically, and a reader never finds configs/ missing."""
    import threading

    with tempfile.TemporaryDirectory() as tmp:
        up, _mp, _m = _world(tmp)
        ov = os.path.join(tmp, "ov")
        FT.build_overlay(up, ov, log=lambda *a: None)
        link = os.path.join(ov, "configs")
        first = os.readlink(link)
        ino = os.stat(os.path.join(ov, first)).st_ino
        FT.build_overlay(up, ov, log=lambda *a: None)
        assert os.readlink(link) == first and os.stat(os.path.join(ov, first)).st_ino == ino
        errors, reads, stop = [], [0], threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    with open(os.path.join(link, "train_configs.yaml")) as fh:
                        assert "agenticFilter" in fh.read()
                    reads[0] += 1
                except Exception as e:                        # noqa: BLE001
                    errors.append(repr(e))

        t = threading.Thread(target=reader)
        t.start()
        cfg = os.path.join(up, "configs", "train_configs.yaml")
        base = open(cfg).read()
        for i in range(12):                                   # the pipeline's config changes
            open(cfg, "w").write(base + f"\nextra_{i % 3}:\n  reweight: False\n")
            FT.build_overlay(up, ov, log=lambda *a: None)
        stop.set()
        t.join()
        assert not errors and reads[0] > 0, (errors[:3], reads)
        assert os.readlink(link) != first and os.path.isdir(os.path.join(ov, first))


def test_link_made_by_another_job_at_once():
    """Two jobs building one overlay can both try to make the same link: the second finds
    it made and carries on; any other entry under that name is still an error."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "dataset.py"), os.path.join(tmp, "ov_dataset.py")
        open(src, "w").close()
        os.symlink(src, dst)                                  # the other job got there first
        FT._link(src, dst)
        assert os.readlink(dst) == src
        assert raises(FileExistsError, FT._link, os.path.join(tmp, "other.py"), dst)
        real = os.path.join(tmp, "real.py")
        open(real, "w").close()
        assert raises(FileExistsError, FT._link, src, real)


def test_filtered_run():
    with tempfile.TemporaryDirectory() as tmp:
        up, mp, masks = _world(tmp)
        r = _cli(tmp, ["--upstream", up, "--masks", masks["adaptive"],
                       "--allow-upstream-drift"],
                 ["--cancer", "brca", "--train_method", "agenticFilter", "--model_path", mp,
                  "--max_train_tiles", "200"])
        assert r.returncode == 0, r.stdout + r.stderr
        out = json.load(open(os.path.join(mp, "out.json")))
        assert out["train"] == _expected(masks["adaptive"], 200), out
        assert out["valid"] == [min(n, 200) for n in SIZES] == out["test"]
        assert out["suffix"] is True and out["cfg"]["pretrained"] is False
        assert out["classes"][0] == "UnveilCancerDataset"
        rec = json.load(open(_record(mp)))
        assert rec["status"] == "finished" and rec["filter"] == "adaptive"
        assert rec["census"] == {"training": {"built": 1, "armed": 1},
                                 "validation": {"built": 1, "armed": 0},
                                 "testing": {"built": 1, "armed": 0}}
        assert rec["upstream_check"]["match"] is False and rec["allow_upstream_drift"] is True


def test_reweight_dataset_is_filtered_too():
    with tempfile.TemporaryDirectory() as tmp:
        up, mp, masks = _world(tmp)
        r = _cli(tmp, ["--upstream", up, "--masks", masks["random_matched"],
                       "--allow-upstream-drift"],
                 ["--cancer", "brca", "--train_method", "agenticRandomFilter",
                  "--model_path", mp, "--reweight_method", "oversample"])
        assert r.returncode == 0, r.stdout + r.stderr
        out = json.load(open(os.path.join(mp, "out.json")))
        assert out["classes"][0] == "UnveilReweightDataset"
        assert out["train"] == _expected(masks["random_matched"], 10 ** 9)


def test_unfiltered_run_and_drift_stop():
    with tempfile.TemporaryDirectory() as tmp:
        up, mp, _masks = _world(tmp)
        args = ["--cancer", "brca", "--train_method", "baseline", "--model_path", mp]
        r = _cli(tmp, ["--upstream", up], args)
        assert r.returncode == 2 and "differ from the version" in r.stderr, r.stderr
        r = _cli(tmp, ["--upstream", up, "--allow-upstream-drift"], args)
        assert r.returncode == 0, r.stdout + r.stderr
        out = json.load(open(os.path.join(mp, "out.json")))
        assert out["train"] == list(SIZES) and out["suffix"] is False
        rec = json.load(open(_record(mp)))
        assert rec["status"] == "finished" and rec["masks"] == []
        assert all(d["armed"] == 0 for d in rec["census"].values())


def test_inference_scores_the_filtered_model_unfiltered():
    with tempfile.TemporaryDirectory() as tmp:
        up, mp, _masks = _world(tmp)
        r = _cli(tmp, ["--upstream", up, "--inference", "--allow-upstream-drift"],
                 ["--cancer", "brca", "--train_method", "agenticFilter", "--model_path", mp,
                  "--inference_only"])
        assert r.returncode == 0, r.stdout + r.stderr
        out = json.load(open(os.path.join(mp, "out.json")))
        assert out["suffix"] is True                    # finds the agenticFilter model's folder
        assert out["train"] == list(SIZES) and out["valid"] == list(SIZES)
        recs = [f for f in os.listdir(mp) if f.startswith("unveil_tilefilter_run_")]
        rec = json.load(open(os.path.join(mp, recs[0])))
        assert len(recs) == 1 and rec["mode"] == "inference" and rec["status"] == "finished"
        assert all(d["armed"] == 0 for d in rec["census"].values())
        assert "nothing was filtered, training included (asserted)" in r.stdout, r.stdout


def test_sig_test_only_reruns_the_filtered_models_bias_test():
    """The pipeline's --sig_test_only reads a finished run's saved results and builds no
    dataset: under --inference it finds the filtered model's folder and passes."""
    with tempfile.TemporaryDirectory() as tmp:
        up, mp, _masks = _world(tmp)
        r = _cli(tmp, ["--upstream", up, "--inference", "--allow-upstream-drift"],
                 ["--cancer", "brca", "--train_method", "agenticFilter", "--model_path", mp,
                  "--sig_test_only"])
        assert r.returncode == 0, r.stdout + r.stderr
        out = json.load(open(os.path.join(mp, "out.json")))
        assert out == {"suffix": True, "sig_test_only": True}, out
        rec = json.load(open(_record(mp)))
        assert rec["mode"] == "inference" and rec["status"] == "finished"
        assert rec["census"] == {} and rec["masks"] == []
        assert "no dataset was built" in r.stdout, r.stdout


def test_census_failure_stops_the_run():
    """Masks given, but the pipeline built no training partition: nothing was filtered,
    so the run must fail rather than pass as a filtered result."""
    with tempfile.TemporaryDirectory() as tmp:
        up, mp, masks = _world(tmp)
        r = _cli(tmp, ["--upstream", up, "--masks", masks["adaptive"],
                       "--allow-upstream-drift"],
                 ["--cancer", "brca", "--train_method", "agenticFilter", "--model_path", mp,
                  "--no_train_partition"])
        assert r.returncode == 2 and "no training dataset was filtered" in r.stderr, r.stderr
        rec = json.load(open(os.path.join(mp, [f for f in os.listdir(mp)
                                                if f.startswith("unveil_tilefilter_run_")][0])))
        assert rec["status"] == "failed" and "no training dataset" in rec["error"]


def test_failure_is_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        up, mp, masks = _world(tmp)
        r = _cli(tmp, ["--upstream", up, "--masks", masks["adaptive"],
                       "--allow-upstream-drift"],
                 ["--cancer", "brca", "--train_method", "agenticFilter", "--model_path", mp,
                  "--fail"])
        assert r.returncode != 0 and "training failed" in r.stderr
        rec = json.load(open(_record(mp)))
        assert rec["status"] == "failed" and "training failed" in rec["error"]


def test_missing_mask_stops_training():
    with tempfile.TemporaryDirectory() as tmp:
        up, mp, masks = _world(tmp)
        sig = os.path.join(tmp, "signals")
        part = build_masks(sig, "adaptive", slides=["TCGA-00", "TCGA-01"])
        p = os.path.join(tmp, "partial.npz")
        write_mask_file(p, part, "adaptive", None, 0, sig)
        r = _cli(tmp, ["--upstream", up, "--masks", p, "--allow-upstream-drift"],
                 ["--cancer", "brca", "--train_method", "agenticFilter", "--model_path", mp])
        assert r.returncode == 2 and "have no mask" in r.stderr, r.stderr
        assert not os.path.exists(os.path.join(mp, "out.json"))


def test_refusals_before_anything_runs():
    with tempfile.TemporaryDirectory() as tmp:
        up, mp, masks = _world(tmp)
        cwd = os.getcwd()
        bad = [
            ([masks["adaptive"]], ["--train_method", "agenticRandomFilter", "--model_path", mp]),
            ([], ["--train_method", "agenticFilter", "--model_path", mp]),
            ([masks["adaptive"]], ["--train_method", "baseline", "--model_path", mp]),
            ([masks["adaptive"]], ["--train_method", "agenticFilter", "--model_path", "rel/p"]),
            ([masks["adaptive"]], ["--train_meth", "agenticFilter", "--model_path", mp]),
            ([masks["adaptive"]], ["--train_method", "agenticFilter", "--train_method",
                                   "agenticFilter", "--model_path", mp]),
            ([masks["adaptive"]], ["--cancer", "GBM LGG", "--train_method", "agenticFilter",
                                   "--model_path", mp]),
        ]
        inference = [
            ([masks["adaptive"]], ["--train_method", "agenticFilter", "--model_path", mp,
                                   "--inference_only"]),             # --inference with masks
            ([masks["adaptive"]], ["--train_method", "agenticFilter", "--model_path", mp,
                                   "--sig_test_only"]),
            ([], ["--train_method", "agenticFilter", "--model_path", mp]),  # would train
            ([], ["--train_method", "agenticFilter", "--model_path", mp, "--inference_only",
                  "--sig_test_only"]),                # both: the pipeline would only re-test
        ]
        bad.append(([masks["adaptive"]], ["--train_method", "agenticFilter", "--model_path",
                                          mp, "--inference_only"]))  # training mode + inference
        bad.append(([masks["adaptive"]], ["--train_method", "agenticFilter", "--model_path",
                                          mp, "--sig_test_only"]))   # training mode + bias test
        bad.append(([], ["--train_method", "baseline", "--model_path", mp, "--sig_test_only"]))
        bad.append(([masks["adaptive"]], ["--train_method", "agenticFilter", "--model_path",
                                          mp, "--inference_on"]))    # abbreviation
        bad.append(([masks["adaptive"]], ["--train_method", "agenticFilter", "--model_path",
                                          mp, "--sig_test"]))        # abbreviation
        for flag, cases in ((False, bad), (True, inference)):
            for m, argv in cases:
                assert raises(FT.AdapterError, FT.run, up, "mutation", argv, masks=m,
                              overlay=os.path.join(tmp, "ov"), inference=flag,
                              log=lambda *a: None), (flag, argv)
                assert os.getcwd() == cwd and not os.path.exists(os.path.join(tmp, "ov"))


if __name__ == "__main__":
    sys.exit(run_module_tests(globals()))
