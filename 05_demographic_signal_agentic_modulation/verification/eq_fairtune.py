"""Check the fairtune adapter against the real fairtune_pipeline (no training).

    python verification/eq_fairtune.py --upstream <fairtune_pipeline> \\
        [--reference-config <train_configs.yaml the paper's runs read>]

Run it with the pipeline's own Python environment. In a temporary directory, which it
removes afterwards, it:
  1. builds the overlay, checks that its configs/ is the overlay's own copy with the three
     methods equal to the baseline block and, with --reference-config, that the overlay's
     train_configs.yaml reads back identical to the one the paper's runs used;
  2. imports the pipeline's dataset and framework from the overlay and checks the five
     callables (and five files) the adapter depends on against the tested version;
  3. installs the injection with masks that drop the last 100 of every slide's 500 tiles
     and builds the pipeline's REAL CancerDataset for the training, validation, testing
     and "all" partitions, and its ReweightDataset for training, on synthetic slides
     (vanilla split, task 4);
  4. checks that only the training partition is filtered, that the filter sees the whole
     bag and max_train_tiles is drawn from the kept tiles, that ReweightDataset is
     replaced and filtered too, that the suffix rule is wrapped, that
     `from dataset import *` (as the entrypoints do) picks up the replacement, and that
     the census assertion passes -- and fails when tampered with.
Prints one line per check; exit 1 if any fails.
"""

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
from unveil_tilefilter import fairtune as FT  # noqa: E402
from unveil_tilefilter.masks import MaskSet, write_mask_file  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f" -- {detail}" if detail and not cond
                                                     else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--reference-config", default=None)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="unveil-eq-fairtune-")
    try:
        return _checks(a, tmp)
    finally:
        os.chdir("/")
        shutil.rmtree(tmp, ignore_errors=True)


def _checks(a, tmp):
    import pandas as pd
    import torch
    import yaml

    upstream = os.path.realpath(a.upstream)
    ov = FT.build_overlay(upstream, os.path.join(tmp, "overlay"), log=lambda *x: None)
    cfg = yaml.load(open(os.path.join(ov, "configs", "train_configs.yaml")),
                    Loader=yaml.FullLoader)
    check("three methods = the pipeline's baseline block",
          all(cfg[m] == cfg["baseline"] for m in FT.METHODS))
    check("configs/ is the overlay's own copy",
          os.path.realpath(os.path.join(ov, "configs")).startswith(os.path.realpath(ov) + os.sep))
    if a.reference_config:
        ref = yaml.load(open(a.reference_config), Loader=yaml.FullLoader)
        check("overlay train_configs.yaml reads back identical to the paper's", cfg == ref,
              f"keys only in one: {sorted(set(cfg) ^ set(ref))}")

    os.chdir(ov)
    sys.path.insert(0, ov)
    import dataset
    import framework

    pins = FT.check_upstream(upstream, {"dataset": dataset, "framework": framework})
    check("the five callables match the tested version", pins["match"], pins["drift"])
    check("the five pinned files match the tested version",
          all(pins["files_match_tested"].values()), pins["files_match_tested"])

    n, drop, maxt = 500, 100, 300
    rows = []
    for i, fold in enumerate((0, 0, 1, 2)):
        fid = f"SLIDE-{i:03d}"
        path = os.path.join(tmp, f"{fid}.pt")
        # row r of the bag holds the value r, so a returned bag tells which tiles it has
        torch.save(torch.arange(n, dtype=torch.float32)[:, None].repeat(1, 8), path)
        rows.append({"folder_id": fid, "path": path, "label": i % 2, "sensitive": i % 2,
                     "case_submitter_id": f"CASE-{i:03d}", "fold": fold})
    df = pd.DataFrame(rows)
    keep = np.ones(n, dtype=bool)
    keep[-drop:] = False
    mfile = os.path.join(tmp, "masks.npz")
    write_mask_file(mfile, {r["folder_id"]: keep for r in rows}, "adaptive", None, 0, tmp)

    orig_cancer, orig_reweight = dataset.CancerDataset, dataset.ReweightDataset
    orig_suffix = framework.should_emit_reweight_suffix
    inj = FT.Injection(MaskSet([mfile]), log=lambda *x: None)
    inj.apply(dataset, framework)
    check("CancerDataset replaced by a subclass",
          dataset.CancerDataset.__name__ == "UnveilCancerDataset"
          and issubclass(dataset.CancerDataset, orig_cancer))
    check("ReweightDataset replaced by a subclass",
          dataset.ReweightDataset.__name__ == "UnveilReweightDataset"
          and issubclass(dataset.ReweightDataset, orig_reweight))
    ns = {}
    exec("from dataset import *", ns)                      # what the entrypoints do
    check("`from dataset import *` gets the replacement",
          ns["CancerDataset"] is dataset.CancerDataset
          and ns["ReweightDataset"] is dataset.ReweightDataset)
    check("get_datasets builds from the module attribute",
          dataset.get_datasets.__globals__ is vars(dataset))

    class Args:
        pretrained = False
        model_arch = ""

        def __init__(self, m):
            self.train_method = m

    pre = Args("baseline")
    pre.pretrained = True
    check("suffix: the three methods get one",
          all(framework.should_emit_reweight_suffix(Args(m)) for m in FT.METHODS))
    check("suffix: baseline keeps none, a pretrained run keeps its own",
          framework.should_emit_reweight_suffix(Args("baseline")) is False
          and framework.should_emit_reweight_suffix(pre) is orig_suffix(pre) is True)

    kept_rows = set(range(n - drop))
    tr = dataset.CancerDataset(df.copy(), task=4, fold_idx=0, split_type="vanilla",
                               max_train_tiles=maxt)
    out = tr[0]
    vals = set(out[0][:, 0].long().tolist())
    check("training: max_train_tiles drawn from the kept tiles",
          out[0].shape == (maxt, 8) and out[1] == maxt and vals <= kept_rows
          and len(vals) == maxt, f"shape {tuple(out[0].shape)}")
    check("training: the rest of the item is the pipeline's",
          out[6] == "SLIDE-000" and out[5] == "CASE-000" and len(out) == 8)
    tr2 = dataset.CancerDataset(df.copy(), task=4, fold_idx=0, split_type="vanilla")
    b = tr2[1][0]
    check("training: without a cap, the bag is exactly the kept tiles in order",
          b[:, 0].long().tolist() == list(range(n - drop)))
    rw = dataset.ReweightDataset(df.copy(), task=4, fold_idx=0, split_type="vanilla")
    check("training through ReweightDataset is filtered too", rw[0][1] == n - drop)
    for fold_idx, name in ((1, "validation"), (2, "testing"), (None, "all")):
        ds = dataset.CancerDataset(df.copy(), task=4, fold_idx=fold_idx, split_type="vanilla",
                                   max_train_tiles=maxt)
        item = ds[0]
        check(f"{name}: not filtered, whole bag", ds.partition_type == name
              and ds._unveil_active is False and item[1] == n)
    s = inj.summary()
    check("census: training armed, nothing else",
          s["training"]["armed"] == s["training"]["built"] == 3
          and all(d["armed"] == 0 for p, d in s.items() if p != "training"), s)
    try:
        inj.assert_census()
        check("census assertion passes", True)
    except FT.AdapterError as e:
        check("census assertion passes", False, str(e))
    inj.census.append({"partition_type": "testing", "armed": True, "rows": 1, "cls": "x"})
    try:
        inj.assert_census()
        check("census assertion fails on a filtered test set", False)
    except FT.AdapterError:
        check("census assertion fails on a filtered test set", True)
    inj.census[:] = [{"partition_type": "training", "armed": False, "rows": 1, "cls": "x"}]
    try:
        inj.assert_census()
        check("census assertion fails when nothing was filtered", False)
    except FT.AdapterError:
        check("census assertion fails when nothing was filtered", True)
    print(f"\n{sum(RESULTS)} of {len(RESULTS)} checks passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
