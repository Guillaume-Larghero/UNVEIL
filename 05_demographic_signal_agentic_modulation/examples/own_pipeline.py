"""End to end on synthetic data: signal -> masks -> a filtered training loop.

    python examples/own_pipeline.py [--workdir DIR] [--seed 0]

Every step here is what you would do with your own data; only the data are made up:
  1. write one feature file per slide (yours come from your feature extractor);
  2. train a small gated-attention demographic classifier (yours is trained by you);
  3. compute the signal with unveil_tilefilter.signal.compute_signals;
  4. build the masks with unveil_tilefilter.masks;
  5. train a toy task model on the TRAINING slides through MaskedBagDataset, and evaluate
     it on whole test slides.
"""

import argparse
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

import torch  # noqa: E402
from torch import nn  # noqa: E402

from toy_models import GatedABMIL  # noqa: E402
from unveil_tilefilter.dataset import MaskedBagDataset  # noqa: E402
from unveil_tilefilter.masks import MaskSet, build_masks, write_mask_file  # noqa: E402
from unveil_tilefilter.signal import compute_signals  # noqa: E402


def make_slides(feat_dir, n_slides, dim, rng):
    """Slides whose task label lives in some tiles and whose group shows in others."""
    os.makedirs(feat_dir, exist_ok=True)
    rows = []
    for i in range(n_slides):
        n = int(rng.integers(100, 600))
        group, label = int(rng.integers(0, 2)), int(rng.integers(0, 2))
        x = rng.standard_normal((n, dim)).astype(np.float32)
        x[: n // 5, 0] += 2.0 * group                   # tiles that show the group
        x[n // 3:, 1] += 1.0 if label else -1.0           # tiles that show the label
        fid = f"SLIDE-{i:03d}"
        torch.save(torch.from_numpy(x), os.path.join(feat_dir, fid + ".pt"))
        rows.append((fid, group, label, "train" if i < int(0.75 * n_slides) else "test"))
    return rows


def train_bag_model(model, items, epochs, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    for _ in range(epochs):
        for bag, y in items:
            opt.zero_grad()
            loss = loss_fn(model(bag).reshape(1), torch.tensor([float(y)]))
            loss.backward()
            opt.step()
    return model.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slides", type=int, default=60)
    a = ap.parse_args()
    work = a.workdir or tempfile.mkdtemp(prefix="unveil-example-")
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    feats = os.path.join(work, "features")
    rows = make_slides(feats, a.slides, 8, rng)
    train = [r for r in rows if r[3] == "train"]
    test = [r for r in rows if r[3] == "test"]
    load = lambda fid: torch.load(os.path.join(feats, fid + ".pt"), weights_only=True)  # noqa: E731

    # the demographic classifier, trained on the training slides only
    clf = train_bag_model(GatedABMIL(8), [(load(f), g) for f, g, _y, _s in train], epochs=3)
    ckpt = os.path.join(work, "demographic_classifier.pt")
    torch.save({"model": clf}, ckpt)

    sig = os.path.join(work, "signals")
    compute_signals(feats, ckpt, sig, code_dirs=[HERE], log=lambda *x: None)
    masks = build_masks(sig, "adaptive")
    mfile = os.path.join(work, "masks", "adaptive.npz")
    meta = write_mask_file(mfile, masks, "adaptive", None, 0, sig)
    print(f"masks: {meta['stats']['slides']} slides, overall keep fraction "
          f"{meta['stats']['keep_frac_overall']:.3f}")

    class TrainBags(torch.utils.data.Dataset):
        def __len__(self):
            return len(train)

        def __getitem__(self, i):
            fid, _g, y, _s = train[i]
            return load(fid), y

    ds = MaskedBagDataset(TrainBags(), MaskSet([mfile]), [r[0] for r in train], bag_index=0,
                          max_tiles=300)
    task = train_bag_model(GatedABMIL(8), [ds[i] for i in range(len(ds))], epochs=8)
    with torch.no_grad():
        preds = [float(task(load(f))) for f, _g, _y, _s in test]   # whole test slides
    acc = np.mean([(p >= 0.5) == bool(y) for p, (_f, _g, y, _s) in zip(preds, test)])
    print(f"task model trained on filtered bags; test accuracy on whole slides {acc:.2f}")
    print(f"files in {work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
