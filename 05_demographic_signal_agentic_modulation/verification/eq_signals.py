"""Recompute stored demographic signals with this package and compare them.

    python verification/eq_signals.py --features-dir <slide .pt files> \\
        --checkpoint <classifier> --model-code <its code> --signal-dir <stored signals> \\
        [--n-slides 20]

Takes every k-th slide of --signal-dir (sorted, --n-slides in all, the largest one
always included), computes its signal from the feature file with the default settings
(gated_abmil, chunk 2048, one thread, float16 storage) into a temporary folder, and
compares attention, tile_score, slide_pred and n_tiles with the stored file exactly.
Run it with the Python environment the stored signals were computed in: a different
torch build can change the last bits of the float sums. Exit 1 on any difference.
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
from unveil_tilefilter.signal import compute_signals, list_signal_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-code", action="append", default=[])
    ap.add_argument("--signal-dir", required=True)
    ap.add_argument("--n-slides", type=int, default=20)
    a = ap.parse_args()
    import torch

    ids = list_signal_ids(a.signal_dir)
    step = max(1, len(ids) // max(1, a.n_slides - 1))
    pick = ids[::step][: a.n_slides - 1]
    sizes = {}
    for fid in ids:
        with np.load(os.path.join(a.signal_dir, fid + ".npz")) as z:
            sizes[fid] = int(z["n_tiles"])
    largest = max(ids, key=sizes.get)
    pick = sorted(set(pick) | {largest})
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "sig")
        m = compute_signals(a.features_dir, a.checkpoint, out, code_dirs=a.model_code,
                            slides=pick, log=lambda *x: None)
        bad = 0
        for fid in pick:
            with np.load(os.path.join(a.signal_dir, fid + ".npz")) as ref, \
                    np.load(os.path.join(out, fid + ".npz")) as new:
                same = {k: (ref[k].dtype == new[k].dtype and np.array_equal(ref[k], new[k]))
                        for k in ("attention", "tile_score", "slide_pred", "n_tiles")}
                ok = all(same.values())
                bad += not ok
                print(f"{'ok  ' if ok else 'DIFF'} {fid}: {sizes[fid]} tiles"
                      + ("" if ok else f" {[k for k, v in same.items() if not v]}"))
    print(f"\n{len(pick) - bad} of {len(pick)} slides identical "
          f"(failed to compute: {m['slides_failed']}; torch {torch.__version__}; "
          f"{time.time() - t0:.0f} s)")
    return 1 if bad or m["slides_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
