"""Rebuild the paper's training masks with this package and compare them with the masks
the paper's runs loaded.

    python verification/eq_masks.py --mask-root <repro package>/masks \\
        --signal-store <dir holding <attribute>/<cancer>/<FM>/<FS|PM>/ signal folders>

Every mask cell under --mask-root (laid out <filter>-<fingerprint>/<attribute>/<cancer>/
<FM>/<FS|PM>.npz, as the reproduction package ships them) is rebuilt from its signal
folder with the default parameters -- evolved_v0 cells as `adaptive`, random_matched and
attention_matched cells as themselves, seed 0 -- and compared slide by slide: the same
slides, and every mask array identical. Exit 1 on any difference.
"""

import argparse
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
from unveil_tilefilter.masks import build_masks  # noqa: E402

FILTER = {"evolved_v0": "adaptive", "random_matched": "random_matched",
          "attention_matched": "attention_matched"}
CELL = re.compile(r"(?P<filter>evolved_v0|random_matched|attention_matched)-[0-9a-f]{16}/"
                  r"(?P<attr>[^/]+)/(?P<cancer>[^/]+)/(?P<fm>[^/]+)/(?P<st>FS|PM)\.npz$")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask-root", required=True)
    ap.add_argument("--signal-store", required=True)
    a = ap.parse_args()
    cells = []
    for dirpath, _dirs, files in os.walk(a.mask_root):
        for f in files:
            p = os.path.join(dirpath, f)
            m = CELL.search(p)
            if f.endswith(".npz") and m:
                cells.append((p, m))
    cells.sort()
    if not cells:
        print("no mask cells found")
        return 1
    bad = 0
    slides = tiles = dropped = 0
    t0 = time.time()
    for p, m in cells:
        sig = os.path.join(a.signal_store, m["attr"], m["cancer"], m["fm"], m["st"])
        new = build_masks(sig, FILTER[m["filter"]], seed=0)
        with np.load(p) as ref:
            keys = sorted(ref.files)
            missing_in_new = sorted(set(keys) - set(new))
            diff = [k for k in keys if k in new and not np.array_equal(ref[k], new[k])]
            slides += len(keys)
            tiles += sum(int(ref[k].size) for k in keys)
            dropped += sum(int((~ref[k]).sum()) for k in keys)
        extra = sorted(set(new) - set(keys))
        ok = not (missing_in_new or diff or extra)
        bad += not ok
        rel = os.path.relpath(p, a.mask_root)
        print(f"{'ok  ' if ok else 'DIFF'} {rel}: {len(keys)} slides"
              + ("" if ok else f"; differ {len(diff)}, missing {len(missing_in_new)}, "
                               f"extra {len(extra)} (e.g. {(diff + missing_in_new + extra)[:3]})"))
    print(f"\n{len(cells) - bad} of {len(cells)} cells identical; {slides} slides, {tiles} tiles, "
          f"{dropped} dropped ({time.time() - t0:.0f} s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
