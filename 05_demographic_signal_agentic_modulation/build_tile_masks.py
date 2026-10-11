import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from tile_filter import METHODS, load_signal, tile_mask


def main():
    parser = argparse.ArgumentParser(description="Per-slide boolean tile masks passed to 02_pathology_task_prediction/train.py --tile-masks.")
    parser.add_argument("--signal-dir", required=True, help="output of export_demographic_signal.py")
    parser.add_argument("--method", choices=METHODS, default="signal_aware")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random_matched control")
    parser.add_argument("--output", required=True, help="npz of masks keyed by slide id")
    parser.add_argument("--summary", default=None, help="csv of tiles kept per slide")
    args = parser.parse_args()

    masks, summary = {}, []
    for path in sorted(Path(args.signal_dir).glob("*.npz")):
        mask = tile_mask(args.method, *load_signal(path), slide_key=path.stem, seed=args.seed)
        masks[path.stem] = mask
        summary.append({"slide_id": path.stem, "n_tiles": len(mask), "n_kept": int(mask.sum())})
    if not masks:
        raise SystemExit(f"no signal files in {args.signal_dir}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial.npz")
    np.savez_compressed(partial, **dict(sorted(masks.items())))
    os.replace(partial, output)
    if args.summary:
        pd.DataFrame(summary).to_csv(args.summary, index=False)
    kept = sum(row["n_kept"] for row in summary) / sum(row["n_tiles"] for row in summary)
    print(f"{args.method}: {len(masks)} slides, {kept:.1%} of tiles kept")


if __name__ == "__main__":
    main()
