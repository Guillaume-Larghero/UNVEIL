"""Step 2: per-slide keep-masks.

The filter runs once per slide, offline, and the boolean masks are stored; training then
only indexes. One mask file holds the masks of many slides (an .npz keyed by folder_id)
and a JSON sidecar, ``<file>.json``, recording how they were built: the filter, every
parameter, the seed, the signal directory, and keep-rate statistics.

Every fold of a cross-validation and every training seed reads the same masks, so the
masks of one slide never depend on which split it is in.
"""

import json
import os
import time

import numpy as np

from . import __version__
from .filters import FilterParams, make_mask, mask_id
from .signal import MANIFEST, list_signal_ids, load_features, load_signal, md5_file


class MaskError(RuntimeError):
    pass


def build_masks(signal_dir, filter_name="adaptive", params=None, seed=0, slides=None,
                features_dir=None, feature_ext=".pt", h5_key="features"):
    """{folder_id: mask} for the slides in ``signal_dir`` (or only ``slides``).

    With ``features_dir``, every slide's feature file is opened and its row count checked
    against the signal's tile count, so a mask can never index a different bag."""
    params = (params or FilterParams()).validate()
    ids = list_signal_ids(signal_dir)
    if slides is not None:
        want = list(dict.fromkeys(slides))
        missing = sorted(set(want) - set(ids))
        if missing:
            raise MaskError(f"{len(missing)} listed slide(s) have no signal in {signal_dir}, "
                            f"e.g. {missing[:3]}")
        ids = sorted(want)
    out = {}
    for fid in ids:
        sig = load_signal(signal_dir, fid)
        if features_dir is not None:
            path = os.path.join(features_dir, fid + feature_ext)
            if not os.path.exists(path):
                raise MaskError(f"{fid}: no feature file {path}")
            rows = int(load_features(path, h5_key).shape[0])
            if rows != sig.n_tiles:
                raise MaskError(f"{fid}: {rows} feature rows in {path}, but the signal has "
                                f"{sig.n_tiles} tiles")
        out[fid] = make_mask(sig, filter_name, params, seed)
    return out


def keep_stats(masks):
    if not masks:
        return {"slides": 0}
    kept = np.array([int(m.sum()) for m in masks.values()], dtype=np.float64)
    total = np.array([int(m.size) for m in masks.values()], dtype=np.float64)
    frac = np.divide(kept, total, out=np.ones_like(kept), where=total > 0)
    return {"slides": len(masks), "tiles_total": int(total.sum()),
            "tiles_kept": int(kept.sum()),
            "keep_frac_overall": round(float(kept.sum() / total.sum()), 6) if total.sum() else 1.0,
            "keep_frac_mean": round(float(frac.mean()), 6),
            "keep_frac_min": round(float(frac.min()), 6),
            "keep_frac_max": round(float(frac.max()), 6),
            "slides_filtered": int((kept < total).sum())}


def write_mask_file(path, masks, filter_name, params, seed, signal_dir, extra=None):
    """Write the masks (keys sorted, atomically) and the sidecar; returns the sidecar dict."""
    if not path.endswith(".npz"):
        raise MaskError(f"{path}: a mask file must end in .npz")
    params = (params or FilterParams()).validate()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ordered = {k: np.asarray(masks[k], dtype=bool) for k in sorted(masks)}
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **ordered)
    os.replace(tmp, path)
    mpath = os.path.join(signal_dir, MANIFEST)
    meta = {
        "package": "unveil_tilefilter", "version": __version__,
        "filter": filter_name, "params": params.to_dict(),
        "seed": int(seed) if filter_name == "random_matched" else None,
        "mask_id": mask_id(filter_name, params, seed),
        "signal_dir": os.path.abspath(signal_dir),
        "signal_manifest_md5": md5_file(mpath) if os.path.exists(mpath) else None,
        "numpy": np.__version__,
        "stats": keep_stats(ordered),
        "mask_file_md5": md5_file(path),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if extra:
        meta["extra"] = extra
    with open(path + ".json.tmp", "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    os.replace(path + ".json.tmp", path + ".json")
    return meta


class MaskSet:
    """Read-side view of one or more mask files.

    A folder_id found in two files is refused, never overwritten: a task built from two
    signal directories (e.g. a two-cancer task) must have disjoint slide sets."""

    def __init__(self, paths=()):
        self._masks = {}
        self.sources = []
        self.meta = []
        for p in paths:
            self.load(p)

    def load(self, path):
        if not os.path.exists(path):
            raise MaskError(f"no mask file at {path}")
        with np.load(path) as d:
            dup = sorted(set(d.files) & set(self._masks))
            if dup:
                raise MaskError(f"{len(dup)} folder_id(s) of {path} are already loaded from "
                                f"{self.sources}, e.g. {dup[:3]}")
            for k in d.files:
                m = d[k]
                if m.dtype != np.bool_ or m.ndim != 1:
                    raise MaskError(f"{path}: mask {k!r} is {m.dtype} {m.shape}, not a 1-D "
                                    f"bool array")
                self._masks[k] = m
        side = path + ".json"
        self.meta.append(json.load(open(side)) if os.path.exists(side) else None)
        self.sources.append(os.path.abspath(path))
        return self

    def has(self, folder_id):
        return str(folder_id) in self._masks

    def get(self, folder_id):
        try:
            return self._masks[str(folder_id)]
        except KeyError:
            raise MaskError(f"no mask for slide {folder_id!r} (loaded from {self.sources})") \
                from None

    def keys(self):
        return self._masks.keys()

    def keep_stats(self):
        return keep_stats(self._masks)

    def __len__(self):
        return len(self._masks)
