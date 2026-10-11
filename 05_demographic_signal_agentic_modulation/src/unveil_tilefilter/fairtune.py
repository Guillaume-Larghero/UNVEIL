"""Step 3b: train with the collaborators' fairtune_pipeline, its code left unmodified.

    unveil-tilefilter fairtune --upstream /path/to/fairtune_pipeline \\
        --masks masks/BRCA_Age_FS.npz --entrypoint mutation -- \\
        <the pipeline's own arguments, e.g. --cancer brca --train_method agenticFilter
         --seed 0 --partition 2 --model_path ...>

Everything after ``--`` goes to the pipeline untouched: seed, split, paths and every other
choice are yours. In order, the adapter:

1. builds an overlay directory: symlinks to the pipeline's code, and ``configs/`` pointing
   to a copy of the pipeline's ``configs/`` with three train methods appended --
   agenticFilter (adaptive masks), agenticRandomFilter (random_matched) and
   agenticAttentionOnlyFilter (attention_matched) -- each a copy of the pipeline's own
   ``baseline`` block, so the model, optimiser and model selection are exactly the
   baseline's;
2. imports the pipeline's ``dataset`` and ``framework`` from the overlay and checks the
   five callables it depends on against the version it was tested on (stop on any
   difference unless ``--allow-upstream-drift``);
3. replaces ``dataset.CancerDataset`` and ``dataset.ReweightDataset`` by subclasses that
   filter the TRAINING partition only -- validation, test and external-cohort data are
   never filtered -- and wraps the result-folder suffix rule so the three methods get a
   folder of their own instead of the baseline's;
4. runs the entrypoint;
5. asserts, from a census of every dataset built, that only training datasets were
   filtered (and that at least one was), and writes a run record.

Without ``--masks`` nothing is filtered and the census asserts so: that is a baseline run.

``--inference`` scores a trained model -- on an external cohort, or a TCGA model on its own
partitions -- with the pipeline's ``--inference_only``, or re-runs the bias test on a
finished run's saved results with the pipeline's ``--sig_test_only``. It refuses
``--masks``, keeps the method name the model was trained under (so the pipeline finds that
model's folder), and asserts that no partition, training included, was filtered.

The pipeline runs from the overlay directory, so every relative path in its arguments
resolves there; give absolute paths. Concurrent runs may share one overlay: each build
leaves an existing identical config copy alone and swaps in a changed one atomically.
"""

import hashlib
import inspect
import json
import os
import re
import runpy
import shutil
import sys
import time

import numpy as np

from . import __version__
from .masks import MaskSet

# train method -> the filter its masks must have been built with
METHODS = {"agenticFilter": "adaptive",
           "agenticRandomFilter": "random_matched",
           "agenticAttentionOnlyFilter": "attention_matched"}

# mutation tasks run through main_genetic.py; the other task types through main.py
ENTRYPOINTS = {"mutation": "main_genetic.py", "stage": "main.py", "grade": "main.py",
               "classification": "main.py", "detection": "main.py"}

# Not linked into the overlay: trees the pipeline writes results into, its own job
# scripts, and configs/ (copied instead). A write through a symlink would land in the
# pipeline's tree; leaving these out makes a stray relative write land in the overlay.
BLOCKED = (
    r"^\.git$", r"^\.hypothesis$", r"^__pycache__$", r"^wandb$", r"^logs$",
    r"_models$", r"_models_adversarialDebiasing$", r"^mutation_models_DEDUP_quarantine$",
    r"^m5_out$", r"^mixed_effects_results$", r"^fairness_analysis$", r"^dataset_stats$",
    r"^configs$", r"^run_scripts_SY$", r"^run_scripts_example$",
    r"^jobmap_.*", r"^recov\d*_.*", r"^cells_.*", r"\.tmp$", r"^temp\.", r"^debug\.py$",
)
_BLOCKED = [re.compile(p) for p in BLOCKED]
OVERLAY_MARKER = ".unveil_overlay.json"
REQUIRED = ("dataset.py", "framework.py", "configs/train_configs.yaml")

PINNED_SOURCES = ("dataset:CancerDataset.__getitem__", "dataset:CancerDataset.__init__",
                  "dataset:ReweightDataset", "dataset:get_datasets",
                  "framework:should_emit_reweight_suffix")
PINNED_FILES = ("dataset.py", "framework.py", "main.py", "main_genetic.py",
                "configs/train_configs.yaml")
TESTED_PINS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fairtune_tested_pins.json")

# Pass-through flags this adapter reads. The pipeline's parser accepts unique prefixes
# (argparse allow_abbrev), so an abbreviation would reach it while this adapter saw
# nothing; abbreviations and repeats of these are refused.
READ_FLAGS = ("--train_method", "--model_path", "--cancer", "--inference_only",
              "--sig_test_only")

# The pipeline's modes that train nothing: score a saved model, or re-run the bias test on
# a finished run's saved results.
SCORING_FLAGS = ("--inference_only", "--sig_test_only")


class AdapterError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# pass-through arguments
# --------------------------------------------------------------------------------------

def _arg_value(argv, name):
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def _arg_values(argv, name):
    vals = None
    for i, a in enumerate(argv):
        if a == name:
            vals = []
            for b in argv[i + 1:]:
                if b.startswith("--"):
                    break
                vals.append(b)
        elif a.startswith(name + "="):
            vals = [a.split("=", 1)[1]]
    return vals


def _arg_count(argv, name):
    return sum(1 for a in argv if a == name or a.startswith(name + "="))


def check_passthrough(argv):
    """Refuse the spellings the pipeline would read differently from this adapter."""
    abbrev = [a for a in argv if a.startswith("--") and a != "--"
              and a.split("=", 1)[0] not in READ_FLAGS
              and any(f.startswith(a.split("=", 1)[0]) for f in READ_FLAGS)]
    if abbrev:
        raise AdapterError(f"spell these flags out in full: {abbrev} (the pipeline expands "
                           f"unique prefixes; this adapter would not see them)")
    doubled = [f for f in READ_FLAGS if _arg_count(argv, f) > 1]
    if doubled:
        raise AdapterError(f"{doubled} given more than once: the pipeline keeps the last, "
                           f"this adapter would read the first")
    bad = [t for t in _arg_values(argv, "--cancer") or []
           if not t or any(c.isspace() for c in t)]
    if bad:
        raise AdapterError(f"--cancer token(s) {bad!r}: pass each cancer as its own token "
                           f"(--cancer GBM LGG), never one quoted token")
    return argv


# --------------------------------------------------------------------------------------
# the overlay
# --------------------------------------------------------------------------------------

def _blocked(name):
    return any(r.search(name) for r in _BLOCKED)


def _append_methods(cfg_path):
    """Append the three methods to the overlay's train_configs.yaml, each a copy of the
    pipeline's own `baseline` block. Returns the baseline block."""
    import yaml

    with open(cfg_path) as fh:
        text = fh.read()
    data = yaml.load(text, Loader=yaml.FullLoader)             # the pipeline's own loader
    base = (data or {}).get("baseline")
    if not isinstance(base, dict):
        raise AdapterError(f"{cfg_path} has no `baseline` block to copy")
    for m in METHODS:
        if m in data and data[m] != base:
            raise AdapterError(f"{cfg_path} already defines {m}, differently from baseline: "
                               f"{data[m]!r} vs {base!r}")
    add = [m for m in METHODS if m not in data]
    if add:
        block = ("\n\n# Added by unveil_tilefilter " + __version__ + ": each method below is "
                 "the `baseline` block above,\n# so a filtered run differs from the baseline "
                 "only in the training tiles it sees.\n"
                 + "".join(yaml.safe_dump({m: base}, sort_keys=False) for m in add))
        with open(cfg_path, "w") as fh:
            fh.write(text + block)
    with open(cfg_path) as fh:
        check = yaml.load(fh, Loader=yaml.FullLoader)
    if any(check.get(m) != base for m in METHODS) or \
            {k: v for k, v in check.items() if k not in add} != data:
        raise AdapterError(f"{cfg_path}: the appended methods did not read back as the "
                           f"baseline block")
    return base


def _tree_digest(root):
    """sha256 over the relative paths and bytes of every file under ``root``."""
    h = hashlib.sha256()
    for dirpath, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            path = os.path.join(dirpath, f)
            h.update(os.path.relpath(path, root).encode("utf-8") + b"\0")
            with open(path, "rb") as fh:
                h.update(fh.read())
            h.update(b"\0")
    return h.hexdigest()


def _link(src, dst):
    """Make dst a symlink to src; another process making the same link at once is fine."""
    try:
        os.symlink(src, dst)
    except FileExistsError:
        if not (os.path.islink(dst) and os.readlink(dst) == src):
            raise


def _install_configs(upstream, overlay):
    """configs/ -> a content-addressed copy of the pipeline's configs/ with the methods
    appended. An identical copy is left alone, and a changed one is swapped in by an
    atomic rename of the symlink, so a run reading its configs is never interrupted."""
    tmp = os.path.join(overlay, f".configs.tmp.{os.getpid()}.{time.time_ns()}")
    shutil.copytree(os.path.join(upstream, "configs"), tmp, symlinks=False)
    try:
        base = _append_methods(os.path.join(tmp, "train_configs.yaml"))
        name = ".configs-" + _tree_digest(tmp)[:16]
        target = os.path.join(overlay, name)
        if os.path.isdir(target):
            shutil.rmtree(tmp)
        else:
            try:
                os.rename(tmp, target)
            except OSError:
                if not os.path.isdir(target):          # not just another process's rename
                    raise
                shutil.rmtree(tmp)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    link = os.path.join(overlay, "configs")
    if os.path.islink(link) and os.readlink(link) == name:
        return base, name
    if os.path.exists(link) and not os.path.islink(link):
        raise AdapterError(f"{link} is a real directory; this overlay was built by another "
                           f"tool or version: use a new --overlay")
    tmp_link = os.path.join(overlay, f".configs.link.{os.getpid()}.{time.time_ns()}")
    os.symlink(name, tmp_link)
    os.replace(tmp_link, link)
    return base, name


def build_overlay(upstream, overlay, log=print):
    """Symlink the pipeline's entries into ``overlay`` and point its configs/ at a copy of
    the pipeline's configs/ with the three methods appended. Safe to run from several
    jobs at once on the same overlay."""
    upstream = os.path.realpath(upstream)
    missing = [r for r in REQUIRED if not os.path.exists(os.path.join(upstream, r))]
    if missing:
        raise AdapterError(f"{upstream} is not a fairtune_pipeline checkout: missing {missing}")
    overlay = os.path.abspath(overlay)
    os.makedirs(overlay, exist_ok=True)
    marker = os.path.join(overlay, OVERLAY_MARKER)
    if not os.path.exists(marker):
        # another job may be writing the marker right now: its temporary file does not
        # count, and the marker is looked for again before refusing
        foreign = [f for f in os.listdir(overlay) if not f.startswith(OVERLAY_MARKER)]
        if foreign and not os.path.exists(marker):
            raise AdapterError(f"{overlay} exists and is not an overlay this tool built; use "
                               f"an empty or new --overlay")
        tmp = f"{marker}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:                          # the marker comes first, so a
            json.dump({"upstream": upstream, "package": "unveil_tilefilter",   # concurrent
                       "version": __version__}, fh, indent=2)                  # build sees it
        os.replace(tmp, marker)
    with open(marker) as fh:
        made_from = json.load(fh).get("upstream")
    if made_from != upstream:
        raise AdapterError(f"{overlay} is an overlay of {made_from}, not {upstream}; use "
                           f"another --overlay")
    linked = skipped = 0
    for name in sorted(os.listdir(upstream)):
        if _blocked(name):
            skipped += 1
            continue
        src, dst = os.path.join(upstream, name), os.path.join(overlay, name)
        if os.path.islink(dst):
            if os.readlink(dst) != src:
                raise AdapterError(f"{dst} points to {os.readlink(dst)}, not {src}")
        elif os.path.exists(dst):
            skipped += 1                       # something written here by an earlier run
            continue
        else:
            _link(src, dst)
        linked += 1
    base, cfg = _install_configs(upstream, overlay)
    log(f"[fairtune] overlay {overlay}: {linked} entries linked, {skipped} skipped, "
        f"configs -> {cfg}, {len(METHODS)} methods = baseline {base}")
    return overlay


# --------------------------------------------------------------------------------------
# the upstream check
# --------------------------------------------------------------------------------------

def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(spec, modules):
    mod, _, attr = spec.partition(":")
    obj = modules[mod]
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def check_upstream(upstream, modules, pins_path=TESTED_PINS):
    """{'match', 'drift', 'sources', 'files'}: the live callables vs the tested ones."""
    with open(pins_path) as fh:
        tested = json.load(fh)
    sources = {}
    for spec in PINNED_SOURCES:
        try:
            src = inspect.getsource(_resolve(spec, modules))
            sources[spec] = hashlib.sha256(src.encode("utf-8")).hexdigest()
        except (AttributeError, OSError, TypeError) as e:
            sources[spec] = f"<unavailable: {type(e).__name__}>"
    files = {r: (_sha256_file(os.path.join(upstream, r))
                 if os.path.exists(os.path.join(upstream, r)) else None)
             for r in PINNED_FILES}
    drift = [s for s in PINNED_SOURCES if sources[s] != tested["sources"].get(s)]
    return {"match": not drift, "drift": drift, "sources": sources, "files": files,
            "files_match_tested": {r: files[r] == tested["files"].get(r) for r in files}}


# --------------------------------------------------------------------------------------
# the injection
# --------------------------------------------------------------------------------------

class Injection:
    """Subclasses the pipeline's datasets so that TRAINING bags are filtered.

    Every dataset the pipeline builds is recorded in ``census`` (in the parent process,
    where datasets are constructed; __getitem__ runs in DataLoader workers, so per-item
    counters would never reach the parent and must not be used to assert anything)."""

    def __init__(self, mask_set=None, log=print):
        self.masks = mask_set
        self.enabled = mask_set is not None
        self.census = []
        self.log = log

    def _mixin(self, base):
        inj = self

        class _Filtered(base):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._unveil_active = False
                self._unveil_logged = False
                if inj.enabled and self.partition_type == "training":
                    if getattr(self, "add_sensitive", None):
                        # __getitem__ appends the sensitive vector to the sample, so a tile
                        # mask would no longer line up with its rows
                        raise AdapterError("filtering is not defined for add_sensitive=True")
                    missing = [f for f in self.df.folder_id.astype(str).unique()
                               if not inj.masks.has(f)]
                    if missing:
                        raise AdapterError(
                            f"{len(missing)} of {self.df.folder_id.nunique()} training slides "
                            f"have no mask, e.g. {missing[:5]}; build masks for them first")
                    self._unveil_active = True
                inj.census.append({"partition_type": self.partition_type,
                                   "armed": self._unveil_active, "rows": len(self.df),
                                   "cls": type(self).__name__})

            def __getitem__(self, idx):
                if not self._unveil_active:
                    return super().__getitem__(idx)
                # The pipeline subsamples max_train_tiles inside __getitem__. Turn that off
                # so the filter sees the whole bag, then subsample the kept tiles here, the
                # same way.
                saved = self.max_train_tiles
                self.max_train_tiles = None
                try:
                    out = super().__getitem__(idx)
                finally:
                    self.max_train_tiles = saved
                sample = out[0]
                folder_id = str(self.df.loc[idx].folder_id)
                mask = inj.masks.get(folder_id)
                if mask.shape[0] != sample.shape[0]:
                    raise AdapterError(f"{folder_id}: mask length {mask.shape[0]} != bag size "
                                       f"{sample.shape[0]}")
                sample = sample[mask]
                if not self._unveil_logged:
                    self._unveil_logged = True
                    inj.log(f"[fairtune] filtering active ({self.partition_type}): "
                            f"{folder_id} {int(mask.sum())}/{mask.shape[0]} tiles kept")
                if saved and sample.shape[0] > saved:
                    sel = np.random.choice(sample.shape[0], saved, replace=False)
                    sample = sample[sel, :]
                return (sample, len(sample)) + tuple(out[2:])

        _Filtered.__name__ = _Filtered.__qualname__ = f"Unveil{base.__name__}"
        return _Filtered

    def apply(self, dataset_mod, framework_mod):
        # get_datasets looks the classes up as module attributes when it builds a dataset,
        # so the unmodified factory hands back these subclasses. ReweightDataset binds
        # CancerDataset as its base when its class is created, so it is replaced too.
        dataset_mod.CancerDataset = self._mixin(dataset_mod.CancerDataset)
        dataset_mod.ReweightDataset = self._mixin(dataset_mod.ReweightDataset)
        original = framework_mod.should_emit_reweight_suffix

        def should_emit_reweight_suffix(args):
            if getattr(args, "train_method", "") in METHODS:
                return True
            return original(args)

        should_emit_reweight_suffix.__wrapped__ = original
        framework_mod.should_emit_reweight_suffix = should_emit_reweight_suffix
        # `from framework import *` copied the original into modules imported earlier
        for mod in list(sys.modules.values()):
            if getattr(mod, "should_emit_reweight_suffix", None) is original:
                mod.should_emit_reweight_suffix = should_emit_reweight_suffix

    def summary(self):
        out = {}
        for e in self.census:
            d = out.setdefault(e["partition_type"], {"built": 0, "armed": 0})
            d["built"] += 1
            d["armed"] += int(e["armed"])
        return out

    def assert_census(self):
        s = self.summary()
        armed = {p: d for p, d in s.items() if d["armed"]}
        if not self.enabled:
            if armed:
                raise AdapterError(f"no masks were given, yet {armed} was filtered: {s}")
            return s
        leaked = {p: d for p, d in armed.items() if p != "training"}
        if leaked:
            raise AdapterError(f"non-training data was filtered: {leaked}. Census: {s}")
        if not s.get("training", {}).get("armed"):
            raise AdapterError(f"no training dataset was filtered; the masks had no effect. "
                               f"Census: {s}")
        return s


# --------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------

def _write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def check_masks_vs_method(mask_set, train_method):
    """A filtered run must train under the method its masks were built for, and an
    unfiltered run under none of the three (or its results would be filed as filtered)."""
    if mask_set is None:
        if train_method in METHODS:
            raise AdapterError(f"--train_method {train_method} without --masks: the run would "
                               f"be filed as filtered but train on whole bags")
        return None
    if train_method not in METHODS:
        raise AdapterError(f"--masks given, so --train_method must be one of {list(METHODS)} "
                           f"(got {train_method!r}); otherwise the filtered run would write "
                           f"into the same folder as an unfiltered one")
    if any(m is None for m in mask_set.meta):
        raise AdapterError("a mask file has no .json sidecar, so the filter it was built with "
                           "is unknown; build masks with `unveil-tilefilter masks`")
    built = sorted({m["filter"] for m in mask_set.meta})
    if built != [METHODS[train_method]]:
        raise AdapterError(f"--train_method {train_method} expects {METHODS[train_method]} "
                           f"masks, but the mask files were built with {built}")
    return built[0]


def check_mode(inference, mask_set, passthrough):
    """--inference scores a trained model, or re-runs its bias test, and filters nothing. A
    training run must not be either: with masks, an inference-only run would score the
    training partition on filtered bags, and a bias-test-only run builds no dataset, so its
    census would fail after the pipeline had rewritten the run's bias-test files."""
    scoring = [f for f in SCORING_FLAGS if _arg_count(passthrough, f) > 0]
    if inference:
        if mask_set is not None:
            raise AdapterError("--inference with --masks: scoring reads whole slides; masks "
                               "are for training only")
        if not scoring:
            raise AdapterError("--inference needs the pipeline's --inference_only (or "
                               "--sig_test_only) after --; without it the pipeline would "
                               "train")
        if len(scoring) > 1:
            raise AdapterError("--inference_only and --sig_test_only together: the pipeline "
                               "would only re-run the bias test, on the saved inference "
                               "results; pass one of them")
    elif scoring:
        raise AdapterError(f"{scoring[0]} in the pipeline's arguments: use --inference, "
                           f"which keeps every partition unfiltered")


def run(upstream, entrypoint, passthrough, masks=(), overlay=None, entrypoint_script=None,
        record=None, allow_upstream_drift=False, seed_numpy=None, check_only=False,
        inference=False, log=print):
    """Run one fairtune_pipeline training with the masks applied to training bags, or,
    with ``inference``, one inference-only run with nothing filtered."""
    upstream = os.path.realpath(upstream)
    passthrough = check_passthrough(list(passthrough))
    train_method = _arg_value(passthrough, "--train_method")
    mask_set = MaskSet(masks) if masks else None
    filter_name = None
    if not check_only:
        check_mode(inference, mask_set, passthrough)
        if not inference:
            filter_name = check_masks_vs_method(mask_set, train_method)
    script = entrypoint_script or ENTRYPOINTS.get(entrypoint)
    if script is None:
        raise AdapterError(f"unknown --entrypoint {entrypoint!r}; choose from "
                           f"{sorted(ENTRYPOINTS)} or pass --entrypoint-script")
    script_path = os.path.join(upstream, script)
    if not os.path.exists(script_path):
        raise AdapterError(f"no entrypoint {script_path}")
    model_path = _arg_value(passthrough, "--model_path")
    if model_path is not None and not os.path.isabs(model_path):
        raise AdapterError(f"--model_path {model_path!r} must be absolute: the pipeline runs "
                           f"from the overlay directory, so a relative path would put the "
                           f"results there")
    if model_path is None and not check_only:
        log("[fairtune] note: no --model_path; the pipeline's default is relative, so its "
            "results go inside the overlay")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    record = os.path.abspath(record or os.path.join(
        model_path or os.getcwd(), f"unveil_tilefilter_run_{stamp}_{os.getpid()}.json"))
    overlay = build_overlay(upstream, overlay or os.path.join(os.getcwd(), "unveil_overlay"),
                            log)

    os.chdir(overlay)
    sys.path.insert(0, overlay)
    import dataset                                             # noqa: F401  the pipeline's
    import framework                                           # noqa: F401
    pins = check_upstream(upstream, {"dataset": dataset, "framework": framework})
    if pins["match"]:
        log("[fairtune] upstream callables match the tested version")
    else:
        msg = (f"the pipeline's {pins['drift']} differ from the version this adapter was "
               f"tested on. Read their diff against the tested version: the adapter relies on "
               f"__getitem__ returning (bag, len, ...), on max_train_tiles being applied "
               f"inside it, on partition_type/df.folder_id/add_sensitive, on get_datasets "
               f"looking the classes up as module attributes, and on the suffix rule.")
        if not allow_upstream_drift:
            raise AdapterError(msg + " If it still holds, rerun with --allow-upstream-drift.")
        log("[fairtune] WARNING: " + msg)
    if check_only:
        log(f"[fairtune] check only: overlay and upstream check done, nothing run")
        return {"overlay": overlay, "upstream_check": pins}

    inj = Injection(mask_set, log)
    inj.apply(dataset, framework)
    rec = {"package": "unveil_tilefilter", "version": __version__, "status": "started",
           "mode": "inference" if inference else "train",
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "upstream": upstream, "overlay": overlay, "entrypoint": script,
           "argv": passthrough, "train_method": train_method, "filter": filter_name,
           "upstream_check": pins, "allow_upstream_drift": bool(allow_upstream_drift),
           "seed_numpy": seed_numpy,
           "masks": [] if mask_set is None else
           [{"path": p, "mask_id": (m or {}).get("mask_id"),
             "mask_file_md5": (m or {}).get("mask_file_md5"),
             "params": (m or {}).get("params"), "seed": (m or {}).get("seed")}
            for p, m in zip(mask_set.sources, mask_set.meta)],
           "mask_stats": None if mask_set is None else mask_set.keep_stats()}
    _write_json(record, rec)
    log(f"[fairtune] {'filter ' + filter_name if mask_set else 'NO filtering'}"
        f"{' (inference)' if inference else ''}; record {record}")
    if seed_numpy is not None:
        # Seeds the MAIN process's numpy RNG. It draws the max_train_tiles subsample only
        # with --n_workers 0; DataLoader workers reseed numpy from the run's torch seed.
        np.random.seed(int(seed_numpy))
    sys.argv = [script_path] + passthrough
    error = None
    try:
        runpy.run_path(script_path, run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            error = f"SystemExit({e.code!r})"
            raise
    except BaseException as e:
        error = f"{type(e).__name__}: {e}"[:500]
        raise
    finally:
        census = inj.summary()
        log(f"[fairtune] dataset census: {census}")
        try:
            if error is None:
                inj.assert_census()
                if mask_set:
                    log("[fairtune] only training data was filtered (asserted)")
                elif census:
                    log("[fairtune] nothing was filtered, training included (asserted)")
                else:
                    log("[fairtune] no dataset was built (e.g. --sig_test_only), so nothing "
                        "was filtered")
            rec.update(status="finished" if error is None else "failed", error=error,
                       census=census, ended_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        except AdapterError as e:
            rec.update(status="failed", error=str(e), census=census,
                       ended_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            _write_json(record, rec)
            raise
        _write_json(record, rec)
    return rec
