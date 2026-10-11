"""Tile-filtering functions and their hard invariants.

A filter maps a slide's tiles plus its demographic signal onto a boolean keep-mask::

    filter_tiles(features, signal, ctx) -> np.ndarray[bool] of shape (N,)

The registry exists so that the evolved program, the random control arm and the
attention-only ablation are all the *same kind of object*, selectable by one
environment variable. That is what makes the R3#5 comparison an apples-to-apples one.

Invariants are enforced by raising. There is no repair path and no default mask: a
filter that cannot produce a valid mask is a bug we need to see, not smooth over.
"""

import hashlib
import inspect
import linecache
import re

import numpy as np

MIN_TILES = 64
MIN_KEEP_FRAC = 0.05

# The arm whose per-slide dose the matched controls copy. Evolution rewrites this
# filter's body, and filter_fingerprint follows it, so the controls re-derive their dose
# automatically whenever the method under test changes.
REFERENCE_FILTER = "evolved_v0"
MATCHED_FILTERS = frozenset({"random_matched", "attention_matched"})

_REGISTRY = {}


class FilterViolation(RuntimeError):
    """A filter broke one of the I-1..I-4 invariants."""


class TileContext:
    __slots__ = ("cancer", "attribute", "slide_type", "fm", "folder_id", "n_tiles",
                 "seed", "n_drop", "group")

    def __init__(self, cancer, attribute, slide_type, fm, folder_id, n_tiles, seed,
                 n_drop=None, group=None):
        self.cancer = cancer
        self.attribute = attribute
        self.slide_type = slide_type
        self.fm = fm
        self.folder_id = folder_id
        self.n_tiles = n_tiles
        self.seed = seed
        # Dose imposed by the reference arm on THIS slide. Populated by apply_filter for
        # the rate-matched controls; None for every other filter.
        self.n_drop = n_drop
        # Demographic group 0/1, None if unknown. Set only by apply_filter_ext, and only
        # for a filter whose reference declares needs_group.
        self.group = group


def register(name):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def available():
    return sorted(_REGISTRY)


def get(filter_id):
    """Resolve a filter id, expanding the parametric ``random_p{k}`` family."""
    if filter_id in _REGISTRY:
        return _REGISTRY[filter_id]
    m = re.fullmatch(r"random_p(\d+)", filter_id)
    if m:
        pct = int(m.group(1))
        if not 0 <= pct < 100:
            raise KeyError(f"random_p{pct} out of range")
        return _make_random(pct)
    raise KeyError(f"unknown filter_id {filter_id!r}; available: {available()} or random_p<k>")


def source_of(filter_id):
    """Source text of the filter, used to key the mask cache."""
    fn = get(filter_id)
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        # Parametric filters are closures built at runtime; their behaviour is fully
        # determined by the factory plus the id, so both go into the key.
        #
        # For the REFERENCE filter this fallback is a hole: a reference built by exec,
        # functools.partial or a closure-returning decorator lands here, and the constant
        # returned does not depend on the body, so the matched fingerprints would stop
        # following it even with the source_of() term at :157-158. It is closed one layer
        # up rather than here -- evolve/candidate.py's `reference_unsourceable` check
        # (:1154-1163) refuses such a program at load time, which is the only place that
        # knows a program is a candidate at all.
        return f"<parametric:{filter_id}>\n" + inspect.getsource(_make_random)


# Module-level tunables that change a filter's behaviour without changing its source
# text. They MUST feed the fingerprint: recalibrating a control arm's drop rate is a
# planned step, and without this the mask cache would silently serve the old masks.
def _tunables():
    return {k: v for k, v in sorted(globals().items())
            if k.isupper() and isinstance(v, (int, float, str))}


def _evolve_block_text():
    """This module's EVOLVE-BLOCK as source text, markers included.

    Read through linecache -- the path inspect.getsource takes for source_of() -- so the
    body and the block can never be hashed from two different versions of the file. The
    marker strings are local on purpose: as module-level UPPERCASE names they would be
    swept into _tunables() and move every fingerprint. They are also ASSEMBLED rather
    than written out: evolve/candidate.py finds the block by substring, so a literal
    marker anywhere in this file -- this line included -- makes the file itself a
    malformed candidate (measured: "2 start marker(s)"), and every C.load fails.
    """
    tag = "# EVOLVE-" + "BLOCK-"
    start, end = tag + "START", tag + "END"
    lines = linecache.getlines(__file__)
    i = [k for k, ln in enumerate(lines) if ln.strip() == start]
    j = [k for k, ln in enumerate(lines) if ln.strip() == end]
    if len(i) != 1 or len(j) != 1 or j[0] < i[0]:
        raise RuntimeError(f"{__file__}: expected exactly one EVOLVE-BLOCK, found "
                           f"{len(i)} start / {len(j)} end marker(s)")
    return "".join(lines[i[0]:j[0] + 1])


def filter_fingerprint(filter_id, feature_fm=None, group_md5=None, control_seed=None):
    """Stable hash of (id, source, the whole invariant chain, tunables).

    Changing the function body -- or any tunable constant it reads -- changes this,
    which changes the mask cache directory, so an edited filter can never reuse masks
    built by the previous version.

    The invariant chain goes in WHOLE -- keep_floor, max_drop, is_exempt, check_mask,
    apply_filter -- because every one of them can change the mask a given filter body
    produces. Only max_drop and is_exempt used to be hashed, which left three live holes:
    keep_floor sets the floor max_drop subtracts from, check_mask decides which masks
    survive at all, and apply_filter resolves the matched controls' dose and the exempt
    passthrough. Editing any of the three changes EVERY arm's masks while the fingerprint
    stays byte-identical, so the cache reports a hit on masks the current code would never
    produce -- the failure this hash exists to make impossible.

    A rate-matched control takes its dose from REFERENCE_FILTER, so the reference's
    code goes into its key too. Without that, evolution rewriting the reference would
    leave the controls matched to the PREVIOUS dose while the cache reported a hit --
    the same silent-staleness failure the tunables clause above exists to prevent.

    For REFERENCE_FILTER -- and so for its matched controls -- "its code" is the WHOLE
    EVOLVE-BLOCK, not just the function body (2026-09-10, review M3). The block spans
    module level, so a candidate can put the logic that decides its mask in a module
    helper or a tuple constant the body merely reads; inspect.getsource of the body sees
    neither, and _tunables() skips non-scalars. Measured before this: two candidates that
    dropped disjoint tile sets (200 vs 600 of 2000) shared one fingerprint, so after one
    was installed the other would have been served its masks.

    2026-09-11 (P0.1): the matched clause hashes source_of(REFERENCE_FILTER) as well as
    the block text. _evolve_block_text reads the FILE through linecache, so it cannot see
    a reference that was replaced IN-PROCESS -- rebinding _REGISTRY[REFERENCE_FILTER] to
    a different body left both matched fingerprints byte-identical while the dose they
    copy had changed. That is the silent-staleness hole above reached by another route;
    tests/test_injection.py:264-284 exercises exactly it.

    2026-09-13 (S-11): a reference that DECLARES inputs (declared_needs) builds masks via
    apply_filter_ext from data this module cannot see, so the key names that data -- the
    feature FM, the group table's md5, the functions deciding what a body is handed -- and
    RAISES if a declared one is missing. A non-zero control_seed moves random_matched's
    key. Undeclared filters at control_seed 0/None keep their old key (FROZEN table).
    """
    parts = [filter_id, source_of(filter_id), repr(_tunables()),
             inspect.getsource(keep_floor), inspect.getsource(max_drop),
             inspect.getsource(is_exempt), inspect.getsource(check_mask),
             inspect.getsource(apply_filter)]
    if filter_id == REFERENCE_FILTER:
        parts += ["<evolve-block>", _evolve_block_text()]
    if filter_id in MATCHED_FILTERS:
        parts += [f"<matched-to:{REFERENCE_FILTER}>",
                  source_of(REFERENCE_FILTER), _evolve_block_text()]
    need_f, need_g = declared_needs(filter_id)
    if need_f or need_g:
        for name, need, value in (("feature_fm", need_f, feature_fm),
                                  ("group_md5", need_g, group_md5)):
            if need and not value:
                raise ValueError(f"{filter_id}: its reference declares an input, so the "
                                 f"key needs {name}= (precompute_masks.fingerprint_args)")
        parts += ["<declared-inputs>", inspect.getsource(declared_needs),
                  inspect.getsource(apply_filter_ext),
                  f"feature_fm={feature_fm if need_f else ''}",
                  f"group_md5={group_md5 if need_g else ''}"]
    if filter_id == "random_matched" and control_seed is not None and int(control_seed):
        parts += [f"<control-seed:{int(control_seed)}>", inspect.getsource(apply_filter_ext)]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def keep_floor(n):
    """Minimum number of tiles a filter is allowed to leave on a slide of ``n`` tiles."""
    return max(MIN_TILES, int(np.ceil(MIN_KEEP_FRAC * n)))


def max_drop(n):
    """Drop budget for a slide of ``n`` tiles.

    Filters must clip their desired drop count to this. Exceeding it is not repaired --
    ``check_mask`` raises -- so that an evolved candidate which ignores the budget is
    scored as invalid rather than quietly rewritten into something else.
    """
    if n <= MIN_TILES:
        return 0
    return n - keep_floor(n)


def is_exempt(n):
    """Slides too small to filter.

    About 1.7% of slides carry fewer than MIN_TILES tiles (some as few as one). A bag
    that small is already at the edge of what attention-based MIL can use, so removing
    tiles cannot plausibly suppress demographic signal more than it degrades the bag.
    They are therefore passed through untouched -- centrally, so that the evolved arm,
    the random control and the attention-only ablation all see exactly the same slides
    and the comparison is not confounded by differing edge-case handling.
    """
    return n <= MIN_TILES


def check_mask(mask, n, filter_id, folder_id):
    """Enforce I-2 and I-3."""
    if not isinstance(mask, np.ndarray):
        raise FilterViolation(f"{filter_id}/{folder_id}: mask is {type(mask)}, not ndarray")
    if mask.dtype != np.bool_:
        raise FilterViolation(f"{filter_id}/{folder_id}: mask dtype {mask.dtype}, expected bool")
    if mask.shape != (n,):
        raise FilterViolation(f"{filter_id}/{folder_id}: mask shape {mask.shape}, expected {(n,)}")
    kept = int(mask.sum())
    floor = n if is_exempt(n) else keep_floor(n)
    if kept < floor:
        raise FilterViolation(
            f"{filter_id}/{folder_id}: kept {kept}/{n} tiles, below floor {floor} "
            f"(drop budget {max_drop(n)})")
    return mask


def apply_filter(filter_id, features, signal, ctx):
    """Run a filter and validate the result. I-1 is checked before the call."""
    n = ctx.n_tiles
    if signal.attention.shape[0] != n:
        raise FilterViolation(
            f"{filter_id}/{ctx.folder_id}: signal length {signal.attention.shape[0]} "
            f"!= tile count {n}")
    if features is not None and features.shape[0] != n:
        raise FilterViolation(
            f"{filter_id}/{ctx.folder_id}: feature rows {features.shape[0]} != tile count {n}")
    if is_exempt(n):
        return np.ones(n, dtype=bool)
    if filter_id in MATCHED_FILTERS:
        # Resolve the dose HERE rather than trusting a caller-supplied value: the whole
        # point of a matched control is that it removes exactly as many tiles as the
        # reference did on this same slide, and a stale or absent count would silently
        # reintroduce the confound the matching exists to remove.
        ref = check_mask(get(REFERENCE_FILTER)(features, signal, ctx), n,
                         REFERENCE_FILTER, ctx.folder_id)
        ctx.n_drop = int(ref.size - ref.sum())
    mask = get(filter_id)(features, signal, ctx)
    return check_mask(mask, n, filter_id, ctx.folder_id)


def declared_needs(filter_id):
    """(needs_features, needs_group): function attributes set in the block after the def
    (`evolved_v0.needs_features = True`), not module constants, which _tunables() would
    sweep into every key. Absent is False; a non-bool raises. Matched controls report the
    REFERENCE's declaration, since they re-run it for the dose. getattr_static: no code of
    the filter's runs."""
    fid = REFERENCE_FILTER if filter_id in MATCHED_FILTERS else filter_id
    fn = get(fid)
    needs = []
    for name in ("needs_features", "needs_group"):
        value = inspect.getattr_static(fn, name, False)
        if type(value) is not bool:
            raise FilterViolation(f"{fid}: {name} must be True or False, got {value!r}")
        needs.append(value)
    return tuple(needs)


def apply_filter_ext(filter_id, features, signal, ctx, control_seed=None):
    """apply_filter (whose text every key hashes, so it is not edited) for declared inputs
    and a control seed. Same invariants, plus: a body gets features / ctx.group only if
    its reference declares them (else None, as through apply_filter) -- features read-only,
    so an in-place edit cannot reach the caller's tensor; matched controls take their dose
    from the reference run on the SAME inputs; control_seed, if not None, replaces
    ctx.seed for random_matched's draw only. ctx is not modified."""
    n = ctx.n_tiles
    if signal.attention.shape[0] != n:
        raise FilterViolation(
            f"{filter_id}/{ctx.folder_id}: signal length {signal.attention.shape[0]} "
            f"!= tile count {n}")
    if features is not None and features.shape[0] != n:
        raise FilterViolation(
            f"{filter_id}/{ctx.folder_id}: feature rows {features.shape[0]} != tile count {n}")
    need_f, need_g = declared_needs(filter_id)
    if need_f and features is None:
        raise FilterViolation(f"{filter_id}/{ctx.folder_id}: features_missing")
    group = ctx.group if need_g else None
    if group is not None and (type(group) is not int or group not in (0, 1)):
        raise FilterViolation(f"{filter_id}/{ctx.folder_id}: ctx.group {group!r} not 0/1/None")
    feats = None
    if need_f:
        feats = np.asarray(memoryview(np.ascontiguousarray(features)).toreadonly())
    if is_exempt(n):
        return np.ones(n, dtype=bool)
    base = (ctx.cancer, ctx.attribute, ctx.slide_type, ctx.fm, ctx.folder_id, n)
    ref_ctx = TileContext(*base, ctx.seed, group=group)
    if filter_id not in MATCHED_FILTERS:
        return check_mask(get(filter_id)(feats, signal, ref_ctx), n, filter_id,
                          ctx.folder_id)
    ref = check_mask(get(REFERENCE_FILTER)(feats, signal, ref_ctx), n,
                     REFERENCE_FILTER, ctx.folder_id)
    seed = (control_seed if filter_id == "random_matched" and control_seed is not None
            else ctx.seed)
    mctx = TileContext(*base, seed, n_drop=int(ref.size - ref.sum()), group=group)
    return check_mask(get(filter_id)(feats, signal, mctx), n, filter_id, ctx.folder_id)


# --------------------------------------------------------------------------------------
# Built-in filters
# --------------------------------------------------------------------------------------

@register("identity")
def identity(features, signal, ctx):
    """Keep every tile. The transparency control: training under this filter must
    reproduce the unmitigated baseline bit-for-bit (AC-P2)."""
    return np.ones(ctx.n_tiles, dtype=bool)


@register("attention_only")
def attention_only(features, signal, ctx):
    """Drop the tiles the demographic classifier attends to most.

    Attention alone, with none of the confidence / uncertainty / heterogeneity /
    spatial terms the reported method combines -- but at a FIXED rate, which does not
    match the reference arm's (measured: 15.00% here against 5.52% for evolved_v0).
    That difference confounds "which tiles" with "how many", so this variant does NOT
    answer R3#5; ``attention_matched`` does. Kept registered for dose-response work.
    """
    drop = min(int(round(ATTENTION_ONLY_DROP_FRAC * ctx.n_tiles)), max_drop(ctx.n_tiles))
    mask = np.ones(ctx.n_tiles, dtype=bool)
    if drop > 0:
        # Ties broken by index so the mask is deterministic (I-4).
        order = np.lexsort((np.arange(ctx.n_tiles), -signal.attention))
        mask[order[:drop]] = False
    return mask


ATTENTION_ONLY_DROP_FRAC = 0.15


def _make_random(pct):
    """Null control: drop the same fraction, chosen without looking at the signal.

    Seeded per slide so the mask is reproducible across folds and reruns (I-4).
    """
    frac = pct / 100.0

    def random_filter(features, signal, ctx):
        n = ctx.n_tiles
        drop = min(int(round(frac * n)), max_drop(n))
        mask = np.ones(n, dtype=bool)
        if drop > 0:
            seed = int(hashlib.sha256(
                f"{ctx.seed}|{ctx.folder_id}|random_p{pct}".encode("utf-8")
            ).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            mask[rng.choice(n, size=drop, replace=False)] = False
        return mask

    random_filter.__name__ = f"random_p{pct}"
    return random_filter


def _matched_drop(ctx, filter_id):
    """Dose handed down by the reference arm for this slide.

    Raises rather than falling back to a rate of its own: a matched control that quietly
    invents its own dose is indistinguishable from a fixed-rate control, and would let
    the confound back in without any visible failure.
    """
    if ctx.n_drop is None:
        raise FilterViolation(
            f"{filter_id}/{ctx.folder_id}: no reference dose. Matched controls must be "
            f"run through apply_filter(), which resolves it from {REFERENCE_FILTER!r}.")
    if not 0 <= ctx.n_drop <= max_drop(ctx.n_tiles):
        raise FilterViolation(
            f"{filter_id}/{ctx.folder_id}: reference dose {ctx.n_drop} outside "
            f"[0, {max_drop(ctx.n_tiles)}] for {ctx.n_tiles} tiles")
    return ctx.n_drop


@register("random_matched")
def random_matched(features, signal, ctx):
    """Null control at the reference arm's own dose.

    Removes exactly as many tiles as REFERENCE_FILTER removed from this slide, chosen
    uniformly at random and without looking at the demographic signal. Because the count
    is identical slide by slide, the only thing separating this arm from the reference is
    *which* tiles go -- which is precisely the claim R3#5 asks us to defend.

    Seeded per slide so the mask is reproducible across folds and reruns (I-4).
    """
    n = ctx.n_tiles
    drop = _matched_drop(ctx, "random_matched")
    mask = np.ones(n, dtype=bool)
    if drop > 0:
        seed = int(hashlib.sha256(
            f"{ctx.seed}|{ctx.folder_id}|random_matched".encode("utf-8")
        ).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        mask[rng.choice(n, size=drop, replace=False)] = False
    return mask


@register("attention_matched")
def attention_matched(features, signal, ctx):
    """Attention-only ablation at the reference arm's own dose.

    Drops the same number of tiles as REFERENCE_FILTER, but ranks them by demographic
    attention alone -- none of the confidence term the reported method mixes in. Any gap
    between this arm and the reference is therefore attributable to that extra term, not
    to how much tissue each one discarded.
    """
    n = ctx.n_tiles
    drop = _matched_drop(ctx, "attention_matched")
    mask = np.ones(n, dtype=bool)
    if drop > 0:
        # Ties broken by index so the mask is deterministic (I-4).
        order = np.lexsort((np.arange(n), -signal.attention))
        mask[order[:drop]] = False
    return mask


# --------------------------------------------------------------------------------------
# EVOLVE-BLOCK-START
# OpenEvolve rewrites the body below. Everything outside this block -- the registry, the
# invariants, the cache keying -- is fixed infrastructure and must stay untouched, so an
# evolved candidate can only change *how tiles are chosen*, never whether the checks run.
# --------------------------------------------------------------------------------------
@register("evolved_v0")
def evolved_v0(features, signal, ctx):
    """E, the S-10 winner, as the starting point of this run.

    What it does -- all of it:
      * evidence = the per-tile demographic probability, flipped when the slide is
        predicted to be the other group (tile_score if slide_pred >= 0.5, else
        1 - tile_score), so a high value always means "evidence for the predicted group";
      * attention enters as a stable percentile rank, not raw, because it is float16
        upstream and exactly 0.0 on most tiles of many slides;
      * score = 0.8 * evidence + 0.2 * attention rank, and the floor(0.30 * n)
        highest-scoring tiles are dropped (stable sort: ties go by tile index), clipped
        to max_drop(n).
    So the dose is the same 30% on every non-exempt slide; the signal decides only WHICH
    tiles go.

    Provenance: S-10 candidate 2d4f1d3c-19e7-45fc-928c-9db0e19ed2c5, selected on G (the
    frozen demographic classifier's AUC drop beyond a dose-matched random mask) and
    installed 2026-09-11. After RETRAINING, its fairness effect was indistinguishable from
    the same-dose random and attention-only comparison arms
    (analysis/R3_5_response_notes_20260912.md, section 2).
    """
    n = ctx.n_tiles
    tile = np.asarray(signal.tile_score, dtype=np.float64)
    attention = np.asarray(signal.attention, dtype=np.float64)

    prob = float(signal.slide_pred)
    evidence = tile if prob >= 0.5 else 1.0 - tile

    # Give tied float16 attention values the same rank.  In particular, all
    # exact-zero attention tiles remain tied rather than being ordered by index.
    att_order = np.argsort(attention, kind="stable")
    att_sorted = attention[att_order]
    starts = np.r_[True, att_sorted[1:] != att_sorted[:-1]]
    groups = np.cumsum(starts) - 1
    denom = float(groups[-1]) if groups[-1] else 1.0
    att_rank = np.empty(n, dtype=np.float64)
    att_rank[att_order] = groups / denom

    # Target tiles that are both demographic evidence and relevant to the
    # demographic MIL model, while retaining a small evidence floor for
    # quantized/zero-attention slides.
    score = evidence * (0.25 + 0.75 * att_rank) + 0.02 * att_rank

    drop = min(int(np.floor(0.33 * n)), max_drop(n))
    mask = np.ones(n, dtype=bool)
    if drop:
        order = np.argsort(-score, kind="stable")
        mask[order[:drop]] = False
    return mask
# EVOLVE-BLOCK-END
# --------------------------------------------------------------------------------------
