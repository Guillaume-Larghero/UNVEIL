"""The tile filters and their invariants.

A filter turns one slide's demographic signal into a boolean keep-mask over its tiles::

    mask = make_mask(signal, "adaptive", FilterParams(), seed=0)

Filters
-------
adaptive           The method. Every tile gets a score that is high when the tile is
                   evidence for the slide's predicted demographic group and the
                   demographic classifier attends to it; the ``drop_frac`` highest-scoring
                   tiles of the slide are dropped.
random_matched     Control. Drops exactly as many tiles as ``adaptive`` drops on the same
                   slide, chosen at random (one fixed draw per slide and ``seed``).
attention_matched  Control. Drops exactly as many tiles as ``adaptive`` drops on the same
                   slide: those with the highest attention.
identity           Keeps every tile. Training under it is the unmitigated baseline.

With the default parameters, ``adaptive``, ``random_matched`` and ``attention_matched``
give exactly the masks of the filters used in the paper (``evolved_v0``, ``random_matched``
and ``attention_matched`` in reference/filters_locked_c844fcf6.py);
tests/test_filters.py checks this against that file.

Invariants. They raise; nothing is repaired or padded.
  * The signal has one value per tile of the slide.
  * A slide with at most ``min_tiles`` tiles is never filtered.
  * A filtered slide keeps at least max(min_tiles, ceil(min_keep_frac * n)) tiles.
  * The mask is a boolean array of shape (n,).
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields

import numpy as np

FILTERS = ("adaptive", "random_matched", "attention_matched", "identity")

# Bump when the arithmetic of any filter changes, so a mask_id never names masks that the
# current code would not produce. Parameters and the random seed are part of the id too.
ALGORITHM_VERSION = 1


class FilterViolation(RuntimeError):
    """A signal or a mask broke one of the invariants above."""


@dataclass(frozen=True)
class FilterParams:
    """Every number the filters use. The defaults are the values used in the paper.

    drop_frac           fraction of a filterable slide's tiles that ``adaptive`` drops:
                        floor(drop_frac * n), capped so the keep floor holds.
    evidence_base,      score = evidence * (evidence_base + attention_gain * rank)
    attention_gain,             + attention_bonus * rank, where evidence is the tile's
    attention_bonus             demographic probability for the slide's predicted group and
                                rank is the tile's attention rank in [0, 1] (tied attention
                                values share a rank).
    decision_threshold  slide_pred >= this means the slide is predicted to be group 1, so
                        a tile's evidence is tile_score; otherwise it is 1 - tile_score.
    min_tiles           slides with at most this many tiles are never filtered.
    min_keep_frac       a filtered slide keeps at least max(min_tiles,
                        ceil(min_keep_frac * n)) tiles.
    """

    drop_frac: float = 0.33
    evidence_base: float = 0.25
    attention_gain: float = 0.75
    attention_bonus: float = 0.02
    decision_threshold: float = 0.5
    min_tiles: int = 64
    min_keep_frac: float = 0.05

    def validate(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"FilterParams.{f.name} must be a finite number, got {v!r}")
        if not 0.0 <= self.drop_frac < 1.0:
            raise ValueError(f"drop_frac must be in [0, 1), got {self.drop_frac}")
        if not 0.0 <= self.decision_threshold <= 1.0:
            raise ValueError(f"decision_threshold must be in [0, 1], got "
                             f"{self.decision_threshold}")
        if int(self.min_tiles) != self.min_tiles or self.min_tiles < 0:
            raise ValueError(f"min_tiles must be a non-negative integer, got {self.min_tiles}")
        if not 0.0 <= self.min_keep_frac <= 1.0:
            raise ValueError(f"min_keep_frac must be in [0, 1], got {self.min_keep_frac}")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown FilterParams field(s): {unknown}")
        return cls(**d).validate()


class SignalBundle:
    """One slide's demographic signal.

    attention   (n,) the classifier's attention over the slide's tiles (softmax weights)
    tile_score  (n,) per-tile probability of demographic group 1, in [0, 1]
    slide_pred  the classifier's slide-level probability of group 1, in [0, 1]

    Arrays are held as float32, which is what the paper's filters read (its signals were
    stored as float16 and read back as float32; both conversions are exact).
    """

    __slots__ = ("attention", "tile_score", "slide_pred", "n_tiles", "folder_id")

    def __init__(self, attention, tile_score, slide_pred, n_tiles=None, folder_id=""):
        self.attention = np.asarray(attention).astype(np.float32)
        self.tile_score = np.asarray(tile_score).astype(np.float32)
        self.slide_pred = float(slide_pred)
        self.n_tiles = int(self.attention.shape[0] if n_tiles is None else n_tiles)
        self.folder_id = str(folder_id)
        if self.attention.ndim != 1 or self.tile_score.ndim != 1:
            raise FilterViolation(f"{folder_id}: attention and tile_score must be 1-D")
        if self.attention.shape != self.tile_score.shape:
            raise FilterViolation(f"{folder_id}: attention {self.attention.shape} != "
                                  f"tile_score {self.tile_score.shape}")
        if self.attention.shape[0] != self.n_tiles:
            raise FilterViolation(f"{folder_id}: signal length {self.attention.shape[0]} "
                                  f"!= n_tiles {self.n_tiles}")


def keep_floor(n, params):
    """The fewest tiles a filtered slide of ``n`` tiles may keep."""
    return max(int(params.min_tiles), int(np.ceil(params.min_keep_frac * n)))


def max_drop(n, params):
    """The most tiles any filter may drop from a slide of ``n`` tiles."""
    if n <= params.min_tiles:
        return 0
    return n - keep_floor(n, params)


def is_exempt(n, params):
    """Slides too small to filter: passed through untouched by every filter alike."""
    return n <= params.min_tiles


def check_mask(mask, n, params, name, folder_id=""):
    if not isinstance(mask, np.ndarray):
        raise FilterViolation(f"{name}/{folder_id}: mask is {type(mask)}, not ndarray")
    if mask.dtype != np.bool_:
        raise FilterViolation(f"{name}/{folder_id}: mask dtype {mask.dtype}, expected bool")
    if mask.shape != (n,):
        raise FilterViolation(f"{name}/{folder_id}: mask shape {mask.shape}, expected {(n,)}")
    kept = int(mask.sum())
    floor = n if is_exempt(n, params) else keep_floor(n, params)
    if kept < floor:
        raise FilterViolation(f"{name}/{folder_id}: kept {kept}/{n} tiles, below the floor "
                              f"{floor}")
    return mask


def attention_rank(attention):
    """Each tile's attention rank in [0, 1]; tiles with equal attention share a rank."""
    attention = np.asarray(attention, dtype=np.float64)
    n = attention.shape[0]
    att_order = np.argsort(attention, kind="stable")
    att_sorted = attention[att_order]
    starts = np.r_[True, att_sorted[1:] != att_sorted[:-1]]
    groups = np.cumsum(starts) - 1
    denom = float(groups[-1]) if groups[-1] else 1.0
    rank = np.empty(n, dtype=np.float64)
    rank[att_order] = groups / denom
    return rank


def adaptive_scores(signal, params):
    """The per-tile score ``adaptive`` drops the highest of."""
    tile = np.asarray(signal.tile_score, dtype=np.float64)
    prob = float(signal.slide_pred)
    evidence = tile if prob >= params.decision_threshold else 1.0 - tile
    rank = attention_rank(signal.attention)
    return (evidence * (params.evidence_base + params.attention_gain * rank)
            + params.attention_bonus * rank)


def _adaptive(signal, n, params):
    score = adaptive_scores(signal, params)
    drop = min(int(np.floor(params.drop_frac * n)), max_drop(n, params))
    mask = np.ones(n, dtype=bool)
    if drop:
        order = np.argsort(-score, kind="stable")
        mask[order[:drop]] = False
    return mask


def random_draw_seed(seed, folder_id):
    """The per-slide integer seed of random_matched's draw."""
    return int(hashlib.sha256(f"{int(seed)}|{folder_id}|random_matched".encode("utf-8"))
               .hexdigest()[:8], 16)


def _random_matched(signal, n, drop, seed):
    mask = np.ones(n, dtype=bool)
    if drop > 0:
        rng = np.random.default_rng(random_draw_seed(seed, signal.folder_id))
        mask[rng.choice(n, size=drop, replace=False)] = False
    return mask


def _attention_matched(signal, n, drop):
    mask = np.ones(n, dtype=bool)
    if drop > 0:
        # highest attention first; ties go to the lower tile index
        order = np.lexsort((np.arange(n), -signal.attention))
        mask[order[:drop]] = False
    return mask


def make_mask(signal, filter_name="adaptive", params=None, seed=0):
    """The keep-mask of one slide under ``filter_name`` (see the module docstring).

    ``seed`` is used by random_matched only: its draw for a slide is fixed by
    (seed, folder_id), so the same slide gets the same tiles in every fold and every run.
    """
    params = (params or FilterParams()).validate()
    if filter_name not in FILTERS:
        raise ValueError(f"unknown filter {filter_name!r}; choose from {FILTERS}")
    n = signal.n_tiles
    fid = signal.folder_id
    if signal.attention.shape[0] != n or signal.tile_score.shape[0] != n:
        raise FilterViolation(f"{filter_name}/{fid}: signal length != tile count {n}")
    if is_exempt(n, params):
        return np.ones(n, dtype=bool)
    if filter_name == "identity":
        return check_mask(np.ones(n, dtype=bool), n, params, filter_name, fid)
    ref = check_mask(_adaptive(signal, n, params), n, params, "adaptive", fid)
    if filter_name == "adaptive":
        return ref
    # The controls copy the method's dose on THIS slide, so only the choice of tiles differs.
    drop = int(ref.size - ref.sum())
    if not 0 <= drop <= max_drop(n, params):
        raise FilterViolation(f"{filter_name}/{fid}: dose {drop} outside "
                              f"[0, {max_drop(n, params)}]")
    if filter_name == "random_matched":
        mask = _random_matched(signal, n, drop, seed)
    else:
        mask = _attention_matched(signal, n, drop)
    return check_mask(mask, n, params, filter_name, fid)


def mask_id(filter_name, params=None, seed=0):
    """A 16-hex id of (algorithm version, filter, parameters, and the seed where it matters).

    Two mask files with the same id were built by the same rule; any change to the rule,
    a parameter, or random_matched's seed gives a different id."""
    params = (params or FilterParams()).validate()
    payload = {"algorithm_version": ALGORITHM_VERSION, "filter": filter_name,
               "params": params.to_dict(),
               "seed": int(seed) if filter_name == "random_matched" else None}
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
