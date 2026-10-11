import hashlib

import numpy as np

MIN_TILES = 64
MIN_KEEP_FRACTION = 0.05
DROP_FRACTION = 0.33
EVIDENCE_FLOOR = 0.25
ATTENTION_FLOOR = 0.02
METHODS = ("signal_aware", "random_matched", "attention_matched")


def load_signal(path):
    with np.load(path) as signal:
        attention = signal["attention"].astype(np.float32)
        tile_score = signal["tile_score"].astype(np.float32)
        slide_pred = float(signal["slide_pred"])
        n_tiles = int(signal["n_tiles"])
    if attention.shape != (n_tiles,) or tile_score.shape != (n_tiles,):
        raise ValueError(f"{path}: signal arrays do not match n_tiles={n_tiles}")
    return attention, tile_score, slide_pred


def keep_floor(n_tiles):
    return max(MIN_TILES, int(np.ceil(MIN_KEEP_FRACTION * n_tiles)))


def max_drop(n_tiles):
    return 0 if n_tiles <= MIN_TILES else n_tiles - keep_floor(n_tiles)


def dense_rank(values):
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    rank = np.cumsum(np.r_[True, ordered[1:] != ordered[:-1]]) - 1
    scaled = np.empty(len(values))
    scaled[order] = rank / (rank[-1] if rank[-1] else 1.0)
    return scaled


def signal_scores(attention, tile_score, slide_pred):
    tile_score = np.asarray(tile_score, dtype=np.float64)
    evidence = tile_score if slide_pred >= 0.5 else 1.0 - tile_score
    attention_rank = dense_rank(np.asarray(attention, dtype=np.float64))
    return evidence * (EVIDENCE_FLOOR + (1 - EVIDENCE_FLOOR) * attention_rank) + ATTENTION_FLOOR * attention_rank


def drop_top(scores, n_drop):
    mask = np.ones(len(scores), dtype=bool)
    if n_drop:
        mask[np.argsort(-scores, kind="stable")[:n_drop]] = False
    return mask


def signal_aware_mask(attention, tile_score, slide_pred):
    n = len(attention)
    return drop_top(signal_scores(attention, tile_score, slide_pred), min(int(np.floor(DROP_FRACTION * n)), max_drop(n)))


def random_matched_mask(n_tiles, n_drop, slide_key, seed=0):
    mask = np.ones(n_tiles, dtype=bool)
    if n_drop:
        stream = int(hashlib.sha256(f"{seed}|{slide_key}|random_matched".encode()).hexdigest()[:8], 16)
        mask[np.random.default_rng(stream).choice(n_tiles, size=n_drop, replace=False)] = False
    return mask


def attention_matched_mask(attention, n_drop):
    mask = np.ones(len(attention), dtype=bool)
    if n_drop:
        mask[np.lexsort((np.arange(len(attention)), -np.asarray(attention, dtype=np.float64)))[:n_drop]] = False
    return mask


def tile_mask(method, attention, tile_score, slide_pred, slide_key, seed=0):
    n = len(attention)
    if len(tile_score) != n:
        raise ValueError(f"{slide_key}: attention has {n} tiles, tile_score has {len(tile_score)}")
    if n <= MIN_TILES:
        return np.ones(n, dtype=bool)
    reference = signal_aware_mask(attention, tile_score, slide_pred)
    if method == "signal_aware":
        mask = reference
    elif method == "random_matched":
        mask = random_matched_mask(n, int(n - reference.sum()), slide_key, seed)
    elif method == "attention_matched":
        mask = attention_matched_mask(attention, int(n - reference.sum()))
    else:
        raise ValueError(f"unknown method {method}; expected one of {METHODS}")
    if mask.sum() < keep_floor(n):
        raise ValueError(f"{slide_key}: kept {int(mask.sum())} of {n} tiles, below the floor of {keep_floor(n)}")
    return mask
