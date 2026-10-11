"""A tile-filter program. The search may rewrite only the code between the EVOLVE-BLOCK markers.

drop_scores(attention, tile_score, slide_pred) receives, for one slide, the demographic
classifier's attention over tiles, its per-tile demographic probability and its slide-level
probability. It returns a drop score per tile and the fraction of tiles to drop. The evaluator
clips the fraction to the dose cap and the keep floor, drops the highest-scoring tiles, and
draws a random control that removes the same number of tiles from the same slide.
"""
import numpy as np


# EVOLVE-BLOCK-START
def drop_scores(attention, tile_score, slide_pred):
    evidence = tile_score if slide_pred >= 0.5 else 1.0 - tile_score
    order = np.argsort(attention, kind="stable")
    ordered = attention[order]
    level = np.cumsum(np.r_[True, ordered[1:] != ordered[:-1]]) - 1
    rank = np.empty(len(attention))
    rank[order] = level / (level[-1] if level[-1] else 1.0)
    return evidence * (0.25 + 0.75 * rank) + 0.02 * rank, 0.33
# EVOLVE-BLOCK-END
