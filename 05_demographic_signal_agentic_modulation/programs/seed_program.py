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
    rank = np.empty(len(attention))
    rank[np.argsort(attention, kind="stable")] = np.linspace(0.0, 1.0, len(attention))
    return 0.8 * evidence + 0.2 * rank, 0.30
# EVOLVE-BLOCK-END
