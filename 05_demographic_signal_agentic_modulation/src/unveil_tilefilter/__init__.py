"""UNVEIL adaptive demographic tile filter -- implementation package.

Four steps, each usable on its own (see README.md):

    signal    per-tile demographic signal from an attention-MIL demographic classifier
    masks     per-slide keep-masks from that signal (the filter)
    training  apply the masks to TRAINING bags only (generic wrapper or fairtune adapter)
    resolve   score a finished run with the paper's resolution rule
"""

__version__ = "1.0.0"
