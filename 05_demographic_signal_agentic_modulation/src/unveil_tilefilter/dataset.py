"""Step 3a: apply the masks in your own training pipeline.

Only TRAINING bags are filtered. Wrap only your training dataset: validation, test and
external-cohort bags must stay whole, because the method changes what the model learns
from, not what it is evaluated on.

    from unveil_tilefilter.dataset import MaskedBagDataset
    from unveil_tilefilter.masks import MaskSet

    train = MaskedBagDataset(my_train_dataset, MaskSet(["masks/train_cell.npz"]),
                             folder_ids=my_train_folder_ids,   # one per item, same order
                             bag_index=0, max_tiles=2000)

Requirements on your dataset:
  * item ``bag_index`` of what it returns is the slide's (n_tiles, dim) bag, rows in the
    order of the slide's feature file (the order the signal was computed in);
  * it does NOT subsample tiles itself. Subsample here instead (``max_tiles``), so the
    filter sees the whole bag and the sample is drawn from the tiles it keeps -- the order
    the paper's pipeline uses.
"""

import numpy as np

from .masks import MaskError


def apply_mask(bag, mask, folder_id=""):
    """The rows of ``bag`` (torch tensor or numpy array) that ``mask`` keeps."""
    mask = np.asarray(mask)
    n = int(bag.shape[0])
    if mask.dtype != np.bool_ or mask.shape != (n,):
        raise MaskError(f"{folder_id}: mask {mask.dtype} {mask.shape} does not fit a bag of "
                        f"{n} tiles; the mask must come from this slide's signal, computed "
                        f"on the same feature file")
    if hasattr(bag, "index_select"):                     # torch.Tensor
        import torch

        return bag[torch.from_numpy(mask)]
    return bag[mask]


class _DictMasks:
    def __init__(self, d):
        self._d = {str(k): v for k, v in d.items()}

    def has(self, fid):
        return str(fid) in self._d

    def get(self, fid):
        try:
            return self._d[str(fid)]
        except KeyError:
            raise MaskError(f"no mask for slide {fid!r}") from None


try:
    from torch.utils.data import Dataset as _Base
except ImportError:                                      # torch is optional for the rest
    _Base = object


class MaskedBagDataset(_Base):
    """A training dataset whose bags are filtered by precomputed masks.

    base          your training dataset
    masks         a MaskSet, or a dict {folder_id: bool array}
    folder_ids    the slide id of each item: a sequence (same length and order as base) or
                  a callable idx -> folder_id
    bag_index     position of the bag in the tuple base[idx] returns (ignored if base
                  returns the bag alone)
    length_index  position of a tile count to update after filtering, if your dataset
                  returns one (the paper's pipeline returns (bag, len(bag), ...))
    max_tiles     if set, bags still longer than this after filtering are subsampled to it
                  with torch.randperm (a DataLoader seeds torch per worker and per epoch)

    Every slide of ``base`` must have a mask; this is checked once, at construction.
    """

    def __init__(self, base, masks, folder_ids, bag_index=0, length_index=None,
                 max_tiles=None):
        self.base = base
        self.masks = _DictMasks(masks) if isinstance(masks, dict) else masks
        self._fid = folder_ids if callable(folder_ids) else list(folder_ids).__getitem__
        if not callable(folder_ids) and len(folder_ids) != len(base):
            raise MaskError(f"{len(folder_ids)} folder_ids for a dataset of {len(base)} items")
        self.bag_index = bag_index
        self.length_index = length_index
        if max_tiles is not None and int(max_tiles) < 1:
            raise ValueError("max_tiles must be a positive integer or None")
        self.max_tiles = None if max_tiles is None else int(max_tiles)
        missing = sorted({str(self._fid(i)) for i in range(len(base))
                          if not self.masks.has(self._fid(i))})
        if missing:
            raise MaskError(f"{len(missing)} training slide(s) have no mask, e.g. "
                            f"{missing[:5]}; build masks for every training slide first")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        single = not isinstance(item, (tuple, list))
        parts = [item] if single else list(item)
        fid = self._fid(idx)
        bag = apply_mask(parts[self.bag_index if not single else 0], self.masks.get(fid), fid)
        if self.max_tiles is not None and bag.shape[0] > self.max_tiles:
            import torch

            sel = torch.randperm(int(bag.shape[0]))[: self.max_tiles]
            bag = bag[sel] if hasattr(bag, "index_select") else bag[sel.numpy()]
        if single:
            return bag
        parts[self.bag_index] = bag
        if self.length_index is not None:
            parts[self.length_index] = int(bag.shape[0])
        return tuple(parts) if isinstance(item, tuple) else parts
