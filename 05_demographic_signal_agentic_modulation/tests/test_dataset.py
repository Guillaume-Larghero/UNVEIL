"""The generic training-dataset wrapper."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import raises, run_module_tests  # noqa: E402

import torch  # noqa: E402

from unveil_tilefilter.dataset import MaskedBagDataset, apply_mask  # noqa: E402
from unveil_tilefilter.masks import MaskError  # noqa: E402


class Bags(torch.utils.data.Dataset):
    """Returns (bag, len(bag), label) like the paper's pipeline; bag rows are 0..n-1."""

    def __init__(self, sizes):
        self.sizes = sizes

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, i):
        n = self.sizes[i]
        return torch.arange(n, dtype=torch.float32)[:, None].repeat(1, 3), n, i % 2


def _masks(sizes):
    rng = np.random.default_rng(0)
    return {f"S{i}": rng.random(n) > 0.4 for i, n in enumerate(sizes)}


def test_filters_bag_and_updates_length():
    sizes = [50, 200, 3000]
    masks = _masks(sizes)
    ds = MaskedBagDataset(Bags(sizes), masks, [f"S{i}" for i in range(3)], bag_index=0,
                          length_index=1)
    for i in range(3):
        bag, n, label = ds[i]
        kept = np.flatnonzero(masks[f"S{i}"])
        assert np.array_equal(bag[:, 0].numpy(), kept) and n == len(kept) and label == i % 2


def test_max_tiles_subsamples_the_kept_tiles():
    sizes = [3000]
    masks = _masks(sizes)
    ds = MaskedBagDataset(Bags(sizes), masks, lambda i: f"S{i}", length_index=1,
                          max_tiles=100)
    torch.manual_seed(1)
    bag, n, _ = ds[0]
    assert bag.shape == (100, 3) and n == 100
    assert set(bag[:, 0].long().tolist()) <= set(np.flatnonzero(masks["S0"]).tolist())
    assert len(set(bag[:, 0].long().tolist())) == 100            # without replacement


def test_refusals():
    sizes = [50, 60]
    masks = _masks(sizes)
    assert raises(MaskError, MaskedBagDataset, Bags(sizes), {"S0": masks["S0"]}, ["S0", "S1"])
    assert raises(MaskError, MaskedBagDataset, Bags(sizes), masks, ["S0"])
    bad = dict(masks, S1=np.ones(61, dtype=bool))
    ds = MaskedBagDataset(Bags(sizes), bad, ["S0", "S1"])
    assert raises(MaskError, ds.__getitem__, 1)
    assert raises(MaskError, apply_mask, np.zeros((5, 2)), np.ones(5, dtype=np.int8))
    assert raises(ValueError, MaskedBagDataset, Bags(sizes), masks, ["S0", "S1"],
                  max_tiles=0)


def test_numpy_bag_and_single_return():
    class One(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, i):
            return np.arange(80)[:, None]

    m = {"S0": np.arange(80) % 2 == 0}
    out = MaskedBagDataset(One(), m, ["S0"])[0]
    assert isinstance(out, np.ndarray) and out[:, 0].tolist() == list(range(0, 80, 2))


if __name__ == "__main__":
    sys.exit(run_module_tests(globals()))
