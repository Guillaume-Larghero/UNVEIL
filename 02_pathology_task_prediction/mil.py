import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

MANIFEST_COLUMNS = ["cohort", "cancer", "task", "patient_id", "slide_id", "label", "fold", "feature_path"]
PREDICTION_COLUMNS = MANIFEST_COLUMNS[:-1]
LEGACY_PREFIXES = {"featureExtractor.attentionlayer.": "attention.", "fc_target.": "classifier."}


class GatedAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        logits = self.attention_weights(self.attention_V(x) * self.attention_U(x)).squeeze(-1)
        return torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=1)


class AttentionMIL(nn.Module):
    def __init__(self, feature_dim, hidden_dims=(256, 128), n_classes=2, dropout=0.25):
        super().__init__()
        self.attention = GatedAttention(feature_dim)
        layers, width = [], feature_dim
        for size in hidden_dims:
            layers += [nn.Linear(width, size), nn.Dropout(dropout), nn.ReLU(), nn.LayerNorm(size)]
            width = size
        self.classifier = nn.Sequential(*layers, nn.Linear(width, n_classes))

    def forward(self, x, mask, return_attention=False):
        weights = self.attention(x, mask)
        logits = self.classifier(torch.bmm(weights.unsqueeze(1), x).squeeze(1))
        return (logits, weights) if return_attention else logits


def load_state_dict(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    renamed = {}
    for key, value in state.items():
        for old, new in LEGACY_PREFIXES.items():
            if key.startswith(old):
                key = new + key[len(old):]
        renamed[key] = value
    return renamed


def load_model(path, dropout=0.0):
    state = load_state_dict(path)
    feature_dim = state["attention.attention_V.0.weight"].shape[1]
    linear = [k for k in state if k.startswith("classifier.") and k.endswith(".weight") and state[k].ndim == 2]
    widths = [state[k].shape[0] for k in sorted(linear, key=lambda k: int(k.split(".")[1]))]
    model = AttentionMIL(feature_dim, hidden_dims=widths[:-1], n_classes=widths[-1], dropout=dropout)
    model.load_state_dict(state)
    return model


def read_manifest(path):
    manifest = pd.read_csv(path, sep="\t" if str(path).endswith(".tsv") else ",")
    missing = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    if manifest["slide_id"].duplicated().any():
        raise ValueError(f"{path} lists some slides more than once")
    return manifest.astype({"patient_id": str, "slide_id": str, "label": int, "fold": int})


def load_features(path):
    features = torch.load(path, map_location="cpu", weights_only=True)
    return features.reshape(-1, features.shape[-1]).float()


class SlideBags(Dataset):
    def __init__(self, slides, max_tiles=None, tile_masks=None):
        self.slides = slides.reset_index(drop=True)
        self.max_tiles = max_tiles
        self.tile_masks = tile_masks
        if tile_masks is not None:
            missing = sorted(set(self.slides["slide_id"]) - set(tile_masks))
            if missing:
                raise KeyError(f"{len(missing)} training slides have no tile mask, e.g. {missing[:3]}")

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, index):
        row = self.slides.loc[index]
        features = load_features(row["feature_path"])
        if self.tile_masks is not None:
            mask = self.tile_masks[row["slide_id"]]
            if len(mask) != len(features):
                raise ValueError(f"{row['slide_id']}: mask has {len(mask)} tiles, features have {len(features)}")
            features = features[torch.from_numpy(mask)]
        if self.max_tiles and len(features) > self.max_tiles:
            features = features[torch.randperm(len(features))[: self.max_tiles]]
        return features, int(row["label"]), index


def pad_bags(batch):
    features, labels, index = zip(*batch)
    lengths = torch.tensor([len(f) for f in features])
    x = torch.zeros(len(features), int(lengths.max()), features[0].shape[1])
    for i, f in enumerate(features):
        x[i, : len(f)] = f
    mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]
    return x, mask, torch.tensor(labels), torch.tensor(index)


def load_tile_masks(path):
    with np.load(path) as stored:
        return {key: stored[key].astype(bool) for key in stored.files}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probabilities, indices, losses = [], [], []
    for x, mask, y, index in loader:
        logits = model(x.to(device), mask.to(device))
        losses.append(nn.functional.cross_entropy(logits, y.to(device), reduction="none").cpu())
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu())
        indices.append(index)
    order = torch.cat(indices).argsort()
    return torch.cat(probabilities)[order].numpy(), float(torch.cat(losses).mean())
