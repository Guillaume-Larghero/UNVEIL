import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_demographic_classifier"))

CHUNK = 4096


def load_classifier(checkpoint):
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    return model.eval()


@torch.inference_mode()
def attention_logits(model, x):
    if getattr(model, "input_batch_norm_features", None):
        x = x.clone()
        x[:, model.input_batch_norm_features] = model.batch_norm(x[:, model.input_batch_norm_features])
    head = model.deep_attention
    logits = torch.cat([head.W(torch.tanh(head.theta1(c)) * torch.sigmoid(head.theta2(c))).squeeze(-1) for c in x.split(CHUNK)])
    return x, logits / head.temperature


@torch.inference_mode()
def bag_prediction(model, x, logits, keep=None):
    if keep is not None:
        keep = torch.as_tensor(keep, dtype=torch.bool)
        x, logits = x[keep], logits[keep]
    attention = torch.softmax(logits, dim=0)
    return float(model.mlp(((x * attention[:, None]).sum(dim=0))[None]).ravel()[0])


@torch.inference_mode()
def demographic_signal(model, x):
    x, logits = attention_logits(model, x)
    attention = torch.softmax(logits, dim=0)
    tile_score = torch.cat([model.mlp(c).squeeze(-1) for c in x.split(CHUNK)])
    return attention.numpy(), tile_score.numpy(), bag_prediction(model, x, logits)


def load_tiles(path):
    x = torch.load(path, map_location="cpu", weights_only=True).float()
    x = x.squeeze(0) if x.ndim == 3 else x
    if not torch.isfinite(x).all():
        raise ValueError(f"{path}: non-finite tile features")
    return x


def main():
    parser = argparse.ArgumentParser(description="Per-tile attention and demographic score from a trained 01_demographic_classifier model.")
    parser.add_argument("--checkpoint", required=True, help="model checkpoint saved by 01_demographic_classifier")
    parser.add_argument("--features-dir", required=True, help="directory of <slide>.pt tile feature tensors")
    parser.add_argument("--slides", default=None, help="text file of slide ids to export; defaults to every .pt file")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    model = load_classifier(args.checkpoint)
    features_dir, output_dir = Path(args.features_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slides = Path(args.slides).read_text().split() if args.slides else sorted(p.stem for p in features_dir.glob("*.pt"))
    for slide in slides:
        attention, tile_score, slide_pred = demographic_signal(model, load_tiles(features_dir / f"{slide}.pt"))
        np.savez(output_dir / f"{slide}.npz", attention=attention.astype(np.float16), tile_score=tile_score.astype(np.float32), slide_pred=np.float32(slide_pred), n_tiles=np.int64(len(attention)))


if __name__ == "__main__":
    main()
