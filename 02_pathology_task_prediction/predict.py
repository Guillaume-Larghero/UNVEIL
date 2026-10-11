import argparse

import torch
from torch.utils.data import DataLoader

from mil import PREDICTION_COLUMNS, SlideBags, load_model, pad_bags, predict, read_manifest


def main():
    parser = argparse.ArgumentParser(description="Slide predictions from a trained attention-MIL checkpoint, e.g. on an external cohort.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True, help="model.pt from train.py or a released checkpoint")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="restrict to these folds; defaults to every slide")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    if args.folds is not None:
        manifest = manifest[manifest["fold"].isin(args.folds)]
    device = torch.device(args.device)
    model = load_model(args.checkpoint).to(device)
    loader = DataLoader(SlideBags(manifest), batch_size=args.batch_size, collate_fn=pad_bags, num_workers=args.num_workers)
    probability, _ = predict(model, loader, device)
    manifest[PREDICTION_COLUMNS].assign(probability=probability).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
