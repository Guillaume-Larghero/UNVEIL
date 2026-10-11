import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mil import AttentionMIL, SlideBags, load_features, load_tile_masks, pad_bags, predict, read_manifest
from unveil.fairness import auc

PREDICTION_COLUMNS = ["cohort", "cancer", "task", "patient_id", "slide_id", "label", "fold"]


def split_fold(manifest, test_fold):
    folds = sorted(manifest["fold"].unique())
    validation_fold = folds[(folds.index(test_fold) + 1) % len(folds)]
    test = manifest["fold"] == test_fold
    validation = manifest["fold"] == validation_fold
    return manifest[~test & ~validation], manifest[validation], manifest[test]


def patient_auc(slides, probability):
    units = slides.assign(probability=probability).groupby(["patient_id", "label"], as_index=False)["probability"].mean()
    return auc(units["label"].to_numpy(), units["probability"].to_numpy())


def class_weights(labels):
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = 1 / counts
    return torch.tensor(weights / weights.sum(), dtype=torch.float32)


def train_epoch(model, loader, optimizer, weights, accumulation, device):
    model.train()
    optimizer.zero_grad()
    total = 0.0
    for step, (x, mask, y, _) in enumerate(loader, 1):
        loss = torch.nn.functional.cross_entropy(model(x.to(device), mask.to(device)), y.to(device), weight=weights, reduction="none").mean()
        (loss / accumulation).backward()
        total += float(loss)
        if step % accumulation == 0 or step == len(loader):
            optimizer.step()
            optimizer.zero_grad()
    return total / len(loader)


def run_fold(manifest, test_fold, args, tile_masks, device):
    train, validation, test = split_fold(manifest, test_fold)
    if train["label"].nunique() < 2:
        raise ValueError(f"fold {test_fold}: the training split has a single label class")
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader = lambda slides, **kw: DataLoader(slides, collate_fn=pad_bags, num_workers=args.num_workers, **kw)
    train_loader = loader(SlideBags(train, args.max_train_tiles, tile_masks), batch_size=args.batch_size, shuffle=True, generator=generator)
    validation_loader = loader(SlideBags(validation), batch_size=args.eval_batch_size)
    test_loader = loader(SlideBags(test), batch_size=args.eval_batch_size)

    feature_dim = load_features(train["feature_path"].iloc[0]).shape[1]
    model = AttentionMIL(feature_dim, args.hidden_dims, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    weights = class_weights(train["label"].to_numpy()).to(device)
    select_on_auc = validation["label"].nunique() == 2

    fold_dir = Path(args.output_dir) / f"fold_{test_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best, best_epoch, history = -np.inf, -1, []
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, weights, args.accumulation_steps, device)
        scheduler.step()
        probability, validation_loss = predict(model, validation_loader, device)
        validation_auc = patient_auc(validation, probability)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "validation_auc": validation_auc})
        score = validation_auc if select_on_auc else -validation_loss
        if score > best:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), fold_dir / "model.pt")
        elif epoch - best_epoch >= args.patience:
            break

    model.load_state_dict(torch.load(fold_dir / "model.pt", map_location=device, weights_only=True))
    probability, _ = predict(model, test_loader, device)
    predictions = test[PREDICTION_COLUMNS].assign(probability=probability)
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    summary = {
        "test_fold": int(test_fold),
        "validation_fold": int(validation["fold"].iloc[0]),
        "selection": "validation_patient_auc" if select_on_auc else "validation_loss",
        "best_epoch": best_epoch,
        "n_slides": {"train": len(train), "validation": len(validation), "test": len(test)},
        "n_patients": {name: int(s["patient_id"].nunique()) for name, s in (("train", train), ("validation", validation), ("test", test))},
        "test_patient_auc": patient_auc(test, probability),
        "feature_dim": feature_dim,
        "tile_masks": args.tile_masks,
    }
    (fold_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Cross-validated attention-MIL training for one binary pathology task.")
    parser.add_argument("--manifest", required=True, help="output of build_task_manifest.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="test folds to run; defaults to all")
    parser.add_argument("--tile-masks", default=None, help="npz of per-slide boolean tile masks, applied to training bags only")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-decay", type=float, default=0.955)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--max-train-tiles", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    if manifest.groupby("patient_id")["fold"].nunique().gt(1).any():
        raise ValueError("some patients appear in more than one fold")
    tile_masks = load_tile_masks(args.tile_masks) if args.tile_masks else None
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))
    folds = args.folds if args.folds is not None else sorted(manifest["fold"].unique())
    predictions = pd.concat([run_fold(manifest, k, args, tile_masks, torch.device(args.device)) for k in folds])
    predictions.to_csv(output / ("predictions.csv" if args.folds is None else f"predictions_folds_{'_'.join(map(str, folds))}.csv"), index=False)


if __name__ == "__main__":
    main()
