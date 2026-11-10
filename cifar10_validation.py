"""
Proper train / validation / test split for the CIFAR-10 CNN.

Before: I picked the best epoch using the TEST set, so the reported accuracy
was slightly optimistic. Now:

    train (45,000 photos)  -> the model learns from these
    validation (5,000)     -> used to pick the best epoch (and tune settings)
    test (10,000)          -> touched ONCE at the very end, for the final score

Needs cifar10_cnn.py in the same folder.

    python cifar10_validation.py --model improved --epochs 30
    python cifar10_validation.py --model baseline --epochs 15

Each run appends a row to validation_results.csv.
"""
import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from cifar10_cnn import (BaselineCNN, ImprovedCNN, train_one_epoch, evaluate,
                         MEAN, STD)

RESULTS_FILE = "validation_results.csv"


def get_split_loaders(augment, val_size, seed, batch_size=128):
    """Split the 50,000 training photos into train + validation."""
    normalize = T.Normalize(MEAN, STD)
    plain_tf = T.Compose([T.ToTensor(), normalize])
    if augment:
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    else:
        train_tf = plain_tf

    # Same photos loaded twice: once with augmentation (for training), once
    # without (for validation). Validation must never be augmented.
    train_aug = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=train_tf)
    train_plain = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=plain_tf)
    test_set = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=plain_tf)

    # Fixed random split (same seed = same split every run)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_aug), generator=g).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    train_set = Subset(train_aug, train_idx)
    val_set = Subset(train_plain, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


def plot_history(history, best_epoch, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"], label="validation")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["val_acc"], label="validation")
    for ax in (ax1, ax2):
        ax.axvline(best_epoch - 1, color="gray", linestyle="--", linewidth=1)
    ax2.set_title("Accuracy (dashed line = epoch chosen on validation)")
    ax2.set_xlabel("epoch"); ax2.legend()
    fig.suptitle(f"{name} model, train / validation")
    fig.tight_layout()
    path = f"{name}_val_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def save_result(row):
    new_file = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["model", "epochs", "best_epoch", "best_val_acc",
                             "test_acc_at_best_val", "minutes"])
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "improved"],
                        default="improved")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    improved = args.model == "improved"
    train_loader, val_loader, test_loader = get_split_loaders(
        augment=improved, val_size=args.val_size, seed=args.seed,
        batch_size=args.batch_size)
    print(f"Photos -> train: {len(train_loader.dataset):,} | "
          f"validation: {len(val_loader.dataset):,} | "
          f"test: {len(test_loader.dataset):,}")

    model = (ImprovedCNN() if improved else BaselineCNN()).to(device)
    criterion = nn.CrossEntropyLoss()
    if improved:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                      weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=3e-3, epochs=args.epochs,
            steps_per_epoch=len(train_loader))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, best_epoch = 0.0, 0
    ckpt = f"{args.model}_valbest.pt"
    start_all = time.time()

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, scheduler, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[k].append(v)

        # Pick the best epoch using VALIDATION accuracy, never the test set
        if va_acc > best_val:
            best_val, best_epoch = va_acc, epoch
            torch.save(model.state_dict(), ckpt)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.1%} | "
              f"val loss {va_loss:.3f} acc {va_acc:.1%} | "
              f"{time.time() - start:.0f}s")

    minutes = round((time.time() - start_all) / 60, 1)

    # The test set is used exactly once, on the checkpoint chosen by validation
    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\nBest validation accuracy: {best_val:.2%} (epoch {best_epoch})")
    print(f"TEST accuracy of that checkpoint: {test_acc:.2%}   <- the honest number")

    plot_history(history, best_epoch, args.model)
    save_result([args.model, args.epochs, best_epoch, f"{best_val:.4f}",
                 f"{test_acc:.4f}", minutes])


if __name__ == "__main__":
    main()
