"""
Ablation study for the improved CIFAR-10 CNN.

Turn ONE improvement off at a time and see how much accuracy drops.
Needs cifar10_cnn.py in the same folder (it reuses its data/train/eval code).

Examples (30 epochs each, about 11 min on a Colab T4):
    python cifar10_ablation.py                  # full model (reference)
    python cifar10_ablation.py --no-dropout
    python cifar10_ablation.py --no-aug
    python cifar10_ablation.py --no-bn
    python cifar10_ablation.py --no-sched

Every run appends a row to ablation_results.csv and prints the table so far.
"""
import argparse
import csv
import os
import time

import torch
import torch.nn as nn

from cifar10_cnn import get_loaders, train_one_epoch, evaluate, plot_history

RESULTS_FILE = "ablation_results.csv"


# --------------------------------------------------------------------------
# Model with switches
# --------------------------------------------------------------------------
def conv_block(in_ch, out_ch, use_bn):
    """Two convs (+ optional BatchNorm) + ReLU, then pool."""
    layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=not use_bn))
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class AblationCNN(nn.Module):
    """Same as ImprovedCNN, but BatchNorm and dropout can be switched off."""

    def __init__(self, use_bn=True, use_dropout=True):
        super().__init__()
        p = (0.1, 0.2, 0.3) if use_dropout else (0.0, 0.0, 0.0)
        self.features = nn.Sequential(
            conv_block(3, 64, use_bn), nn.Dropout(p[0]),
            conv_block(64, 128, use_bn), nn.Dropout(p[1]),
            conv_block(128, 256, use_bn), nn.Dropout(p[2]),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------
# Results table helpers
# --------------------------------------------------------------------------
def save_result(row):
    new_file = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["experiment", "best_test_acc", "final_test_acc",
                             "final_train_acc", "minutes"])
        writer.writerow(row)


def print_table():
    with open(RESULTS_FILE) as f:
        rows = list(csv.reader(f))
    print("\n| Experiment | Best test acc | Final test acc | Final train acc | Min |")
    print("|---|---|---|---|---|")
    for r in rows[1:]:
        print(f"| {r[0]} | {float(r[1]):.2%} | {float(r[2]):.2%} | "
              f"{float(r[3]):.2%} | {r[4]} |")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-dropout", action="store_true")
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--no-bn", action="store_true")
    parser.add_argument("--no-sched", action="store_true",
                        help="constant learning rate 1e-3 instead of OneCycle")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Experiment name, e.g. "full" or "no_dropout" or "no_dropout_no_aug"
    off = [n for n, flag in [("dropout", args.no_dropout), ("aug", args.no_aug),
                             ("bn", args.no_bn), ("sched", args.no_sched)] if flag]
    name = "full" if not off else "no_" + "_no_".join(off)
    print(f"Experiment: {name}")

    torch.manual_seed(args.seed)  # same starting point for fair comparison
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(augment=not args.no_aug,
                                            batch_size=args.batch_size)
    model = AblationCNN(use_bn=not args.no_bn,
                        use_dropout=not args.no_dropout).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=5e-4)
    if args.no_sched:
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=3e-3, epochs=args.epochs,
            steps_per_epoch=len(train_loader))

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    start_all = time.time()

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, scheduler, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        for k, v in zip(history, (tr_loss, tr_acc, te_loss, te_acc)):
            history[k].append(v)
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.1%} | "
              f"test loss {te_loss:.3f} acc {te_acc:.1%} | "
              f"{time.time() - start:.0f}s")

    minutes = round((time.time() - start_all) / 60, 1)
    best = max(history["test_acc"])
    print(f"\nBest test accuracy: {best:.2%}")

    plot_history(history, name)  # saves <name>_curves.png
    save_result([name, f"{best:.4f}", f"{history['test_acc'][-1]:.4f}",
                 f"{history['train_acc'][-1]:.4f}", minutes])
    print_table()


if __name__ == "__main__":
    main()
