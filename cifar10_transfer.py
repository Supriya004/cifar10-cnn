"""
Transfer learning on CIFAR-10 with a pretrained ResNet18.

Tests the hypothesis from the confusion matrix: does a model pretrained on
ImageNet separate cats from dogs better than my CNN trained from scratch?

Uses a proper train / validation / test split: the best epoch is chosen on
VALIDATION accuracy and the test set is used once, at the end.

Needs cifar10_cnn.py, cifar10_validation.py and confusion_matrix.py in the
same folder.

    python cifar10_transfer.py --mode head --epochs 3   # train only the new last layer
    python cifar10_transfer.py --mode full --epochs 5   # fine-tune the whole network

Control experiment (same ResNet18, but random weights instead of ImageNet ones):
    python cifar10_transfer.py --mode full --epochs 5 --init scratch
    python cifar10_transfer.py --mode full --epochs 20 --init scratch --lr 1e-3
Scratch runs are saved as scratch_full_* so they never overwrite the pretrained
model's files.

Each run appends a row to transfer_results.csv and prints per-class accuracy.
"""
import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from cifar10_cnn import train_one_epoch, evaluate, CLASSES, MEAN, STD
from cifar10_validation import get_split_loaders, plot_history
from confusion_matrix import collect_predictions, build_matrix, report

RESULTS_FILE = "transfer_results.csv"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TransferNet(nn.Module):
    """Pretrained ResNet18 with a new 10-class output layer.

    Takes the same normalized 32x32 batches as the other scripts, converts
    them to the ImageNet normalization the pretrained weights expect, and
    upscales them (pretrained networks work much better on bigger images).
    """

    def __init__(self, size=128, freeze_backbone=False, pretrained=True):
        super().__init__()
        weights = (torchvision.models.ResNet18_Weights.DEFAULT
                   if pretrained else None)  # None = random initialization
        self.backbone = torchvision.models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(512, 10)  # new layer, randomly initialized
        self.size = size
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                p.requires_grad = name.startswith("fc.")

        self.register_buffer("c_mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("c_std", torch.tensor(STD).view(1, 3, 1, 1))
        self.register_buffer("i_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("i_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone and mode:
            # keep the frozen part fully fixed (BatchNorm statistics too)
            for name, module in self.backbone.named_children():
                if name != "fc":
                    module.eval()
        return self

    def forward(self, x):
        x = x * self.c_std + self.c_mean          # undo CIFAR normalization
        x = (x - self.i_mean) / self.i_std        # apply ImageNet normalization
        x = F.interpolate(x, size=self.size, mode="bilinear",
                          align_corners=False)    # 32x32 -> size x size
        return self.backbone(x)


def save_result(row):
    new_file = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["mode", "epochs", "size", "best_epoch",
                             "best_val_acc", "test_acc_at_best_val", "minutes"])
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["head", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init", choices=["pretrained", "scratch"],
                        default="pretrained",
                        help="scratch = same ResNet18 with random weights (control)")
    args = parser.parse_args()
    if args.init == "scratch" and args.mode == "head":
        parser.error("--init scratch only makes sense with --mode full")

    torch.manual_seed(args.seed)

    lr = args.lr or (1e-3 if args.mode == "head" else 3e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = args.init == "pretrained"
    run_name = f"transfer_{args.mode}" if pretrained else f"scratch_{args.mode}"
    print(f"Mode: {args.mode} | init: {args.init} | epochs: {args.epochs} | "
          f"size: {args.size} | lr: {lr} | device: {device}")

    train_loader, val_loader, test_loader = get_split_loaders(
        augment=True, val_size=args.val_size, seed=args.seed,
        batch_size=args.batch_size)
    model = TransferNet(size=args.size,
                        freeze_backbone=(args.mode == "head"),
                        pretrained=pretrained).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,} of {total:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, best_epoch = 0.0, 0
    ckpt = f"{run_name}_valbest.pt"
    start_all = time.time()

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, scheduler, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[k].append(v)

        # Pick the best epoch on VALIDATION accuracy, never on the test set
        if va_acc > best_val:
            best_val, best_epoch = va_acc, epoch
            torch.save(model.state_dict(), ckpt)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.1%} | "
              f"val loss {va_loss:.3f} acc {va_acc:.1%} | "
              f"{time.time() - start:.0f}s")

    minutes = round((time.time() - start_all) / 60, 1)

    # Test set used exactly once, on the checkpoint chosen by validation
    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nBest validation accuracy: {best_val:.2%} (epoch {best_epoch})")
    print(f"TEST accuracy of that checkpoint: {test_acc:.2%}")

    plot_history(history, best_epoch, run_name)
    save_result([args.mode if pretrained else f"{args.mode}-scratch",
                 args.epochs, args.size, best_epoch,
                 f"{best_val:.4f}", f"{test_acc:.4f}", minutes])

    # Per-class results: the real test of the cat vs dog hypothesis
    preds, labels = collect_predictions(model, test_loader, device)
    cm = build_matrix(labels, preds, len(CLASSES))
    report(cm, CLASSES)

    cat, dog = CLASSES.index("cat"), CLASSES.index("dog")
    print(f"\ncat <-> dog swaps: {cm[dog, cat] + cm[cat, dog]} photos "
          f"(dog->cat {cm[dog, cat]}, cat->dog {cm[cat, dog]})")
    print("Improved CNN from scratch for reference: 157 swaps "
          "(dog->cat 81, cat->dog 76), cat 82.5%, dog 86.5% "
          "(that checkpoint was picked on the test set, so slightly optimistic)")


if __name__ == "__main__":
    main()
