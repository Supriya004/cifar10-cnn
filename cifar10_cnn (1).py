"""
CIFAR-10 image classifier: from a basic CNN to a ~90% accuracy CNN.

Run (Colab/Kaggle: switch runtime to GPU first):
    python cifar10_cnn.py --model baseline --epochs 15
    python cifar10_cnn.py --model improved --epochs 30

Compare the two runs. The gap between them is the whole lesson.
"""
import argparse
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

CLASSES = ("plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
MEAN = (0.4914, 0.4822, 0.4465)   # per-channel mean of CIFAR-10 train set
STD = (0.2470, 0.2435, 0.2616)    # per-channel std


# --------------------------------------------------------------------------
# STEP 1: Data
# --------------------------------------------------------------------------
def get_loaders(augment: bool, batch_size: int = 128):
    """Return train/test DataLoaders. `augment` adds random crops + flips."""
    normalize = T.Normalize(MEAN, STD)

    if augment:
        # Augmentation = show the network slightly different versions of each
        # image every epoch, so it can't just memorize the training set.
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    else:
        train_tf = T.Compose([T.ToTensor(), normalize])

    test_tf = T.Compose([T.ToTensor(), normalize])  # never augment test data

    train_set = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256,
                             shuffle=False, num_workers=2)
    return train_loader, test_loader


# --------------------------------------------------------------------------
# STEP 2: Baseline model
# --------------------------------------------------------------------------
class BaselineCNN(nn.Module):
    """3 conv layers + 2 linear layers. Simple, expect ~70-75% accuracy."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # input: 3 x 32 x 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # -> 32 x 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # -> 64 x 8 x 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # -> 128 x 4 x 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, 10),  # 10 raw scores (logits), one per class
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------
# STEP 3: Improved model (BatchNorm + Dropout + deeper blocks)
# --------------------------------------------------------------------------
def conv_block(in_ch, out_ch):
    """Two conv layers, each followed by BatchNorm + ReLU, then pool."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ImprovedCNN(nn.Module):
    """VGG-style network. With augmentation + OneCycle LR: ~88-92%."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 64),      # -> 64 x 16 x 16
            nn.Dropout(0.1),
            conv_block(64, 128),    # -> 128 x 8 x 8
            nn.Dropout(0.2),
            conv_block(128, 256),   # -> 256 x 4 x 4
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------
# STEP 4: Training and evaluation loops
# --------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()  # enables dropout + batchnorm updates
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()              # clear old gradients
        outputs = model(images)            # forward pass
        loss = criterion(outputs, labels)  # how wrong are we?
        loss.backward()                    # backprop: compute gradients
        optimizer.step()                   # update weights
        if scheduler is not None:
            scheduler.step()               # OneCycleLR steps every batch

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()  # disables dropout, uses running stats in batchnorm
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


# --------------------------------------------------------------------------
# STEP 5: Inspect results
# --------------------------------------------------------------------------
def plot_history(history, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["test_loss"], label="test")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["test_acc"], label="test")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.suptitle(f"{name} model")
    fig.tight_layout()
    fig.savefig(f"{name}_curves.png", dpi=120)
    print(f"Saved {name}_curves.png")


@torch.no_grad()
def show_predictions(model, loader, device, name, n=8):
    model.eval()
    images, labels = next(iter(loader))
    preds = model(images[:n].to(device)).argmax(1).cpu()

    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.6))
    for i, ax in enumerate(axes):
        img = (images[i] * std + mean).clamp(0, 1)  # undo normalization
        ax.imshow(img.permute(1, 2, 0))
        ok = preds[i] == labels[i]
        ax.set_title(f"{CLASSES[preds[i]]}\n(true: {CLASSES[labels[i]]})",
                     fontsize=8, color="green" if ok else "red")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{name}_predictions.png", dpi=120)
    print(f"Saved {name}_predictions.png")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "improved"],
                        default="baseline")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    improved = args.model == "improved"
    train_loader, test_loader = get_loaders(augment=improved,
                                            batch_size=args.batch_size)

    model = (ImprovedCNN() if improved else BaselineCNN()).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()

    if improved:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                      weight_decay=5e-4)
        # OneCycle: LR warms up to max_lr then anneals down. Big win for free.
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=3e-3, epochs=args.epochs,
            steps_per_epoch=len(train_loader))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = None

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, scheduler, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)

        for k, v in zip(history, (tr_loss, tr_acc, te_loss, te_acc)):
            history[k].append(v)

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), f"{args.model}_best.pt")

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.1%} | "
              f"test loss {te_loss:.3f} acc {te_acc:.1%} | "
              f"{time.time() - start:.0f}s")

    print(f"\nBest test accuracy: {best_acc:.2%}")
    plot_history(history, args.model)

    model.load_state_dict(torch.load(f"{args.model}_best.pt",
                                     map_location=device))
    show_predictions(model, test_loader, device, args.model)


if __name__ == "__main__":
    main()
