"""
Confusion matrix for the improved CIFAR-10 CNN.

Shows WHICH classes the model mixes up, not just how often it is wrong.
Needs cifar10_cnn.py in the same folder and a trained checkpoint:

    python cifar10_cnn.py --model improved --epochs 30     # makes improved_best.pt
    python confusion_matrix.py

Outputs:
    confusion_matrix.png   - 10x10 grid (rows = true class, columns = predicted)
    misclassified.png      - 16 test photos the model got wrong
    plus per-class accuracy and the most common mistakes printed in the console
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from cifar10_cnn import ImprovedCNN, get_loaders, CLASSES, MEAN, STD


# --------------------------------------------------------------------------
# 1. Get predictions for all 10,000 test photos
# --------------------------------------------------------------------------
@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        preds = model(images.to(device)).argmax(1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


# --------------------------------------------------------------------------
# 2. Build the matrix: cm[true, predicted] = how many photos
# --------------------------------------------------------------------------
def build_matrix(labels, preds, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(labels, preds):
        cm[true, pred] += 1
    return cm


# --------------------------------------------------------------------------
# 3. Report: per-class accuracy and the biggest mistakes
# --------------------------------------------------------------------------
def report(cm, classes):
    per_class = cm.diagonal() / cm.sum(axis=1)
    print("\nAccuracy per class (worst first):")
    for i in np.argsort(per_class):
        print(f"  {classes[i]:<6} {per_class[i]:.1%}")

    mistakes = []
    for t in range(len(classes)):
        for p in range(len(classes)):
            if t != p:
                mistakes.append((cm[t, p], classes[t], classes[p]))
    mistakes.sort(reverse=True)

    print("\nMost common mistakes (true -> predicted):")
    for count, t, p in mistakes[:10]:
        print(f"  {t:<6} -> {p:<6} {count} photos")
    print(f"\nOverall accuracy: {cm.diagonal().sum() / cm.sum():.2%}")


# --------------------------------------------------------------------------
# 4. Plots
# --------------------------------------------------------------------------
def plot_matrix(cm, classes, path="confusion_matrix.png"):
    row_pct = cm / cm.sum(axis=1, keepdims=True)  # colour by % of each true class

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (improved CNN, test set)")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if row_pct[i, j] > 0.5 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, label="share of the true class")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


def plot_misclassified(dataset, labels, preds, classes,
                       path="misclassified.png", n=16):
    wrong = np.where(preds != labels)[0][:n]
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4.6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, idx in zip(axes.flat, wrong):
        img = (dataset[idx][0] * std + mean).clamp(0, 1)  # undo normalization
        ax.imshow(img.permute(1, 2, 0))
        ax.set_title(f"pred: {classes[preds[idx]]}\ntrue: {classes[labels[idx]]}",
                     fontsize=8, color="red")
    fig.suptitle("Photos the improved CNN got wrong")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="improved_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, test_loader = get_loaders(augment=False)
    model = ImprovedCNN().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    preds, labels = collect_predictions(model, test_loader, device)
    cm = build_matrix(labels, preds, len(CLASSES))

    report(cm, CLASSES)
    plot_matrix(cm, CLASSES)
    plot_misclassified(test_loader.dataset, labels, preds, CLASSES)


if __name__ == "__main__":
    main()
