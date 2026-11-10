"""
Gradio demo: upload a photo, get the top-3 CIFAR-10 predictions.

Serves the fine-tuned ResNet18 (transfer_full_valbest.pt). Standalone: it does
not import the training scripts, so it can be deployed on its own.

Files needed next to this one:
    transfer_full_valbest.pt   (made by: python cifar10_transfer.py --mode full)
"""
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

CLASSES = ("plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
WEIGHTS = "transfer_full_valbest.pt"


class TransferNet(nn.Module):
    """Same structure as in cifar10_transfer.py (names must match the checkpoint)."""

    def __init__(self, size=128):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None)  # our weights get loaded below
        self.backbone.fc = nn.Linear(512, 10)
        self.size = size
        self.register_buffer("c_mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("c_std", torch.tensor(STD).view(1, 3, 1, 1))
        self.register_buffer("i_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("i_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = x * self.c_std + self.c_mean
        x = (x - self.i_mean) / self.i_std
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        return self.backbone(x)


model = TransferNet()
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
model.eval()

# The model was trained on 32x32 photos, so shrink whatever is uploaded
preprocess = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


def center_square(img):
    """Crop the middle square, so a wide photo isn't squashed when shrunk."""
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def predict(img):
    if img is None:
        return None
    x = preprocess(center_square(img.convert("RGB"))).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a photo"),
    outputs=gr.Label(num_top_classes=3, label="Top 3 predictions"),
    title="CIFAR-10 image classifier",
    description=(
        "Fine-tuned ResNet18 (95.5% test accuracy on CIFAR-10). It only knows "
        "10 classes (plane, car, bird, cat, deer, dog, frog, horse, ship, truck) "
        "and always picks one of them, even for photos of something else. "
        "Photos are cropped to a centered square and shrunk to 32x32 first, "
        "as in the training data, so keep the animal or object big and centered."
    ),
)

if __name__ == "__main__":
    demo.launch()
