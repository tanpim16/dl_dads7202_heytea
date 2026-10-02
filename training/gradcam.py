"""
GradCAM implementation.
Usage:
    gcam = GradCAM(model, target_layer)
    cam, pred_idx = gcam(img_tensor)   # img_tensor: (C, H, W) on device
"""
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import IMAGENET_MEAN, IMAGENET_STD, CLASSES, IMG_SIZE
from models import get_gradcam_layer


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._activations = None
        self._gradients   = None

        # tensor hook แทน register_full_backward_hook: full backward hook พังเมื่อ output ของ layer
        # ถูกแก้แบบ in-place ต่อ (VGG: Conv -> ReLU(inplace=True)) -> RuntimeError ตอน backward
        target_layer.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, module, inp, out):
        self._activations = out.detach().clone()
        out.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self._gradients = grad.detach()

    def __call__(self, img_tensor: torch.Tensor, class_idx: int = None):
        """
        img_tensor : (C, H, W) on the model's device — no batch dim.
        Returns (cam_np [H, W] in [0,1], predicted class index).
        """
        self.model.eval()
        x = img_tensor.unsqueeze(0)          # (1, C, H, W)

        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()
            self.model.zero_grad()
            logits[0, class_idx].backward()

        # Grad-weighted average over spatial dims
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze()   # (H, W)
        cam = torch.relu(cam).cpu().numpy()

        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


# ── Visualization helpers ─────────────────────────────────────────────────────

def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = (tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def _overlay(img_np: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(alpha * heatmap + (1 - alpha) * img_np, 0, 1)


# ── Main visualization function ───────────────────────────────────────────────

def visualize_gradcam(model, arch, loader, device,
                      n_correct=3, n_wrong=3, save_path=None):
    """
    Show GradCAM for n_correct correctly classified + n_wrong misclassified
    samples. Separating correct/wrong cases gives richer error analysis.
    """
    target_layer = get_gradcam_layer(model, arch)
    gcam = GradCAM(model.to(device), target_layer)

    correct_samples, wrong_samples = [], []

    for imgs, labels in loader:
        for i in range(len(imgs)):
            if len(correct_samples) >= n_correct and len(wrong_samples) >= n_wrong:
                break
            img_t = imgs[i].to(device)
            cam, pred_idx = gcam(img_t)
            img_np  = _denormalize(imgs[i])
            overlay = _overlay(img_np, cam)
            entry = {
                "original": img_np,
                "overlay":  overlay,
                "gt":   CLASSES[labels[i].item()].replace("cha_", ""),
                "pred": CLASSES[pred_idx].replace("cha_", ""),
                "correct": labels[i].item() == pred_idx,
            }
            if entry["correct"] and len(correct_samples) < n_correct:
                correct_samples.append(entry)
            elif not entry["correct"] and len(wrong_samples) < n_wrong:
                wrong_samples.append(entry)
        if len(correct_samples) >= n_correct and len(wrong_samples) >= n_wrong:
            break

    samples = correct_samples + wrong_samples
    n = len(samples)
    if n == 0:
        print("No samples collected for GradCAM.")
        return

    fig, axes = plt.subplots(n, 2, figsize=(6, 3.2 * n))
    if n == 1:
        axes = [axes]

    for i, s in enumerate(samples):
        border = "green" if s["correct"] else "red"
        axes[i][0].imshow(s["original"])
        axes[i][0].set_title(f"GT: {s['gt']}", fontsize=9)
        axes[i][1].imshow(s["overlay"])
        axes[i][1].set_title(f"Pred: {s['pred']}", fontsize=9, color=border)
        for ax in axes[i]:
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(2)

    fig.suptitle(f"GradCAM — {arch}  (green=correct, red=wrong)", fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved GradCAM → {save_path}")
    plt.close(fig)
    return fig
