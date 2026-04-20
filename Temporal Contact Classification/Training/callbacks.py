"""
Training callbacks: t-SNE visualization of feature space (contact vs no-contact).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def collect_backbone_center_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Backbone outputs per-frame features; take the **center** time index → (N, F).

    Returns
    -------
    features : (N, F) float32
    labels : (N,) int
    """
    model.eval()
    feats: list[np.ndarray] = []
    labs: list[np.ndarray] = []
    n = 0
    t = model.window_size
    ci = (t - 1) // 2

    for x, y in tqdm(loader, desc="t-SNE features", leave=False):
        if n >= max_samples:
            break
        x = x.to(device)
        b = x.size(0)
        flat = x.reshape(b * t, x.size(2), x.size(3), x.size(4))
        z = model.backbone(flat)  # (B*T, F)
        z = z.reshape(b, t, -1)
        zc = z[:, ci, :].cpu().numpy()
        feats.append(zc)
        labs.append(y.cpu().numpy().astype(np.int64))
        n += b

    if not feats:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    F = np.concatenate(feats, axis=0)[:max_samples]
    L = np.concatenate(labs, axis=0)[:max_samples]
    return F.astype(np.float32), L


def run_tsne_and_plot(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = "t-SNE (backbone center-frame features)",
    perplexity: float = 30.0,
    random_state: int = 42,
) -> None:
    """Fit 2D t-SNE and save a scatter plot (contact vs no-contact)."""
    if len(features) < 4:
        print("t-SNE: not enough samples; skipping plot.")
        return

    n = len(features)
    perp = min(perplexity, max(2, n - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    xy = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(7, 6))
    for lab, name, c in [(0, "No contact", "tab:blue"), (1, "Contact", "tab:orange")]:
        m = labels == lab
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.6, c=c, label=name)
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE plot saved → {save_path}")


class TSNECallback:
    """
    Every ``every_n_epochs``, run t-SNE on a subset of the validation loader.

    Uses **center-frame backbone features** so you can see whether contact vs
    no-contact clusters separate before / while training the temporal head.
    """

    def __init__(
        self,
        val_loader: DataLoader,
        device: torch.device,
        out_dir: str,
        every_n_epochs: int = 5,
        max_samples: int = 2000,
    ) -> None:
        self.device = device
        self.out_dir = out_dir
        self.every_n_epochs = every_n_epochs
        self.max_samples = max_samples
        self.loader = val_loader

    def __call__(self, model: nn.Module, epoch: int) -> None:
        if self.every_n_epochs <= 0:
            return
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        feat, lab = collect_backbone_center_features(
            model, self.loader, self.device, max_samples=self.max_samples
        )
        path = os.path.join(self.out_dir, f"tsne_epoch_{epoch+1:03d}.png")
        run_tsne_and_plot(
            feat,
            lab,
            path,
            title=f"t-SNE epoch {epoch+1} (backbone center features)",
        )
