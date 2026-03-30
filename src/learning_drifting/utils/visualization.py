from pathlib import Path

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from torch.distributions import Independent, Normal
from learning_drifting.train.prepare_dataset import SyntheticDataset

def plot_loss_curve(losses: list[float], output_path: Path) -> None:
    """Plot raw and smoothed trainig loss curve,"""

    steps = np.arange(1, len(losses) + 1)
    smoothed_losses = gaussian_filter1d(losses, sigma=5)
    blue = "#1f77b4"
    plt.figure(figsize=(6, 5))
    plt.plot(steps, losses, color=blue, alpha=0.3)
    plt.plot(steps, smoothed_losses, color=blue, linewidth=2)
    plt.title("Training dynamics", fontsize=16)
    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print("Training curves saved to", output_path)


@torch.no_grad()
def plot_drifting_samples(
    model,
    dataset: SyntheticDataset,
    output_dir: str | Path = ".",
    filename: str = "sampling.png",
    num_samples: int = 100_000,
    noise_dim: int = 2,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    # ground truth samples
    gt_samples = dataset.sample(num_samples).detach().cpu().numpy()

    # generated samples
    noise = torch.randn(num_samples, noise_dim, device=device)
    gen_samples = model(noise).detach().cpu().numpy()

    # plotting range
    square_range = dataset.get_square_range(samples=torch.from_numpy(gt_samples))

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    # ground truth
    h = axes[0].hist2d(
        gt_samples[:, 0],
        gt_samples[:, 1],
        bins=300,
        range=square_range,
    )
    cmax = torch.quantile(torch.from_numpy(h[0]), 0.99).item()
    norm = cm.colors.Normalize(vmin=0.0, vmax=cmax)
    axes[0].cla()
    axes[0].hist2d(
        gt_samples[:, 0],
        gt_samples[:, 1],
        bins=300,
        norm=norm,
        range=square_range,
    )
    axes[0].set_title("Ground Truth")
    axes[0].set_aspect("equal")
    axes[0].axis("off")

    # generated
    axes[1].hist2d(
        gen_samples[:, 0],
        gen_samples[:, 1],
        bins=300,
        norm=norm,
        range=square_range,
    )
    axes[1].set_title("Generated")
    axes[1].set_aspect("equal")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(fig)

    # print(f"Drifting samples saved to {output_dir / filename}")


@torch.no_grad()
def plot_training_snapshots(
    model,
    dataset: SyntheticDataset,
    checkpoint_steps: list[int],
    checkpoint_dir: str | Path,
    output_path: str | Path,
    num_samples: int = 100_000,
    noise_dim: int = 2,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    # ground truth
    gt_samples = dataset.sample(num_samples).detach().cpu().numpy()
    square_range = dataset.get_square_range(samples=torch.from_numpy(gt_samples))

    ncols = len(checkpoint_steps) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(2.2 * ncols, 2.4))

    # GT panel
    h = axes[0].hist2d(
        gt_samples[:, 0],
        gt_samples[:, 1],
        bins=300,
        range=square_range,
    )
    cmax = torch.quantile(torch.from_numpy(h[0]), 0.99).item()
    norm = cm.colors.Normalize(vmin=0.0, vmax=cmax)
    axes[0].cla()
    axes[0].hist2d(
        gt_samples[:, 0],
        gt_samples[:, 1],
        bins=300,
        norm=norm,
        range=square_range,
    )
    axes[0].set_title("Ground Truth")
    axes[0].set_aspect("equal")
    axes[0].axis("off")

    # each checkpoint
    for i, step in enumerate(checkpoint_steps, start=1):
        ckpt_path = checkpoint_dir / f"ckpt_step_{step:06d}.pth"
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        noise = torch.randn(num_samples, noise_dim, device=device)
        gen_samples = model(noise).detach().cpu().numpy()

        axes[i].hist2d(
            gen_samples[:, 0],
            gen_samples[:, 1],
            bins=300,
            norm=norm,
            range=square_range,
        )
        axes[i].set_title(f"step {step}")
        axes[i].set_aspect("equal")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Training snapshot figure saved to {output_path}")