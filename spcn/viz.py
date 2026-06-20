import os
import numpy as np
import matplotlib.pyplot as plt


def plot_error_curve(history, output_dir="output"):
    """Plot mean error across epochs."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(history, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.title("Error Convergence During Training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_curve.png"), dpi=150)
    plt.close()


def plot_activations(grid, patterns, output_dir="output"):
    """Plot activation heatmaps for each pattern across all layers."""
    os.makedirs(output_dir, exist_ok=True)

    num_patterns = len(patterns)
    num_layers = grid.depth

    for pattern_idx, pattern in enumerate(patterns):
        grid.state[:, :, 0] = pattern.astype(np.float32)

        from .neuron import compute_predictions
        compute_predictions(grid)

        fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 4))
        if num_layers == 1:
            axes = [axes]

        for z in range(num_layers):
            im = axes[z].imshow(grid.state[:, :, z], cmap="hot", vmin=0, vmax=1)
            axes[z].set_title(f"Layer Z={z}")
            axes[z].set_xlabel("X")
            axes[z].set_ylabel("Y")
            plt.colorbar(im, ax=axes[z])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"activations_pattern_{pattern_idx}.png"), dpi=150)
        plt.close()
