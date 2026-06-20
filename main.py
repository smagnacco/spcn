import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spcn.grid import Grid
from spcn.training import train
from spcn.viz import plot_error_curve, plot_activations
from data.patterns import generate_patterns


GRID_W = 4
GRID_H = 4
GRID_D = 3
RADIUS = 2
LR = 0.1
LAMBDA = 0.0001
ALPHA = 0.5
EPOCHS = 200


def main():
    print("Initializing Spatial Predictive Coding Network")
    print(f"Grid: {GRID_W}×{GRID_H}×{GRID_D}, Radius: {RADIUS}")
    print()

    grid = Grid(GRID_W, GRID_H, GRID_D, radius=RADIUS)

    patterns = generate_patterns()
    print(f"Generated {len(patterns)} binary patterns")
    print()

    initial_error = None
    history = train(grid, patterns, epochs=EPOCHS, learning_rate=LR, lambda_penalty=LAMBDA, alpha=ALPHA)

    if history:
        initial_error = history[0]
        final_error = history[-1]
        reduction = (initial_error - final_error) / initial_error * 100
    else:
        initial_error = 0
        final_error = 0
        reduction = 0

    print()
    print("=" * 50)
    print("Training complete.")
    print(f"Initial error : {initial_error:.4f}")
    print(f"Final error   : {final_error:.4f}")
    print(f"Reduction     : {reduction:.1f}%")
    print("=" * 50)
    print()

    plot_error_curve(history)
    print("✓ Saved error curve to output/error_curve.png")

    plot_activations(grid, patterns)
    print("✓ Saved activation heatmaps to output/")

    from spcn.neuron import compute_predictions, compute_errors

    pattern_predictions_z1 = []
    pattern_errors_z1 = []
    for pattern in patterns:
        grid.state[:, :, 0] = pattern.astype(np.float32)
        compute_predictions(grid)
        compute_errors(grid)
        pred_z1 = grid.prediction[:, :, 1].copy()
        err_z1 = np.abs(grid.error[:, :, 1]).copy()
        pattern_predictions_z1.append(pred_z1)
        pattern_errors_z1.append(err_z1)

    max_pred_diff = 0
    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            diff = np.mean(np.abs(pattern_predictions_z1[i] - pattern_predictions_z1[j]))
            max_pred_diff = max(max_pred_diff, diff)

    max_error_diff = 0
    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            diff = np.mean(np.abs(pattern_errors_z1[i] - pattern_errors_z1[j]))
            max_error_diff = max(max_error_diff, diff)

    similarity_threshold = 0.05
    patterns_separable = max_pred_diff > similarity_threshold

    print(f"Patterns separable in Z=1: {'YES' if patterns_separable else 'NO'}")
    print(f"  Max pairwise prediction difference: {max_pred_diff:.4f}")
    print(f"  Max pairwise error difference: {max_error_diff:.4f}")
    print()

    if reduction >= 50 and patterns_separable and EPOCHS < 10:
        print("✓ PHASE 1 SUCCESS")
    else:
        print("Phase 1 results:")
        print(f"  - Error reduction >= 50%: {reduction >= 50} ({reduction:.1f}%)")
        print(f"  - Patterns separable: {patterns_separable}")


if __name__ == "__main__":
    main()
