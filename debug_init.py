#!/usr/bin/env python3
"""Debug weight initialization and signal propagation"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spcn.grid import Grid
from spcn.neuron import compute_predictions, compute_errors, compute_digit_output

GRID_W = 28
GRID_H = 28
GRID_D = 4
RADIUS = 2
NUM_DIGITS = 10

# Test weight initialization
print("=" * 60)
print("Weight Initialization Debug")
print("=" * 60)
print()

grid = Grid(GRID_W, GRID_H, GRID_D, radius=RADIUS, num_digits=NUM_DIGITS, device="GPU")

# Check weight scales
weights_mean = np.mean(np.abs(grid.weights))
weights_std = np.std(grid.weights)
weights_max = np.max(np.abs(grid.weights))

print(f"Weights mean(abs): {weights_mean:.4f}")
print(f"Weights std: {weights_std:.4f}")
print(f"Weights max(abs): {weights_max:.4f}")
print()

digit_weights_mean = np.mean(np.abs(grid.digit_weights))
digit_weights_std = np.std(grid.digit_weights)
print(f"Digit weights mean(abs): {digit_weights_mean:.4f}")
print(f"Digit weights std: {digit_weights_std:.4f}")
print()

# Compute expected scale
fan_in = (2 * RADIUS + 1) ** 2  # 25 for radius=2
expected_scale = np.sqrt(2.0 / fan_in)
print(f"Expected Kaiming scale for radius={RADIUS}: {expected_scale:.4f}")
print(f"Old scale (0.1): 0.1000")
print()

# Test forward pass with sample input
print("-" * 60)
print("Signal Propagation Test")
print("-" * 60)
print()

# Create a random input
sample_input = np.random.uniform(0, 1, (GRID_W, GRID_H)).astype(np.float32)
grid.state[:, :, 0] = sample_input

# Compute predictions
compute_predictions(grid)

# Check prediction magnitudes at each layer
for z in range(1, GRID_D):
    pred_mean = np.mean(np.abs(grid.prediction[:, :, z]))
    pred_std = np.std(grid.prediction[:, :, z])
    pred_var = np.var(grid.prediction[:, :, z])
    print(f"Layer {z} predictions - mean(abs): {pred_mean:.4f}, std: {pred_std:.4f}, var: {pred_var:.4f}")

print()

# Compute errors
compute_errors(grid)

for z in range(1, GRID_D):
    error_mean = np.mean(np.abs(grid.error[:, :, z]))
    error_std = np.std(grid.error[:, :, z])
    print(f"Layer {z} errors - mean(abs): {error_mean:.4f}, std: {error_std:.4f}")

print()

# Compute digit output
compute_digit_output(grid)
print(f"Digit output: {grid.digit_output}")
print(f"Digit output variance: {np.var(grid.digit_output):.4f}")
print()

print("=" * 60)
