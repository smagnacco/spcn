#!/usr/bin/env python3
"""Debug training: check weight updates and gradient flow"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spcn.grid import Grid
from spcn.neuron import (
    compute_predictions, compute_errors, update_weights, update_states,
    compute_digit_output, update_digit_weights
)
from data.mnist_loader import load_mnist

GRID_W = 28
GRID_H = 28
GRID_D = 4
RADIUS = 2
NUM_DIGITS = 10
LR = 0.1

print("=" * 60)
print("Training Debug: Weight Updates & Signal Flow")
print("=" * 60)
print()

# Load data
(X_train, y_train), (X_test, y_test) = load_mnist()
print(f"Loaded MNIST: train={X_train.shape}, test={X_test.shape}")
print()

# Initialize grid
grid = Grid(GRID_W, GRID_H, GRID_D, radius=RADIUS, num_digits=NUM_DIGITS, device="GPU")

# Store initial weights for comparison
initial_weights = grid.weights.copy()
initial_digit_weights = grid.digit_weights.copy()

print("-" * 60)
print("Single Training Step Analysis")
print("-" * 60)
print()

# One training step
img = X_train[0]
label = y_train[0]

print(f"Input image: {img.shape}, label={label}")
grid.state[:, :, 0] = img.astype(np.float32)

# Forward pass
compute_predictions(grid)
compute_errors(grid)
compute_digit_output(grid)

print(f"Initial digit_output: {grid.digit_output}")
print(f"Digit output for label {label}: {grid.digit_output[label]:.4f}")
print()

# Target
target = np.zeros(NUM_DIGITS, dtype=np.float32)
target[label] = 1.0
grid.digit_target = target

# Before update
pred_before = grid.digit_output.copy()

# Updates
update_weights(grid, learning_rate=LR, lambda_penalty=0.0001)
update_states(grid, alpha=0.5)
update_digit_weights(grid, learning_rate=LR)

# Compute new predictions
compute_predictions(grid)
compute_digit_output(grid)

print(f"After update digit_output: {grid.digit_output}")
print(f"Digit output for label {label}: {grid.digit_output[label]:.4f}")
print()

# Check weight changes
weight_delta = grid.weights - initial_weights
digit_weight_delta = grid.digit_weights - initial_digit_weights

print(f"Weight change stats:")
print(f"  Mean(|delta|): {np.mean(np.abs(weight_delta)):.6f}")
print(f"  Max(|delta|): {np.max(np.abs(weight_delta)):.6f}")
print(f"  Fraction changed >0.001: {np.sum(np.abs(weight_delta) > 0.001) / weight_delta.size * 100:.1f}%")
print()

print(f"Digit weight change stats:")
print(f"  Mean(|delta|): {np.mean(np.abs(digit_weight_delta)):.6f}")
print(f"  Max(|delta|): {np.max(np.abs(digit_weight_delta)):.6f}")
print()

# Check error signals at each layer
print("Error signal magnitudes:")
for z in range(1, GRID_D):
    error_mean = np.mean(np.abs(grid.error[:, :, z]))
    error_max = np.max(np.abs(grid.error[:, :, z]))
    print(f"  Layer {z}: mean={error_mean:.4f}, max={error_max:.4f}")
print()

# Classification error
class_error = target[label] - pred_before[label]
print(f"Classification error for correct class: {class_error:.4f}")
print()

print("=" * 60)
