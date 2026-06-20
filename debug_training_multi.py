#!/usr/bin/env python3
"""Debug multi-sample training"""

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
N_SAMPLES = 10

print("=" * 60)
print(f"Multi-Sample Training (first {N_SAMPLES} samples)")
print("=" * 60)
print()

# Load data
(X_train, y_train), (X_test, y_test) = load_mnist()

# Initialize grid
grid = Grid(GRID_W, GRID_H, GRID_D, radius=RADIUS, num_digits=NUM_DIGITS, device="GPU")

print("Sample | Label | Pred(correct) Before | Pred(correct) After | Accuracy")
print("-" * 70)

correct = 0
for i in range(N_SAMPLES):
    img = X_train[i]
    label = y_train[i]

    # Forward
    grid.state[:, :, 0] = img.astype(np.float32)
    compute_predictions(grid)
    compute_errors(grid)
    compute_digit_output(grid)

    pred_before = grid.digit_output[label]
    pred_class_before = np.argmax(grid.digit_output)

    # Target
    target = np.zeros(NUM_DIGITS, dtype=np.float32)
    target[label] = 1.0
    grid.digit_target = target

    # Update
    update_weights(grid, learning_rate=LR, lambda_penalty=0.0001)
    update_states(grid, alpha=0.5)
    update_digit_weights(grid, learning_rate=LR)

    # Forward again
    compute_predictions(grid)
    compute_digit_output(grid)

    pred_after = grid.digit_output[label]
    pred_class_after = np.argmax(grid.digit_output)

    is_correct = (pred_class_after == label)
    correct += is_correct

    print(f"{i:6d} | {label:5d} | {pred_before:19.4f} | {pred_after:19.4f} | {int(is_correct)}")

print()
print(f"Accuracy after {N_SAMPLES} samples: {correct}/{N_SAMPLES} = {100*correct/N_SAMPLES:.1f}%")
print()
print("=" * 60)
