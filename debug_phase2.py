#!/usr/bin/env python3
"""Debug Phase 2: check what train_mnist actually reports"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spcn.grid import Grid
from spcn.training import train_mnist
from data.mnist_loader import load_mnist

GRID_W = 28
GRID_H = 28
GRID_D = 4
RADIUS = 2
NUM_DIGITS = 10
LR = 0.1
LAMBDA = 0.0001
ALPHA = 0.5
EPOCHS = 3

print("=" * 60)
print("Phase 2 Debug: Check train_mnist Output")
print("=" * 60)
print()

(X_train, y_train), (X_test, y_test) = load_mnist()
print(f"Loaded: train={X_train.shape}, test={X_test.shape}")
print()

grid = Grid(GRID_W, GRID_H, GRID_D, radius=RADIUS, num_digits=NUM_DIGITS, device="GPU")
print(f"Grid initialized on GPU")
print()

history = train_mnist(
    grid, X_train, y_train, X_test, y_test,
    epochs=EPOCHS, learning_rate=LR, lambda_penalty=LAMBDA, alpha=ALPHA
)

print()
print("=" * 60)
print("History Returned:")
print("=" * 60)
for key in history:
    print(f"{key}: {history[key]}")
print()

print("Final Test Accuracy:", history["test_acc"][-1] if history["test_acc"] else "N/A")
print()

# Now manually check test set on same grid
print("-" * 60)
print("Manual Test on Test Set:")
print("-" * 60)

from spcn.neuron import compute_predictions, compute_digit_output

test_correct = 0
for i in range(min(100, len(X_test))):
    img = X_test[i]
    label = y_test[i]

    grid.state[:, :, 0] = img.astype(np.float32)
    compute_predictions(grid)
    compute_digit_output(grid)

    pred = np.argmax(grid.digit_output)
    if pred == label:
        test_correct += 1

manual_test_acc = test_correct / min(100, len(X_test))
print(f"Manual test accuracy: {manual_test_acc:.3f} ({test_correct}/100)")
print()

print("=" * 60)
