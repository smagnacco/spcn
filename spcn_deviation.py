"""Experimental deviations from SPEC.md, kept separate from the spec-compliant spcn/ package.

Deviations from spec:
1. Kaiming weight init (sqrt(2/fan_in)) instead of spec's randn() * 0.01.
2. Classification loss is backpropagated into the top hidden layer's error
   signal, breaking the spec's "fully local, no global backprop" constraint.

Use this only for the phase2_deviation.py experiment track. spcn/ and phase2.py
remain spec-compliant.
"""

import numpy as np
from tqdm import tqdm

from spcn.grid import Grid as SpecGrid
from spcn.neuron import (
    compute_predictions, compute_errors, update_weights, update_states,
    compute_digit_output, update_digit_weights,
)


class DeviationGrid(SpecGrid):
    """Grid with Kaiming-scaled weights instead of spec's randn() * 0.01."""

    def __init__(self, width, height, depth, radius=2, num_digits=None, device="AUTO"):
        super().__init__(width, height, depth, radius=radius, num_digits=num_digits, device=device)

        offset_size = 2 * radius + 1
        fan_in = offset_size * offset_size
        kaiming_scale = np.sqrt(2.0 / fan_in)
        self.weights = np.random.randn(width, height, depth, offset_size, offset_size).astype(np.float32) * kaiming_scale

        if num_digits:
            fan_in_digit = width * height * depth
            kaiming_scale_digit = np.sqrt(2.0 / fan_in_digit)
            self.digit_weights = np.random.randn(width, height, depth, num_digits).astype(np.float32) * kaiming_scale_digit * 100


def backprop_digit_error(grid, scale=0.1):
    """Inject classification error into the top hidden layer's error signal.

    Deviation from spec: spec's update_weights/update_states use only the
    local prediction error (state_below - prediction). Here we add a term
    derived from the classification loss so internal layers receive
    discriminative pressure, not just reconstruction pressure.
    """
    if not hasattr(grid, 'digit_weights'):
        return

    digit_error = grid.digit_target - grid.digit_output  # (num_digits,)
    top = grid.depth - 1

    layer_grad = np.zeros((grid.width, grid.height), dtype=np.float32)
    for d in range(grid.num_digits):
        layer_grad += digit_error[d] * grid.digit_weights[:, :, top, d]

    grid.error[:, :, top] += scale * layer_grad


def train_mnist_deviation(grid, X_train, y_train, X_test, y_test,
                           epochs=10, learning_rate=0.01, lambda_penalty=0.0001,
                           alpha=0.5, backprop_scale=0.1):
    """Same loop as spcn.training.train_mnist, plus backprop_digit_error injection."""
    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    n_train = min(200, len(X_train))
    n_test = min(100, len(X_test))

    for epoch in tqdm(range(epochs), desc="MNIST Training (deviation)"):
        epoch_loss = 0.0
        correct = 0

        for i in range(n_train):
            img = X_train[i]
            label = y_train[i]

            grid.state[:, :, 0] = img.astype(np.float32)
            compute_predictions(grid)
            compute_errors(grid)
            compute_digit_output(grid)

            target = np.zeros(grid.num_digits, dtype=np.float32)
            target[label] = 1.0
            grid.digit_target = target

            backprop_digit_error(grid, scale=backprop_scale)

            update_weights(grid, learning_rate=learning_rate, lambda_penalty=lambda_penalty)
            update_states(grid, alpha=alpha)
            update_digit_weights(grid, learning_rate=learning_rate)

            loss = np.mean((grid.digit_output - target) ** 2)
            epoch_loss += loss

            pred = np.argmax(grid.digit_output)
            if pred == label:
                correct += 1

        train_acc = correct / n_train
        avg_loss = epoch_loss / n_train
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)

        test_correct = 0
        for i in range(n_test):
            img = X_test[i]
            label = y_test[i]

            grid.state[:, :, 0] = img.astype(np.float32)
            compute_predictions(grid)
            compute_digit_output(grid)

            pred = np.argmax(grid.digit_output)
            if pred == label:
                test_correct += 1

        test_acc = test_correct / n_test
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch + 1:2d} | loss: {avg_loss:.4f} | train_acc: {train_acc:.3f} | test_acc: {test_acc:.3f}")

    return history
