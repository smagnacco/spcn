import numpy as np
from tqdm import tqdm
from .neuron import (
    compute_predictions, compute_errors, update_weights, update_states,
    compute_digit_output, update_digit_weights
)


def train(grid, patterns, epochs=200, learning_rate=0.01, lambda_penalty=0.001, alpha=0.1):
    """Train the network on a sequence of patterns."""
    history = []

    for epoch in tqdm(range(epochs), desc="Training"):
        pattern = patterns[np.random.randint(0, len(patterns))]

        grid.state[:, :, 0] = pattern.astype(np.float32)

        compute_predictions(grid)

        compute_errors(grid)

        update_weights(grid, learning_rate=learning_rate, lambda_penalty=lambda_penalty)
        update_states(grid, alpha=alpha)

        mean_error = np.mean(np.abs(grid.error[:, :, 1:]))
        history.append(mean_error)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d} | mean_error: {mean_error:.4f}")

    return history


def train_mnist(grid, X_train, y_train, X_test, y_test, epochs=10, learning_rate=0.01, lambda_penalty=0.0001, alpha=0.5):
    """Train network on MNIST with digit classification head."""
    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    n_train = min(200, len(X_train))
    n_test = min(100, len(X_test))

    for epoch in tqdm(range(epochs), desc="MNIST Training"):
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
