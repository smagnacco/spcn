import numpy as np
from tqdm import tqdm
from .neuron import (
    compute_predictions, compute_errors, update_weights, update_states,
    compute_digit_output, update_digit_weights, clamp_top_layer, update_contrastive
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


def train_mnist(grid, X_train, y_train, X_test, y_test, epochs=10, learning_rate=0.01,
                 lambda_penalty=0.0001, alpha=0.5, settle_iters=8,
                 n_train=None, n_test=None, weight_decay=0.0001, homeostatic_rate=0.01,
                 digit_learning_rate=0.005):
    """Train network on MNIST with top-down label conditioning.

    Each sample clamps the input pixels at the bottom layer and biases the
    top layer toward the digit prototype, then runs `settle_iters` local
    predict/error/update cycles so hidden layers reconcile bottom-up sensory
    evidence with top-down category expectation -- purely local, no
    backprop. The top-layer clamp strength fades from full to zero across
    the settle loop: early iterations inject the label as a strong prior so
    gradient-free local learning has a target to organize around, but the
    final iterations run unclamped so the readout head (digit_weights) is
    trained on what the network itself settles to from the image, not on
    the clamp value -- otherwise digit_weights degenerates into a trivial
    prototype-to-label lookup that ignores the image entirely.
    """
    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    n_train = len(X_train) if n_train is None else min(n_train, len(X_train))
    n_test = len(X_test) if n_test is None else min(n_test, len(X_test))

    for epoch in tqdm(range(epochs), desc="MNIST Training"):
        epoch_loss = 0.0
        correct = 0

        for i in range(n_train):
            img = X_train[i]
            label = y_train[i]

            target = np.zeros(grid.num_digits, dtype=np.float32)
            target[label] = 1.0
            grid.digit_target = target

            grid.state[:, :, 0] = img.astype(np.float32)

            for it in range(settle_iters):
                blend = 1.0 - it / (settle_iters - 1) if settle_iters > 1 else 0.0
                if blend > 0:
                    clamp_top_layer(grid, label, blend=blend)
                compute_predictions(grid)
                compute_errors(grid)
                update_weights(grid, learning_rate=learning_rate, lambda_penalty=lambda_penalty,
                               weight_decay=weight_decay)
                update_states(grid, alpha=alpha, homeostatic_rate=homeostatic_rate)
                grid.state[:, :, 0] = img.astype(np.float32)

            compute_digit_output(grid)
            update_digit_weights(grid, learning_rate=digit_learning_rate, weight_decay=weight_decay)

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
            for _ in range(settle_iters):
                compute_predictions(grid)
                compute_errors(grid)
                update_states(grid, alpha=alpha, homeostatic_rate=homeostatic_rate)
                grid.state[:, :, 0] = img.astype(np.float32)

            compute_digit_output(grid)

            pred = np.argmax(grid.digit_output)
            if pred == label:
                test_correct += 1

        test_acc = test_correct / n_test
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch + 1:2d} | loss: {avg_loss:.4f} | train_acc: {train_acc:.3f} | test_acc: {test_acc:.3f}")

    return history


def train_mnist_contrastive(grid, X_train, y_train, X_test, y_test, epochs=10, learning_rate=0.01,
                             lambda_penalty=0.0001, alpha=0.5, settle_iters=8,
                             n_train=None, n_test=None, weight_decay=0.0001, homeostatic_rate=0.01,
                             digit_learning_rate=0.005, contrastive_rate=0.1, prototype_decay=0.05):
    """Phase 3: train_mnist plus a contrastive Hebbian update on the top
    layer's settled state.

    Phase 2 established (see FINDINGS_PHASE2.md) that local reconstruction
    error plus a fading top-down prototype clamp gives the top layer real,
    input-dependent variance, but that variance is not organized along
    class-discriminative axes -- the network reached mode collapse, always
    predicting the same class. update_contrastive adds the missing piece:
    after each sample settles (clamp fully faded), its top-layer state is
    pulled toward a running per-class prototype for its own label and
    pushed away from a different class's prototype. This is still a local,
    pairwise rule -- no global loss, no backprop -- it only reads the
    current sample's state and per-class running averages.
    """
    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    n_train = len(X_train) if n_train is None else min(n_train, len(X_train))
    n_test = len(X_test) if n_test is None else min(n_test, len(X_test))

    for epoch in tqdm(range(epochs), desc="MNIST Contrastive Training"):
        epoch_loss = 0.0
        correct = 0

        for i in range(n_train):
            img = X_train[i]
            label = y_train[i]

            target = np.zeros(grid.num_digits, dtype=np.float32)
            target[label] = 1.0
            grid.digit_target = target

            grid.state[:, :, 0] = img.astype(np.float32)

            for it in range(settle_iters):
                blend = 1.0 - it / (settle_iters - 1) if settle_iters > 1 else 0.0
                if blend > 0:
                    clamp_top_layer(grid, label, blend=blend)
                compute_predictions(grid)
                compute_errors(grid)
                update_weights(grid, learning_rate=learning_rate, lambda_penalty=lambda_penalty,
                               weight_decay=weight_decay)
                update_states(grid, alpha=alpha, homeostatic_rate=homeostatic_rate)
                update_contrastive(grid, label, learning_rate=contrastive_rate,
                                    prototype_decay=prototype_decay)
                grid.state[:, :, 0] = img.astype(np.float32)

            compute_digit_output(grid)
            update_digit_weights(grid, learning_rate=digit_learning_rate, weight_decay=weight_decay)

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
            for _ in range(settle_iters):
                compute_predictions(grid)
                compute_errors(grid)
                update_states(grid, alpha=alpha, homeostatic_rate=homeostatic_rate)
                grid.state[:, :, 0] = img.astype(np.float32)

            compute_digit_output(grid)

            pred = np.argmax(grid.digit_output)
            if pred == label:
                test_correct += 1

        test_acc = test_correct / n_test
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch + 1:2d} | loss: {avg_loss:.4f} | train_acc: {train_acc:.3f} | test_acc: {test_acc:.3f}")

    return history
