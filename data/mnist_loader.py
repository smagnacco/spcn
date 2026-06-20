import numpy as np


def generate_synthetic_digits(n_train=1000, n_test=200):
    """Generate synthetic digit patterns (simple binary shapes)."""
    np.random.seed(42)

    X_train = np.random.binomial(1, 0.3, (n_train, 28, 28)).astype(np.float32)
    y_train = np.random.randint(0, 10, n_train).astype(np.int32)

    X_test = np.random.binomial(1, 0.3, (n_test, 28, 28)).astype(np.float32)
    y_test = np.random.randint(0, 10, n_test).astype(np.int32)

    for i in range(n_train):
        digit = y_train[i]
        mask = np.zeros((28, 28))
        region = (digit % 4) * 2
        mask[region:region+14, region:region+14] = 1
        X_train[i] = X_train[i] * 0.5 + mask * 0.5

    for i in range(n_test):
        digit = y_test[i]
        mask = np.zeros((28, 28))
        region = (digit % 4) * 2
        mask[region:region+14, region:region+14] = 1
        X_test[i] = X_test[i] * 0.5 + mask * 0.5

    return (X_train, y_train), (X_test, y_test)


def load_mnist():
    """Load synthetic MNIST (network unavailable)."""
    return generate_synthetic_digits()
