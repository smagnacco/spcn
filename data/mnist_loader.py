import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist(n_train=5000, n_test=1000, seed=42):
    """Load real handwritten digits from MNIST (28x28, sklearn fetch_openml).

    Downloads once and caches under sklearn's data home on first call.
    Pixels are normalized to [0, 1] to match the network's sigmoid-activation
    range; labels are 0-9 digit classes.
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int32)

    X = X.reshape(-1, 28, 28)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X))

    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train + n_test]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return (X_train, y_train), (X_test, y_test)
