import numpy as np


class SimpleMLP:
    """Baseline MLP with similar param count to SPCN."""

    def __init__(self, input_size=28 * 28, hidden_size=512, output_size=10, learning_rate=0.01):
        self.lr = learning_rate

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)

        self.cache = None

    def sigmoid(self, x):
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        """X shape: (batch, 28, 28) or (28, 28)"""
        if X.ndim == 3:
            batch_size = X.shape[0]
            X_flat = X.reshape(batch_size, -1)
        else:
            X_flat = X.reshape(1, -1)

        z1 = np.dot(X_flat, self.W1) + self.b1
        a1 = self.sigmoid(z1)

        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)

        self.cache = (X_flat, a1)
        return a2

    def backward(self, y):
        """y: one-hot labels shape (batch, 10)"""
        X_flat, a1 = self.cache

        dz2 = (self.forward(X_flat.reshape(-1, 28, 28)) - y) * self.sigmoid_derivative(self.forward(X_flat.reshape(-1, 28, 28)))
        dW2 = np.dot(a1.T, dz2) / y.shape[0]
        db2 = np.mean(dz2, axis=0)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(a1)
        dW1 = np.dot(X_flat.T, dz1) / y.shape[0]
        db1 = np.mean(dz1, axis=0)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, X):
        """Return predicted class."""
        if X.ndim == 3:
            output = self.forward(X)
        else:
            output = self.forward(X.reshape(1, 28, 28))
        return np.argmax(output, axis=1 if output.ndim > 1 else 0)

    def count_params(self):
        """Count total parameters."""
        return (
            self.W1.size + self.b1.size +
            self.W2.size + self.b2.size
        )
