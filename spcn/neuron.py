import numpy as np
from .connections import is_valid_neighbor


def sigmoid(x, grid=None):
    """Sigmoid with optional OpenVINO acceleration."""
    if grid and hasattr(grid, 'sigmoid_openvino'):
        return grid.sigmoid_openvino(x)
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def compute_predictions(grid):
    """Top-down: each neuron predicts from lower layer."""
    for z in range(grid.depth - 1, 0, -1):
        for x in range(grid.width):
            for y in range(grid.height):
                prediction_sum = 0.0
                for dx in range(-grid.radius, grid.radius + 1):
                    for dy in range(-grid.radius, grid.radius + 1):
                        if is_valid_neighbor(x, y, dx, dy, grid.width, grid.height):
                            nx, ny = x + dx, y + dy
                            offset_idx_x = dx + grid.radius
                            offset_idx_y = dy + grid.radius
                            weight = grid.weights[x, y, z, offset_idx_x, offset_idx_y]
                            state_below = grid.state[nx, ny, z - 1]
                            prediction_sum += weight * state_below

                grid.prediction[x, y, z] = sigmoid(prediction_sum, grid)


def compute_errors(grid):
    """Bottom-up: each neuron calculates error as state_below - prediction."""
    for z in range(1, grid.depth):
        grid.error[:, :, z] = grid.state[:, :, z - 1] - grid.prediction[:, :, z]


def update_weights(grid, learning_rate=0.01, lambda_penalty=0.001):
    """Update weights using local learning rule with distance penalty."""
    for z in range(1, grid.depth):
        for x in range(grid.width):
            for y in range(grid.height):
                error = grid.error[x, y, z]
                for dx in range(-grid.radius, grid.radius + 1):
                    for dy in range(-grid.radius, grid.radius + 1):
                        if is_valid_neighbor(x, y, dx, dy, grid.width, grid.height):
                            nx, ny = x + dx, y + dy
                            offset_idx_x = dx + grid.radius
                            offset_idx_y = dy + grid.radius
                            state_below = grid.state[nx, ny, z - 1]

                            distance_cost = dx * dx + dy * dy

                            delta_w = learning_rate * error * state_below - lambda_penalty * distance_cost

                            grid.weights[x, y, z, offset_idx_x, offset_idx_y] += delta_w


def update_states(grid, alpha=0.1):
    """Update neuron states based on error signal."""
    for z in range(1, grid.depth):
        grid.state[:, :, z] = sigmoid(grid.state[:, :, z] + alpha * grid.error[:, :, z], grid)


def compute_digit_output(grid):
    """Top layer output: 10 digit predictions from top layer state."""
    if not hasattr(grid, 'digit_weights'):
        return

    for d in range(grid.num_digits):
        digit_sum = np.sum(grid.digit_weights[:, :, grid.depth - 1, d] * grid.state[:, :, grid.depth - 1])
        grid.digit_output[d] = sigmoid(digit_sum, grid)


def update_digit_weights(grid, learning_rate=0.01):
    """Update digit layer weights using error from classification."""
    if not hasattr(grid, 'digit_weights'):
        return

    for d in range(grid.num_digits):
        error = grid.digit_target[d] - grid.digit_output[d]
        delta_w = learning_rate * error * grid.state[:, :, grid.depth - 1]
        grid.digit_weights[:, :, grid.depth - 1, d] += delta_w
