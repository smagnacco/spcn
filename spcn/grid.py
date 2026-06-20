import numpy as np
try:
    from .openvino_backend import OpenVINOGrid as BaseGrid
    USE_OPENVINO = True
except ImportError:
    USE_OPENVINO = False
    BaseGrid = object


class GridBase:
    def __init__(self, width, height, depth, radius=2, num_digits=None, device="AUTO"):
        self.width = width
        self.height = height
        self.depth = depth
        self.radius = radius
        self.num_digits = num_digits
        self.device = device

        self.state = np.random.uniform(0, 0.1, (width, height, depth)).astype(np.float32)
        self.prediction = np.zeros((width, height, depth), dtype=np.float32)
        self.error = np.zeros((width, height, depth), dtype=np.float32)

        offset_size = 2 * radius + 1
        fan_in = offset_size * offset_size
        kaiming_scale = np.sqrt(2.0 / fan_in)
        self.weights = np.random.randn(width, height, depth, offset_size, offset_size).astype(np.float32) * kaiming_scale

        if num_digits:
            fan_in_digit = width * height * depth
            kaiming_scale_digit = np.sqrt(2.0 / fan_in_digit)
            self.digit_weights = np.random.randn(width, height, depth, num_digits).astype(np.float32) * kaiming_scale_digit * 100
            self.digit_output = np.zeros(num_digits, dtype=np.float32)
            self.digit_target = np.zeros(num_digits, dtype=np.float32)

    def reset_state(self):
        self.state[:, :, 1:] = 0
        self.prediction = np.zeros((self.width, self.height, self.depth), dtype=np.float32)
        self.error = np.zeros((self.width, self.height, self.depth), dtype=np.float32)


class Grid(GridBase if not USE_OPENVINO else BaseGrid):
    """Grid using OpenVINO if available, NumPy fallback."""
    pass


if not USE_OPENVINO:
    Grid = GridBase
