import numpy as np
try:
    from .openvino_backend import OpenVINOGrid as BaseGrid
    USE_OPENVINO = True
except ImportError:
    USE_OPENVINO = False
    BaseGrid = object


def build_digit_prototypes(width, height, num_digits, high=0.9, low=0.05):
    """Partition the top layer into num_digits non-overlapping rectangular
    regions, one per digit class. Used as a top-down spatial prior: clamping
    the top layer toward prototypes[label] during training injects the
    classification signal as a local target rather than a separate readout.
    """
    cols = int(np.ceil(np.sqrt(num_digits)))
    rows = int(np.ceil(num_digits / cols))
    cell_w = max(1, width // cols)
    cell_h = max(1, height // rows)

    prototypes = np.full((num_digits, width, height), low, dtype=np.float32)
    for d in range(num_digits):
        r, c = divmod(d, cols)
        x0, x1 = c * cell_w, min(width, (c + 1) * cell_w)
        y0, y1 = r * cell_h, min(height, (r + 1) * cell_h)
        prototypes[d, x0:x1, y0:y1] = high
    return prototypes


class GridBase:
    def __init__(self, width, height, depth, radius=2, z_radius=1, gamma=1.0, num_digits=None, device="AUTO"):
        self.width = width
        self.height = height
        self.depth = depth
        self.radius = radius
        self.z_radius = z_radius
        self.gamma = gamma
        self.num_digits = num_digits
        self.device = device

        self.state = np.random.uniform(0, 0.1, (width, height, depth)).astype(np.float32)
        self.prediction = np.zeros((width, height, depth), dtype=np.float32)
        self.error = np.zeros((width, height, depth), dtype=np.float32)

        offset_xy = 2 * radius + 1
        offset_z = z_radius + 1
        fan_in = offset_xy * offset_xy * offset_z
        weight_scale = 1.0 / np.sqrt(fan_in)
        self.weights = np.random.randn(
            width, height, depth, offset_xy, offset_xy, offset_z
        ).astype(np.float32) * weight_scale

        if num_digits:
            digit_fan_in = width * height
            digit_weight_scale = 1.0 / np.sqrt(digit_fan_in)
            self.digit_weights = np.random.randn(width, height, num_digits).astype(np.float32) * digit_weight_scale
            self.digit_output = np.zeros(num_digits, dtype=np.float32)
            self.digit_target = np.zeros(num_digits, dtype=np.float32)
            self.digit_prototypes = build_digit_prototypes(width, height, num_digits)

    def reset_state(self):
        self.state = np.random.uniform(0, 0.1, (self.width, self.height, self.depth)).astype(np.float32)
        self.prediction = np.zeros((self.width, self.height, self.depth), dtype=np.float32)
        self.error = np.zeros((self.width, self.height, self.depth), dtype=np.float32)


class Grid(GridBase if not USE_OPENVINO else BaseGrid):
    """Grid using OpenVINO if available, NumPy fallback."""
    pass


if not USE_OPENVINO:
    Grid = GridBase
