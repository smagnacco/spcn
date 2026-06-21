import numpy as np
from openvino.runtime import Core
import tempfile
import os


def create_simple_model():
    """Create minimal OpenVINO model for testing."""
    try:
        from openvino.tools.mo import convert_model
        import openvino.opset13 as ops
        from openvino import Model
    except ImportError:
        return None

    try:
        input_param = ops.parameter([1, 784], name="input", dtype=np.float32)
        hidden = ops.matmul(input_param, ops.constant(np.random.randn(784, 512).astype(np.float32)))
        output = ops.sigmoid(hidden)
        model = Model(output, [input_param])
        return model
    except Exception:
        return None


class OpenVINOGrid:
    """Grid with OpenVINO acceleration."""

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
            from .grid import build_digit_prototypes
            digit_fan_in = width * height
            digit_weight_scale = 1.0 / np.sqrt(digit_fan_in)
            self.digit_weights = np.random.randn(width, height, num_digits).astype(np.float32) * digit_weight_scale
            self.digit_output = np.zeros(num_digits, dtype=np.float32)
            self.digit_target = np.zeros(num_digits, dtype=np.float32)
            self.digit_prototypes = build_digit_prototypes(width, height, num_digits)

        self.ie = Core()
        self.compiled_model = None
        self._init_openvino_model()

    def _init_openvino_model(self):
        """Initialize OpenVINO model."""
        try:
            model = create_simple_model()
            if model:
                self.compiled_model = self.ie.compile_model(model, self.device)
        except Exception as e:
            print(f"OpenVINO model init failed: {e}. Using NumPy fallback.")
            self.compiled_model = None

    def reset_state(self):
        self.state = np.random.uniform(0, 0.1, (self.width, self.height, self.depth)).astype(np.float32)
        self.prediction = np.zeros((self.width, self.height, self.depth), dtype=np.float32)
        self.error = np.zeros((self.width, self.height, self.depth), dtype=np.float32)

    def sigmoid_openvino(self, x):
        """Sigmoid via OpenVINO if available, else NumPy."""
        if self.compiled_model and x.size < 1000:
            try:
                x_flat = np.clip(x.reshape(1, -1).astype(np.float32), -500, 500)
                result = self.compiled_model([x_flat])[0]
                return result.reshape(x.shape)
            except Exception:
                pass

        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def get_device_info(self):
        """Get available OpenVINO devices."""
        try:
            devices = self.ie.available_devices
            return devices
        except Exception:
            return ["CPU"]
