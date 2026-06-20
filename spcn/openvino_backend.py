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
        self.state[:, :, 1:] = 0
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
