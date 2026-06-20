#!/usr/bin/env python3
"""Test OpenVINO GPU inference performance"""

import time
import numpy as np

try:
    import openvino as ov
except ImportError:
    print("Error: OpenVINO not installed. Run: source ~/openvino_env/bin/activate")
    exit(1)

def test_inference():
    """Test GPU vs CPU inference performance"""

    print("=" * 60)
    print("OpenVINO GPU/CPU Inference Test")
    print("=" * 60)
    print()

    # Initialize OpenVINO Core
    core = ov.Core()
    print(f"Available devices: {core.available_devices}")
    print()

    # Create a simple model using function interface
    from openvino import opset13

    # Build a simple neural network model
    input_shape = [1, 3, 224, 224]  # Batch, Channels, Height, Width
    param = opset13.parameter(input_shape, ov.Type.f32, "input_image")

    # Conv2D weights: [output_channels, input_channels, kernel_h, kernel_w]
    conv_weights = opset13.constant(np.random.randn(16, 3, 3, 3).astype(np.float32))
    conv = opset13.convolution(param, conv_weights, [1, 1], [1, 1], [1, 1], [1, 1])

    relu = opset13.relu(conv)
    result = opset13.result(relu)

    model = ov.Model([result], [param])
    print("Model created: Simple Conv2D -> ReLU")
    print(f"Input shape: {input_shape}")
    print()

    # Prepare input data
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Test on CPU
    print("-" * 60)
    print("Testing on CPU")
    print("-" * 60)
    compiled_model_cpu = core.compile_model(model, "CPU")

    start = time.time()
    for i in range(10):
        output_cpu = compiled_model_cpu([input_data])
    cpu_time = time.time() - start
    print(f"10 inferences on CPU: {cpu_time*1000:.2f}ms")
    print(f"Average per inference: {cpu_time*100:.2f}ms")
    print()

    # Test on GPU (if available)
    if "GPU" in core.available_devices:
        print("-" * 60)
        print("Testing on GPU")
        print("-" * 60)
        try:
            compiled_model_gpu = core.compile_model(model, "GPU")

            start = time.time()
            for i in range(10):
                output_gpu = compiled_model_gpu([input_data])
            gpu_time = time.time() - start
            print(f"10 inferences on GPU: {gpu_time*1000:.2f}ms")
            print(f"Average per inference: {gpu_time*100:.2f}ms")
            print()

            # Compare results
            speedup = cpu_time / gpu_time
            print("-" * 60)
            if speedup > 1:
                print(f"✓ GPU is {speedup:.2f}x FASTER than CPU")
            else:
                print(f"✗ GPU is {1/speedup:.2f}x SLOWER than CPU")
                print("  (Small models may be faster on CPU due to overhead)")
            print("-" * 60)
        except Exception as e:
            print(f"⚠️  GPU inference failed: {e}")
    else:
        print("⚠️  GPU device not available for testing")
        print()

if __name__ == "__main__":
    test_inference()
