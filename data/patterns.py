import numpy as np


def generate_patterns():
    """Generate 4 simple binary patterns for Phase 1."""
    patterns = []

    pattern_0 = np.zeros((4, 4), dtype=np.float32)
    pattern_0[:2, :2] = 1.0
    patterns.append(pattern_0)

    pattern_1 = np.zeros((4, 4), dtype=np.float32)
    pattern_1[:2, 2:] = 1.0
    patterns.append(pattern_1)

    pattern_2 = np.zeros((4, 4), dtype=np.float32)
    pattern_2[2:, :2] = 1.0
    patterns.append(pattern_2)

    pattern_3 = np.zeros((4, 4), dtype=np.float32)
    np.fill_diagonal(pattern_3, 1.0)
    patterns.append(pattern_3)

    return patterns
