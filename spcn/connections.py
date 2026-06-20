import numpy as np


def build_connections(width, height, radius=2):
    """Generate list of valid connection offsets (dx, dy) and their costs."""
    connections = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            cost = dx * dx + dy * dy
            connections.append((dx, dy, cost))
    return connections


def get_neighbor_indices(x, y, dx, dy, width, height):
    """Get neighbor coordinates, returning None if out of bounds."""
    nx, ny = x + dx, y + dy
    if 0 <= nx < width and 0 <= ny < height:
        return nx, ny
    return None


def is_valid_neighbor(x, y, dx, dy, width, height):
    """Check if neighbor at offset (dx, dy) from (x, y) is within bounds."""
    nx, ny = x + dx, y + dy
    return 0 <= nx < width and 0 <= ny < height
