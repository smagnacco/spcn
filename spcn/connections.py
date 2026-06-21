import numpy as np


def build_connections(width, height, radius=2, z_radius=1, gamma=1.0):
    """Generate list of valid connection offsets (dx, dy, dz) and their costs.

    dz ranges over [-z_radius, 0]: a voxel only predicts from its own layer
    (dz=0, lateral) or from layers below it (dz<0, feedforward), never above.
    cost = dx^2 + dy^2 + gamma * dz^2, so gamma controls how much harder it is
    to route information across layers (vertical) vs within a layer (lateral).
    """
    connections = []
    for dz in range(-z_radius, 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                cost = dx * dx + dy * dy + gamma * dz * dz
                connections.append((dx, dy, dz, cost))
    return connections


def get_neighbor_indices(x, y, z, dx, dy, dz, width, height, depth):
    """Get neighbor coordinates, returning None if out of bounds."""
    nx, ny, nz = x + dx, y + dy, z + dz
    if 0 <= nx < width and 0 <= ny < height and 0 <= nz < depth:
        return nx, ny, nz
    return None


def is_valid_neighbor(x, y, z, dx, dy, dz, width, height, depth):
    """Check if neighbor at offset (dx, dy, dz) from (x, y, z) is within bounds."""
    nx, ny, nz = x + dx, y + dy, z + dz
    return 0 <= nx < width and 0 <= ny < height and 0 <= nz < depth
