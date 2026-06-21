import numpy as np
from .connections import build_connections


def sigmoid(x, grid=None):
    """Sigmoid with optional OpenVINO acceleration."""
    if grid and hasattr(grid, 'sigmoid_openvino'):
        return grid.sigmoid_openvino(x)
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def _get_connections(grid):
    """Cache the (dx,dy,dz,cost) offset list on the grid."""
    cached = getattr(grid, "_connections", None)
    if cached is None or getattr(grid, "_connections_key", None) != (grid.radius, grid.z_radius, grid.gamma):
        cached = build_connections(grid.width, grid.height, radius=grid.radius,
                                     z_radius=grid.z_radius, gamma=grid.gamma)
        grid._connections = cached
        grid._connections_key = (grid.radius, grid.z_radius, grid.gamma)
    return cached


def _shifted(volume, dx, dy, dz):
    """Return `volume` shifted so that result[x,y,z] == volume[x+dx, y+dy, z+dz],
    with out-of-bounds positions filled with 0."""
    w, h, d = volume.shape
    out = np.zeros_like(volume)

    src_x0, src_x1 = max(0, dx), min(w, w + dx)
    src_y0, src_y1 = max(0, dy), min(h, h + dy)
    src_z0, src_z1 = max(0, dz), min(d, d + dz)

    dst_x0, dst_x1 = max(0, -dx), min(w, w - dx)
    dst_y0, dst_y1 = max(0, -dy), min(h, h - dy)
    dst_z0, dst_z1 = max(0, -dz), min(d, d - dz)

    if src_x0 >= src_x1 or src_y0 >= src_y1 or src_z0 >= src_z1:
        return out

    out[dst_x0:dst_x1, dst_y0:dst_y1, dst_z0:dst_z1] = volume[src_x0:src_x1, src_y0:src_y1, src_z0:src_z1]
    return out


def compute_predictions(grid):
    """Each neuron predicts itself from same-layer and lower-layer neighbors,
    reachable within radius (lateral) and z_radius (depth). Vectorized: one
    array op per offset instead of one per (voxel, offset) pair."""
    connections = _get_connections(grid)
    prediction_sum = np.zeros_like(grid.state)

    for dx, dy, dz, _cost in connections:
        ox, oy, oz = dx + grid.radius, dy + grid.radius, -dz
        weight = grid.weights[:, :, :, ox, oy, oz]
        neighbor_state = _shifted(grid.state, dx, dy, dz)
        prediction_sum += weight * neighbor_state

    grid.prediction[:] = sigmoid(prediction_sum, grid)
    grid.prediction[:, :, 0] = 0.0


def compute_errors(grid):
    """Each neuron's error is its own state minus its prediction (skip input layer 0)."""
    grid.error[:, :, 1:] = grid.state[:, :, 1:] - grid.prediction[:, :, 1:]


def update_weights(grid, learning_rate=0.01, lambda_penalty=0.001, weight_decay=0.0):
    """Update weights using local Hebbian rule, gated by a distance-decaying
    factor so far connections grow more slowly than near ones, plus a general
    L2 weight decay so weights can't grow unbounded purely by memorizing
    training samples. Vectorized: one array op per offset instead of one per
    (voxel, offset) pair."""
    connections = _get_connections(grid)

    error = np.zeros_like(grid.error)
    error[:, :, 1:] = grid.error[:, :, 1:]

    for dx, dy, dz, cost in connections:
        ox, oy, oz = dx + grid.radius, dy + grid.radius, -dz
        neighbor_state = _shifted(grid.state, dx, dy, dz)
        decay = np.exp(-lambda_penalty * cost)
        weight = grid.weights[:, :, :, ox, oy, oz]
        delta_w = learning_rate * (error * neighbor_state * decay - weight_decay * weight)
        grid.weights[:, :, :, ox, oy, oz] += delta_w


def update_states(grid, alpha=0.1, target_rate=0.1, homeostatic_rate=0.01):
    """Update neuron states based on error signal, with homeostatic
    regulation pulling each voxel's running activity toward target_rate so
    purely local learning can't collapse all activity to 0 or saturate to a
    trivial constant plateau."""
    if not hasattr(grid, "avg_activity") or grid.avg_activity.shape != grid.state.shape:
        grid.avg_activity = np.full_like(grid.state, target_rate)

    drive = grid.state[:, :, 1:] + alpha * grid.error[:, :, 1:]
    drive -= homeostatic_rate * (grid.avg_activity[:, :, 1:] - target_rate)

    grid.state[:, :, 1:] = sigmoid(drive, grid)
    grid.avg_activity[:, :, 1:] += homeostatic_rate * (grid.state[:, :, 1:] - grid.avg_activity[:, :, 1:])


def compute_digit_output(grid):
    """Top layer output: num_digits class predictions read out from the top
    layer's settled state via a local linear head."""
    if not hasattr(grid, 'digit_weights'):
        return

    top_state = grid.state[:, :, grid.depth - 1]
    digit_sum = np.einsum('xy,xyd->d', top_state, grid.digit_weights)
    grid.digit_output[:] = sigmoid(digit_sum, grid)


def update_digit_weights(grid, learning_rate=0.01, weight_decay=0.0):
    """Update digit-head weights using local error between target and output,
    plus L2 decay -- this is the network's only fully-dense, unconstrained
    layer (no distance penalty applies to a global classification readout),
    so it is the most overfit-prone and benefits most from decay."""
    if not hasattr(grid, 'digit_weights'):
        return

    top_state = grid.state[:, :, grid.depth - 1]
    error = grid.digit_target - grid.digit_output
    delta_w = learning_rate * (np.einsum('xy,d->xyd', top_state, error) - weight_decay * grid.digit_weights)
    grid.digit_weights += delta_w


def clamp_top_layer(grid, label, blend=1.0):
    """Bias the top layer's state toward the digit prototype pattern for
    `label`, acting as a top-down prior during the settle loop. blend=1.0
    hard-clamps; lower values mix the prototype with the current state so
    the clamp itself can be eased local-error-style rather than forced."""
    if not hasattr(grid, 'digit_prototypes'):
        return
    prototype = grid.digit_prototypes[label]
    top = grid.depth - 1
    grid.state[:, :, top] = (1 - blend) * grid.state[:, :, top] + blend * prototype


def _backward_spread(grid, signal):
    """Spread a per-voxel signal one hop "backward" along existing
    connections: for each (dx,dy,dz) offset and its weight w[x,y,z,...],
    voxel (x+dx,y+dy,z+dz) receives w * signal[x,y,z]. This mirrors
    compute_predictions' forward gather, run in reverse, so a signal
    placed at the top layer can reach lower layers using only the
    network's own existing weights -- still local (each voxel's
    contribution depends only on its own outgoing weights and its own
    signal value), just propagated through the stack instead of confined
    to one layer.
    """
    connections = _get_connections(grid)
    spread = np.zeros_like(grid.state)
    for dx, dy, dz, _cost in connections:
        ox, oy, oz = dx + grid.radius, dy + grid.radius, -dz
        weight = grid.weights[:, :, :, ox, oy, oz]
        contribution = weight * signal
        spread += _shifted(contribution, -dx, -dy, -dz)
    return np.tanh(spread)


def update_contrastive(grid, label, learning_rate=0.01, neg_label=None,
                        prototype_decay=0.05, weight_decay=0.0, depth_falloff=0.5):
    """Contrastive Hebbian update on the weights throughout the network,
    not just the top layer.

    Each class keeps a running EMA "prototype" of the top-layer state seen
    for that class so far (grid.class_prototypes), updated purely locally
    -- no other sample's data is read, no global loss. The discriminative
    signal -- "move toward your own class's prototype, away from a
    different class's prototype" -- starts at the top layer and is spread
    backward one hop at a time using the network's own existing weights
    (_backward_spread), with its magnitude shrinking by depth_falloff per
    hop. At each layer it is treated like compute_errors' reconstruction
    error and fed into the same Hebbian weight-update mechanism as
    update_weights.

    Two earlier versions of this function were tried and measured
    insufficient (see FINDINGS_PHASE2.md / PHASE3 notes): nudging
    grid.state directly differentiated prototypes but didn't survive to
    free-running test-time dynamics; updating only the top layer's weights
    was too small a signal (1 update/sample vs. update_weights' 6/sample
    across the whole network) to outcompete the reconstruction term's pull
    toward its flat fixed point. Spreading the signal through every layer,
    every settle iteration, is the attempt to make the discriminative
    pressure comparable in scale to the reconstruction pressure it has to
    compete with.
    """
    if not hasattr(grid, 'digit_prototypes'):
        return

    top = grid.depth - 1
    num_digits = grid.num_digits

    if not hasattr(grid, 'class_prototypes') or grid.class_prototypes.shape[0] != num_digits:
        grid.class_prototypes = grid.digit_prototypes.copy()
        grid.class_prototype_seen = np.zeros(num_digits, dtype=bool)

    if neg_label is None:
        neg_label = np.random.randint(0, num_digits)
        while neg_label == label and num_digits > 1:
            neg_label = np.random.randint(0, num_digits)

    top_state = grid.state[:, :, top]
    pos_prototype = grid.class_prototypes[label]
    neg_prototype = grid.class_prototypes[neg_label]

    signal = np.zeros_like(grid.state)
    signal[:, :, top] = (pos_prototype - top_state) - (top_state - neg_prototype)

    connections = _get_connections(grid)
    for layer in range(grid.depth):
        layer_error = signal[:, :, layer]
        if not np.any(layer_error):
            signal = _backward_spread(grid, signal) * depth_falloff
            continue

        for dx, dy, dz, cost in connections:
            nz = layer + dz
            if not (0 <= nz < grid.depth):
                continue
            ox, oy, oz = dx + grid.radius, dy + grid.radius, -dz
            neighbor_state_full = _shifted(grid.state, dx, dy, dz)
            neighbor_state = neighbor_state_full[:, :, layer]
            weight = grid.weights[:, :, layer, ox, oy, oz]
            delta_w = learning_rate * (layer_error * neighbor_state - weight_decay * weight)
            grid.weights[:, :, layer, ox, oy, oz] += delta_w

        signal = _backward_spread(grid, signal) * depth_falloff

    if grid.class_prototype_seen[label]:
        grid.class_prototypes[label] += prototype_decay * (top_state - grid.class_prototypes[label])
    else:
        grid.class_prototypes[label] = top_state.copy()
        grid.class_prototype_seen[label] = True
