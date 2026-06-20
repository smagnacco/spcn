# Spatial Predictive Coding Network (SPCN)

A biologically-inspired neural network based on predictive coding principles, operating on a 3D voxel grid with spatially-constrained, distance-penalized connections.

## Installation

```bash
pip install -r requirements.txt
```

## Running Phase 1

```bash
python main.py
```

This will train the network on 4 synthetic binary patterns and generate:
- Error convergence curve
- Activation heatmaps for each layer
- Summary statistics

## Architecture

- **Grid**: W×H×D voxel lattice
- **Phase 1**: 4×4×3 grid (4×4 input, 3 layers)
- **Learning**: Fully local, no backpropagation
- **Connectivity**: Distance-penalized (cost = dx² + dy²)
