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

## Continual Learning Phases (2–5)

Beyond Phase 1, the project asks a single question: **can a purely local rule
(no backprop, no central controller) resist catastrophic forgetting in
class-incremental continual learning?** Protocol throughout: Split-MNIST, 3
sequential tasks (T1=0-3, T2=4-6, T3=7-9), single shared 10-way head, masked CE
in training, unmasked argmax in eval. Baselines: MLP naive and MLP+EWC.

Each phase is documented in its own findings file:

- **Phase 2 — capacity ceiling** (`FINDINGS_PHASE2.md`, `POSTMORTEM_PHASE2.md`):
  the local predictive-coding rule with a generic top-down prior has no
  discriminative term — it cannot pull same-class representations together and
  push different-class apart. Structural ceiling on *which function* the rule
  can learn.

- **Phase 4 — scalar representation ceiling** (`FINDINGS_PHASE4.md`): top-k
  sparsity + a contrastive term lift the Phase-2 ceiling (each task now learns,
  peak ~0.89). The SPCN even produces **emergent orthogonal allocation** between
  tasks from strictly local rules — disjoint cores, no explicit masks or
  controller (unlike PackNet/HAT). But its BWT-vs-Plasticity Pareto frontier
  stays **interior to EWC**: a scalar activation cannot preserve one neuron's
  role in one task and adapt it to another at once. This **emergent orthogonal
  allocation remains the project's best result.**

- **Phase 5 — complex activation, CLOSED** (`FINDINGS_PHASE5.md`,
  `PHASE5_DESIGN.md`, `continual_learning_phase5.py`): tested whether a complex
  activation `z = r·e^(iθ)` (magnitude = intensity, phase = task/context) breaks
  the scalar ceiling. **It does not, in this substrate.** The decisive sanity —
  the complex cosine between tasks on shared neurons, which had to reach ≈ −0.5 —
  comes out ≈ 0 from **both** extremes of phase-sharing topology: recruitment ON
  (disjoint cores → nothing shared to separate) and OFF (full overlap → every
  neuron averages all tasks). These are the only two topologies, not samples of
  a continuum: any intermediate point needs a mechanism to commit each neuron to
  a dominant task, and that mechanism *is* spatial allocation.

  **Structural finding:** phase superposition needs per-neuron competition to
  separate tasks, and that competition is functionally equivalent to spatial
  allocation. In this substrate (linear phase accumulator + projected readout)
  phase adds no new degree of freedom over Phase-4's spatial partition. The
  project closes on Phase-4 emergent allocation as its best result. *Not tested
  (open door):* substrates with native phase competition — coupled oscillators,
  Kuramoto dynamics — where synchronization could force per-neuron phase
  commitment without an explicit allocation step.

```bash
python continual_learning_3tasks.py    # Phase 4 main experiment + sweep
python continual_learning_phase5.py     # Phase 5a' complex activation
```
