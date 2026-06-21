# Phase 2 Findings: Why Top-Down-Conditioned Predictive Coding Hits a Classification Ceiling

## Summary

Phase 2 (top-down label conditioning via a faded clamp on the top layer,
real MNIST data, vectorized 3D connectivity) was implemented and debugged
through four distinct bugs. After fixing all four, the network still
cannot classify above chance (~10-12%) on real MNIST. Root-cause analysis
via an argmax-distribution probe shows this is not a remaining bug but a
**structural ceiling**: pure local predictive-coding + a generic top-down
prior has no mechanism that pulls same-class representations together and
pushes different-class representations apart, which is what classification
requires. Phase 3 addresses this with a contrastive Hebbian update.

## Bugs found and fixed, in order

1. **Connectivity was not actually 3D.** Original code only wired each
   voxel to the single layer directly below it. Fixed in `connections.py`
   / `grid.py` / `neuron.py` to support real `(dx, dy, dz)` neighborhoods
   with `cost = dx² + dy² + γ·dz²`.

2. **Distance penalty subtracted cost unconditionally**, driving far
   weights toward unbounded negative values regardless of correlation
   sign. Fixed to gate the Hebbian term multiplicatively
   (`exp(-λ·cost)`) instead.

3. **Pure-Python nested loops** over every voxel × neighbor offset — not
   viable past toy grid sizes. Vectorized in `neuron.py` to one NumPy
   array op per offset (~75 ops) instead of one per (voxel, offset) pair.
   ~4x speedup measured on the toy grid, confirmed scaling to 16×16×6.

4. **No homeostasis** — local-only learning has a trivial fixed point
   (collapse to 0, or saturate to a flat constant) with nothing to prevent
   it. Added an intrinsic-plasticity-style EMA term in `update_states`
   pulling each voxel toward a target firing rate.

5. **Synthetic MNIST stand-in reused the same 4 region masks across 10
   digit classes** (`region = digit % 4`), so digits `{0,4,8}`, `{1,5,9}`,
   etc. were structurally identical except for random noise — no
   real per-digit signal existed in the data to learn. Replaced with real
   MNIST via `sklearn.datasets.fetch_openml('mnist_784')`.

6. **Hard top-down clamp held for the entire settle loop**, including the
   final iteration before the classifier head read out the top layer.
   `digit_weights` learned a trivial "prototype → label" lookup that never
   touched the image. Fixed by fading the clamp from full strength to zero
   across the settle iterations, so the final readout reflects the
   network's own bottom-up computation.

7. **Random weight initialization at flat scale (`0.01`) was a contracting
   map.** Summing ~75 neighbor contributions with small iid weights is, by
   basic statistics, an averaging operation: every layer converged to the
   same input-independent flat value within ~4 free-running iterations,
   confirmed by direct measurement (state std 0.011 → 0.0024, plateaued).
   Fixed with Kaiming-style initialization (`1/sqrt(fan_in)`), restoring
   stable input-dependent variance at the settle fixed point (std ~0.0225,
   no longer collapsing).

8. **Digit-head learning rate was shared with the main settle loop
   (`0.1`), too high for a single-sample update.** One `update_digit_weights`
   call drove `digit_output` to near-perfect saturation (0.9999988) for
   whichever sample was just seen — meaning across 1000 samples/epoch the
   head was oscillating, not converging. Decoupled into a separate
   `digit_learning_rate` (0.005), confirmed it no longer saturates on a
   single update.

## The remaining ceiling (not a bug)

After fixing all eight issues above, train and test accuracy on real
MNIST are both still pinned near chance, and loss decreases smoothly
without accuracy moving. An argmax-distribution probe across 200 test
images gave a definitive answer:

```
Predicted class counts: Counter({5: 200})
Num unique predicted classes used: 1
digit_output std per class: ~0.0001 (effectively constant)
```

**Every single test image is classified as the same digit.** This is mode
collapse, not noise: `digit_weights` converged to a static bias vector
that ignores the image. The earlier fixes (weight init, decoupled LR) did
restore real, input-dependent variance in the top layer's settled state —
but that variance is not *organized along class-discriminative axes*.
Nothing in the local learning rule pushes same-label states toward shared
regions of state space or pushes different-label states apart.

`update_weights` only minimizes local reconstruction error
(`state - prediction`); the top-down clamp only pulls the top layer toward
a generic spatial prior during early settle iterations. Neither mechanism
teaches the hidden layers "make 3s and 8s look different." A linear
readout head can only separate classes if the upstream representation
already separates them — and nothing here is responsible for making that
true. This is the structural property of the learning rule, not a
parameter to retune.

## Path forward: Phase 3

Phase 3 adds a **contrastive Hebbian update**: in addition to the existing
reconstruction-error-driven Hebbian term, when a sample's settled top-layer
state is compared against a running per-class prototype, it is pulled
toward its own class's prototype and pushed away from a different class's
prototype. This is a local, pairwise rule (no global loss, no backprop) —
the standard way predictive-coding-style local learning is given a
discriminative objective in the literature. See `phase3.py` / the
`update_contrastive` function in `spcn/neuron.py`.

## Phase 3 results and a second finding

Three implementations of the contrastive update were tried, in order of
increasing reach through the network:

1. **State-nudge only**: pull/push applied directly to `grid.state` at the
   top layer after settling. This produced well-differentiated
   `class_prototypes` (pairwise distances ~0.18-0.19 across all 10
   classes) — the contrastive math itself works — but had zero effect on
   free-running test-time dynamics, since the nudge only ever touched
   `grid.state`, not the weights that generate it. Test-time states still
   collapsed to one fixed point regardless of input (nearest-prototype
   classification on test states: 198/200 to one class).

2. **Top-layer weight update only**: rewrote the update to adjust the top
   layer's incoming weights via the same Hebbian mechanism as
   `update_weights`, once per sample. Measured no improvement — the
   reconstruction-error term updates the same weights 6x per sample
   (once per settle iteration) across the whole network; one contrastive
   update per sample was too weak to compete.

3. **Multi-layer weight update, every settle iteration**: the contrastive
   error signal starts at the top layer and is spread backward through
   every layer using the network's own existing weights
   (`_backward_spread`), with magnitude bounded by `tanh` to prevent the
   numerical blowup an earlier unbounded version hit (NaN within one
   epoch). Applied at every settle iteration, matching the reconstruction
   term's frequency.

This third version is qualitatively different from every earlier result
in this project: **train accuracy climbs smoothly and monotonically across
epochs for the first time** — 9.9% → 18.4% over 15 epochs, loss decreasing
in lockstep, no saturation, no collapse. This confirms the multi-layer
contrastive signal is a real, working discriminative learning mechanism.

But: **test accuracy stays flat at 11.5% (chance), and the argmax probe
shows 200/200 test predictions still collapse to a single class.** Train
accuracy improves on samples the network has seen; nothing it has learned
transfers to unseen samples. This is a different failure mode than phase
2's pure mode collapse — it is overfitting in its classic form, consistent
with the capacity/sample-size mismatch flagged earlier (~110K parameters,
1000 training samples) now compounded by a contrastive mechanism that may
be exploiting per-sample idiosyncrasies pulled toward a running prototype
rather than learning a class-general code. Closing this gap is a distinct
problem from the one phase 3 set out to solve, and needs its own pass
(e.g. far more training samples, regularizing the contrastive update
itself, or evaluating whether the per-class prototype is stable enough
across the dataset to generalize at all).

## Phase 3 follow-up: more training samples partially closes the gap

Tested the most direct fix for the overfitting diagnosis above:
`data/mnist_loader.py`'s defaults were raised from 1000/200 to 5000/1000
(train/test), and `phase3.py` was rerun (5 epochs, same contrastive
configuration as the 15-epoch run, `contrastive_rate=0.05`).

```
Epoch  1 | loss: 0.0957 | train_acc: 0.127 | test_acc: 0.107
Epoch  2 | loss: 0.0920 | train_acc: 0.156 | test_acc: 0.110
Epoch  3 | loss: 0.0902 | train_acc: 0.173 | test_acc: 0.175
Epoch  4 | loss: 0.0899 | train_acc: 0.177 | test_acc: 0.093
Epoch  5 | loss: 0.0900 | train_acc: 0.179 | test_acc: 0.160

SPCN final test accuracy: 0.160   (was 0.115 flat at 1000 samples)
MLP baseline test accuracy: 0.260 (was ~0.175-0.190 at 1000 samples)
```

**Test accuracy moved off the chance floor for the first time** — peaking
at 17.5% (epoch 3), settling at 16.0% (epoch 5), versus a flat 11.5%
across all 15 epochs at the smaller sample size. This confirms the
overfitting diagnosis: more distinct training examples does produce real,
if modest, generalization. Train accuracy kept its healthy smooth climb
(12.7% → 17.9%), same pattern as the 1000-sample run.

However the result is not yet stable: test accuracy oscillates across
epochs (10.7% → 11.0% → 17.5% → 9.3% → 16.0%) rather than improving
monotonically, a different signature from either mode collapse or clean
overfitting — closer to a noisy, early-stage learning regime. The MLP
baseline also improved with more data (19% → 26%) and remains about 10
points ahead of the SPCN at this scale.

**Net assessment**: the contrastive Hebbian update (phase 3) and more
training data (this follow-up) together turn the network from a
non-functional classifier (phase 2: structural mode collapse, chance
accuracy regardless of data or training length) into a functional but
weak and unstable one (phase 3 + more data: real but noisy generalization,
trailing a same-sized MLP baseline by ~10 points). Stopping here as a
documented checkpoint. Candidates for a future pass, not pursued further
in this session: more epochs to see whether the test-accuracy oscillation
damps out, a lower or decaying contrastive/learning rate to reduce
epoch-to-epoch variance, and a larger training set still (5000 remains
small relative to the ~110K-parameter network).
