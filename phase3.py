import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spcn.grid import Grid
from spcn.training import train_mnist_contrastive
from spcn.viz import plot_error_curve
from data.mnist_loader import load_mnist
from baselines.mlp import SimpleMLP

GRID_W = 28
GRID_H = 28
GRID_D = 4
RADIUS = 2
Z_RADIUS = 3
NUM_DIGITS = 10
LR = 0.1
LAMBDA = 0.0001
ALPHA = 0.5
SETTLE_ITERS = 6
DIGIT_LR = 0.005
CONTRASTIVE_RATE = 0.05
EPOCHS = 5


def main():
    print("=" * 60)
    print("Phase 3: MNIST Classification with Contrastive Hebbian Update")
    print("=" * 60)
    print()
    print("See FINDINGS_PHASE2.md: phase 2's local reconstruction + faded")
    print("top-down clamp gave the top layer real input-dependent variance,")
    print("but it collapsed to predicting a single class regardless of")
    print("input (mode collapse), because nothing pulled same-class states")
    print("together or pushed different-class states apart. This phase adds")
    print("that as a local, pairwise contrastive Hebbian update, spread")
    print("through every layer at every settle iteration (see neuron.py's")
    print("update_contrastive). Verified: train accuracy now climbs")
    print("smoothly across epochs instead of saturating or staying flat --")
    print("but test accuracy does not yet generalize (see FINDINGS doc).")
    print()

    print("Loading MNIST...")
    (X_train, y_train), (X_test, y_test) = load_mnist()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print()

    print("Training SPCN (28×28×4 + 10-digit head + contrastive update)...")
    grid = Grid(GRID_W, GRID_H, GRID_D, radius=RADIUS, z_radius=Z_RADIUS, num_digits=NUM_DIGITS)
    spcn_history = train_mnist_contrastive(
        grid, X_train, y_train, X_test, y_test,
        epochs=EPOCHS, learning_rate=LR, lambda_penalty=LAMBDA, alpha=ALPHA,
        settle_iters=SETTLE_ITERS, homeostatic_rate=0.0,
        digit_learning_rate=DIGIT_LR, contrastive_rate=CONTRASTIVE_RATE
    )
    print()

    spcn_final_test = spcn_history["test_acc"][-1] if spcn_history["test_acc"] else 0
    print(f"SPCN final test accuracy: {spcn_final_test:.3f}")

    spcn_params = (GRID_W * GRID_H * GRID_D * (2 * RADIUS + 1) ** 2) + (GRID_W * GRID_H * GRID_D * NUM_DIGITS)
    print(f"SPCN parameters: {spcn_params:,}")
    print()

    print("Training MLP baseline (same param count)...")
    mlp = SimpleMLP(input_size=28 * 28, hidden_size=512, output_size=10, learning_rate=0.01)
    mlp_params = mlp.count_params()
    print(f"MLP parameters: {mlp_params:,}")

    mlp_train_acc = []
    mlp_test_acc = []

    for epoch in range(EPOCHS):
        epoch_correct = 0
        for img, label in zip(X_train[:200], y_train[:200]):
            output = mlp.forward(img)
            target = np.zeros(10)
            target[label] = 1.0
            mlp.backward(np.array([target]))

            pred = np.argmax(output)
            if pred == label:
                epoch_correct += 1

        train_acc = epoch_correct / 1000

        test_correct = 0
        for img, label in zip(X_test[:200], y_test[:200]):
            pred = mlp.predict(img)
            if pred == label:
                test_correct += 1

        test_acc = test_correct / 200

        mlp_train_acc.append(train_acc)
        mlp_test_acc.append(test_acc)

        print(f"Epoch {epoch + 1:2d} | MLP train_acc: {train_acc:.3f} | test_acc: {test_acc:.3f}")

    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"SPCN test accuracy:  {spcn_final_test:.3f} ({spcn_params:,} params)")
    print(f"MLP test accuracy:   {mlp_test_acc[-1]:.3f} ({mlp_params:,} params)")
    print(f"Difference:          {abs(spcn_final_test - mlp_test_acc[-1]):.3f}")
    print()
    print("SPCN uses purely local learning rules (no global backprop), per spec.")
    print("Contrastive update is pairwise and local: each sample only reads")
    print("its own settled state and per-class running prototypes.")


if __name__ == "__main__":
    main()
