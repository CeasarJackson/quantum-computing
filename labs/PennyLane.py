#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/PennyLane.py

Purpose
-------
Train a four-qubit variational binary classifier using PennyLane's Autograd
interface and parameter-shift differentiation. RX feature encoding, two
trainable RY/RZ layers, CNOT entanglement, and Pauli-Z readout are used.
Synthetic parity labels are encoded as -1/+1 for mean-squared-error training.
Supports independently sampled evaluation or disjoint parity inputs, with a
majority-class classical baseline for comparison.

Prerequisites
-------------
Python 3.13, PennyLane 0.45.1, NumPy, and labs/applog.py.
Use the project's existing .venv; no additional environment is created.

Security
--------
No credentials, network calls, or external data. Synthetic samples only.
Dry-run validates configuration and dependencies without training or writing
model artifacts. Shared applog creates a timestamped log unless --help is used.

Usage
-----
    python labs/PennyLane.py --help
    python labs/PennyLane.py --dry-run
    python labs/PennyLane.py --steps 50 --samples 20 --test-samples 20 --seed 42
    python labs/PennyLane.py --steps 5 --samples 8 --verbose
    python labs/PennyLane.py --split disjoint --train-unique 12 --steps 50
    python labs/PennyLane.py --split disjoint --split-seed 42 --init-seed 47

Log files
---------
logs/PennyLane/PennyLane_<YYYYmmdd_HHMMSS>.log (from project root).

Exit Codes
----------
0   Successful training or dry-run.
1   Dependency, training, or numerical failure.
2   Invalid command-line argument.
130 Interrupted by user.
===============================================================================
"""

import argparse
import sys

import numpy as np

from applog import get_logger

# The circuit uses four qubits and 2 trainable rotations per qubit per layer.
N_WIRES = 4
N_LAYERS = 2
N_PARAMETERS = N_WIRES * N_LAYERS * 2


def positive_int(value):
    """Parse a strictly positive integer for command-line arguments."""
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return result


def positive_float(value):
    """Parse a finite, strictly positive float for command-line arguments."""
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and > 0")
    return result


def parse_args(argv=None):
    """Return validated command-line configuration."""
    parser = argparse.ArgumentParser(
        description="Four-qubit PennyLane variational binary classifier"
    )
    parser.add_argument("--steps", type=positive_int, default=50)
    parser.add_argument("--samples", type=positive_int, default=20)
    parser.add_argument("--test-samples", type=positive_int, default=20)
    parser.add_argument("--split", choices=("sampled", "disjoint"), default="sampled")
    parser.add_argument("--train-unique", type=positive_int, default=12)
    parser.add_argument("--learning-rate", type=positive_float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Independent partition seed; defaults to --seed",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=None,
        help="Independent initialization seed; defaults to --seed + 1",
    )
    parser.add_argument(
        "--diff-method",
        choices=("parameter-shift", "best"),
        default="parameter-shift",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def make_dataset(samples, seed):
    """Create reproducible RX angles and balanced-sign parity targets."""
    rng = np.random.default_rng(seed)
    # Each bit becomes an angle of 0 or pi; its sign is a Pauli-Z eigenvalue.
    # The parity task is deliberately non-linear in the four input bits.
    # The random generator is local so no global NumPy state is modified.
    bits = rng.integers(0, 2, size=(samples, N_WIRES))
    angles = bits.astype(float) * np.pi
    labels = 1.0 - 2.0 * (bits.sum(axis=1) % 2)
    return angles, labels


def make_disjoint_dataset(train_unique, seed):
    """Partition all 16 binary inputs into nonoverlapping unique subsets."""
    if not 1 <= train_unique < 16:
        raise ValueError("--train-unique must be between 1 and 15")
    # Exhaustively enumerate bit patterns instead of drawing with replacement.
    integers = np.arange(16, dtype=np.uint8)
    shifts = np.arange(N_WIRES - 1, -1, -1, dtype=np.uint8)
    bits = ((integers[:, None] >> shifts) & 1).astype(int)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(bits))
    train_bits = bits[order[:train_unique]]
    test_bits = bits[order[train_unique:]]
    # Verify disjointness before any circuit optimization can begin.
    train_set = {tuple(row) for row in train_bits}
    test_set = {tuple(row) for row in test_bits}
    if train_set & test_set:
        raise AssertionError("Disjoint split unexpectedly overlaps")
    train_y = 1.0 - 2.0 * (train_bits.sum(axis=1) % 2)
    test_y = 1.0 - 2.0 * (test_bits.sum(axis=1) % 2)
    return train_bits * np.pi, train_y, test_bits * np.pi, test_y


def majority_baseline(train_targets, test_targets):
    """Score a classical constant classifier learned from training labels."""
    # A tie predicts +1; test labels never influence this baseline decision.
    majority = 1.0 if np.sum(train_targets) >= 0 else -1.0
    train_accuracy = float(np.mean(train_targets == majority))
    test_accuracy = float(np.mean(test_targets == majority))
    return majority, train_accuracy, test_accuracy


def make_circuit(qml, method):
    """Build a differentiable QNode with feature encoding and entanglement."""
    device = qml.device("default.qubit", wires=N_WIRES)

    @qml.qnode(device, interface="autograd", diff_method=method)
    def circuit(features, weights):
        """Encode a sample, apply trainable gates, and measure qubit zero."""
        # RX embeds each classical input as a qubit rotation.
        for wire in range(N_WIRES):
            qml.RX(features[wire], wires=wire)
        # RY and RZ are learned rotations; repeated layers add expressivity.
        for layer in range(N_LAYERS):
            for wire in range(N_WIRES):
                qml.RY(weights[layer, wire, 0], wires=wire)
                qml.RZ(weights[layer, wire, 1], wires=wire)
            # CNOTs entangle neighboring wires, then close the four-wire ring.
            # Entanglement lets the readout respond to multi-bit correlations.
            for wire in range(N_WIRES - 1):
                qml.CNOT(wires=[wire, wire + 1])
            qml.CNOT(wires=[N_WIRES - 1, 0])
        # Pauli-Z lies in [-1, 1], matching the encoded class targets.
        return qml.expval(qml.PauliZ(0))

    return circuit


def loss_function(circuit, features, targets, weights, pnp):
    """Return the mean squared error of predictions against -1/+1 labels."""
    predictions = pnp.stack([circuit(row, weights) for row in features])
    return pnp.mean((predictions - targets) ** 2)


def evaluate(circuit, features, targets, weights):
    """Return MSE and accuracy for a fixed, non-optimizing dataset."""
    # The threshold at zero converts the Z expectation into a class sign.
    predictions = np.asarray([float(circuit(row, weights)) for row in features])
    if not np.all(np.isfinite(predictions)):
        raise FloatingPointError("Non-finite circuit predictions")
    mse = float(np.mean((predictions - targets) ** 2))
    accuracy = float(np.mean(np.where(predictions >= 0, 1.0, -1.0) == targets))
    return mse, accuracy


def train(args, log):
    """Train and evaluate a reproducible classifier with analytic gradients."""
    import pennylane as qml
    from pennylane import numpy as pnp

    # Preserve the original --seed behavior when overrides are omitted.
    split_seed = args.seed if args.split_seed is None else args.split_seed
    init_seed = args.seed + 1 if args.init_seed is None else args.init_seed
    log.info("Split seed=%d; initialization seed=%d", split_seed, init_seed)

    if args.split == "disjoint":
        # Train and test on mutually exclusive four-bit configurations.
        features, targets, test_features, test_targets = make_disjoint_dataset(
            args.train_unique, split_seed
        )
        log.ok(
            "Disjoint input split verified: %d train, %d test, overlap=0",
            len(targets),
            len(test_targets),
        )
    else:
        # Independent random draws can repeat the same four-bit inputs.
        features, targets = make_dataset(args.samples, split_seed)
        test_features, test_targets = make_dataset(args.test_samples, split_seed + 2)
    baseline_label, baseline_train, baseline_test = majority_baseline(
        targets, test_targets
    )
    circuit = make_circuit(qml, args.diff_method)
    rng = np.random.default_rng(init_seed)
    # Shape: (layers, qubits, [RY, RZ]); 2 * 4 * 2 = 16 parameters.
    initial = rng.normal(0.0, 0.1, (N_LAYERS, N_WIRES, 2))
    weights = pnp.array(initial, requires_grad=True)
    # Parameter-shift computes analytic gate gradients via shifted circuits.
    # Gradient descent updates only the trainable rotations, not the data.
    optimizer = qml.GradientDescentOptimizer(stepsize=args.learning_rate)

    def objective(current_weights):
        """Expose only the differentiable weights to the optimizer."""
        return loss_function(circuit, features, targets, current_weights, pnp)

    initial_loss = float(objective(weights))
    log.info(
        "Train=%d; test=%d; steps=%d; parameters=%d",
        len(targets),
        len(test_targets),
        args.steps,
        N_PARAMETERS,
    )
    log.info("Initial MSE: %.8f", initial_loss)
    for step in range(1, args.steps + 1):
        weights = optimizer.step(objective, weights)
        if args.verbose or step == 1 or step == args.steps or step % 10 == 0:
            current_loss = float(objective(weights))
            log.info("Step %d/%d: MSE=%.8f", step, args.steps, current_loss)

    # Training metrics describe fit; test metrics describe unseen draws.
    train_mse, train_accuracy = evaluate(circuit, features, targets, weights)
    test_mse, test_accuracy = evaluate(circuit, test_features, test_targets, weights)
    log.ok(
        "Training complete: MSE=%.8f, accuracy=%.1f%%",
        train_mse,
        100.0 * train_accuracy,
    )
    log.ok(
        "Held-out test: MSE=%.8f, accuracy=%.1f%%",
        test_mse,
        100.0 * test_accuracy,
    )
    log.info("Initial training MSE=%.8f; final=%.8f", initial_loss, train_mse)
    log.info(
        "Classical majority baseline (label=%+d): train=%.1f%%, test=%.1f%%",
        int(baseline_label),
        100.0 * baseline_train,
        100.0 * baseline_test,
    )
    if args.split == "sampled":
        log.warn_(
            "Only 16 possible parity inputs; train/test overlap is possible. "
            "Use --split disjoint for unique unseen input combinations."
        )
    else:
        log.info(
            "Disjoint accuracy measures unseen combinations, but only %d "
            "test inputs exist in this split.",
            len(test_targets),
        )
    return 0


def main(argv=None):
    """Validate the environment, then execute a dry-run or model training."""
    args = parse_args(argv)
    log = get_logger("PennyLane")
    try:
        import pennylane as qml

        log.info("Python: %s", sys.version.split()[0])
        log.info("PennyLane: %s; NumPy: %s", qml.__version__, np.__version__)
        log.info("Differentiation: %s", args.diff_method)
        if args.split == "disjoint" and not 1 <= args.train_unique < 16:
            raise ValueError("--train-unique must be between 1 and 15")
        if args.dry_run:
            # Avoid QNode execution and optimizer updates in dry-run mode.
            log.ok("Dry-run passed: inputs and dependencies validated")
            return 0
        return train(args, log)
    except (ImportError, RuntimeError, ValueError, FloatingPointError) as exc:
        log.fail("PennyLane classifier failed: %s", exc)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
