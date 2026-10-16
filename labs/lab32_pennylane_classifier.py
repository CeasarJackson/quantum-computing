#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab32_pennylane_classifier.py

Purpose
-------
Labs 1-31 used Qiskit and our own simulators. labs/PennyLane.py trains a
four-qubit variational classifier with PennyLane (autograd, parameter-shift
gradients), and labs/pennylane_repeated_eval.py repeats it over seeds and
compares it with classical baselines. Lab 32 puts both scripts under the same
discipline as the rest of the course: an INDEPENDENT test of every claim they
rest on, so that a number they print can be trusted.

The lab verifies:
  1. Data: parity labels, the exhaustive 16-input partition, disjointness,
     reproducibility, and rejection of invalid split sizes.
  2. Baselines: the majority rule uses only training labels (ties give +1);
     a linear model cannot learn parity (accuracy at most 75% on all inputs).
  3. Circuit: PennyLane's <Z0> equals a from-scratch NumPy state-vector
     simulation of the same RX / RY / RZ / CNOT circuit (1e-10).
  4. Gradient: the parameter-shift gradient equals central finite differences
     of the NumPy model, and equals PennyLane's backpropagation gradient.
  5. Training: the loss falls, the same seeds reproduce identical output, and
     a different initialisation gives a different result.
  6. The repeated-evaluation helpers: seed parsing, the ridge baseline, atomic
     CSV export, and one real (short) classifier run through its subprocess.
  7. Statistics: with only 4 unseen inputs per split, 3 of the 5 possible test
     accuracies cannot be told from a coin flip, and a coin flip itself scatters
     by 25 percentage points. Seed-to-seed swings of the size seen in the
     repeated evaluation are therefore expected noise, not a finding.
  8. Stronger classical comparators: a 16-term polynomial that contains the
     parity function and a 1-nearest-neighbour model both fail to learn parity
     from 12 of the 16 inputs, so the quantum model's success is not simply
     "any nonlinear model would do".

Prerequisites
-------------
PennyLane 0.45.1 in the project's .venv (see requirements.txt).

Security
--------
Handles no secrets and makes no network calls. Writes only to a temporary
directory that it removes.

Usage
-----
    python3 labs/lab32_pennylane_classifier.py
    python3 labs/lab32_pennylane_classifier.py --steps 20 --seed 5
    python3 labs/lab32_pennylane_classifier.py --quick

Log files
---------
logs/lab32/lab32_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid command-line usage.
130 User interrupted execution.
===============================================================================
"""

import argparse
import csv
import importlib.util
import itertools
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

from applog import get_logger
from PennyLane import (
    N_LAYERS,
    N_WIRES,
    loss_function,
    majority_baseline,
    make_circuit,
    make_dataset,
    make_disjoint_dataset,
    parse_args,
    train,
)
from pennylane_repeated_eval import (
    export_csv,
    linear_baseline,
    nearest_neighbour_baseline,
    polynomial_baseline,
    run_quantum,
    seed_list,
)

log = get_logger("lab32")

CLASSIFIER = Path(__file__).resolve().parent / "PennyLane.py"


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


class Recorder:
    """Stand-in logger that stores formatted messages instead of printing.

    PennyLane.train() reports everything through a logger, so capturing the
    messages is how its results are read back without changing the script.
    """

    def __init__(self):
        """Start with no messages."""
        self.lines = []

    def _store(self, fmt, *args):
        """Format a message the way logging would and keep it."""
        self.lines.append(fmt % args if args else fmt)

    info = ok = warn_ = fail = _store


# ----------------------------------------------------------------------------
# From-scratch NumPy reference for the classifier circuit
# ----------------------------------------------------------------------------
def one_qubit(state, gate, wire):
    """Apply a 2x2 gate to one axis of a (2,)*n state tensor."""
    # Contract the gate with one qubit axis, then put the result axis back.
    out = np.tensordot(gate, state, axes=([1], [wire]))
    return np.moveaxis(out, 0, wire)


def rx(theta):
    """RX(theta) = exp(-i theta X / 2)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def ry(theta):
    """RY(theta) = exp(-i theta Y / 2)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta):
    """RZ(theta) = exp(-i theta Z / 2)."""
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


def cnot(state, control, target):
    """CNOT on a state tensor: flip the target axis where control = 1."""
    out = state.copy()
    index = [slice(None)] * state.ndim
    index[control] = 1
    # Removing the control axis shifts later axes down by one.
    axis = target - (target > control)
    out[tuple(index)] = np.flip(state[tuple(index)], axis=axis)
    return out


def reference_expval(features, weights):
    """<Z_0> of the classifier circuit, simulated without PennyLane."""
    # The state is a rank-4 tensor so each gate acts on one axis, as in textbooks.
    # Start in |0000>; RX then encodes the four input angles.
    state = np.zeros((2,) * N_WIRES, dtype=complex)
    state[(0,) * N_WIRES] = 1.0
    for wire in range(N_WIRES):
        state = one_qubit(state, rx(features[wire]), wire)
    for layer in range(N_LAYERS):
        for wire in range(N_WIRES):
            state = one_qubit(state, ry(weights[layer, wire, 0]), wire)
            state = one_qubit(state, rz(weights[layer, wire, 1]), wire)
        # Open chain of CNOTs, then the CNOT that closes the ring.
        for wire in range(N_WIRES - 1):
            state = cnot(state, wire, wire + 1)
        state = cnot(state, N_WIRES - 1, 0)
    prob = np.abs(state) ** 2
    sign = np.array([1.0, -1.0]).reshape((2,) + (1,) * (N_WIRES - 1))
    return float(np.sum(prob * sign))


def reference_loss(features, targets, weights):
    """Mean squared error of the NumPy model, same form as the classifier."""
    preds = np.array([reference_expval(row, weights) for row in features])
    return float(np.mean((preds - targets) ** 2))


def wilson(successes, trials, z=1.96):
    """Wilson score interval for a binomial proportion."""
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return centre - half, centre + half


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_data(r, ctx):
    """Check 1: datasets, labels and splits."""
    # Everything downstream trusts these labels and splits.
    log.info("=== Check 1: data ===")
    # A large sample makes a missing label or angle value very unlikely to hide.
    # Angles are mapped back to bits so labels are recomputed independently.
    angles, labels = make_dataset(200, ctx["seed"])
    bits = np.rint(angles / np.pi).astype(int)
    r.append(
        check(
            np.allclose(labels, 1 - 2 * (bits.sum(axis=1) % 2))
            and set(np.unique(angles)) <= {0.0, np.pi},
            "sampled data: angles are 0 or pi and labels are the parity sign",
        )
    )
    ok_all, ok_labels = True, True
    for k in range(1, 16):
        for seed in (ctx["seed"], ctx["seed"] + 1):
            tr_x, tr_y, te_x, te_y = make_disjoint_dataset(k, seed)
            keys = [tuple(np.rint(row / np.pi).astype(int)) for row in tr_x] + [
                tuple(np.rint(row / np.pi).astype(int)) for row in te_x
            ]
            # Exactly the 16 distinct bit patterns, each used once.
            ok_all &= len(keys) == 16 and set(keys) == set(
                itertools.product((0, 1), repeat=4)
            )
            ok_all &= len(tr_y) == k and len(te_y) == 16 - k
            for x, y in ((tr_x, tr_y), (te_x, te_y)):
                ok_labels &= np.allclose(
                    y, 1 - 2 * (np.rint(x / np.pi).sum(axis=1) % 2)
                )
    r.append(check(ok_all, "disjoint split: all 16 inputs used once, any train size"))
    r.append(check(ok_labels, "and every label equals the parity sign of its bits"))
    a1 = make_disjoint_dataset(12, 5)[0]
    a2 = make_disjoint_dataset(12, 5)[0]
    a3 = make_disjoint_dataset(12, 6)[0]
    r.append(
        check(
            np.array_equal(a1, a2) and not np.array_equal(a1, a3),
            "the split is reproducible per seed and changes with the seed",
        )
    )
    bad = 0
    for k in (0, 16, -3):
        try:
            make_disjoint_dataset(k, 1)
        except ValueError:
            bad += 1
    r.append(check(bad == 3, "train sizes 0, 16 and -3 are rejected"))


def check_baselines(r, ctx):
    """Check 2: the classical comparators."""
    # Baselines set the bar the quantum model has to beat on identical splits.
    log.info("=== Check 2: baselines ===")
    # Two +1 labels against one -1 label: the constant classifier must say +1.
    # Test labels are chosen to differ so leakage would change the answer.
    y_train = np.array([1.0, 1.0, -1.0])
    label, tr_acc, te_acc = majority_baseline(y_train, np.array([1.0, -1.0]))
    flipped = majority_baseline(y_train, np.array([-1.0, -1.0]))[0]
    tie = majority_baseline(np.array([1.0, -1.0]), np.array([1.0]))[0]
    r.append(
        check(
            label == 1.0 and flipped == 1.0 and abs(tr_acc - 2 / 3) < 1e-12,
            "majority label comes from training labels only",
        )
    )
    r.append(check(tie == 1.0, "a tie predicts +1"))
    all_x, all_y, _, _ = make_disjoint_dataset(15, 1)
    lin = linear_baseline(all_x, all_y, all_x, all_y, 1e-6)[0]
    log.info("ridge-linear accuracy on 15 inputs it was trained on: %.1f%%", lin)
    r.append(
        check(
            lin <= 75.0, "a linear model cannot learn parity (training accuracy <= 75%)"
        )
    )


def check_circuit(r, ctx):
    """Check 3: PennyLane circuit against the NumPy reference."""
    import pennylane as qml

    # A wrong gate order would still train, so the circuit is verified first.
    log.info("=== Check 3: circuit against a from-scratch simulation ===")
    # Use the same QNode that the classifier trains, not a copy of it.
    # Agreement with an independent NumPy simulation shows that the gate order,
    # the CNOT ring and the readout wire are what the documentation says.
    circuit = make_circuit(qml, "parameter-shift")
    # Local generator: no global NumPy state is touched by this check.
    rng = np.random.default_rng(ctx["seed"])
    worst = 0.0
    for _ in range(20):
        features = rng.choice([0.0, np.pi], size=N_WIRES)
        if rng.random() < 0.5:
            # Also test generic angles, not only the 0 / pi data encoding.
            features = rng.uniform(0, 2 * np.pi, N_WIRES)
        weights = rng.normal(0, 1.0, (N_LAYERS, N_WIRES, 2))
        got = float(circuit(features, weights))
        worst = max(worst, abs(got - reference_expval(features, weights)))
    r.append(
        check(worst < 1e-10, f"20 random circuits agree (max difference {worst:.1e})")
    )
    ctx["qml"] = qml


def check_gradient(r, ctx):
    """Check 4: parameter-shift gradient against finite differences."""
    qml = ctx["qml"]
    from pennylane import numpy as pnp

    # Gradients decide whether training can work at all, so check them three ways.
    log.info("=== Check 4: gradient ===")
    rng = np.random.default_rng(ctx["seed"] + 1)
    features, targets, _, _ = make_disjoint_dataset(6, ctx["seed"])
    weights0 = rng.normal(0, 0.7, (N_LAYERS, N_WIRES, 2))
    shift = make_circuit(qml, "parameter-shift")
    best = make_circuit(qml, "best")

    def grad_of(circuit):
        """Gradient of the classifier loss through the given QNode."""

        def objective(w):
            """Loss as a function of the weights only."""
            return loss_function(circuit, features, targets, w, pnp)

        return np.asarray(qml.grad(objective)(pnp.array(weights0, requires_grad=True)))

    g_shift, g_best = grad_of(shift), grad_of(best)
    fd = np.zeros_like(weights0)
    eps = 1e-6
    for idx in np.ndindex(*weights0.shape):
        up, dn = weights0.copy(), weights0.copy()
        up[idx] += eps
        dn[idx] -= eps
        fd[idx] = (
            reference_loss(features, targets, up)
            - reference_loss(features, targets, dn)
        ) / (2 * eps)
    err_fd = float(np.abs(g_shift - fd).max())
    err_best = float(np.abs(g_shift - g_best).max())
    log.info(
        "largest gradient entry %.3f; shift-vs-FD %.1e; shift-vs-best %.1e",
        np.abs(g_shift).max(),
        err_fd,
        err_best,
    )
    r.append(check(err_fd < 1e-6, "parameter-shift equals central finite differences"))
    r.append(check(err_best < 1e-8, "and equals the backpropagation gradient"))


def run_train(ctx, init_seed):
    """Run PennyLane.train() on a small disjoint problem; return its messages."""
    argv = [
        "--split",
        "disjoint",
        "--train-unique",
        "12",
        "--steps",
        str(ctx["steps"]),
        "--split-seed",
        "42",
        "--init-seed",
        str(init_seed),
    ]
    # train() logs through the Recorder, so no files or subprocesses are needed.
    # The message text is the same text the repeated-evaluation script parses.
    rec = Recorder()
    code = train(parse_args(argv), rec)
    return code, rec.lines


def parse_mse(lines, prefix):
    """First MSE value on a recorded line that starts with `prefix`."""
    for line in lines:
        if line.startswith(prefix):
            # The initial line reads "Initial MSE: x"; the others "MSE=x,".
            tail = line.split("MSE: ")[1] if "MSE: " in line else line.split("MSE=")[1]
            return float(tail.split(",")[0].split(";")[0])
    raise ValueError(f"no line starting with {prefix!r}")


def check_training(r, ctx):
    """Check 5: loss falls, runs are reproducible, seeds matter."""
    # Training is the slowest check, so --quick shortens it to six steps.
    log.info("=== Check 5: training (%d steps) ===", ctx["steps"])
    code1, first = run_train(ctx, 43)
    code2, second = run_train(ctx, 43)
    _, other = run_train(ctx, 44)
    initial = parse_mse(first, "Initial MSE")
    final = parse_mse(first, "Training complete")
    log.info("training MSE %.5f -> %.5f", initial, final)
    r.append(check(code1 == 0 and code2 == 0, "train() returns 0"))
    r.append(check(final < initial, "the training loss falls"))
    r.append(check(first == second, "identical seeds give identical output"))
    r.append(
        check(
            parse_mse(other, "Training complete") != final,
            "a different initialisation seed gives a different result",
        )
    )


def check_helpers(r, ctx):
    """Check 6: the repeated-evaluation helpers."""
    # These helpers are the glue between the classifier and its CSV report.
    log.info("=== Check 6: repeated-evaluation helpers ===")
    r.append(
        check(
            seed_list("42, 43,44") == [42, 43, 44],
            "seed_list parses a list with spaces",
        )
    )
    bad = 0
    for text in ("1,1", "a,b", "1,,2"):
        try:
            seed_list(text)
        except argparse.ArgumentTypeError:
            bad += 1
    r.append(check(bad == 3, "duplicate, non-numeric and empty seeds are rejected"))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.csv"
        rows = [{"a": 1, "b": 2.5}, {"a": 3, "b": 4.5}]
        export_csv(path, ["a", "b"], rows)
        with path.open(newline="") as stream:
            back = list(csv.DictReader(stream))
        r.append(
            check(
                [dict(x) for x in back]
                == [{"a": "1", "b": "2.5"}, {"a": "3", "b": "4.5"}]
                and not list(Path(tmp).glob("*.tmp")),
                "CSV export round-trips and leaves no temporary file",
            )
        )
    args = argparse.Namespace(train_unique=12, steps=2, learning_rate=0.1)
    tr_mse, tr_acc, te_mse, te_acc = run_quantum(CLASSIFIER, args, 42, 43)
    log.info(
        "2-step subprocess run: train MSE %.3f acc %.1f%% | test MSE %.3f acc %.1f%%",
        tr_mse,
        tr_acc,
        te_mse,
        te_acc,
    )
    r.append(
        check(
            all(np.isfinite([tr_mse, tr_acc, te_mse, te_acc]))
            and 0 <= tr_acc <= 100
            and te_acc in (0.0, 25.0, 50.0, 75.0, 100.0),
            "the subprocess run returns parsed metrics; 4 test inputs give "
            "accuracies in steps of 25%",
        )
    )


def check_statistics(r, ctx):
    """Check 7: how little a 4-input test can say."""
    # Pure arithmetic: no PennyLane call is needed for this check.
    log.info("=== Check 7: statistics of a 4-input test ===")
    # Four held-out inputs is the size of the default 12/4 partition.
    # Wilson intervals behave better than the normal approximation at n = 4.
    n = 4
    inside = []
    for k in range(n + 1):
        lo, hi = wilson(k, n)
        inside.append(lo <= 0.5 <= hi)
        log.info("%d of %d correct: 95%% interval %.2f - %.2f", k, n, lo, hi)
    r.append(
        check(
            sum(inside) == 3,
            "3 of the 5 possible outcomes cannot be told from a coin flip",
        )
    )
    # Exact distribution of the number correct for a model that guesses.
    probs = [math.comb(n, k) / 2**n for k in range(n + 1)]
    mean = sum(p * k / n for k, p in enumerate(probs))
    sd = math.sqrt(sum(p * (k / n - mean) ** 2 for k, p in enumerate(probs)))
    log.info(
        "guessing: mean accuracy %.0f%%, standard deviation %.0f points",
        100 * mean,
        100 * sd,
    )
    r.append(
        check(
            abs(mean - 0.5) < 1e-12 and abs(sd - 0.25) < 1e-12,
            "a coin-flip model scatters by 25 percentage points on 4 inputs",
        )
    )
    # Five random splits: how often do guessing models look as spread as the
    # repeated evaluation (sample SD over 5 seeds of about 41 points)?
    rng = np.random.default_rng(ctx["seed"])
    spreads = []
    for _ in range(2000):
        accs = rng.binomial(n, 0.5, 5) / n * 100
        spreads.append(np.std(accs, ddof=1))
    frac = float(np.mean(np.array(spreads) >= 25))
    log.info(
        "share of 5-seed guessing runs with sample SD >= 25 points: %.0f%%", 100 * frac
    )
    r.append(
        check(
            frac > 0.2,
            "large seed-to-seed swings are common even for pure guessing",
        )
    )


def check_classical(r, ctx):
    """Check 8: the stronger classical comparators."""
    # A linear baseline cannot represent parity, so a skeptic asks whether ANY
    # classical model learns it from 12 of the 16 inputs. Two more are tried.
    log.info("=== Check 8: nonlinear classical baselines ===")
    wins = {"linear": [], "polynomial": [], "1-NN": []}
    fits = []
    for split_seed in range(42, 47):
        trx, try_, tex, tey = make_disjoint_dataset(12, split_seed)
        wins["linear"].append(linear_baseline(trx, try_, tex, tey, 1.0)[1])
        poly = polynomial_baseline(trx, try_, tex, tey, 1.0)
        wins["polynomial"].append(poly[1])
        fits.append(poly[0])
        wins["1-NN"].append(nearest_neighbour_baseline(trx, try_, tex, tey)[1])
    for name, vals in wins.items():
        log.info(
            "%-10s test accuracy per split %s, mean %.0f%%", name, vals, np.mean(vals)
        )
    r.append(
        check(
            min(fits) == 100.0,
            "the 16-term polynomial fits the 12 training inputs exactly",
        )
    )
    r.append(
        check(
            all(np.mean(v) < 60 for v in wins.values()),
            "yet no classical comparator learns parity from 12 of 16 inputs",
        )
    )
    # Leave-one-out: all four direct neighbours are in the training set, and
    # parity gives every one of them the opposite label.
    wrong = 0
    for split_seed in range(10):
        trx, try_, tex, tey = make_disjoint_dataset(15, split_seed)
        wrong += nearest_neighbour_baseline(trx, try_, tex, tey)[1] == 0.0
    r.append(check(wrong == 10, "1-NN is wrong on all 10 leave-one-out inputs tried"))


def main(argv=None):
    """Run every Lab 32 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 32: PennyLane classifier tests")
    ap.add_argument("--steps", type=int, default=12, help="training steps in check 5")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true", help="smaller, faster run")
    args = ap.parse_args(argv)
    if not 2 <= args.steps <= 200:
        ap.error(f"--steps must be 2..200. Got {args.steps}.")
    # Quick mode only shortens the training in check 5.
    ctx = {"seed": args.seed, "steps": 6 if args.quick else args.steps}
    # find_spec avoids importing the package just to prove it is present.
    if importlib.util.find_spec("pennylane") is None:
        log.fail("PennyLane is not installed in this environment")
        return 1
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_data(r, ctx)
    check_baselines(r, ctx)
    check_circuit(r, ctx)
    check_gradient(r, ctx)
    check_training(r, ctx)
    check_helpers(r, ctx)
    check_statistics(r, ctx)
    check_classical(r, ctx)
    bad = r.count(False)
    if bad:
        log.fail("%d of %d checks failed", bad, len(r))
        return 1
    log.ok("All %d checks passed", len(r))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
