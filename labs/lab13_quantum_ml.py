#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab13_quantum_ml.py

Purpose
-------
Lab 13 builds two quantum machine-learning models on small 2-D datasets and
measures them honestly against classical baselines.

  A. QUANTUM KERNEL. Each point x is encoded into a 2-qubit state |phi(x)>;
     the kernel k(x, x') = |<phi(x')|phi(x)>|^2 is then used in kernel ridge
     regression. Two feature maps: a plain product map (RY rotations; classical
     to simulate) and a ZZ map (entangling; the standard "ZZFeatureMap").
  B. VARIATIONAL CLASSIFIER with DATA RE-UPLOADING. Layers of [encode x,
     trainable rotations, CNOT]; the output is <Z> on qubit 0, scaled and
     shifted by two classical numbers. Gradients come from the parameter-shift
     rule (Lab 8) and are checked against finite differences.

Datasets: concentric circles (not linearly separable) and noisy moons.
Baselines: logistic regression (linear) and an RBF-kernel ridge classifier.

What this lab does NOT claim: there is no quantum speedup or advantage here.
These are tiny, classically simulated models on easy data; a classical RBF
kernel does as well. The point is to learn how the pieces work and how to
test them (kernel properties, gradient correctness, ablations, shot noise).

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab13_quantum_ml.py
    python3 labs/lab13_quantum_ml.py --seed 7 --epochs 400

Log files
---------
logs/lab13/lab13_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid command-line usage.
130 User interrupted execution.
===============================================================================
"""

import argparse
import math
import sys
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from applog import get_logger

log = get_logger("lab13")

N_TRAIN = N_TEST = 150
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0)  # ridge strengths tried by CV
CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)  # control = qubit 0 (first factor)
Z0 = np.array([1.0, 1.0, -1.0, -1.0])  # diagonal of Z (x) I


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def make_circles(n, rng, noise=0.08):
    """Two concentric rings: label -1 outer (radius 1), +1 inner (0.45)."""
    # Angle t sweeps the full circle; radius separates the two classes.
    t = rng.uniform(0, 2 * math.pi, n)
    label = rng.integers(0, 2, n)
    radius = np.where(label == 0, 1.0, 0.45) + rng.normal(0, noise, n)
    return np.c_[radius * np.cos(t), radius * np.sin(t)], 2 * label - 1


def make_moons(n, rng, noise=0.2):
    """Two interleaved half circles, with Gaussian noise."""
    # Two half-circles offset against each other give the classic moons.
    t = rng.uniform(0, math.pi, n)
    label = rng.integers(0, 2, n)
    x = np.where(label == 0, np.cos(t), 1 - np.cos(t))
    y = np.where(label == 0, np.sin(t), 0.5 - np.sin(t))
    pts = np.c_[x, y] + rng.normal(0, noise, (n, 2))
    return pts, 2 * label - 1


def to_angles(train, test):
    """Scale features to [0, pi] using TRAIN statistics only (no test leakage)."""
    # Scale with TRAIN statistics only so no test information leaks in.
    lo, hi = train.min(axis=0), train.max(axis=0)
    scale = lambda x: np.clip((x - lo) / (hi - lo), 0, 1) * math.pi  # noqa: E731
    return scale(train), scale(test)


# ----------------------------------------------------------------------------
# Part A: quantum kernels
# ----------------------------------------------------------------------------
def feature_state(angles, kind):
    """Statevector |phi(x)> of a 2-qubit feature map for one data point."""
    # Angle encoding: one qubit per feature, product state, no entanglement.
    a0, a1 = angles
    qc = QuantumCircuit(2)
    if kind == "product":
        qc.ry(a0, 0)  # each feature rotates its own qubit
        qc.ry(a1, 1)
    else:  # "zz": Hadamards, 2 repetitions of
        qc.h([0, 1])  # phases plus a ZZ interaction
        for _ in range(2):
            qc.p(2 * a0, 0)
            qc.p(2 * a1, 1)
            qc.cx(0, 1)
            qc.p(2 * (math.pi - a0) * (math.pi - a1), 1)  # product feature
            qc.cx(0, 1)
    return Statevector.from_instruction(qc).data


def states(points, kind):
    """Stack feature states for many points: an (N, 4) complex array."""
    return np.array([feature_state(p, kind) for p in points])


def kernel(s_a, s_b):
    """Fidelity kernel matrix |<b|a>|^2 between two sets of states."""
    # Fidelity kernel |<a|b>|^2 is what a swap test would estimate on hardware.
    return np.abs(s_a.conj() @ s_b.T) ** 2


def ridge_fit(k, y, lam):
    """Kernel ridge regression weights: (K + lam I)^-1 y."""
    # Solve (K + lam I) a = y; lam keeps the system well conditioned.
    return np.linalg.solve(k + lam * np.eye(len(y)), y)


def cv_lambda(k, y, rng, folds=5):
    """Pick the ridge strength by cross-validation on the TRAINING set only.

    Returns (best lambda, its mean validation accuracy).
    """
    # Shuffle once so every fold sees a mix of both classes.
    idx = rng.permutation(len(y))
    parts = np.array_split(idx, folds)
    best, best_acc = LAMBDAS[0], -1.0
    for lam in LAMBDAS:
        accs = []
        for f in range(folds):
            va = parts[f]
            tr = np.concatenate([parts[g] for g in range(folds) if g != f])
            w = ridge_fit(k[np.ix_(tr, tr)], y[tr], lam)
            accs.append(np.mean(np.sign(k[np.ix_(va, tr)] @ w) == y[va]))
        if np.mean(accs) > best_acc:
            best, best_acc = lam, float(np.mean(accs))
    return best, best_acc


def kernel_accuracy(k_train, k_test, y_train, y_test, rng):
    """CV-tuned kernel ridge classifier -> (cv acc, train acc, test acc, lambda)."""
    lam, cv_acc = cv_lambda(k_train, y_train, rng)
    w = ridge_fit(k_train, y_train, lam)
    return (
        cv_acc,
        float(np.mean(np.sign(k_train @ w) == y_train)),
        float(np.mean(np.sign(k_test @ w) == y_test)),
        lam,
    )


def rbf(a, b, gamma):
    """Classical RBF kernel exp(-gamma |x - x'|^2)."""
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
    return np.exp(-gamma * d)


def logistic_accuracy(x_train, y_train, x_test, y_test):
    """Linear baseline: logistic regression by plain gradient descent."""
    # Plain gradient descent; the classical baseline for the comparison.
    w, b = np.zeros(2), 0.0
    target = (y_train + 1) / 2
    for _ in range(3000):
        p = 1 / (1 + np.exp(-(x_train @ w + b)))
        w -= 0.5 * (x_train.T @ (p - target)) / len(y_train)
        b -= 0.5 * float(np.mean(p - target))
    return float(np.mean(np.sign(x_test @ w + b) == y_test))


# ----------------------------------------------------------------------------
# Part B: variational classifier with data re-uploading (batched NumPy)
# ----------------------------------------------------------------------------
def ry_matrix(theta):
    """Single-qubit RY(theta)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz_matrix(theta):
    """Single-qubit RZ(theta)."""
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])


def ry_batch(theta):
    """RY for a whole vector of angles: an (N, 2, 2) array."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    out = np.zeros((len(theta), 2, 2), dtype=complex)
    out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1] = c, -s, s, c
    return out


def vqc_probs(angles, theta, layers):
    """Output probabilities of the 2-qubit circuit for every data point."""
    # Batch all samples through each layer at once to avoid Python loops.
    n = len(angles)
    psi = np.zeros((n, 4), dtype=complex)
    psi[:, 0] = 1.0
    th = theta.reshape(layers, 4)
    for lay in range(layers):
        # Encode: RY(x0) on qubit 0 and RY(x1) on qubit 1 (a Kronecker product
        # per sample). Re-uploading repeats this every layer.
        enc = np.einsum(
            "nij,nkl->nikjl", ry_batch(angles[:, 0]), ry_batch(angles[:, 1])
        ).reshape(n, 4, 4)
        psi = np.einsum("nij,nj->ni", enc, psi)
        # Trainable single-qubit rotations (RY then RZ on each qubit).
        rot = np.kron(
            rz_matrix(th[lay, 1]) @ ry_matrix(th[lay, 0]),
            rz_matrix(th[lay, 3]) @ ry_matrix(th[lay, 2]),
        )
        psi = psi @ rot.T
        psi = psi @ CNOT.T  # entangle the two qubits
    return np.abs(psi) ** 2


def vqc_expect(angles, theta, layers):
    """<Z> on qubit 0 for every data point (the raw model output)."""
    return vqc_probs(angles, theta, layers) @ Z0


def vqc_loss(angles, y, theta, layers, scale, shift):
    """Mean squared error between scaled output and the +-1 labels."""
    return float(np.mean((scale * vqc_expect(angles, theta, layers) + shift - y) ** 2))


def vqc_gradient(angles, y, theta, layers, scale, shift):
    """Loss gradient: parameter-shift for circuit angles, calculus for a, b."""
    # The shift rule is exact, not an approximation like finite differences.
    f = vqc_expect(angles, theta, layers)
    resid = 2 * (scale * f + shift - y) / len(y)  # dLoss/d(prediction)
    grad = np.zeros_like(theta)
    for j in range(len(theta)):
        up, dn = theta.copy(), theta.copy()
        up[j] += math.pi / 2  # exact shift rule for RY/RZ
        dn[j] -= math.pi / 2
        df = (vqc_expect(angles, up, layers) - vqc_expect(angles, dn, layers)) / 2
        grad[j] = float(np.sum(resid * scale * df))
    return grad, float(np.sum(resid * f)), float(np.sum(resid))


def train_vqc(angles, y, layers, epochs, seed):
    """Adam optimiser on the VQC; returns (theta, scale, shift, loss history)."""
    # Seeded generator so every run starts from the same angles.
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-math.pi, math.pi, 4 * layers)
    scale, shift = 1.0, 0.0
    m = np.zeros(4 * layers + 2)
    v = np.zeros_like(m)
    history = []
    for ep in range(1, epochs + 1):
        g_t, g_s, g_b = vqc_gradient(angles, y, theta, layers, scale, shift)
        g = np.concatenate([g_t, [g_s, g_b]])
        m = 0.9 * m + 0.1 * g  # Adam moments with bias fix
        v = 0.999 * v + 0.001 * g**2
        step = 0.1 * (m / (1 - 0.9**ep)) / (np.sqrt(v / (1 - 0.999**ep)) + 1e-8)
        theta -= step[:-2]
        scale -= step[-2]
        shift -= step[-1]
        history.append(vqc_loss(angles, y, theta, layers, scale, shift))
    return theta, scale, shift, history


def vqc_accuracy(angles, y, theta, layers, scale, shift):
    """Classification accuracy of sign(scale * <Z> + shift)."""
    pred = np.sign(scale * vqc_expect(angles, theta, layers) + shift)
    return float(np.mean(pred == y))


def qiskit_expectation(angle_pair, theta, layers):
    """The same circuit built in Qiskit; returns <Z> on qubit 0 (reference)."""
    # Cross-check the NumPy model against a Qiskit circuit.
    qc = QuantumCircuit(2)
    th = theta.reshape(layers, 4)
    for lay in range(layers):
        qc.ry(float(angle_pair[0]), 0)
        qc.ry(float(angle_pair[1]), 1)
        qc.ry(th[lay, 0], 0)
        qc.rz(th[lay, 1], 0)
        qc.ry(th[lay, 2], 1)
        qc.rz(th[lay, 3], 1)
        qc.cx(0, 1)
    obs = SparsePauliOp("IZ")  # Qiskit: rightmost char = qubit 0
    return float(np.real(Statevector.from_instruction(qc).expectation_value(obs)))


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def tune_family(name, builders, y_tr, y_te, rng):
    """Try each (label, k_train, k_test) candidate; keep the best by CV accuracy.

    Every kernel family gets the same treatment (its own bandwidth tuned on
    the training set only), so the comparison is not rigged either way.
    """
    # Selection uses cross-validation accuracy only; the test set stays unseen.
    best = None
    for label, k_tr, k_te in builders:
        cv_acc, tr, te, lam = kernel_accuracy(k_tr, k_te, y_tr, y_te, rng)
        if best is None or cv_acc > best[0]:
            best = (cv_acc, te, label, lam, tr)
    log.info(
        "%s: best bandwidth %s (cv %.3f, train %.3f) -> TEST %.3f",
        name,
        best[2],
        best[0],
        best[4],
        best[1],
    )
    return best[1]


def check_kernels(name, data, rng, r):
    """Part A on one dataset: kernel properties and accuracy table."""
    a_tr, a_te, x_tr, x_te, y_tr, y_te = data
    log.info(
        "=== Quantum kernels on %s (%d train / %d test) ===", name, len(y_tr), len(y_te)
    )
    table = {}
    for kind in ("product", "zz"):
        candidates = []
        for scale in (0.25, 0.5, 1.0, 2.0):  # feature-map bandwidth
            s_tr, s_te = states(a_tr * scale, kind), states(a_te * scale, kind)
            k_tr, k_te = kernel(s_tr, s_tr), kernel(s_te, s_tr)
            eig = np.linalg.eigvalsh(k_tr)
            r.append(
                check(
                    np.allclose(k_tr, k_tr.T)
                    and np.allclose(np.diag(k_tr), 1)
                    and k_tr.min() > -1e-12
                    and k_tr.max() < 1 + 1e-12
                    and eig.min() > -1e-9,
                    f"{name}/{kind}/scale {scale}: symmetric, diag 1, in [0, 1], "
                    f"PSD (min eig {eig.min():.1e})",
                )
            )
            candidates.append((f"scale {scale}", k_tr, k_te))
        table[kind] = tune_family(f"{name}/{kind}", candidates, y_tr, y_te, rng)
    candidates = [
        (f"gamma {g}", rbf(x_tr, x_tr, g), rbf(x_te, x_tr, g))
        for g in (0.5, 2.0, 8.0, 16.0)
    ]
    table["rbf"] = tune_family(f"{name}/rbf", candidates, y_tr, y_te, rng)
    table["linear"] = logistic_accuracy(x_tr, y_tr, x_te, y_te)
    log.info("%s linear baseline test %.3f", name, table["linear"])
    return table


def check_vqc(name, data, epochs, seed, r):
    """Part B on one dataset: gradients, training, ablation, Qiskit, shots."""
    a_tr, a_te, _, _, y_tr, y_te = data
    log.info("=== Variational classifier on %s ===", name)
    # Gradient check first: a wrong gradient trains silently and badly.
    layers = 2
    rng = np.random.default_rng(seed)
    th = rng.uniform(-math.pi, math.pi, 4 * layers)
    g, _, _ = vqc_gradient(a_tr, y_tr, th, layers, 1.3, 0.1)
    eps = 1e-6
    fd = np.array(
        [
            (
                vqc_loss(a_tr, y_tr, th + eps * e, layers, 1.3, 0.1)
                - vqc_loss(a_tr, y_tr, th - eps * e, layers, 1.3, 0.1)
            )
            / (2 * eps)
            for e in np.eye(len(th))
        ]
    )
    r.append(
        check(
            np.allclose(g, fd, atol=1e-6),
            f"{name}: parameter-shift gradient equals finite "
            f"differences (max diff {np.max(np.abs(g - fd)):.1e})",
        )
    )

    accs, models = {}, {}
    for lay in (1, 2, 3):
        t0 = time.time()
        theta, a, b, hist = train_vqc(a_tr, y_tr, lay, epochs, seed)
        models[lay] = (theta, a, b)
        accs[lay] = vqc_accuracy(a_te, y_te, theta, lay, a, b)
        log.info(
            "%s: %d layer(s), %d params: loss %.3f -> %.3f, test acc %.3f" " (%.1f s)",
            name,
            lay,
            4 * lay,
            hist[0],
            hist[-1],
            accs[lay],
            time.time() - t0,
        )
        r.append(
            check(hist[-1] < hist[0], f"{name}: {lay}-layer training lowers the loss")
        )
    # Ablation: with ONE layer the data enter once, so the model is a low-degree
    # trigonometric function of x. Re-uploading raises the degree.
    r.append(
        check(
            accs[3] >= accs[1],
            f"{name}: re-uploading helps or ties "
            f"(1 layer {accs[1]:.3f} -> 3 layers {accs[3]:.3f})",
        )
    )

    theta, a, b = models[3]
    # Cross-check the fast NumPy model against Qiskit on a few test points.
    ref = [qiskit_expectation(p, theta, 3) for p in a_te[:5]]
    mine = vqc_expect(a_te[:5], theta, 3)
    r.append(
        check(
            np.allclose(ref, mine, atol=1e-10),
            f"{name}: NumPy circuit equals Qiskit on 5 test points",
        )
    )

    # Finite shots: sample the measured bit on qubit 0 instead of reading the
    # exact expectation, as hardware would.
    rng = np.random.default_rng(seed + 1)
    shots = 1000
    p_plus = (1 + vqc_expect(a_te, theta, 3)) / 2  # P(qubit 0 reads 0)
    est = 2 * rng.binomial(shots, np.clip(p_plus, 0, 1)) / shots - 1
    shot_acc = float(np.mean(np.sign(a * est + b) == y_te))
    log.info(
        "%s: exact test acc %.3f, with %d shots per point %.3f",
        name,
        accs[3],
        shots,
        shot_acc,
    )
    r.append(
        check(
            abs(shot_acc - accs[3]) < 0.05,
            f"{name}: {shots}-shot readout changes accuracy by < 0.05",
        )
    )
    return accs


def main(argv=None):
    """Run every Lab 13 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 13: quantum machine learning")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--epochs", type=int, default=300, help="Adam steps for each variational model"
    )
    args = ap.parse_args(argv)
    if args.epochs < 50:
        ap.error(f"--epochs must be >= 50. Got {args.epochs}.")
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    # results[dataset][model] = test accuracy, used by the final comparison.
    results = {}
    for name, maker in (("circles", make_circles), ("moons", make_moons)):
        rng = np.random.default_rng(args.seed)
        x_tr, y_tr = maker(N_TRAIN, rng)
        x_te, y_te = maker(N_TEST, rng)
        a_tr, a_te = to_angles(x_tr, x_te)
        data = (a_tr, a_te, x_tr, x_te, y_tr, y_te)
        results[name] = check_kernels(name, data, rng, r)
        results[name]["vqc"] = check_vqc(name, data, args.epochs, args.seed, r)

    log.info("=== Check: honest comparison ===")
    c, m = results["circles"], results["moons"]
    # Circles are not linearly separable, so a linear model stays far behind.
    r.append(
        check(
            c["linear"] < min(c["zz"], c["product"]) - 0.2,
            f"circles: linear {c['linear']:.2f} trails the quantum "
            f"kernels ({c['product']:.2f}, {c['zz']:.2f}) by > 0.2",
        )
    )
    # The honest claim: nothing quantum BEATS the tuned classical kernel. The
    # margin 0.02 is about two standard errors of an accuracy on 150 points.
    best_m = max(m["product"], m["zz"], m["vqc"][3])
    best_c = max(c["product"], c["zz"], c["vqc"][3])
    r.append(
        check(
            m["rbf"] >= max(m["product"], m["zz"], m["vqc"][3]) - 0.02
            and c["rbf"] >= max(c["product"], c["zz"], c["vqc"][3]) - 0.02,
            f"no quantum advantage: best quantum model vs classical RBF is "
            f"{best_m:.3f} vs {m['rbf']:.3f} (moons), {best_c:.3f} vs "
            f"{c['rbf']:.3f} (circles)",
        )
    )
    # Which feature map wins depends on the random sample, so report it
    # instead of asserting it: the choice of map is a modelling decision.
    log.info(
        "moons: ZZ map %.3f vs product map %.3f (feature-map choice "
        "moves accuracy by %.3f)",
        m["zz"],
        m["product"],
        abs(m["zz"] - m["product"]),
    )
    r.append(
        check(
            c["vqc"][3] > 0.9 and m["vqc"][3] > 0.8,
            f"variational model: circles {c['vqc'][3]:.2f}, moons "
            f"{m['vqc'][3]:.2f} with 12 trainable angles",
        )
    )
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
