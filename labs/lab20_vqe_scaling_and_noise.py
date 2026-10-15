#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab20_vqe_scaling_and_noise.py

Purpose
-------
Lab 16 prepared the Heisenberg ground state with a Hamiltonian Variational
Ansatz (HVA): singlet pairs followed by layers of exp(-i a H_odd) exp(-i b
H_even). It worked perfectly at 4 and 6 spins but stalled at 8. Lab 20 asks the
two questions a real experiment must answer:

  A. SCALING. How many layers does an n-spin chain need, and does a better
     optimiser (WARM START: begin each depth from the previous solution plus a
     zero-angle layer) fix the stall seen at n = 8?
  B. NOISE. Each layer costs CNOTs, and every CNOT adds error. At what depth
     does adding layers stop helping on a noisy machine? The noise strengths
     include 0.37% per CNOT (the effective strength measured on ibm_fez in
     Lab 17) and the 1% assumed in Lab 14.

The lab verifies:
  1. Warm-started energy error NEVER increases with depth (a zero-angle layer
     is the identity), and it falls far enough to be useful.
  2. The number of layers needed grows with chain length.
  3. The best state has overlap with the exact ground state obeying the
     rigorous bound  1 - error/gap, and any angles keep S^2 = 0, S_z = 0.
  4. The Qiskit circuit (noiseless density matrix) reproduces the NumPy energy.
  5. With noise, a finite number of layers is best, and noisier hardware wants
     a SHALLOWER circuit. No noisy energy can fall below the exact E0.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab20_vqe_scaling_and_noise.py
    python3 labs/lab20_vqe_scaling_and_noise.py --qubits 8 --layers 8
    python3 labs/lab20_vqe_scaling_and_noise.py --quick

Log files
---------
logs/lab20/lab20_<YYYYmmdd_HHMMSS>.log

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
from functools import reduce

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from applog import get_logger

log = get_logger("lab20")

J = 1.0
I2 = np.eye(2, dtype=complex)
PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
BASIS = ["cx", "rz", "sx", "x"]
NOISE_LEVELS = (0.0, 0.001, 0.0037, 0.01)  # depolarizing strength per CNOT


# ----------------------------------------------------------------------------
# Model and circuits (same construction as Lab 16)
# ----------------------------------------------------------------------------
def site(n, p, q):
    """Dense n-qubit operator with Pauli `p` on qubit q (qubit 0 rightmost)."""
    ops = [PAULI[p] if k == q else I2 for k in reversed(range(n))]
    return reduce(np.kron, ops)


def bond(n, q):
    """Heisenberg term on bond (q, q+1): XX + YY + ZZ (times J)."""
    return J * sum(site(n, p, q) @ site(n, p, q + 1) for p in "XYZ")


def groups(n):
    """Dense (H_even, H_odd): bonds starting on even / odd qubits.

    Bonds inside one group touch disjoint qubits, so they commute and the
    group exponential factorises exactly. Even and odd groups do not commute.
    """
    even = sum(bond(n, q) for q in range(0, n - 1, 2))
    odd = sum(bond(n, q) for q in range(1, n - 1, 2)) if n > 2 else 0 * even
    return even, odd


def total_sz(n):
    """Total S_z = (1/2) sum Z_i."""
    return 0.5 * sum(site(n, "Z", q) for q in range(n))


def total_s2(n):
    """Total S^2 = (1/4) (3n + 2 sum_{i<j} sigma_i . sigma_j)."""
    pair = sum(
        site(n, p, i) @ site(n, p, j)
        for i in range(n)
        for j in range(i + 1, n)
        for p in "XYZ"
    )
    return 0.25 * (3 * n * np.eye(2**n) + 2 * pair)


def heis_sparse(n):
    """The same Hamiltonian as a SparsePauliOp (independent construction)."""
    terms = [(p * 2, [q, q + 1], J) for q in range(n - 1) for p in "XYZ"]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def expval(op, psi):
    """<psi|op|psi> as a real number."""
    return float((psi.conj() @ op @ psi).real)


def bond_gate(qc, a, b, angle):
    """Apply exp(-i angle (XX+YY+ZZ)) on qubits a,b.

    RXX(theta) = exp(-i theta/2 XX), so theta = 2*angle for each Pauli. The
    three rotations commute, so their order is irrelevant.
    """
    qc.rxx(2 * angle, a, b)
    qc.ryy(2 * angle, a, b)
    qc.rzz(2 * angle, a, b)


def layer(qc, n, parity, angle):
    """Apply exp(-i angle H_group) for the even (0) or odd (1) bond group."""
    for q in range(parity, n - 1, 2):
        bond_gate(qc, q, q + 1, J * angle)


def singlets(n):
    """Circuit preparing singlets (|01>-|10>)/sqrt2 on pairs (0,1),(2,3),..."""
    qc = QuantumCircuit(n)
    for a in range(0, n, 2):
        # H, CX makes the Bell state (|00>+|11>)/sqrt2; Z flips the sign of
        # |11> and X on the target swaps 0<->1, giving (|01> - |10>)/sqrt2.
        qc.h(a)
        qc.cx(a, a + 1)
        qc.z(a)
        qc.x(a + 1)
    return qc


def hva_circuit(n, params):
    """Singlet start, then layers [exp(-i a H_odd) exp(-i b H_even)]."""
    qc = singlets(n)
    for k in range(len(params) // 2):
        layer(qc, n, 1, params[2 * k])
        layer(qc, n, 0, params[2 * k + 1])
    return qc


def hva_energy(params, n, h, even_eig, odd_eig, psi0):
    """Energy of the HVA state, using precomputed eigendecompositions."""
    psi = psi0
    for k in range(len(params) // 2):
        for (d, v), ang in ((odd_eig, params[2 * k]), (even_eig, params[2 * k + 1])):
            psi = v @ (np.exp(-1j * ang * d) * (v.conj().T @ psi))
    return expval(h, psi)


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


# ----------------------------------------------------------------------------
# Optimisation
# ----------------------------------------------------------------------------
def build(n):
    """Return everything the optimiser needs for an n-spin chain."""
    # Full Hamiltonian = even-bond group + odd-bond group (Lab 16).
    even, odd = groups(n)
    h = even + odd
    # Eigendecompositions of the two bond groups let us apply exp(-i a H)
    # to a vector with two matrix-vector products instead of a matrix exp.
    eigs_even, eigs_odd = np.linalg.eigh(even), np.linalg.eigh(odd)
    psi0 = Statevector.from_instruction(singlets(n)).data
    # Exact ground state and gap, for error bars and the overlap bound.
    w, v = np.linalg.eigh(h)
    return {
        "n": n,
        "h": h,
        "even": eigs_even,
        "odd": eigs_odd,
        "psi0": psi0,
        "e0": w[0],
        "gap": w[1] - w[0],
        "gs": v[:, 0],
    }


def energy_of(params, m):
    """Energy of the HVA state for the chain description m."""
    return hva_energy(params, m["n"], m["h"], m["even"], m["odd"], m["psi0"])


def optimise_depths(m, max_layers, rng, restarts=2):
    """Best energy for p = 1..max_layers, warm-started from the previous depth.

    Warm start: the p-layer search begins at the (p-1)-layer optimum with one
    extra layer whose two angles are 0. A zero-angle layer is the identity, so
    the new search starts at exactly the old energy and cannot end higher.
    A few random restarts are also tried, in case a different basin is better.
    """
    # out[p-1] = (energy error, angles) at depth p; prev = last best angles.
    out, prev = [], np.zeros(0)
    for p in range(1, max_layers + 1):
        starts = [np.concatenate([prev, np.zeros(2)])]
        starts += [rng.uniform(-0.8, 0.8, 2 * p) for _ in range(restarts)]
        best = None
        for x0 in starts:
            res = minimize(energy_of, x0, args=(m,), method="BFGS")
            if best is None or res.fun < best.fun:
                best = res
        # Keep the winner as the next depth's starting point.
        prev = best.x
        out.append((best.fun - m["e0"], best.x))
    return out


def layers_needed(errs, tol=1e-3):
    """Smallest depth whose energy error is below tol (None if never)."""
    # Walk depths upward and stop at the first one that is good enough.
    for p, e in enumerate(errs, start=1):
        if e < tol:
            return p
    return None


# ----------------------------------------------------------------------------
# Noisy circuit evaluation
# ----------------------------------------------------------------------------
def noisy_energy(m, params, lam):
    """Energy of the transpiled HVA circuit with depolarizing CNOT noise."""
    from qiskit_aer import AerSimulator  # noqa: PLC0415
    from qiskit_aer.noise import NoiseModel, depolarizing_error  # noqa: PLC0415

    # Level 3 resynthesises each bond gate into the optimal 3 CNOTs (Lab 16);
    # level 1 would leave 6, doubling the noise for no reason.
    qc = transpile(
        hva_circuit(m["n"], params),
        basis_gates=BASIS,
        optimization_level=3,
        seed_transpiler=3,
    )
    ncx = qc.count_ops().get("cx", 0)
    qc.save_density_matrix()
    nm = NoiseModel()
    if lam > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(lam, 2), ["cx"])
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    rho = np.asarray(sim.run(qc).result().data(0)["density_matrix"].data)
    return float(np.trace(rho @ m["h"]).real), ncx


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_scaling(n, layers, seed, r):
    """Checks 1 to 3: ideal optimisation, scaling with n, symmetry."""
    log.info("=== Check 1: warm-started HVA on n=%d (up to %d layers) ===", n, layers)
    # One seeded generator keeps the random restarts reproducible.
    rng = np.random.default_rng(seed)
    m = build(n)
    res = optimise_depths(m, layers, rng)
    errs = [e for e, _ in res]
    for p, e in enumerate(errs, start=1):
        log.info("p=%d (%2d angles): energy error %.3e", p, 2 * p, e)
    r.append(
        check(
            all(b <= a + 1e-9 for a, b in zip(errs, errs[1:])),
            "energy error never increases with depth (warm start)",
        )
    )
    r.append(
        check(
            all(e > -1e-9 for e in errs),
            "variational principle: no energy below the exact E0",
        )
    )
    r.append(
        check(
            errs[-1] < 0.25 * errs[0] or errs[-1] < 1e-3,
            f"depth {layers} cuts the error from {errs[0]:.2e} to " f"{errs[-1]:.2e}",
        )
    )
    # Overlap with the unique ground state obeys E - E0 >= gap (1 - overlap).
    best = res[-1][1]
    sv = Statevector.from_instruction(hva_circuit(n, best)).data
    ov = abs(np.vdot(m["gs"], sv)) ** 2
    bound = 1 - errs[-1] / m["gap"]
    r.append(
        check(
            ov >= bound - 1e-9,
            f"overlap {ov:.6f} obeys the bound " f"1 - error/gap = {bound:.6f}",
        )
    )
    e_circ = float((sv.conj() @ m["h"] @ sv).real)
    r.append(
        check(
            abs(e_circ - (m["e0"] + errs[-1])) < 1e-7,
            "Qiskit statevector energy equals the NumPy optimiser's",
        )
    )
    wild = rng.uniform(-3, 3, 2 * layers)
    sw = Statevector.from_instruction(hva_circuit(n, wild)).data
    if n <= 8:
        r.append(
            check(
                abs(expval(total_s2(n), sw)) < 1e-9
                and abs(expval(total_sz(n), sw)) < 1e-9,
                "random angles keep S^2 = 0 and S_z = 0 exactly",
            )
        )
    else:
        log.info("n=%d: skipping the dense S^2 check (matrix too large)", n)
    log.info("=== Check 2: layers needed versus chain length ===")
    # need[size] = smallest depth reaching error < 1e-3 for that chain length.
    need = {}
    for size in [s for s in (4, 6, 8) if s <= n]:
        if size == n:
            need[size] = layers_needed(errs)
        else:
            mm = build(size)
            need[size] = layers_needed(
                [e for e, _ in optimise_depths(mm, min(layers, 8), rng)]
            )
    log.info(
        "layers needed for error < 1e-3: %s (None = not reached within " "%d layers)",
        need,
        layers,
    )
    # A depth that was never reached counts as "more than we tried" (infinity),
    # so the ordering test stays meaningful when the largest chain stalls.
    seq = [need[s] if need[s] is not None else math.inf for s in sorted(need)]
    if len(seq) < 2:
        log.info("only one chain length in range: nothing to compare")
    else:
        r.append(
            check(
                all(b >= a for a, b in zip(seq, seq[1:])),
                "longer chains need at least as many layers",
            )
        )
    return m, res


def check_noise(r, quick):
    """Checks 4 and 5: depth versus noise on a 6-spin chain."""
    log.info("=== Check 3: noisy circuits, n=6 ===")
    n = 6
    m = build(n)
    rng = np.random.default_rng(11)
    res = optimise_depths(m, 3, rng)
    # Parameters for depth p come from the noise-free optimisation.
    depth_params = {0: np.zeros(0)}
    depth_params.update({p + 1: res[p][1] for p in range(3)})
    # table[(noise, depth)] = noisy energy; cx_counts[depth] = CNOTs used.
    table = {}
    cx_counts = {}
    for lam in NOISE_LEVELS:
        for p in range(4):
            e, ncx = noisy_energy(m, depth_params[p], lam)
            table[lam, p] = e
            cx_counts[p] = ncx
    log.info("CNOTs per depth: %s", cx_counts)
    for lam in NOISE_LEVELS:
        log.info(
            "lambda=%.4f: energy error by depth p=0..3: %s",
            lam,
            " ".join(f"{table[lam, p] - m['e0']:+.3f}" for p in range(4)),
        )
    # Noiseless circuit must reproduce the NumPy optimiser's energies.
    ok = all(abs((table[0.0, p + 1] - m["e0"]) - res[p][0]) < 1e-6 for p in range(3))
    r.append(
        check(
            ok,
            "noiseless density-matrix circuits reproduce the "
            "NumPy energies at every depth",
        )
    )
    r.append(
        check(
            all(
                table[lam, p] > m["e0"] - 1e-9 for lam in NOISE_LEVELS for p in range(4)
            ),
            "no noisy state beats the exact ground energy",
        )
    )
    # Depth with the lowest noisy energy, for each noise level.
    best = {
        lam: min(range(4), key=lambda p: table[lam, p] - m["e0"])
        for lam in NOISE_LEVELS
    }
    log.info("best depth by noise level: %s", best)
    r.append(check(best[0.0] == 3, "noiseless: deepest circuit is best"))
    r.append(
        check(
            best[0.01] < 3,
            "at 1% per CNOT extra layers HURT: the "
            f"best depth is {best[0.01]}, not 3",
        )
    )
    seq = [best[lam] for lam in NOISE_LEVELS]
    r.append(
        check(
            all(b <= a for a, b in zip(seq, seq[1:])),
            "noisier hardware prefers a shallower circuit",
        )
    )
    gain = table[0.0037, 0] - table[0.0037, best[0.0037]]
    log.info(
        "at the measured ibm_fez level, the best depth (%d) improves on "
        "the bare singlet start by %.3f energy units",
        best[0.0037],
        gain,
    )
    r.append(
        check(gain >= -1e-9, "the best depth is never worse than doing " "nothing")
    )


def main(argv=None):
    """Run every Lab 20 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 20: VQE scaling and noise")
    ap.add_argument("--qubits", type=int, default=8, help="even, 4..8")
    ap.add_argument("--layers", type=int, default=6, help="max HVA layers 2..8")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true", help="n=6, 4 layers")
    args = ap.parse_args(argv)
    if args.qubits % 2 or not 4 <= args.qubits <= 8:
        ap.error(f"--qubits must be even and 4..8. Got {args.qubits}.")
    if not 2 <= args.layers <= 8:
        ap.error(f"--layers must be 2..8. Got {args.layers}.")
    n, layers = (6, 4) if args.quick else (args.qubits, args.layers)
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_scaling(n, layers, args.seed, r)
    check_noise(r, args.quick)
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
