#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab14_hamiltonian_simulation.py

Purpose
-------
Lab 14 simulates the time evolution of a small spin chain, the problem that
Feynman proposed quantum computers for. The model is the open-boundary
transverse-field Ising chain

    H = A + B,   A = -J * sum_i Z_i Z_{i+1},   B = -h * sum_i X_i

whose two parts do not commute, so exp(-iHt) cannot be split exactly. The lab
verifies, in order:

  1. The Hamiltonian: our own Kronecker-product build equals Qiskit's
     SparsePauliOp; it is Hermitian; exp(-iHt) is unitary and conserves energy.
  2. Trotter-Suzuki product formulas, built in NumPy: first order has error
     proportional to 1/r and second order 1/r^2 for r steps (slopes fitted on a
     log-log plot); if A and B commute (h = 0) the formula is exact.
  3. Qiskit circuits made of RZZ and RX gates reproduce the NumPy product
     formulas exactly, and Qiskit's own PauliEvolutionGate agrees too.
  4. Dynamics: the magnetisation <Z_0>(t) after a quench from |0...0>, exact
     versus Trotter. Trotter energy is not conserved but stays bounded.
  5. Hardware cost: the CNOT count of a transpiled step is 2(n-1) for both
     orders, so second order is a free improvement in accuracy.
  6. Noise: with depolarizing CNOT errors, more Trotter steps cut formula
     error but add hardware error, so fidelity peaks at a finite step count.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab14_hamiltonian_simulation.py
    python3 labs/lab14_hamiltonian_simulation.py --qubits 5 --time 1.5

Log files
---------
logs/lab14/lab14_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid command-line usage.
130 User interrupted execution.
===============================================================================
"""

import argparse
import sys
from functools import reduce

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.linalg import expm

from applog import get_logger

log = get_logger("lab14")

J = 1.0  # Ising coupling
H_FIELD = 1.0  # transverse field (critical point)
BASIS = ["cx", "rz", "sx", "x"]  # a typical native gate set
I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)


# ----------------------------------------------------------------------------
# Hamiltonian
# ----------------------------------------------------------------------------
def kron_all(ops):
    """Kronecker product of a list of matrices (first element = leftmost)."""
    # Kronecker product of a list of single-qubit matrices.
    return reduce(np.kron, ops)


def site_op(n, pauli, sites):
    """Dense n-qubit operator with `pauli` on each qubit in `sites`.

    Qiskit is little-endian (qubit 0 is the RIGHTMOST tensor factor), so the
    list is built from qubit n-1 down to qubit 0.
    """
    # Little endian: qubit 0 is the rightmost factor, matching Qiskit.
    return kron_all([pauli if q in sites else I2 for q in reversed(range(n))])


def dense_terms(n, j=J, h=H_FIELD):
    """Return (A, B) as dense matrices built only with Kronecker products."""
    # Split H = A + B into Z-type and X-type parts; each part commutes internally.
    a = -j * sum(site_op(n, PZ, (q, q + 1)) for q in range(n - 1))
    b = -h * sum(site_op(n, PX, (q,)) for q in range(n))
    return a, b


def pauli_terms(n, j=J, h=H_FIELD):
    """Return (A, B) as Qiskit SparsePauliOps (independent of dense_terms)."""
    # Pauli-string list is what the circuit builders consume.
    zz = [("ZZ", [q, q + 1], -j) for q in range(n - 1)]
    xx = [("X", [q], -h) for q in range(n)]
    return (
        SparsePauliOp.from_sparse_list(zz, num_qubits=n),
        SparsePauliOp.from_sparse_list(xx, num_qubits=n),
    )


# ----------------------------------------------------------------------------
# Product formulas (NumPy reference)
# ----------------------------------------------------------------------------
def exact_u(a, b, t):
    """Exact propagator exp(-i (A+B) t)."""
    # Exact propagator is the reference for every Trotter error below.
    return expm(-1j * (a + b) * t)


def trotter_u(a, b, t, r, order):
    """Product-formula propagator with r steps of size t/r.

    order 1:  (e^{-iB dt} e^{-iA dt})^r                     error ~ t^2/r
              (A acts first, matching the circuit: ZZ layer, then RX layer)
    order 2:  (e^{-iB dt/2} e^{-iA dt} e^{-iB dt/2})^r      error ~ t^3/r^2
    """
    # Divide the time into r equal steps of length dt.
    dt = t / r
    if order == 1:
        step = expm(-1j * b * dt) @ expm(-1j * a * dt)
    else:
        half_b = expm(-1j * b * dt / 2)
        step = half_b @ expm(-1j * a * dt) @ half_b
    return np.linalg.matrix_power(step, r)


def op_error(u, v):
    """Operator-norm distance ||U - V||_2 (largest singular value)."""
    # Spectral norm is the worst-case error over all input states.
    return float(np.linalg.norm(u - v, 2))


# ----------------------------------------------------------------------------
# Circuits
# ----------------------------------------------------------------------------
def trotter_circuit(n, t, r, order, j=J, h=H_FIELD):
    """Circuit for the product formula, one RZZ per bond and one RX per site.

    RZZ(theta) = exp(-i theta/2 Z(x)Z) and RX(theta) = exp(-i theta/2 X), so
    exp(-i A dt) needs theta = -2 j dt and exp(-i B dt) needs theta = -2 h dt.
    """
    # First-order product formula: one X layer then one ZZ layer per step.
    dt = t / r
    qc = QuantumCircuit(n)

    def x_layer(frac):
        """Apply exp(-i B frac*dt) as RX on every qubit."""
        # Rx(2*h*dt) implements exp(-i h dt X) on every qubit.
        for q in range(n):
            qc.rx(-2 * h * dt * frac, q)

    def zz_layer():
        """Apply exp(-i A dt) as RZZ on every bond."""
        # Rzz on neighbours; gates here all commute so order is free.
        for q in range(n - 1):
            qc.rzz(-2 * j * dt, q, q + 1)

    for _ in range(r):
        if order == 1:
            zz_layer()
            x_layer(1.0)
        else:
            # The symmetric split puts the ZZ layer ONCE in the middle, so the
            # CNOT count per step equals that of first order.
            x_layer(0.5)
            zz_layer()
            x_layer(0.5)
    return qc


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def slope(rs, errs):
    """Least-squares slope of log(error) against log(r)."""
    # Log-log slope gives the empirical order of the product formula.
    return float(np.polyfit(np.log(rs), np.log(errs), 1)[0])


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_hamiltonian(n, t, r):
    """Check 1: Hamiltonian construction, unitarity, energy conservation."""
    log.info("=== Check 1: the Hamiltonian (n=%d qubits) ===", n)
    a, b = dense_terms(n)
    pa, pb = pauli_terms(n)
    h = a + b
    u = exact_u(a, b, t)
    psi0 = np.zeros(2**n, dtype=complex)
    psi0[0] = 1.0
    psi_t = u @ psi0
    e0, e1 = (psi0.conj() @ h @ psi0).real, (psi_t.conj() @ h @ psi_t).real
    evals = np.linalg.eigvalsh(h)
    log.info(
        "ground energy %.4f, spectrum width %.4f, <H> in |0..0> = %.4f",
        evals[0],
        evals[-1] - evals[0],
        e0,
    )
    r.append(
        check(
            np.allclose(pa.to_matrix(), a) and np.allclose(pb.to_matrix(), b),
            "Kronecker build equals Qiskit SparsePauliOp (A and B)",
        )
    )
    r.append(check(np.allclose(h, h.conj().T), "H is Hermitian"))
    r.append(
        check(
            np.allclose(u.conj().T @ u, np.eye(2**n), atol=1e-12),
            "exp(-iHt) is unitary",
        )
    )
    r.append(
        check(
            abs(e1 - e0) < 1e-10,
            f"exact evolution conserves energy " f"({e0:.6f} -> {e1:.6f})",
        )
    )
    comm = np.linalg.norm(a @ b - b @ a, 2)
    r.append(
        check(
            comm > 1e-3,
            f"[A, B] != 0 (norm {comm:.3f}): that is why " "Trotter error exists",
        )
    )


def check_orders(n, t, r):
    """Check 2: error scaling of first and second order formulas."""
    log.info("=== Check 2: Trotter error versus step count (t=%.2f) ===", t)
    a, b = dense_terms(n)
    u = exact_u(a, b, t)
    steps = [8, 16, 32, 64, 128]
    err = {o: [op_error(u, trotter_u(a, b, t, k, o)) for k in steps] for o in (1, 2)}
    for i, k in enumerate(steps):
        log.info("r=%3d: first order %.2e   second order %.2e", k, err[1][i], err[2][i])
    s1, s2 = slope(steps, err[1]), slope(steps, err[2])
    log.info("fitted slopes: first %.2f (theory -1), second %.2f (theory -2)", s1, s2)
    r.append(check(abs(s1 + 1) < 0.15, "first-order error falls as 1/r"))
    r.append(check(abs(s2 + 2) < 0.2, "second-order error falls as 1/r^2"))
    r.append(
        check(
            all(e2 < e1 for e1, e2 in zip(err[1], err[2])),
            "second order beats first order at every r",
        )
    )
    # Halving the step must quarter the second-order error (asymptotically).
    r.append(
        check(
            3.5 < err[2][2] / err[2][3] < 4.5,
            f"doubling r cuts second-order error "
            f"{err[2][2] / err[2][3]:.2f}x (expect ~4x)",
        )
    )
    # With h = 0 the two parts commute and any single step is exact.
    a0, b0 = dense_terms(n, h=0.0)
    ex0 = exact_u(a0, b0, t)
    r.append(
        check(
            op_error(ex0, trotter_u(a0, b0, t, 1, 1)) < 1e-12,
            "commuting case (h=0): ONE step is exact",
        )
    )
    # Error comes from the commutator, so it grows with the field strength.
    e_small = op_error(
        exact_u(*dense_terms(n, h=0.2), t), trotter_u(*dense_terms(n, h=0.2), t, 8, 1)
    )
    e_big = op_error(
        exact_u(*dense_terms(n, h=1.0), t), trotter_u(*dense_terms(n, h=1.0), t, 8, 1)
    )
    r.append(
        check(
            e_small < e_big,
            f"error grows with non-commutation "
            f"(h=0.2: {e_small:.3f}, h=1.0: {e_big:.3f})",
        )
    )


def check_circuits(n, t, r):
    """Check 3: Qiskit circuits equal the NumPy formulas."""
    log.info("=== Check 3: Qiskit circuits ===")
    a, b = dense_terms(n)
    k = 4
    for order in (1, 2):
        qc = trotter_circuit(n, t, k, order)
        diff = op_error(Operator(qc).data, trotter_u(a, b, t, k, order))
        r.append(
            check(
                diff < 1e-10,
                f"order {order}: circuit equals NumPy "
                f"product formula (diff {diff:.1e})",
            )
        )
    pa, pb = pauli_terms(n)
    # The full Hamiltonian as one Pauli sum, the input Qiskit's gate expects.
    h_op = pa + pb
    # Operator(gate) would use the gate's EXACT matrix, hiding the Trotter
    # error. Wrapping it in a circuit and decomposing forces the synthesised
    # product formula, which is what hardware would run.
    exact = exact_u(a, b, t)

    def synthesised(**kw):
        """Operator of the decomposed PauliEvolutionGate (a real product formula)."""
        circ = QuantumCircuit(n)
        circ.append(PauliEvolutionGate(h_op, time=t, **kw), range(n))
        return Operator(circ.decompose(reps=2)).data

    d1 = op_error(synthesised(synthesis=LieTrotter(reps=k)), exact)
    d2 = op_error(synthesised(synthesis=SuzukiTrotter(2, reps=k)), exact)
    log.info(
        "PauliEvolutionGate error vs exact at r=%d: Lie %.2e, Suzuki-2 %.2e", k, d1, d2
    )
    r.append(
        check(0 < d2 < d1, "Qiskit's PauliEvolutionGate: Suzuki-2 beats Lie-Trotter")
    )
    # With no synthesis argument the gate uses ONE Lie-Trotter step, so it
    # should match our own r=1 first-order formula up to term ordering.
    one = op_error(synthesised(), exact)
    r.append(
        check(
            one > d1,
            f"one default step (error {one:.3f}) is worse than "
            f"{k} steps ({d1:.3f})",
        )
    )


def check_dynamics(n, t, r):
    """Check 4: quench dynamics of <Z_0> and Trotter energy drift."""
    log.info("=== Check 4: quench dynamics from |0...0> ===")
    a, b = dense_terms(n)
    z0 = site_op(n, PZ, (0,))
    h = a + b
    psi0 = np.zeros(2**n, dtype=complex)
    psi0[0] = 1.0
    times = np.linspace(0, t, 9)
    exact_m, trot_m, trot_e = [], [], []
    for tau in times:
        pe = exact_u(a, b, tau) @ psi0
        # The step size is fixed (dt = 0.125) so r grows with tau.
        steps = max(1, int(round(tau / 0.125)))
        pt = trotter_u(a, b, tau, steps, 2) @ psi0
        exact_m.append((pe.conj() @ z0 @ pe).real)
        trot_m.append((pt.conj() @ z0 @ pt).real)
        trot_e.append((pt.conj() @ h @ pt).real)
    dev = max(abs(x - y) for x, y in zip(exact_m, trot_m))
    log.info("<Z_0>(t) exact : %s", " ".join(f"{x:+.3f}" for x in exact_m))
    log.info("<Z_0>(t) Trotter: %s", " ".join(f"{x:+.3f}" for x in trot_m))
    log.info(
        "max |exact - Trotter| = %.4f; Trotter <H> range %.4f..%.4f",
        dev,
        min(trot_e),
        max(trot_e),
    )
    r.append(
        check(
            abs(exact_m[0] - 1) < 1e-12 and min(exact_m) < 0.9,
            "quench starts at <Z_0>=1 and the magnetisation decays",
        )
    )
    r.append(
        check(
            dev < 0.02,
            f"second-order Trotter (dt=0.125) tracks <Z_0> "
            f"within 0.02 (max error {dev:.4f})",
        )
    )
    spread = max(trot_e) - min(trot_e)
    r.append(
        check(
            0 < spread < 0.2,
            f"Trotter energy is not conserved but "
            f"stays bounded (spread {spread:.4f})",
        )
    )


def check_cost(n, t, r):
    """Check 5: CNOT counts after transpiling to a native gate set."""
    log.info("=== Check 5: hardware cost ===")
    counts = {}
    for order in (1, 2):
        for k in (1, 4, 8):
            qc = transpile(
                trotter_circuit(n, t, k, order),
                basis_gates=BASIS,
                optimization_level=1,
                seed_transpiler=7,
            )
            counts[(order, k)] = qc.count_ops().get("cx", 0)
    log.info("CX counts (order, r): %s", counts)
    expected = {key: 2 * (n - 1) * key[1] for key in counts}
    r.append(
        check(
            counts == expected,
            "CX count = 2(n-1) per step, for BOTH " "orders (each RZZ costs 2 CNOTs)",
        )
    )
    # Second order costs the same CNOTs but adds only single-qubit rotations.
    q1 = transpile(
        trotter_circuit(n, t, 8, 1),
        basis_gates=BASIS,
        optimization_level=1,
        seed_transpiler=7,
    )
    q2 = transpile(
        trotter_circuit(n, t, 8, 2),
        basis_gates=BASIS,
        optimization_level=1,
        seed_transpiler=7,
    )
    log.info("r=8 depth: first order %d, second order %d", q1.depth(), q2.depth())
    r.append(
        check(
            q2.count_ops().get("cx") == q1.count_ops().get("cx"),
            "second order accuracy costs no extra CNOTs",
        )
    )


def noisy_fidelity(n, t, k, order, p2):
    """Fidelity of the noisy circuit's output with the exact evolved state."""
    # Transpile to the hardware basis so CX counts are realistic.
    qc = transpile(
        trotter_circuit(n, t, k, order),
        basis_gates=BASIS,
        optimization_level=1,
        seed_transpiler=7,
    )
    qc.save_density_matrix()
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    rho = sim.run(qc).result().data(0)["density_matrix"].data
    a, b = dense_terms(n)
    psi = exact_u(a, b, t)[:, 0]
    return float((psi.conj() @ np.asarray(rho) @ psi).real)


def check_noise(n, t, r, p2=0.01):
    """Check 6: formula error falls with r, hardware error rises."""
    log.info("=== Check 6: noisy hardware (CX depolarizing p=%.3f) ===", p2)
    steps = [1, 2, 4, 8, 16, 32]
    fid = {o: [noisy_fidelity(n, t, k, o, p2) for k in steps] for o in (1, 2)}
    for i, k in enumerate(steps):
        log.info(
            "r=%2d (%3d CX): fidelity first %.4f  second %.4f",
            k,
            2 * (n - 1) * k,
            fid[1][i],
            fid[2][i],
        )
    best = {o: steps[int(np.argmax(fid[o]))] for o in (1, 2)}
    log.info("best step count: first order r=%d, second order r=%d", best[1], best[2])
    r.append(
        check(
            0 < min(best.values()) and best[1] < steps[-1] and best[2] < steps[-1],
            "fidelity peaks at a FINITE step count: more steps are "
            "not always better on noisy hardware",
        )
    )
    r.append(
        check(
            max(fid[2]) > max(fid[1]) - 1e-3,
            f"best second-order fidelity {max(fid[2]):.4f} >= best "
            f"first-order {max(fid[1]):.4f}",
        )
    )
    ideal = noisy_fidelity(n, t, 8, 2, 0.0)
    r.append(
        check(ideal > 0.99, f"noiseless r=8 second order: fidelity " f"{ideal:.4f}")
    )
    r.append(
        check(
            fid[2][-1] < max(fid[2]) - 0.02,
            "too many steps loses fidelity to gate noise",
        )
    )


def main(argv=None):
    """Run every Lab 14 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 14: Hamiltonian simulation")
    ap.add_argument("--qubits", type=int, default=4, help="chain length (3..6)")
    ap.add_argument("--time", type=float, default=1.0, help="evolution time")
    args = ap.parse_args(argv)
    if not 3 <= args.qubits <= 6:
        ap.error(f"--qubits must be 3..6. Got {args.qubits}.")
    if not 0.2 <= args.time <= 2.0:
        ap.error(f"--time must be 0.2..2.0. Got {args.time}.")
    # Short aliases used by every check below.
    n, t = args.qubits, args.time
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_hamiltonian(n, t, r)
    check_orders(n, t, r)
    check_circuits(n, t, r)
    check_dynamics(n, max(t, 1.0), r)
    check_cost(n, t, r)
    check_noise(n, t, r)
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
