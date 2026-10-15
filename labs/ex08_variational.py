#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/ex08_variational.py

Purpose
-------
Lab 8 optional exercises, each as predict-then-verify:
  1. Gradient descent from seeds 1..5, then with a too-large step (lr=1.5).
  2. A fifth parameter and an extra entangling layer in the VQE ansatz.
  3. QAOA on a 5-node graph (edges (3,4) and (4,0) added): p=1 vs p=2.
  4. The real H2 molecule from PySCF (STO-3G, R = 0.735 A): build the
     Hamiltonian by Jordan-Wigner, run a 1-parameter VQE, compare with FCI.
     Needs `pip install pyscf`; the exercise is skipped (not failed) if the
     package is missing.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/ex08_variational.py

Log files
---------
logs/ex08/ex08_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid command-line usage.
130 User interrupted execution.
===============================================================================
"""

import math
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize_scalar

import lab08_variational as lab08
from applog import get_logger
from exlib import Tally, base_parser

log = get_logger("ex08")

BASE_ANSATZ = lab08.ansatz  # keep the original so we can restore it


def steps_to_accuracy(history, e0):
    """First iteration whose energy error is below chemical accuracy."""
    # Scan in iteration order so the first hit is the earliest step that is
    # already inside chemical accuracy (a later dip would not count).
    for i, e in enumerate(history):
        if e - e0 < lab08.CHEM_ACC:
            return i
    return None


def ex1(t, ham, e0):
    """Exercise 1: seeds 1..5 and an oversized learning rate."""
    t.ask("Exercise 1: gradient descent from seeds 1-5, lr = 0.4 and 1.5")
    t.predict(
        "lr=0.4 converges from every seed (smooth landscape); lr=1.5"
        " overshoots every step, so the energy bounces around instead"
        " of settling and never reaches chemical accuracy"
    )
    # Two step sizes: one safe, one deliberately too large.
    for lr in (0.4, 1.5):
        hits, finals, ups = [], [], []
        for seed in range(1, 6):
            # Seeded random start so each run is reproducible.
            theta0 = np.random.default_rng(seed).uniform(-math.pi, math.pi, 4)
            _, hist = lab08.gradient_descent(theta0, ham, lr=lr, steps=200)
            # Record iterations to reach chemical accuracy.
            hits.append(steps_to_accuracy(hist, e0))
            finals.append(hist[-1] - e0)
            # Largest rise in energy between steps detects overshoot.
            ups.append(float(np.max(np.diff(hist))))
        t.note(
            "lr=%.1f: steps to 1.6 mHa %s; final errors %s; largest"
            " single-step energy RISE %s",
            lr,
            hits,
            [f"{x:.1e}" for x in finals],
            [f"{x:.2f}" for x in ups],
        )
        if lr == 0.4:
            t.check(
                all(h is not None for h in hits),
                "lr=0.4: all five seeds reach chemical accuracy",
            )
            t.check(max(ups) < 1e-9, "lr=0.4: energy never rises (monotone descent)")
        else:
            t.check(
                all(h is None for h in hits),
                "lr=1.5: no seed reaches chemical accuracy",
            )
            t.check(
                all(u > 0.05 for u in ups) and min(finals) > 0.1,
                "lr=1.5: energy rises on overshoot and ends >0.1 Ha high",
            )


def ansatz_five(theta):
    """Lab ansatz plus a fifth Ry on qubit 0: it merges with theta[2]."""
    # Start from the lab's own four-parameter ansatz so the only change is
    # the extra gate under test.
    qc = BASE_ANSATZ(theta[:4])
    qc.ry(theta[4], 0)  # Ry(a) Ry(b) = Ry(a + b): redundant
    return qc


def ansatz_layer(theta):
    """Six angles: three Ry layers separated by two CNOT entanglers."""
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)
    qc.ry(theta[2], 0)
    qc.ry(theta[3], 1)
    qc.cx(0, 1)
    qc.ry(theta[4], 0)
    qc.ry(theta[5], 1)
    return qc


def ex2(t, ham, e0):
    """Exercise 2: fifth parameter and extra entangling layer."""
    t.ask("Exercise 2: a 5th parameter and an extra entangling layer")
    t.predict(
        "a 5th Ry on the same qubit is redundant (only theta2+theta4"
        " matters): same minimum, a flat direction, no new power. An"
        " extra CNOT layer adds directions and should converge in"
        " fewer steps on average; all variants reach the same E0"
    )
    # Three ansatz variants compared on identical starting seeds.
    variants = (
        ("base (4 params)", BASE_ANSATZ, 4),
        ("fifth Ry (5)", ansatz_five, 5),
        ("extra layer (6)", ansatz_layer, 6),
    )
    means = {}
    try:
        for name, fn, n in variants:
            # energy() looks up the module global, so patching swaps the circuit.
            lab08.ansatz = fn  # energy() reads this module global
            hits = []
            for seed in range(1, 6):
                rng = np.random.default_rng(seed)
                theta0 = rng.uniform(-math.pi, math.pi, 4)
                if n == 5:
                    theta0 = np.append(theta0, 0.0)  # redundant angle at 0
                elif n == 6:
                    theta0 = np.append(theta0, rng.uniform(-math.pi, math.pi, 2))
                _, hist = lab08.gradient_descent(theta0, ham, steps=200)
                hits.append(steps_to_accuracy(hist, e0))
                t.check(hist[-1] - e0 < 1e-9, f"{name}, seed {seed}: converges to E0")
            # Mean steps summarizes the five seeds in one number.
            means[name] = float(np.mean(hits))
            t.note(
                "%s: steps to 1.6 mHa per seed %s, mean %.1f", name, hits, means[name]
            )
        # Prove the redundancy directly: shifting mass between theta2 and
        # theta4 must leave the energy unchanged to machine precision.
        lab08.ansatz = ansatz_five
        th = np.array([0.3, -0.7, 0.5, 0.2, 0.4])
        moved = th + np.array([0, 0, 0.35, 0, -0.35])
        t.check(
            abs(lab08.energy(th, ham) - lab08.energy(moved, ham)) < 1e-12,
            "5th parameter is redundant: E depends only on theta2+theta4",
        )
    finally:
        lab08.ansatz = BASE_ANSATZ  # always restore the lab's ansatz
    t.check(
        means["extra layer (6)"] < means["base (4 params)"],
        "the extra entangling layer converges faster on average",
    )


def ex3(t):
    """Exercise 3: QAOA on a 5-node graph."""
    t.ask("Exercise 3: QAOA MaxCut on 5 nodes (edges (3,4) and (4,0) added)")
    t.predict(
        "the new edges 0-3-4 make a triangle (odd cycle), so not every"
        " edge can be cut: max cut = 5 of 7. p=2 should beat p=1;"
        " neither reaches ratio 1"
    )
    # Remember the 4-node graph so we can restore it afterwards.
    old_edges, old_circ = list(lab08.EDGES), lab08.qaoa_circuit
    try:
        lab08.EDGES[:] = old_edges + [(3, 4), (4, 0)]
        # expected_cut() calls qaoa_circuit without n, so widen the default.
        lab08.qaoa_circuit = lambda g, b, n=5: old_circ(g, b, n)
        # Brute force over all 32 cuts gives the true optimum.
        cuts = [lab08.cut_value(x) for x in range(32)]
        best = max(cuts)
        t.note(
            "%d edges, max cut %d, random average %.2f",
            len(lab08.EDGES),
            best,
            sum(cuts) / 32,
        )
        t.check(
            best == 5 and abs(sum(cuts) / 32 - 3.5) < 1e-12,
            "brute force: max cut 5, random guessing 3.5",
        )
        rng = np.random.default_rng(0)
        # Coarse grid first: QAOA landscapes have many local optima.
        grid = [
            (g, b)
            for g in np.linspace(0, 2 * math.pi, 25, endpoint=False)
            for b in np.linspace(0, math.pi, 13, endpoint=False)
        ]
        scored = sorted(grid, key=lambda gb: -lab08.expected_cut(list(gb), 1))
        e1, x1 = lab08.best_qaoa(1, rng, [np.array(g) for g in scored[:3]])
        starts = [np.array([x1[0], 0.0, x1[1], 0.0])]
        starts += [rng.uniform(0, math.pi, 4) for _ in range(5)]
        e2, _ = lab08.best_qaoa(2, rng, starts)
    finally:
        lab08.EDGES[:] = old_edges  # restore the 4-node graph
        lab08.qaoa_circuit = old_circ
    t.note(
        "p=1: <C>=%.4f ratio %.4f;  p=2: <C>=%.4f ratio %.4f",
        e1,
        e1 / best,
        e2,
        e2 / best,
    )
    t.check(
        3.5 < e1 < e2 < best,
        "random 3.5 < p=1 < p=2 < max cut 5: depth helps, never perfect",
    )
    t.check(
        e1 / best > 0.85 and e2 / best > e1 / best,
        "approximation ratio improves with p (and p=1 already > 0.85)",
    )


# ----------------------------------------------------------------------------
# Exercise 4: real H2 from PySCF
# ----------------------------------------------------------------------------
def site_op(p, op, n):
    """Embed 2x2 `op` on qubit p of n qubits, little-endian (qubit 0 = LSB)."""
    # Identity on every qubit except p; the Kronecker product below then
    # places `op` on exactly that qubit.
    mats = [np.eye(2)] * n
    mats[p] = op
    out = np.eye(1)
    for m in reversed(mats):  # highest qubit is the leftmost factor
        out = np.kron(out, m)
    return out


def annihilation(p, n):
    """Jordan-Wigner a_p = Z_0 ... Z_{p-1} (|0><1|)_p, a 2^n matrix."""
    z = np.diag([1.0, -1.0])
    lower = np.array([[0.0, 1.0], [0.0, 0.0]])  # |0><1|
    out = np.eye(2**n)
    for q in range(p):
        out = out @ site_op(q, z, n)  # Z string keeps the fermion signs
    return out @ site_op(p, lower, n)


def h2_hamiltonian(h1, eri, e_nuc, n_mo):
    """Second-quantized electronic H plus nuclear repulsion, as a matrix."""
    # Two spin orbitals (up and down) per spatial orbital, hence 2 * n_mo
    # qubits after the Jordan-Wigner mapping.
    n = 2 * n_mo
    a = [annihilation(p, n) for p in range(n)]
    ad = [x.T for x in a]  # real matrices: dagger = transpose
    spin_orb = lambda i, s: i + n_mo * s  # noqa: E731
    h = e_nuc * np.eye(2**n)
    for i in range(n_mo):
        for j in range(n_mo):
            for s in (0, 1):
                h = h + h1[i, j] * ad[spin_orb(i, s)] @ a[spin_orb(j, s)]
    for i in range(n_mo):
        for j in range(n_mo):
            for k in range(n_mo):
                for l in range(n_mo):
                    for s in (0, 1):
                        for u in (0, 1):
                            # chemist's (ij|kl): 1/2 a+_i a+_k a_l a_j
                            h = h + 0.5 * eri[i, j, k, l] * (
                                ad[spin_orb(i, s)]
                                @ ad[spin_orb(k, u)]
                                @ a[spin_orb(l, u)]
                                @ a[spin_orb(j, s)]
                            )
    return h


def h2_ansatz(phi):
    """cos(phi)|HF> + sin(phi)|double>, particle-number conserving."""
    qc = QuantumCircuit(4)
    qc.ry(2 * phi, 1)  # virtual orbital amplitude sin(phi)
    qc.cx(1, 3)  # both spins of orbital 1 move together
    qc.x([0, 2])
    qc.cx(1, 0)  # occupied orbital is the complement
    qc.cx(1, 2)
    return qc


def ex4(t):
    """Exercise 4: the real H2 molecule."""
    t.ask("Exercise 4: H2 from PySCF, Jordan-Wigner, 1-parameter VQE")
    try:
        from pyscf import fci, gto, scf  # noqa: PLC0415
        from pyscf import ao2mo  # noqa: PLC0415
    except ImportError:
        log.warn_(
            "pyscf is not installed: skipping Exercise 4 "
            "(pip install pyscf, then re-run)"
        )
        return
    t.predict(
        "in a 2-orbital basis the singlet FCI space is spanned by the"
        " HF determinant and the double excitation, so a 1-parameter"
        " ansatz reaches FCI exactly (~ -1.137 Ha); <HF|H|HF> equals"
        " the SCF energy (~ -1.117 Ha)"
    )
    # Standard textbook geometry: H-H at 0.735 Angstrom, STO-3G basis.
    mol = gto.M(atom="H 0 0 0; H 0 0 0.735", basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run()
    mo = mf.mo_coeff
    n_mo = mo.shape[1]
    # Transform integrals from atomic to molecular orbitals.
    h1 = mo.T @ mf.get_hcore() @ mo
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mo), n_mo)
    e_fci = float(fci.FCI(mf).kernel()[0])
    h = h2_hamiltonian(h1, eri, float(mol.energy_nuc()), n_mo)
    t.note("SCF %.8f Ha, FCI %.8f Ha, %d spin orbitals", mf.e_tot, e_fci, 2 * n_mo)
    t.check(np.allclose(h, h.T), "Jordan-Wigner Hamiltonian is symmetric")
    # The matrix covers every particle number; FCI is the 2-electron sector.
    # Number operator isolates the 2-electron sector of Fock space.
    number = sum(annihilation(p, 4).T @ annihilation(p, 4) for p in range(4))
    keep = np.isclose(np.diag(number), 2)
    e_sector = float(np.linalg.eigvalsh(h[np.ix_(keep, keep)])[0])
    t.check(
        abs(e_sector - e_fci) < 1e-8,
        f"2-electron ground energy {e_sector:.8f} equals PySCF FCI",
    )
    ham = SparsePauliOp.from_operator(h)
    ham = ham.simplify()
    t.note("Hamiltonian has %d Pauli terms on 4 qubits", len(ham))
    # Qubits 0 and 2 occupied = orbital 0 with both spins.
    hf = Statevector.from_label("0101")  # qubits 0 and 2 set: orbital 0, both spins
    e_hf = float(np.real(hf.expectation_value(ham)))
    t.check(
        abs(e_hf - mf.e_tot) < 1e-8, f"<HF|H|HF> = {e_hf:.8f} reproduces the SCF energy"
    )

    def energy(phi):
        """VQE energy for one angle."""
        sv = Statevector.from_instruction(h2_ansatz(phi))
        return float(np.real(sv.expectation_value(ham)))

    # One angle: a bounded scalar search is the right tool, no gradients needed.
    res = minimize_scalar(
        energy,
        bounds=(-math.pi / 2, math.pi / 2),
        method="bounded",
        options={"xatol": 1e-10},
    )
    t.note(
        "VQE: phi=%.5f, E=%.8f Ha, error vs FCI %.2e Ha",
        res.x,
        res.fun,
        res.fun - e_fci,
    )
    t.check(abs(energy(0.0) - e_hf) < 1e-9, "phi = 0 is exactly the HF state")
    t.check(
        abs(res.fun - e_fci) < 1e-7 and res.fun - e_fci > -1e-9,
        "1-parameter VQE matches FCI (and never undercuts it)",
    )
    t.check(
        e_hf - e_fci > lab08.CHEM_ACC,
        f"HF misses by {e_hf - e_fci:.4f} Ha (correlation energy), more"
        " than chemical accuracy; VQE recovers it",
    )


def main(argv=None):
    """Run all four Lab 8 exercises; return the exit code."""
    base_parser("Lab 8 exercises").parse_args(argv)
    ham = lab08.hamiltonian()
    e0 = float(np.linalg.eigvalsh(ham.to_matrix())[0])
    t = Tally(log)
    ex1(t, ham, e0)
    ex2(t, ham, e0)
    ex3(t)
    ex4(t)
    return t.finish()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
