#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab16_heisenberg.py

Purpose
-------
Lab 16 moves from the Ising chain of Labs 14-15 to the Heisenberg
antiferromagnet, the model of magnetism in which every pair of neighbouring
spins interacts through all three Paulis:

    H = J sum_i ( X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1} ),   J = 1.

It adds two ideas: SYMMETRY (H conserves total S_z and total spin S^2, so a
good circuit should too) and GROUND-STATE PREPARATION with a variational
circuit (the Hamiltonian Variational Ansatz, HVA). The lab verifies:

  1. The model: H commutes with S_z and S^2; a two-spin singlet has energy -3;
     the 4-site open chain has E0 = -3 - 2 sqrt(3); the ground state is a
     spin singlet; energy per site approaches 1 - 4 ln 2 as the chain grows.
  2. Trotter evolution: the three terms on ONE bond commute, so a bond gate is
     exact and the only error comes from even/odd bond ordering. Slopes -1 and
     -2 again, and S_z is conserved EXACTLY even with a coarse step.
  3. Circuits: RXX/RYY/RZZ gates match the NumPy formula; the transpiler
     merges the three into 3 CNOTs per bond (naively 6).
  4. Quench from the Neel state |0101..>: staggered magnetisation, exact versus
     Trotter.
  5. Ground state by HVA: start from a product of singlets, apply p layers of
     exp(-i theta_k H_odd) exp(-i phi_k H_even), optimise 2p angles. Energy
     error falls with p, the singlet symmetry is preserved for ANY angles, and
     the circuit agrees with Qiskit's own expectation value.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab16_heisenberg.py
    python3 labs/lab16_heisenberg.py --qubits 8 --layers 4 --seed 3

Log files
---------
logs/lab16/lab16_<YYYYmmdd_HHMMSS>.log

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
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from scipy.linalg import expm
from scipy.optimize import minimize

from applog import get_logger

log = get_logger("lab16")

J = 1.0
I2 = np.eye(2, dtype=complex)
PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


# ----------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------
def site(n, p, q):
    """Dense n-qubit operator with Pauli `p` on qubit q (qubit 0 rightmost)."""
    # Little endian placement of a Pauli on qubit q in the full space.
    ops = [PAULI[p] if k == q else I2 for k in reversed(range(n))]
    return reduce(np.kron, ops)


def bond(n, q):
    """Heisenberg term on bond (q, q+1): XX + YY + ZZ (times J)."""
    # Heisenberg bond: XX + YY + ZZ, all three Paulis with equal weight.
    return J * sum(site(n, p, q) @ site(n, p, q + 1) for p in "XYZ")


def groups(n):
    """Dense (H_even, H_odd): bonds starting on even / odd qubits.

    Bonds inside one group touch disjoint qubits, so they commute and the
    group exponential factorises exactly. Even and odd groups do not commute.
    """
    # Even and odd bonds each commute internally, so a group is one layer.
    even = sum(bond(n, q) for q in range(0, n - 1, 2))
    odd = sum(bond(n, q) for q in range(1, n - 1, 2)) if n > 2 else 0 * even
    return even, odd


def total_sz(n):
    """Total S_z = (1/2) sum Z_i."""
    # Total Sz is conserved by the Hamiltonian; used as a symmetry check.
    return 0.5 * sum(site(n, "Z", q) for q in range(n))


def total_s2(n):
    """Total S^2 = (1/4) (3n + 2 sum_{i<j} sigma_i . sigma_j)."""
    # S^2 = sum_i s_i^2 + sum pairs of s_i.s_j; singlet gives zero.
    pair = sum(
        site(n, p, i) @ site(n, p, j)
        for i in range(n)
        for j in range(i + 1, n)
        for p in "XYZ"
    )
    return 0.25 * (3 * n * np.eye(2**n) + 2 * pair)


def heis_sparse(n):
    """The same Hamiltonian as a SparsePauliOp (independent construction)."""
    # Sparse form scales to larger n than the dense matrices.
    terms = [(p * 2, [q, q + 1], J) for q in range(n - 1) for p in "XYZ"]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def neel_index(n):
    """Basis index of |0101..>: X on odd qubits (qubit 0 is the LSB)."""
    # Neel state has alternating up and down spins.
    return sum(1 << q for q in range(1, n, 2))


def stag_mag(psi, n):
    """Staggered magnetisation (1/n) sum (-1)^i <Z_i>."""
    # Staggered magnetisation measures antiferromagnetic order.
    probs = np.abs(psi) ** 2
    idx = np.arange(len(probs))
    return float(
        np.mean([(-1) ** q * np.dot(probs, 1 - 2 * ((idx >> q) & 1)) for q in range(n)])
    )


def expval(op, psi):
    """<psi|op|psi> as a real number."""
    return float((psi.conj() @ op @ psi).real)


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def slope(xs, ys):
    """Least-squares slope of log(y) against log(x)."""
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


# ----------------------------------------------------------------------------
# Product formulas and circuits
# ----------------------------------------------------------------------------
def trotter_u(n, t, r, order):
    """Product-formula propagator with r steps (see Lab 14)."""
    # Order 1: A then B. Order 2 (symmetric) cancels the leading error term.
    even, odd = groups(n)
    dt = t / r
    if order == 1:
        step = expm(-1j * odd * dt) @ expm(-1j * even * dt)
    else:
        half = expm(-1j * even * dt / 2)
        step = half @ expm(-1j * odd * dt) @ half
    return np.linalg.matrix_power(step, r)


def bond_gate(qc, a, b, angle):
    """Apply exp(-i angle (XX+YY+ZZ)) on qubits a,b.

    RXX(theta) = exp(-i theta/2 XX), so theta = 2*angle for each Pauli. The
    three rotations commute, so their order is irrelevant.
    """
    # RXX, RYY, RZZ commute on one bond so exp of the bond is exact.
    qc.rxx(2 * angle, a, b)
    qc.ryy(2 * angle, a, b)
    qc.rzz(2 * angle, a, b)


def layer(qc, n, parity, angle):
    """Apply exp(-i angle H_group) for the even (0) or odd (1) bond group."""
    for q in range(parity, n - 1, 2):
        bond_gate(qc, q, q + 1, J * angle)


def trotter_circuit(n, t, r, order, neel=False):
    """Circuit for r steps of the product formula (optionally from Neel)."""
    qc = QuantumCircuit(n)
    if neel:
        for q in range(1, n, 2):
            qc.x(q)
    dt = t / r
    for _ in range(r):
        if order == 1:
            layer(qc, n, 0, dt)
            layer(qc, n, 1, dt)
        else:
            layer(qc, n, 0, dt / 2)
            layer(qc, n, 1, dt)
            layer(qc, n, 0, dt / 2)
    return qc


def singlets(n):
    """Circuit preparing singlets (|01>-|10>)/sqrt2 on pairs (0,1),(2,3),..."""
    # Singlet start state lies in the Sz=0, S=0 sector of the ground state.
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


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_model(n, r):
    """Check 1: symmetries, exact energies, size scaling."""
    log.info("=== Check 1: the Heisenberg model ===")
    even, odd = groups(n)
    h = even + odd
    sz, s2 = total_sz(n), total_s2(n)
    r.append(
        check(
            np.allclose(h, heis_sparse(n).to_matrix()),
            "Kronecker build equals Qiskit SparsePauliOp",
        )
    )
    r.append(
        check(
            np.linalg.norm(h @ sz - sz @ h) < 1e-12,
            "[H, S_z] = 0: total S_z is conserved",
        )
    )
    r.append(
        check(
            np.linalg.norm(h @ s2 - s2 @ h) < 1e-10,
            "[H, S^2] = 0: total spin is conserved (SU(2) symmetry)",
        )
    )
    r.append(
        check(
            np.linalg.norm(even @ odd - odd @ even) > 1e-3,
            "even and odd bond groups do NOT commute (Trotter error)",
        )
    )
    # Bonds inside a group commute: the group exponential is exact.
    b0, b2 = bond(n, 0), (bond(n, 2) if n > 3 else bond(n, 0))
    r.append(
        check(
            np.linalg.norm(b0 @ b2 - b2 @ b0) < 1e-12, "bonds inside one group commute"
        )
    )
    # Two spins: singlet energy -3, triplet +1 (XX+YY+ZZ = 2 SWAP - 1).
    ev2 = np.linalg.eigvalsh(bond(2, 0))
    r.append(
        check(
            np.allclose(ev2, [-3, 1, 1, 1]),
            "two spins: singlet -3, triplet +1 (threefold)",
        )
    )
    w, v = np.linalg.eigh(h)
    gs = v[:, 0]
    e4 = np.linalg.eigvalsh(sum(bond(4, q) for q in range(3)))[0]
    r.append(
        check(
            abs(e4 - (-3 - 2 * math.sqrt(3))) < 1e-10,
            f"4-site open chain: E0 = -3 - 2 sqrt(3) = {e4:.6f}",
        )
    )
    r.append(
        check(
            abs(expval(s2, gs)) < 1e-9 and w[1] - w[0] > 1e-3,
            f"n={n}: ground state is a unique singlet (S^2=0), "
            f"gap {w[1] - w[0]:.4f}",
        )
    )
    # Energy per site for growing open chains approaches 1 - 4 ln 2 from above.
    e_inf = 1 - 4 * math.log(2)
    per = {}
    for m in (4, 6, 8, 10):
        hm = heis_sparse(m).to_matrix(sparse=True)
        from scipy.sparse.linalg import eigsh  # noqa: PLC0415

        per[m] = eigsh(hm, k=1, which="SA")[0][0] / m
    log.info(
        "E0/n for n=4,6,8,10: %s ; infinite chain %.4f",
        {m: round(float(x), 4) for m, x in per.items()},
        e_inf,
    )
    vals = list(per.values())
    r.append(
        check(
            all(a > b for a, b in zip(vals, vals[1:])) and all(x > e_inf for x in vals),
            "E0/n decreases with n toward 1 - 4 ln 2 (never below it)",
        )
    )
    return gs


def check_trotter(n, t, r):
    """Check 2 and 4: Trotter error, symmetry, Neel quench."""
    log.info("=== Check 2: Trotter error and symmetry (t=%.2f) ===", t)
    even, odd = groups(n)
    h = even + odd
    u = expm(-1j * h * t)
    steps = [8, 16, 32, 64, 128]
    err = {
        o: [np.linalg.norm(u - trotter_u(n, t, k, o), 2) for k in steps] for o in (1, 2)
    }
    for i, k in enumerate(steps):
        log.info("r=%3d: first order %.2e  second order %.2e", k, err[1][i], err[2][i])
    s1, s2 = slope(steps, err[1]), slope(steps, err[2])
    log.info("fitted slopes: first %.2f, second %.2f", s1, s2)
    r.append(
        check(
            abs(s1 + 1) < 0.15 and abs(s2 + 2) < 0.2,
            "error falls as 1/r (first order) and 1/r^2 (second)",
        )
    )
    # Even a deliberately crude single step keeps S_z exactly: each bond gate
    # commutes with S_z, so the whole circuit does.
    sz = total_sz(n)
    crude = trotter_u(n, t, 1, 1)
    psi = np.zeros(2**n, dtype=complex)
    psi[neel_index(n)] = 1
    out = crude @ psi
    var = expval(sz @ sz, out) - expval(sz, out) ** 2
    r.append(
        check(
            var < 1e-12 and abs(expval(sz, out)) < 1e-12,
            f"one crude step still conserves S_z exactly " f"(variance {var:.1e})",
        )
    )
    r.append(
        check(
            np.linalg.norm(crude @ sz - sz @ crude) < 1e-10,
            "Trotter unitary commutes with S_z (symmetry-preserving)",
        )
    )
    log.info("=== Check 4: quench from the Neel state ===")
    times = np.linspace(0, max(t, 1.5), 7)
    ex, tr, en = [], [], []
    for tau in times:
        pe = expm(-1j * h * tau) @ psi
        k = max(1, int(round(tau / 0.125)))
        pt = trotter_u(n, tau, k, 2) @ psi
        ex.append(stag_mag(pe, n))
        tr.append(stag_mag(pt, n))
        en.append(expval(h, pt))
    log.info("staggered M exact  : %s", " ".join(f"{x:+.3f}" for x in ex))
    log.info("staggered M Trotter: %s", " ".join(f"{x:+.3f}" for x in tr))
    dev = max(abs(a - b) for a, b in zip(ex, tr))
    r.append(
        check(
            abs(ex[0] - 1) < 1e-12 and min(ex) < 0.5, "Neel order starts at 1 and melts"
        )
    )
    r.append(
        check(
            dev < 0.03,
            f"second-order Trotter (dt=0.125) tracks it " f"(max error {dev:.4f})",
        )
    )
    r.append(
        check(
            max(en) - min(en) < 0.5,
            "Trotter energy stays bounded " f"(spread {max(en) - min(en):.4f})",
        )
    )


def check_circuits(n, t, r):
    """Check 3: Qiskit circuits and CNOT cost."""
    log.info("=== Check 3: Qiskit circuits and CNOT cost ===")
    k = 3
    for order in (1, 2):
        qc = trotter_circuit(n, t, k, order)
        diff = np.linalg.norm(Operator(qc).data - trotter_u(n, t, k, order), 2)
        r.append(
            check(
                diff < 1e-10,
                f"order {order}: circuit equals NumPy "
                f"product formula (diff {diff:.1e})",
            )
        )
    # One bond gate in isolation: exp(-i a (XX+YY+ZZ)) needs only 3 CNOTs.
    one = QuantumCircuit(2)
    bond_gate(one, 0, 1, 0.37)
    naive = (
        transpile(one, basis_gates=["cx", "rz", "sx", "x"], optimization_level=0)
        .count_ops()
        .get("cx", 0)
    )
    best = (
        transpile(
            one,
            basis_gates=["cx", "rz", "sx", "x"],
            optimization_level=3,
            seed_transpiler=1,
        )
        .count_ops()
        .get("cx", 0)
    )
    log.info("CNOTs for one bond gate: gate-by-gate %d, resynthesised %d", naive, best)
    r.append(
        check(
            naive == 6 and best == 3,
            "three commuting rotations (6 CNOTs) collapse to the "
            "3-CNOT optimum when two-qubit blocks are resynthesised",
        )
    )
    full = transpile(
        trotter_circuit(n, t, 1, 1),
        basis_gates=["cx", "rz", "sx", "x"],
        optimization_level=3,
        seed_transpiler=1,
    )
    r.append(
        check(
            full.count_ops().get("cx", 0) == 3 * (n - 1),
            f"one first-order step on n={n}: "
            f"{full.count_ops().get('cx', 0)} CNOTs = 3(n-1)",
        )
    )


def hva_energy(params, n, h, even_eig, odd_eig, psi0):
    """Energy of the HVA state, using precomputed eigendecompositions."""
    # Apply layers in order to the start state to get the trial state.
    psi = psi0
    for k in range(len(params) // 2):
        for (d, v), ang in ((odd_eig, params[2 * k]), (even_eig, params[2 * k + 1])):
            psi = v @ (np.exp(-1j * ang * d) * (v.conj().T @ psi))
    return expval(h, psi)


def check_hva(n, layers, seed, gs, r):
    """Check 5: variational ground-state preparation."""
    log.info("=== Check 5: ground state by the Hamiltonian Variational Ansatz ===")
    even, odd = groups(n)
    h = even + odd
    e0 = np.linalg.eigvalsh(h)[0]
    psi0 = Statevector.from_instruction(singlets(n)).data
    s2 = total_s2(n)
    r.append(
        check(abs(expval(s2, psi0)) < 1e-10, "singlet-product start state has S^2 = 0")
    )
    # The start state has energy -3 per singlet; the odd bonds contribute 0.
    r.append(
        check(
            abs(expval(h, psi0) + 1.5 * n) < 1e-10,
            f"dimer start energy {expval(h, psi0):.4f} = -3 n/2",
        )
    )
    eigs = []
    for mat in (odd, even):
        d, v = np.linalg.eigh(mat)
        eigs.append((d, v))
    rng = np.random.default_rng(seed)
    errs = []
    best_params = None
    for p in range(1, layers + 1):
        best = None
        # A few random restarts: the landscape has local minima.
        for _ in range(6):
            x0 = rng.uniform(-0.8, 0.8, 2 * p)
            res = minimize(
                hva_energy, x0, args=(n, h, eigs[1], eigs[0], psi0), method="BFGS"
            )
            if best is None or res.fun < best.fun:
                best = res
        errs.append(best.fun - e0)
        best_params = best.x
        log.info(
            "p=%d (%d angles): E = %.8f, error %.2e (E0 = %.8f)",
            p,
            2 * p,
            best.fun,
            best.fun - e0,
            e0,
        )
    r.append(
        check(
            all(e > -1e-9 for e in errs),
            "variational principle: no energy below the exact E0",
        )
    )
    r.append(
        check(
            all(b <= a + 1e-9 for a, b in zip(errs, errs[1:])),
            "energy error does not increase with more layers",
        )
    )
    # Small chains are solved exactly; for larger ones demand a clear gain
    # (error cut to a quarter of the one-layer error) instead of exactness.
    r.append(
        check(
            errs[-1] < max(1e-3, 0.25 * errs[0]),
            f"p={layers}: error {errs[-1]:.1e} vs {errs[0]:.1e} at p=1 "
            "(exact, or cut to a quarter or better)",
        )
    )
    # Qiskit cross-check of the best circuit.
    qc = hva_circuit(n, best_params)
    sv = Statevector.from_instruction(qc)
    e_qiskit = float(sv.expectation_value(heis_sparse(n)).real)
    r.append(
        check(
            abs(e_qiskit - (e0 + errs[-1])) < 1e-7,
            f"Qiskit circuit energy {e_qiskit:.8f} matches NumPy",
        )
    )
    ov = abs(np.vdot(gs, sv.data)) ** 2
    log.info("overlap of HVA state with the exact ground state: %.6f", ov)
    # Rigorous bound: E - E0 >= gap * (1 - overlap), so a small energy error
    # forces a large overlap with the unique ground state.
    gap = np.linalg.eigvalsh(h)[1] - e0
    r.append(
        check(
            ov >= 1 - errs[-1] / gap - 1e-9,
            f"overlap {ov:.6f} obeys the bound 1 - error/gap = "
            f"{1 - errs[-1] / gap:.6f}",
        )
    )
    # Symmetry: ANY angles keep the state a singlet with S_z = 0 and a
    # Hamiltonian-compatible structure, because every gate commutes with S^2.
    wild = rng.uniform(-3, 3, 2 * layers)
    sw = Statevector.from_instruction(hva_circuit(n, wild)).data
    r.append(
        check(
            abs(expval(s2, sw)) < 1e-9 and abs(expval(total_sz(n), sw)) < 1e-9,
            "random angles still give S^2 = 0 and S_z = 0 exactly",
        )
    )
    # The singlet sector (Catalan number) is tiny compared with 2^n, so a few
    # angles can already span much of it; once 2p reaches its size the fit
    # can become exact.
    log.info(
        "n=%d: %d angles vs a singlet sector of dimension %d (the full "
        "space has %d)",
        n,
        2 * layers,
        math.comb(n, n // 2) // (n // 2 + 1),
        2**n,
    )


def main(argv=None):
    """Run every Lab 16 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 16: Heisenberg chain")
    ap.add_argument("--qubits", type=int, default=6, help="even, 4..8")
    ap.add_argument("--time", type=float, default=1.0, help="evolution time")
    ap.add_argument("--layers", type=int, default=3, help="max HVA layers 1..5")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    if args.qubits % 2 or not 4 <= args.qubits <= 8:
        ap.error(f"--qubits must be even and 4..8. Got {args.qubits}.")
    if not 0.2 <= args.time <= 2.0:
        ap.error(f"--time must be 0.2..2.0. Got {args.time}.")
    if not 1 <= args.layers <= 5:
        ap.error(f"--layers must be 1..5. Got {args.layers}.")
    n, t = args.qubits, args.time
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    gs = check_model(n, r)
    check_trotter(n, t, r)
    check_circuits(n, t, r)
    check_hva(n, args.layers, args.seed, gs, r)
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
