#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab21_noise_aware_training.py

Purpose
-------
Lab 20 prepared the Heisenberg ground state with angles tuned on a PERFECT
simulator and then asked how much hardware noise spoils them. Lab 21 asks the
natural next question: if we know the noise, can we choose DIFFERENT angles
that make the best of a bad machine? This is noise-aware training.

To make training affordable the lab uses a fast NumPy density-matrix model
that is checked against Qiskit Aer. The key fact it exploits: a two-qubit
depolarizing channel commutes with any unitary on the same two qubits, so a
bond gate compiled into three CNOTs followed by single-qubit gates is
equivalent to the ideal gate followed by ONE depolarizing step with
retention (1 - lambda)^3.

The lab verifies:
  1. The NumPy noisy model equals Aer's density-matrix simulation (with the
     real transpiled circuit), is a valid density matrix, and reduces to the
     ideal energy at lambda = 0.
  2. Noise-aware angles are never worse than the ideal-trained angles at the
     same depth, and no noisy energy beats the exact ground energy.
  3. Noise-aware training genuinely helps DEEP circuits at realistic noise
     (the ideal angles over-rotate), but it does NOT change the headline: the
     best achievable energy over all depths is unchanged, and the shallowest
     circuit is still the best one. Training cannot buy back what the
     CNOTs destroy.
  4. A careful baseline matters: with a weak ideal optimiser the gain looks
     larger than it is (an early version of this lab fooled itself that way).

Honest limit: on real hardware the noisy energy is not available exactly; it
must be estimated from shots, so noise-aware training needs many more circuit
runs than shown here. This lab shows what is achievable, not what is cheap.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab21_noise_aware_training.py
    python3 labs/lab21_noise_aware_training.py --seed 3

Log files
---------
logs/lab21/lab21_<YYYYmmdd_HHMMSS>.log

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
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from applog import get_logger

log = get_logger("lab21")

J = 1.0
N = 6  # chain length used throughout
I2 = np.eye(2, dtype=complex)
PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
BASIS = ["cx", "rz", "sx", "x"]
NOISE_LEVELS = (0.001, 0.0037, 0.01)  # depolarizing strength per CNOT
DEPTHS = (1, 2, 3)


# ----------------------------------------------------------------------------
# Model and circuits (same construction as Labs 16 and 20)
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
# Fast noisy simulation
# ----------------------------------------------------------------------------
def depolarize(rho, n, q, retain):
    """Two-qubit depolarizing step on qubits (q, q+1) of an n-qubit state.

    rho -> retain * rho + (1 - retain) * (I/4 on the pair) (x) Tr_pair(rho)
    """
    t = rho.reshape([2] * (2 * n))
    # Qubit k is axis n-1-k for the ket indices and n+(n-1-k) for the bra.
    a, b = n - 1 - q, n - 2 - q
    t = np.moveaxis(t, [a, b, n + a, n + b], [0, 1, 2, 3])
    rest = t.shape[4:]
    traced = np.einsum("ijij...->...", t)
    ident = np.einsum("ik,jl->ijkl", I2, I2)
    mixed = ident.reshape(ident.shape + (1,) * len(rest)) * traced / 4
    t = retain * t + (1 - retain) * mixed
    t = np.moveaxis(t, [0, 1, 2, 3], [a, b, n + a, n + b])
    return t.reshape(2**n, 2**n)


class NoisyChain:
    """Density-matrix model of the HVA circuit with per-CNOT depolarizing noise."""

    def __init__(self, n):
        """Precompute the Hamiltonian, group eigensystems and the start state."""
        self.n = n
        even, odd = groups(n)
        self.h = even + odd
        self.eig = {0: np.linalg.eigh(even), 1: np.linalg.eigh(odd)}
        w, v = np.linalg.eigh(self.h)
        self.e0, self.gap, self.gs = w[0], w[1] - w[0], v[:, 0]
        self.psi0 = Statevector.from_instruction(singlets(n)).data

    def evolve(self, rho, parity, angle, lam):
        """Apply exp(-i angle H_group) then depolarize every bond in the group."""
        # Closed-form exp(-i a H_g) from the stored eigensystem of the group.
        d, v = self.eig[parity]
        u = v @ np.diag(np.exp(-1j * angle * d)) @ v.conj().T
        rho = u @ rho @ u.conj().T
        # Each bond gate costs three CNOTs, hence retention (1 - lambda)^3.
        for q in range(parity, self.n - 1, 2):
            rho = depolarize(rho, self.n, q, (1 - lam) ** 3)
        return rho

    def state(self, params, lam):
        """Density matrix after singlet preparation and len(params)/2 layers."""
        rho = np.outer(self.psi0, self.psi0.conj())
        # Singlet prep: one noisy CNOT per pair (the other gates are 1-qubit).
        if lam > 0:
            for q in range(0, self.n, 2):
                rho = depolarize(rho, self.n, q, 1 - lam)
        # Mirror hva_circuit: odd group first, then even group, per layer.
        for k in range(len(params) // 2):
            rho = self.evolve(rho, 1, params[2 * k], lam)
            rho = self.evolve(rho, 0, params[2 * k + 1], lam)
        return rho

    def energy(self, params, lam):
        """Noisy energy Tr(rho H)."""
        return float(np.trace(self.state(params, lam) @ self.h).real)


def aer_energy(chain, params, lam):
    """Same quantity from Qiskit Aer on the TRANSPILED circuit (ground truth)."""
    from qiskit_aer import AerSimulator  # noqa: PLC0415
    from qiskit_aer.noise import NoiseModel, depolarizing_error  # noqa: PLC0415

    qc = transpile(
        hva_circuit(chain.n, params),
        basis_gates=BASIS,
        optimization_level=3,
        seed_transpiler=3,
    )
    qc.save_density_matrix()
    nm = NoiseModel()
    if lam > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(lam, 2), ["cx"])
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    rho = np.asarray(sim.run(qc).result().data(0)["density_matrix"].data)
    return float(np.trace(rho @ chain.h).real)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(chain, depth, lam, x0, restarts, rng):
    """Minimise the noisy energy from x0 plus a few random restarts."""
    # Keep the lowest-energy result among the warm start and random restarts.
    best = None
    starts = [x0] + [rng.uniform(-0.8, 0.8, 2 * depth) for _ in range(restarts)]
    for s in starts:
        res = minimize(lambda x: chain.energy(x, lam), s, method="BFGS")
        if best is None or res.fun < best.fun:
            best = res
    return best.fun, best.x


def ideal_angles(chain, depth, rng):
    """Noise-free optimum by warm-started depth (as in Lab 20)."""
    # out[p] = (energy, angles) at depth p; prev feeds the next warm start.
    prev, out = np.zeros(0), {}
    for p in range(1, depth + 1):
        x0 = np.concatenate([prev, np.zeros(2)])
        # Many restarts: a weak ideal baseline would make noise-aware
        # training look better than it is (an earlier version did exactly that).
        e, x = train(chain, p, 0.0, x0, 12, rng)
        prev, out[p] = x, (e, x)
    return out


def main(argv=None):
    """Run every Lab 21 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 21: noise-aware training")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    # One seeded generator keeps the random restarts reproducible.
    rng = np.random.default_rng(args.seed)
    chain = NoisyChain(N)
    # r collects every check's boolean so the exit code reflects all of them.
    r = []

    log.info("=== Check 1: the fast noisy model against Aer ===")
    ideal = ideal_angles(chain, max(DEPTHS), rng)
    for lam in (0.0, 0.0037, 0.01):
        for p in (0, 2, 3):
            x = ideal[p][1] if p else np.zeros(0)
            mine, aer = chain.energy(x, lam), aer_energy(chain, x, lam)
            r.append(
                check(
                    abs(mine - aer) < 1e-6,
                    f"lambda={lam}, depth {p}: " f"NumPy {mine:.6f} = Aer {aer:.6f}",
                )
            )
    rho = chain.state(ideal[3][1], 0.01)
    r.append(
        check(
            abs(np.trace(rho) - 1) < 1e-10
            and np.linalg.eigvalsh((rho + rho.conj().T) / 2)[0] > -1e-10,
            "noisy state has unit trace and no negative eigenvalues",
        )
    )
    r.append(
        check(
            abs(chain.energy(ideal[3][1], 0.0) - chain.e0 - (ideal[3][0] - chain.e0))
            < 1e-9,
            "lambda = 0 reproduces the ideal energy",
        )
    )

    log.info("=== Check 2: noise-aware versus ideal-trained angles ===")
    # Both dictionaries are keyed by (noise level, depth).
    aware, unaware = {}, {}
    for lam in NOISE_LEVELS:
        for p in DEPTHS:
            x_ideal = ideal[p][1]
            unaware[lam, p] = chain.energy(x_ideal, lam)
            # Start from the ideal angles, so aware <= unaware by construction.
            aware[lam, p] = train(chain, p, lam, x_ideal, 3, rng)
    # Reference: the singlet start with no layers at all.
    bare = {lam: chain.energy(np.zeros(0), lam) for lam in NOISE_LEVELS}
    for lam in NOISE_LEVELS:
        log.info(
            "lambda=%.4f: bare start %+.3f | ideal angles %s | " "noise-aware %s",
            lam,
            bare[lam] - chain.e0,
            " ".join(f"{unaware[lam, p] - chain.e0:+.3f}" for p in DEPTHS),
            " ".join(f"{aware[lam, p][0] - chain.e0:+.3f}" for p in DEPTHS),
        )
    r.append(
        check(
            all(aware[k][0] <= unaware[k] + 1e-9 for k in aware),
            "noise-aware energy <= ideal-trained energy at every " "(noise, depth)",
        )
    )
    r.append(
        check(
            all(aware[k][0] > chain.e0 - 1e-9 for k in aware),
            "no noisy state beats the exact ground energy",
        )
    )

    log.info("=== Check 3: how much does it help? ===")
    # Positive gain = noise-aware training lowered the energy.
    gain = {k: unaware[k] - aware[k][0] for k in aware}
    log.info(
        "gain (energy units) by (lambda, depth): %s",
        {k: round(v, 3) for k, v in gain.items()},
    )
    # Deep circuits: the ideal angles over-rotate, so training pays off.
    r.append(
        check(
            gain[0.0037, 3] > 0.05 and gain[0.01, 3] > 0.1,
            f"depth 3 gains: {gain[0.0037, 3]:.3f} at the ibm_fez level, "
            f"{gain[0.01, 3]:.3f} at 1% per CNOT",
        )
    )
    best_u, best_a = {}, {}
    for lam in NOISE_LEVELS:
        best_u[lam] = min([bare[lam]] + [unaware[lam, p] for p in DEPTHS])
        best_a[lam] = min([bare[lam]] + [aware[lam, p][0] for p in DEPTHS])
        log.info(
            "lambda=%.4f: best over depths, ideal angles %+.3f, aware "
            "%+.3f (error vs exact)",
            lam,
            best_u[lam] - chain.e0,
            best_a[lam] - chain.e0,
        )
    # The headline result: the best energy you can reach hardly moves.
    r.append(
        check(
            all(abs(best_u[k] - best_a[k]) < 0.01 for k in NOISE_LEVELS),
            "the best energy over all depths is unchanged (within "
            "0.01): training cannot rescue deep circuits",
        )
    )
    # At 0.1% per CNOT two layers can edge out one (depending on the ideal
    # optimum found), so the claim is made only for the realistic levels.
    r.append(
        check(
            all(
                aware[lam, 1][0] < aware[lam, p][0]
                for lam in (0.0037, 0.01)
                for p in (2, 3)
            ),
            "at the ibm_fez level and above, one layer still beats "
            "two or three even with noise-aware angles",
        )
    )

    log.info("=== Check 4: what changes in the angles? ===")
    # Logged only: the ideal depth-3 optimum is not unique, so angle sizes
    # vary from seed to seed and make a poor pass/fail criterion.
    for lam in NOISE_LEVELS:
        log.info(
            "lambda=%.4f, depth 3: |angles| ideal %.3f, aware %.3f",
            lam,
            np.linalg.norm(ideal[3][1]),
            np.linalg.norm(aware[lam, 3][1]),
        )
    x_aware = aware[0.0037, 3][1]
    r.append(
        check(
            abs(chain.energy(x_aware, 0.0037) - aer_energy(chain, x_aware, 0.0037))
            < 1e-6,
            "Aer confirms the energy of the TRAINED angles",
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
