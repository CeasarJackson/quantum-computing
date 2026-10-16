#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab23_sparse_heisenberg.py

Purpose
-------
Labs 16 and 20 used DENSE matrices, which stop at about 8 spins: a dense
Hamiltonian for n spins has 4^n entries (n = 14 would need about 4 GB). Lab 23
removes that wall with two standard tricks and pushes the Heisenberg ground
state preparation to 14 spins and beyond:

  1. SPARSE Hamiltonian + Lanczos (scipy eigsh) for the exact ground energy.
  2. A STATE-VECTOR simulator that applies each bond gate directly to the 2^n
     amplitudes (cost 2^n per gate, memory 2^n) and an ADJOINT gradient that
     gives all 2p derivatives of the energy for about the price of three
     circuit runs, instead of 2p+1 runs for finite differences.

With these, the warm-started Hamiltonian Variational Ansatz of Lab 20 can be
studied on longer chains, and its depth requirement measured, not guessed.

The lab verifies:
  1. The sparse Hamiltonian and the gate-by-gate simulator agree with the dense
     Lab 16/20 machinery at n = 8 (energies and states).
  2. The adjoint gradient equals a finite-difference gradient.
  3. Exact ground energy per site decreases with n toward 1 - 4 ln 2, and the
     gap closes (about 1/n) as the chain grows.
  4. Warm-started HVA error never rises with depth, at each size.
  5. The depth needed grows with the chain length (a light-cone effect), and
     total S_z stays exactly zero for any angles.
  6. Memory: dense versus sparse, in plain numbers.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab23_sparse_heisenberg.py
    python3 labs/lab23_sparse_heisenberg.py --max-qubits 16 --layers 10
    python3 labs/lab23_sparse_heisenberg.py --quick

Log files
---------
logs/lab23/lab23_<YYYYmmdd_HHMMSS>.log

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
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

from applog import get_logger

log = get_logger("lab23")

# The two-qubit Heisenberg term XX + YY + ZZ in the basis |00>,|01>,|10>,|11>.
BOND = np.array(
    [[1, 0, 0, 0], [0, -1, 2, 0], [0, 2, -1, 0], [0, 0, 0, 1]], dtype=complex
)
# Its eigenvectors split into the symmetric triplet states and the singlet.
BOND_W, BOND_V = np.linalg.eigh(BOND)


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


# ----------------------------------------------------------------------------
# Operators and gates on the state vector
# ----------------------------------------------------------------------------
def sparse_hamiltonian(n):
    """H = sum over bonds of XX+YY+ZZ as a scipy sparse matrix."""
    # Sum bond terms as sparse Kronecker products; memory is O(n 2^n).
    h = sp.csr_matrix((2**n, 2**n), dtype=complex)
    for q in range(n - 1):
        # Qubit q is the (q)th least significant bit, so the pair (q+1, q)
        # sits between identity blocks of size 2^(n-q-2) and 2^q.
        left = sp.identity(2 ** (n - q - 2), format="csr")
        right = sp.identity(2**q, format="csr")
        h = h + sp.kron(sp.kron(left, sp.csr_matrix(BOND)), right, format="csr")
    return h


def group_terms(n, parity):
    """Bond positions of one brick-wall group: parity 0 = even, 1 = odd."""
    return list(range(parity, n - 1, 2))


def apply_two_qubit(psi, mat, q, n):
    """Apply a 4x4 matrix to qubits (q+1, q) of an n-qubit state vector."""
    # View the state as (high, 4, low) so a two-qubit gate is one matmul.
    t = psi.reshape(2 ** (n - q - 2), 4, 2**q)
    return np.einsum("ab,ibj->iaj", mat, t).reshape(-1)


def bond_unitary(angle):
    """exp(-i angle (XX+YY+ZZ)) as a 4x4 matrix."""
    return (BOND_V * np.exp(-1j * angle * BOND_W)) @ BOND_V.conj().T


def apply_group(psi, n, parity, angle):
    """Apply exp(-i angle H_group): all bond gates of the group (they commute)."""
    # Build the bond unitary once per group, apply to every bond.
    u = bond_unitary(angle)
    for q in group_terms(n, parity):
        psi = apply_two_qubit(psi, u, q, n)
    return psi


def apply_group_h(psi, n, parity):
    """Apply H_group itself (sum of bond terms) to a state vector."""
    # Hamiltonian action by the same view trick, no matrix is stored.
    out = np.zeros_like(psi)
    for q in group_terms(n, parity):
        out += apply_two_qubit(psi, BOND, q, n)
    return out


def singlet_product(n):
    """State with a singlet on each pair (0,1), (2,3), ..."""
    # Singlet pairs on bonds (0,1), (2,3)... give a product start state.
    pair = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
    psi = np.array([1.0 + 0j])
    for _ in range(n // 2):
        psi = np.kron(pair, psi)
    return psi


# ----------------------------------------------------------------------------
# Energy and adjoint gradient
# ----------------------------------------------------------------------------
def layer_plan(params):
    """Ordered list of (parity, angle) steps: per layer odd first, then even."""
    steps = []
    for k in range(len(params) // 2):
        steps.append((1, params[2 * k]))
        steps.append((0, params[2 * k + 1]))
    return steps


def energy_and_grad(params, n, h, psi0):
    """Energy <H> of the HVA state and its exact gradient (adjoint method).

    Forward pass stores the state after every step. Backward pass carries
    |phi> = (later steps)^dagger H |psi_final>, and the derivative with respect
    to step k's angle is 2 Im <phi_k| H_k |psi_k>, where H_k is the step's group
    Hamiltonian (the generator of its rotation).
    """
    # Reverse sweep of the stored steps gives the exact gradient.
    steps = layer_plan(params)
    states = [psi0]
    for parity, angle in steps:
        states.append(apply_group(states[-1], n, parity, angle))
    final = states[-1]
    hpsi = h @ final
    energy = float(np.vdot(final, hpsi).real)
    grad = np.zeros(len(steps))
    phi = hpsi
    for k in range(len(steps) - 1, -1, -1):
        parity, angle = steps[k]
        hk_psi = apply_group_h(states[k + 1], n, parity)
        grad[k] = 2 * np.vdot(phi, hk_psi).imag
        # Undo this step on the adjoint state (apply U^dagger).
        phi = apply_group(phi, n, parity, -angle)
    return energy, grad


def optimise_depths(n, h, e0, max_layers, rng, restarts=1):
    """Warm-started best energy error for depth p = 1..max_layers."""
    # Warm start each depth from the previous optimum.
    psi0 = singlet_product(n)
    prev, out = np.zeros(0), []
    for p in range(1, max_layers + 1):
        starts = [np.concatenate([prev, np.zeros(2)])]
        starts += [rng.uniform(-0.8, 0.8, 2 * p) for _ in range(restarts)]
        best = None
        for x0 in starts:
            res = minimize(
                energy_and_grad,
                x0,
                args=(n, h, psi0),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 400},
            )
            if best is None or res.fun < best.fun:
                best = res
        prev = best.x
        out.append((best.fun - e0, best.x))
    return out


def ground_energy(h, k=2):
    """Lowest eigenvalues of the sparse Hamiltonian (Lanczos)."""
    # Lowest eigenvalue by Lanczos; fast for sparse matrices.
    w = eigsh(h, k=k, which="SA", return_eigenvectors=False)
    return np.sort(w.real)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_against_dense(r, rng):
    """Checks 1 and 2: agreement with dense linear algebra, and the gradient."""
    log.info("=== Check 1: sparse and gate-by-gate code versus dense (n=8) ===")
    n = 8
    # Dense reference built independently with Kronecker products.
    dense = np.zeros((2**n, 2**n), dtype=complex)
    for q in range(n - 1):
        dense += np.kron(np.kron(np.eye(2 ** (n - q - 2)), BOND), np.eye(2**q))
    h = sparse_hamiltonian(n)
    r.append(
        check(
            abs(h - sp.csr_matrix(dense)).max() < 1e-12,
            "sparse H equals the dense Kronecker-product H",
        )
    )
    w = np.linalg.eigvalsh(dense)
    ws = ground_energy(h)
    r.append(
        check(
            abs(w[0] - ws[0]) < 1e-9 and abs(w[1] - ws[1]) < 1e-9,
            f"Lanczos finds the same two lowest levels ({ws[0]:.6f}, " f"{ws[1]:.6f})",
        )
    )
    psi0 = singlet_product(n)
    r.append(
        check(
            abs(np.vdot(psi0, h @ psi0).real + 1.5 * n) < 1e-10,
            "singlet-product start has energy -3n/2",
        )
    )
    # Gate-by-gate evolution must equal multiplying by the dense exponential.
    from scipy.linalg import expm  # noqa: PLC0415

    angle = 0.37
    even = sum(
        np.kron(np.kron(np.eye(2 ** (n - q - 2)), BOND), np.eye(2**q))
        for q in group_terms(n, 0)
    )
    # Use one perturbed (non-symmetric) input for both the gate-by-gate code and
    # the dense matrix exponential, so the comparison is not a special case.
    base = psi0 + 0.1 * rng.normal(size=2**n)
    got = apply_group(base, n, 0, angle)
    ref = expm(-1j * angle * even) @ base
    r.append(
        check(
            np.linalg.norm(got - ref) < 1e-10,
            "bond-gate application equals the dense matrix exponential",
        )
    )
    log.info("=== Check 2: adjoint gradient versus finite differences ===")
    x = rng.uniform(-1, 1, 6)
    e, g = energy_and_grad(x, n, h, psi0)
    eps = 1e-6
    fd = np.array(
        [
            (
                energy_and_grad(x + eps * np.eye(6)[i], n, h, psi0)[0]
                - energy_and_grad(x - eps * np.eye(6)[i], n, h, psi0)[0]
            )
            / (2 * eps)
            for i in range(6)
        ]
    )
    r.append(
        check(
            np.max(np.abs(g - fd)) < 1e-6,
            "adjoint gradient equals "
            f"central differences (max diff {np.max(np.abs(g - fd)):.1e})",
        )
    )


def check_scaling(r, sizes):
    """Check 3: exact ground energy per site and the closing gap."""
    log.info("=== Check 3: exact ground state up to n=%d ===", sizes[-1])
    e_inf = 1 - 4 * math.log(2)
    per, gaps, info = [], [], {}
    for n in sizes:
        h = sparse_hamiltonian(n)
        t0 = time.perf_counter()
        w = ground_energy(h)
        per.append(w[0] / n)
        gaps.append(w[1] - w[0])
        info[n] = (h, w[0])
        log.info(
            "n=%2d: E0 = %10.6f, E0/n = %.5f, gap = %.4f (%.1f s)",
            n,
            w[0],
            w[0] / n,
            w[1] - w[0],
            time.perf_counter() - t0,
        )
    r.append(
        check(
            all(a > b for a, b in zip(per, per[1:])) and all(x > e_inf for x in per),
            f"E0/n falls toward 1 - 4 ln 2 = {e_inf:.4f} and stays above",
        )
    )
    r.append(
        check(
            all(a > b for a, b in zip(gaps, gaps[1:])),
            "the gap shrinks as the chain grows (gapless limit)",
        )
    )
    return info


def check_hva(r, sizes, info, layers, rng):
    """Checks 4 and 5: warm-started HVA across sizes."""
    log.info("=== Check 4: warm-started HVA on each size (up to %d layers) ===", layers)
    needed = {}
    best_params = {}
    for n in sizes:
        h, e0 = info[n]
        t0 = time.perf_counter()
        res = optimise_depths(n, h, e0, layers, rng)
        errs = [e for e, _ in res]
        best_params[n] = res[-1][1]
        log.info(
            "n=%2d: error by depth: %s (%.1f s)",
            n,
            " ".join(f"{e:.1e}" for e in errs),
            time.perf_counter() - t0,
        )
        r.append(
            check(
                all(b <= a + 1e-9 for a, b in zip(errs, errs[1:]))
                and errs[0] > -1e-9
                and min(errs) > -1e-9,
                f"n={n}: error never rises with depth and never goes "
                "below the exact ground energy",
            )
        )
        # Error per site below 1e-3: size-independent accuracy target.
        needed[n] = next((p for p, e in enumerate(errs, 1) if e / n < 1e-3), None)
    log.info("layers needed for error per site < 1e-3: %s (None = not reached)", needed)
    seq = [needed[n] if needed[n] is not None else math.inf for n in sizes]
    r.append(
        check(
            all(b >= a for a, b in zip(seq, seq[1:])),
            "longer chains need at least as many layers (light-cone)",
        )
    )
    log.info("=== Check 5: symmetry for random angles ===")
    n = sizes[-1]
    psi = singlet_product(n)
    for parity, angle in layer_plan(rng.uniform(-3, 3, 8)):
        psi = apply_group(psi, n, parity, angle)
    # Total S_z = (1/2) sum Z_i is diagonal: its expectation and variance are
    # cheap sums over the probability distribution.
    idx = np.arange(2**n)
    sz = 0.5 * (n - 2 * np.array([bin(i).count("1") for i in idx]))
    prob = np.abs(psi) ** 2
    mean = float(prob @ sz)
    var = float(prob @ (sz**2)) - mean**2
    r.append(
        check(
            abs(mean) < 1e-9 and var < 1e-9,
            f"n={n}: random angles keep S_z = 0 exactly "
            f"(mean {mean:.1e}, variance {var:.1e})",
        )
    )


def check_memory(r, sizes):
    """Check 6: why dense linear algebra stops where it does."""
    log.info("=== Check 6: memory, dense versus sparse ===")
    # Deduplicate so a size is never logged twice.
    for n in sorted({8, 12, sizes[-1], 20, 30}):
        dense = (4.0**n) * 16 / 2**30
        vec = (2.0**n) * 16 / 2**30
        log.info("n=%2d: dense H %.3g GB | state vector %.3g GB", n, dense, vec)
    r.append(
        check(
            4.0**14 * 16 / 2**30 > 3.9,
            "a dense 14-spin Hamiltonian needs more than 4 GB, while its "
            "state vector needs 0.00024 GB",
        )
    )


def main(argv=None):
    """Run every Lab 23 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 23: sparse Heisenberg chains")
    # Default 12 keeps the run near 40 s; 14 takes about 2.5 min (4x per +2).
    ap.add_argument("--max-qubits", type=int, default=12, help="even, 8..18")
    ap.add_argument("--layers", type=int, default=8, help="max HVA layers 2..12")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true", help="max 12 qubits, 6 layers")
    args = ap.parse_args(argv)
    if args.max_qubits % 2 or not 8 <= args.max_qubits <= 18:
        ap.error(f"--max-qubits must be even and 8..18. Got {args.max_qubits}.")
    if not 2 <= args.layers <= 12:
        ap.error(f"--layers must be 2..12. Got {args.layers}.")
    top, layers = (12, 6) if args.quick else (args.max_qubits, args.layers)
    rng = np.random.default_rng(args.seed)
    sizes = list(range(8, top + 1, 2))
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_against_dense(r, rng)
    info = check_scaling(r, sizes)
    check_hva(r, sizes, info, layers, rng)
    check_memory(r, sizes)
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
