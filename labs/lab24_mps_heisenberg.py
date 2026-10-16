#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab24_mps_heisenberg.py

Purpose
-------
Lab 23 stored all 2^n amplitudes, which stops near n = 30 (16 GB) and was
already slow at n = 14. Lab 24 removes that wall for NEARLY-AREA-LAW states by
writing a matrix-product-state (MPS) simulator from scratch:

  * The state is a chain of small tensors A[i] with shape (left bond, 2, right
    bond). Memory is n * chi^2 numbers instead of 2^n, where chi is the "bond
    dimension" (the Schmidt rank we are willing to keep across each cut).
  * A two-qubit gate on neighbours is applied by merging two tensors, applying
    the 4x4 matrix, and splitting again with an SVD. Keeping only the chi
    largest singular values is the only approximation (the "truncation").
  * Keeping the tensors in MIXED-CANONICAL form (everything left of the
    "centre" is left-orthogonal, everything right of it right-orthogonal) makes
    the SVD truncation optimal and the energy a sum of local terms.

The Hamiltonian Variational Ansatz (HVA) of Labs 16, 20 and 23 is a brick-wall
circuit of nearest-neighbour gates, so it is exactly the circuit an MPS
handles well. Angles trained on a 12-spin chain with the exact simulator are
then reused on much longer chains, which tests whether the ansatz is local
(a bulk property independent of n).

The lab verifies:
  1. The MPS equals the state-vector simulator when nothing is truncated.
  2. Truncation error falls as chi grows and vanishes at the exact rank.
  3. Entanglement stays far below the volume-law maximum.
  4. Trained angles transfer: the energy per site is the same for n = 16...64.
  5. Cost grows linearly with n, so n = 128 runs in seconds.
  6. The bulk energy error falls with circuit depth, against a finite-size
     fit of the exact ground energy.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab24_mps_heisenberg.py
    python3 labs/lab24_mps_heisenberg.py --chi 128 --layers 8
    python3 labs/lab24_mps_heisenberg.py --quick

Log files
---------
logs/lab24/lab24_<YYYYmmdd_HHMMSS>.log

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

log = get_logger("lab24")

# The two-qubit Heisenberg term XX + YY + ZZ in the basis |00>,|01>,|10>,|11>.
# It is symmetric under swapping the two qubits, so bit-order conventions
# between the state vector (little endian) and the MPS (site 0 first) cancel.
BOND = np.array(
    [[1, 0, 0, 0], [0, -1, 2, 0], [0, 2, -1, 0], [0, 0, 0, 1]], dtype=complex
)
BOND_W, BOND_V = np.linalg.eigh(BOND)
# Infinite-chain ground energy per site of the Heisenberg model (Bethe ansatz).
E_INF = 1 - 4 * math.log(2)


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def bond_unitary(angle):
    """exp(-i angle (XX+YY+ZZ)) as a 4x4 matrix."""
    return (BOND_V * np.exp(-1j * angle * BOND_W)) @ BOND_V.conj().T


def group_terms(n, parity):
    """Bond positions of one brick-wall group: parity 0 = even, 1 = odd."""
    return list(range(parity, n - 1, 2))


def layer_plan(params):
    """Ordered (parity, angle) steps: per layer the odd group, then the even."""
    steps = []
    for k in range(len(params) // 2):
        steps.append((1, params[2 * k]))
        steps.append((0, params[2 * k + 1]))
    return steps


# ----------------------------------------------------------------------------
# Matrix product state
# ----------------------------------------------------------------------------
class MPS:
    """Open-boundary matrix product state of n spin-1/2 sites.

    Attributes
    ----------
    a        list of complex tensors, a[i].shape == (left, 2, right)
    centre   index of the orthogonality centre (the only non-isometric tensor)
    chi_max  largest bond dimension kept by any SVD
    cutoff   singular values below cutoff * (largest) are always dropped
    discarded  running total of the squared singular values thrown away
    """

    def __init__(self, tensors, centre, chi_max, cutoff=1e-12):
        """Store the tensors and the truncation settings."""
        # Tensors are (left, physical, right); site i is qubit i.
        self.a = tensors
        self.centre = centre
        self.chi_max = chi_max
        self.cutoff = cutoff
        self.discarded = 0.0

    @property
    def n(self):
        """Number of sites."""
        return len(self.a)

    @classmethod
    def singlet_product(cls, n, chi_max):
        """Singlets on the pairs (0,1), (2,3), ... as an exact MPS.

        A singlet is a rank-2 state, so an SVD splits it into two tensors that
        share a bond of dimension 2. Between pairs the bond dimension is 1.
        """
        # Each pair is a bond-dimension-1 pair of tensors made from a singlet SVD.
        pair = np.array([[0, 1], [-1, 0]], dtype=complex) / math.sqrt(2)
        u, s, vh = np.linalg.svd(pair)
        tensors = []
        for _ in range(n // 2):
            tensors.append(u.reshape(1, 2, 2))
            tensors.append((s[:, None] * vh).reshape(2, 2, 1))
        # Every tensor is left-orthogonal, so the centre is the last site.
        return cls(tensors, n - 1, chi_max)

    def copy(self):
        """Deep copy (the tensors are modified in place by gates)."""
        # Deep copy so a trial truncation cannot alter the original.
        new = MPS([t.copy() for t in self.a], self.centre, self.chi_max, self.cutoff)
        new.discarded = self.discarded
        return new

    def move_centre(self, target):
        """Shift the orthogonality centre with QR (no approximation)."""
        # QR sweeps move the orthogonality centre without changing the state.
        while self.centre < target:
            c = self.centre
            left, d, right = self.a[c].shape
            # QR makes site c left-orthogonal; R moves onto the next site.
            q, rr = np.linalg.qr(self.a[c].reshape(left * d, right))
            self.a[c] = q.reshape(left, d, -1)
            self.a[c + 1] = np.tensordot(rr, self.a[c + 1], axes=(1, 0))
            self.centre += 1
        while self.centre > target:
            c = self.centre
            left, d, right = self.a[c].shape
            # The mirror image: QR of the transpose makes site c right-orth.
            q, rr = np.linalg.qr(self.a[c].reshape(left, d * right).T)
            self.a[c] = q.T.reshape(-1, d, right)
            self.a[c - 1] = np.tensordot(self.a[c - 1], rr.T, axes=(2, 0))
            self.centre -= 1

    def apply_bond(self, i, mat, to):
        """Apply a 4x4 gate to sites (i, i+1); `to` is where the centre ends."""
        # The centre must sit on one of the two sites for the SVD to be optimal.
        if self.centre < i:
            self.move_centre(i)
        elif self.centre > i + 1:
            self.move_centre(i + 1)
        left, right = self.a[i].shape[0], self.a[i + 1].shape[2]
        theta = np.tensordot(self.a[i], self.a[i + 1], axes=(2, 0))
        theta = np.einsum("ab,lbr->lar", mat, theta.reshape(left, 4, right))
        u, s, vh = np.linalg.svd(
            theta.reshape(left * 2, 2 * right), full_matrices=False
        )
        keep = max(1, min(int(np.sum(s > self.cutoff * s[0])), self.chi_max))
        weights = s**2
        # Bookkeeping: the fraction of the state norm that truncation removed.
        self.discarded += float(weights[keep:].sum() / weights.sum())
        s = s[:keep] / np.linalg.norm(s[:keep])
        u, vh = u[:, :keep], vh[:keep]
        if to == "right":
            self.a[i] = u.reshape(left, 2, keep)
            self.a[i + 1] = (s[:, None] * vh).reshape(keep, 2, right)
            self.centre = i + 1
        else:
            self.a[i] = (u * s).reshape(left, 2, keep)
            self.a[i + 1] = vh.reshape(keep, 2, right)
            self.centre = i

    def apply_group(self, parity, angle):
        """Apply exp(-i angle H_group): every bond gate of one parity."""
        # Gates in a group touch disjoint bonds so order within it is free.
        bonds = group_terms(self.n, parity)
        if not bonds:
            return
        u = bond_unitary(angle)
        # Sweep away from the nearer end so the centre moves as little as it can.
        if self.centre <= self.n // 2:
            for q in bonds:
                self.apply_bond(q, u, "right")
        else:
            for q in reversed(bonds):
                self.apply_bond(q, u, "left")

    def energy(self):
        """<H> / <psi|psi> as a sum of local two-site expectation values."""
        # Centre at the left end so the norm is the local tensor norm.
        self.move_centre(0)
        total = 0.0
        for i in range(self.n - 1):
            theta = np.tensordot(self.a[i], self.a[i + 1], axes=(2, 0))
            left, right = theta.shape[0], theta.shape[3]
            flat = theta.reshape(left, 4, right)
            h_theta = np.einsum("ab,lbr->lar", BOND, flat)
            total += np.vdot(flat, h_theta).real / np.vdot(flat, flat).real
            if i < self.n - 2:
                self.move_centre(i + 1)
        return total

    def entropy(self, i):
        """Von Neumann entropy (nats) across the cut between sites i and i+1."""
        # With the centre at the cut, Schmidt values are the singular values.
        self.move_centre(i)
        left, d, right = self.a[i].shape
        s = np.linalg.svd(self.a[i].reshape(left * d, right), compute_uv=False)
        p = s**2 / np.sum(s**2)
        p = p[p > 1e-16]
        return float(-np.sum(p * np.log(p)))

    def bond_dims(self):
        """Bond dimension on each of the n-1 internal cuts."""
        # Bond dimensions show where the entanglement lives.
        return [t.shape[2] for t in self.a[:-1]]

    def to_vector(self):
        """Contract to the full 2^n vector (site q = bit q, little endian)."""
        # Contract left to right; only for small n as the vector is 2^n long.
        t = self.a[0][0]
        for tensor in self.a[1:]:
            t = np.tensordot(t, tensor, axes=(-1, 0))
        t = t.reshape((2,) * self.n)
        return t.transpose(range(self.n - 1, -1, -1)).reshape(-1)


def run_hva(n, params, chi):
    """Build the HVA state on n sites with bond-dimension limit chi."""
    mps = MPS.singlet_product(n, chi)
    for parity, angle in layer_plan(params):
        mps.apply_group(parity, angle)
    return mps


# ----------------------------------------------------------------------------
# Exact state-vector tools (used for training and as the reference)
# ----------------------------------------------------------------------------
def sparse_hamiltonian(n):
    """H = sum over bonds of XX+YY+ZZ as a scipy sparse matrix."""
    h = sp.csr_matrix((2**n, 2**n), dtype=complex)
    for q in range(n - 1):
        left = sp.identity(2 ** (n - q - 2), format="csr")
        right = sp.identity(2**q, format="csr")
        h = h + sp.kron(sp.kron(left, sp.csr_matrix(BOND)), right, format="csr")
    return h


def apply_two_qubit(psi, mat, q, n):
    """Apply a 4x4 matrix to qubits (q+1, q) of an n-qubit state vector."""
    t = psi.reshape(2 ** (n - q - 2), 4, 2**q)
    return np.einsum("ab,ibj->iaj", mat, t).reshape(-1)


def apply_group(psi, n, parity, angle):
    """State-vector version of MPS.apply_group."""
    u = bond_unitary(angle)
    for q in group_terms(n, parity):
        psi = apply_two_qubit(psi, u, q, n)
    return psi


def apply_group_h(psi, n, parity):
    """Apply the group Hamiltonian (sum of its bond terms) to a vector."""
    out = np.zeros_like(psi)
    for q in group_terms(n, parity):
        out += apply_two_qubit(psi, BOND, q, n)
    return out


def singlet_vector(n):
    """Singlet-product start state as a full vector."""
    pair = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
    psi = np.array([1.0 + 0j])
    for _ in range(n // 2):
        psi = np.kron(pair, psi)
    return psi


def energy_and_grad(params, n, h, psi0):
    """HVA energy and its exact adjoint gradient (see Lab 23 for the proof)."""
    # Reverse sweep reusing the stored steps gives the exact gradient.
    steps = layer_plan(params)
    states = [psi0]
    for parity, angle in steps:
        states.append(apply_group(states[-1], n, parity, angle))
    hpsi = h @ states[-1]
    energy = float(np.vdot(states[-1], hpsi).real)
    grad = np.zeros(len(steps))
    phi = hpsi
    for k in range(len(steps) - 1, -1, -1):
        parity, angle = steps[k]
        grad[k] = 2 * np.vdot(phi, apply_group_h(states[k + 1], n, parity)).imag
        phi = apply_group(phi, n, parity, -angle)
    return energy, grad


def train_angles(n, h, e0, max_layers, rng):
    """Warm-started L-BFGS training for depths 1..max_layers on n spins."""
    # Warm start each depth from the previous optimum.
    psi0 = singlet_vector(n)
    prev, out = np.zeros(0), []
    for p in range(1, max_layers + 1):
        # Start from the previous depth (extra layer at zero = identity) plus
        # perturbed and random points; keep whichever ends lowest.
        warm = np.concatenate([prev, np.zeros(2)])
        # The landscape has many plateaus: a jiggled warm start and two random
        # starts escape them far more often than one random start does.
        starts = [warm, warm + rng.normal(0, 0.1, 2 * p)]
        starts += [rng.uniform(-0.8, 0.8, 2 * p) for _ in range(2)]
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


def exact_ground(n):
    """Lowest eigenvalue of the n-site Hamiltonian (sparse Lanczos)."""
    # Sparse ground state is the reference for the exact-match checks.
    h = sparse_hamiltonian(n)
    return h, float(eigsh(h, k=1, which="SA", return_eigenvectors=False)[0].real)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_exact_match(r, rng):
    """Check 1: with no truncation the MPS is the state-vector simulation."""
    log.info("=== Check 1: MPS versus the state-vector simulator (n=10) ===")
    n = 10
    # The start state must match as a full vector, qubit for qubit.
    # (A singlet flips sign under qubit exchange, so the two codes may differ by
    # a global phase of -1 per pair; physics only sees the overlap's size.)
    start = MPS.singlet_product(n, 64).to_vector()
    r.append(
        check(
            abs(abs(np.vdot(start, singlet_vector(n))) - 1) < 1e-12,
            "singlet-product MPS equals the state vector (up to phase)",
        )
    )
    # Moving the centre around must not change the state.
    mps = MPS.singlet_product(n, 64)
    mps.move_centre(0)
    mps.move_centre(n - 1)
    moved = mps.to_vector()
    r.append(
        check(
            np.linalg.norm(moved - start) < 1e-12,
            "moving the orthogonality centre leaves the state unchanged",
        )
    )
    h = sparse_hamiltonian(n)
    params = rng.uniform(-1, 1, 8)
    psi = singlet_vector(n)
    for parity, angle in layer_plan(params):
        psi = apply_group(psi, n, parity, angle)
    # chi = 2^(n/2) = 32 is the largest rank any cut can have: no truncation.
    mps = run_hva(n, params, 32)
    overlap = abs(np.vdot(psi, mps.to_vector()))
    r.append(
        check(
            abs(overlap - 1) < 1e-10 and mps.discarded < 1e-20,
            f"4 random layers: overlap {overlap:.12f}, nothing discarded",
        )
    )
    e_vec = float(np.vdot(psi, h @ psi).real)
    e_mps = mps.energy()
    r.append(
        check(
            abs(e_vec - e_mps) < 1e-10,
            f"local energy sum equals <psi|H|psi> ({e_mps:.8f})",
        )
    )


def check_truncation(r, ctx):
    """Check 2: truncation error versus bond dimension at n=16."""
    n = ctx["trunc_n"]
    log.info("=== Check 2: truncation error versus chi (n=%d) ===", n)
    h, e0 = ctx["exact"][n]
    params = ctx["angles"][-1]
    exact = energy_and_grad(params, n, h, singlet_vector(n))[0]
    errs, lost, rows = [], [], []
    top = 2 ** (n // 2)
    for chi in [c for c in (2, 4, 8, 16, 32, 64, 128, 256) if c <= top]:
        mps = run_hva(n, params, chi)
        errs.append(abs(mps.energy() - exact))
        lost.append(mps.discarded)
        rows.append(f"chi={chi}: |dE|={errs[-1]:.1e}, discarded={mps.discarded:.1e}")
    for row in rows:
        log.info(row)
    # The discarded weight must fall monotonically. The energy error usually
    # does too, but a small chi can get lucky by cancellation, so for the energy
    # we test the trend (chi = 8 to chi = 64) rather than every step.
    r.append(
        check(
            all(b <= a + 1e-12 for a, b in zip(lost, lost[1:])),
            "discarded weight never rises as chi grows",
        )
    )
    chis = [c for c in (2, 4, 8, 16, 32, 64, 128, 256) if c <= top]
    r.append(
        check(
            errs[chis.index(top // 4)] < errs[chis.index(8)] / 10,
            f"energy error drops by more than 10x from chi=8 to " f"chi={top // 4}",
        )
    )
    r.append(
        check(errs[-1] < 1e-9, f"chi = 2^(n/2) = {top} reproduces the exact energy")
    )
    r.append(
        check(
            errs[-3] < 1e-3,
            f"chi = {top // 4} is within 1e-3 of exact ({errs[-3]:.1e})",
        )
    )


def check_entanglement(r, ctx):
    """Check 3: the entanglement of the trained state is far below volume law."""
    n = 24
    log.info("=== Check 3: entanglement of the trained state (n=%d) ===", n)
    params = ctx["angles"][-1]
    mps = MPS.singlet_product(n, ctx["chi"])
    entropies = []
    for k, (parity, angle) in enumerate(layer_plan(params), 1):
        mps.apply_group(parity, angle)
        if k % 2 == 0:
            entropies.append(mps.entropy(n // 2 - 1))
    log.info(
        "half-chain entropy after each layer: %s",
        " ".join(f"{s:.2f}" for s in entropies),
    )
    max_bond = max(mps.bond_dims())
    s_max = (n // 2) * math.log(2)
    log.info(
        "largest bond dimension %d of a possible %d; final S = %.3f of " "max %.3f",
        max_bond,
        2 ** (n // 2),
        entropies[-1],
        s_max,
    )
    r.append(
        check(
            entropies[-1] < 0.5 * s_max,
            "final half-chain entropy is below half the volume-law bound",
        )
    )
    # Each layer holds one gate across a cut per group, so a cut's Schmidt rank
    # can grow by at most 4 per gate: S(k layers) <= ln(2 * 4^(2k)).
    r.append(
        check(
            all(
                s <= math.log(2) + 2 * k * math.log(4) + 1e-9
                for k, s in enumerate(entropies, 1)
            ),
            "entropy respects the light-cone bound ln(2) + 2k ln 4",
        )
    )
    r.append(
        check(
            max_bond <= ctx["chi"],
            f"bond dimension stays within the chi limit ({ctx['chi']})",
        )
    )


def check_transfer(r, ctx):
    """Check 4: angles trained at n=12 give the same energy density at any n."""
    log.info("=== Check 4: transfer of the trained angles to longer chains ===")
    params = ctx["angles"][-1]
    energies = {}
    for n in ctx["transfer_sizes"]:
        t0 = time.perf_counter()
        mps = run_hva(n, params, ctx["chi"])
        e = mps.energy()
        energies[n] = e
        extra = ""
        if n in ctx["exact"]:
            h, _ = ctx["exact"][n]
            ev = energy_and_grad(params, n, h, singlet_vector(n))[0]
            extra = f" | state vector {ev / n:+.6f}"
            if n <= ctx["chi_exact_limit"]:
                r.append(
                    check(
                        abs(ev - e) < 1e-3,
                        f"n={n}: MPS within 1e-3 of the state vector",
                    )
                )
        log.info(
            "n=%3d: E/n = %+.6f (discarded %.1e, %.1f s)%s",
            n,
            e / n,
            mps.discarded,
            time.perf_counter() - t0,
            extra,
        )
    # E/n contains a surface term that falls as 1/n, so compare the BULK energy
    # density: the slope of E against n between neighbouring sizes.
    sizes = sorted(energies)
    slopes = [(energies[b] - energies[a]) / (b - a) for a, b in zip(sizes, sizes[1:])]
    log.info(
        "bulk slopes dE/dn between sizes: %s",
        " ".join(f"{a}-{b}: {s:+.5f}" for a, b, s in zip(sizes, sizes[1:], slopes)),
    )
    tail = slopes[-3:]
    r.append(
        check(
            max(tail) - min(tail) < 1e-4,
            f"bulk energy density agrees for the longest chains "
            f"(spread {max(tail) - min(tail):.1e})",
        )
    )
    r.append(
        check(
            all(v > E_INF for v in slopes[1:]),
            f"every bulk slope lies above the infinite-chain {E_INF:.5f}",
        )
    )
    ctx["bulk_slope"] = slopes[-1]


def check_scaling_cost(r, ctx):
    """Check 5: wall time grows linearly with n at fixed chi."""
    log.info("=== Check 5: cost versus chain length (chi=%d) ===", ctx["chi"])
    params = ctx["angles"][-1]
    times = {}
    for n in ctx["cost_sizes"]:
        t0 = time.perf_counter()
        mps = run_hva(n, params, ctx["chi"])
        mps.energy()
        times[n] = time.perf_counter() - t0
        log.info("n=%3d: %.2f s (%.1f ms per site)", n, times[n], 1000 * times[n] / n)
    sizes = ctx["cost_sizes"]
    # Linear scaling means a constant cost per site; short chains have not yet
    # reached the saturated bond dimension, so compare only the longest two.
    ratio = (times[sizes[-1]] / sizes[-1]) / (times[sizes[-2]] / sizes[-2])
    r.append(
        check(
            ratio < 2.0,
            f"doubling n from {sizes[-2]} to {sizes[-1]} changes the cost "
            f"per site by x{ratio:.2f} (linear total = x1; dense = "
            f"x2^{sizes[-2]})",
        )
    )
    mb = sum(t.nbytes for t in mps.a) / 2**20
    r.append(
        check(
            mb < 50,
            f"n={sizes[-1]}: the MPS needs {mb:.2f} MB; a full vector "
            f"would need 2^{sizes[-1]} amplitudes",
        )
    )


def check_depth_bulk(r, ctx):
    """Check 6: bulk energy error per site versus depth, against a fit."""
    log.info("=== Check 6: bulk error versus depth (fit of exact energies) ===")
    sizes = sorted(n for n in ctx["exact"] if n >= 8)
    energies = [ctx["exact"][n][1] for n in sizes]
    # Open chain: E0(n) = e*n + b + c/n (bulk, surface and Casimir-like terms).
    mat = np.array([[n, 1.0, 1.0 / n] for n in sizes])
    (e_bulk, surf, casimir), *_ = np.linalg.lstsq(mat, energies, rcond=None)
    log.info(
        "fit E0(n) = %.5f n %+.5f %+.4f/n (Bethe: %.5f n)", e_bulk, surf, casimir, E_INF
    )
    r.append(
        check(
            abs(e_bulk - E_INF) < 3e-3,
            "fitted bulk energy matches 1 - 4 ln 2 within 3e-3",
        )
    )
    # Bulk error of depth p: slope of E between two long chains, minus the
    # fitted bulk density (the surface term cancels in the difference).
    n1, n2 = ctx["transfer_sizes"][-2:]
    errs = []
    for params in ctx["angles"]:
        e1 = run_hva(n1, params, ctx["chi"]).energy()
        e2 = run_hva(n2, params, ctx["chi"]).energy()
        errs.append((e2 - e1) / (n2 - n1) - e_bulk)
    log.info(
        "bulk error per site (n=%d..%d) by depth: %s",
        n1,
        n2,
        " ".join(f"{e:.1e}" for e in errs),
    )
    r.append(
        check(
            all(b <= a + 1e-3 for a, b in zip(errs, errs[1:])),
            "bulk error per site falls (within 1e-3) as depth grows",
        )
    )
    r.append(
        check(
            errs[-1] < errs[0] / 2,
            f"{len(errs)} layers cut the bulk error by more than 2x",
        )
    )


def main(argv=None):
    """Run every Lab 24 check; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 24: MPS Heisenberg chains")
    ap.add_argument("--chi", type=int, default=64, help="bond dimension 8..512")
    ap.add_argument("--layers", type=int, default=6, help="HVA layers 2..10")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true", help="smaller, faster run")
    args = ap.parse_args(argv)
    if not 8 <= args.chi <= 512:
        ap.error(f"--chi must be 8..512. Got {args.chi}.")
    if not 2 <= args.layers <= 10:
        ap.error(f"--layers must be 2..10. Got {args.layers}.")
    rng = np.random.default_rng(args.seed)
    quick = args.quick
    train_n, layers = (10, min(args.layers, 4)) if quick else (12, args.layers)
    exact_top = 14 if quick else 18
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_exact_match(r, rng)
    log.info(
        "--- exact data: ground energies up to n=%d, training at n=%d ---",
        exact_top,
        train_n,
    )
    exact = {}
    for n in range(8, exact_top + 1, 2):
        exact[n] = exact_ground(n)
    h, e0 = exact[train_n]
    trained = train_angles(train_n, h, e0, layers, rng)
    log.info(
        "training errors at n=%d: %s", train_n, " ".join(f"{e:.1e}" for e, _ in trained)
    )
    ctx = {
        "exact": exact,
        "angles": [x for _, x in trained],
        "chi": args.chi,
        "transfer_sizes": [12, 16, 24, 32] if quick else [12, 16, 24, 32, 48, 64],
        "cost_sizes": [32, 64, 128],
        "chi_exact_limit": 16,
        "trunc_n": 12 if quick else 16,
    }
    check_truncation(r, ctx)
    check_entanglement(r, ctx)
    check_transfer(r, ctx)
    check_scaling_cost(r, ctx)
    check_depth_bulk(r, ctx)
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
