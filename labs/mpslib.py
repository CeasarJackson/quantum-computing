#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/mpslib.py

Purpose
-------
Shared matrix-product-state (MPS) toolkit for Labs 25 and later. Lab 24 built
the MPS circuit simulator inside one file; Labs 25 and 26 need the same
machinery plus a DMRG ground-state solver, so it lives here once:

  * MPS            : tensors, QR centre moves, two-site gates with SVD
                     truncation, local energies, entropy, full-vector export.
  * overlap        : <a|b> for two MPS by a left-to-right contraction.
  * heisenberg_mpo : the XXX Hamiltonian as a bond-dimension-5 MPO.
  * DMRG           : two-site density-matrix renormalisation group.
  * hva helpers    : bond gates, brick-wall layer plan, circuit runner.
  * state-vector   : sparse H, adjoint-gradient HVA training for n up to ~14.

Conventions: site i of the MPS is qubit i; every tensor has shape
(left bond, 2, right bond); the Hamiltonian is H = sum_i (XX + YY + ZZ) on
neighbours (Pauli matrices, so the spin-1/2 coupling is 4 S.S).

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    from mpslib import MPS, DMRG, run_hva, overlap
(Logging stays in the labs: a lab creates its logger with
`from applog import get_logger`; this library only returns numbers.)

Log files
---------
None. This is a library module.

Exit Codes
----------
Library module; it never exits the process.
===============================================================================
"""

import math

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

# The two-qubit Heisenberg term XX + YY + ZZ in the basis |00>,|01>,|10>,|11>.
# It is symmetric under exchanging the two qubits, so bit-order conventions
# between a state vector and the MPS cancel.
BOND = np.array(
    [[1, 0, 0, 0], [0, -1, 2, 0], [0, 2, -1, 0], [0, 0, 0, 1]], dtype=complex
)
BOND_W, BOND_V = np.linalg.eigh(BOND)
# Infinite-chain ground energy per site of the Heisenberg model (Bethe ansatz).
E_INF = 1 - 4 * math.log(2)

# Pauli matrices for the MPO.
PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


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
    a          list of complex tensors, a[i].shape == (left, 2, right)
    centre     index of the orthogonality centre (the only non-isometric site)
    chi_max    largest bond dimension kept by any SVD
    cutoff     singular values below cutoff * (largest) are always dropped
    discarded  running total of the squared singular values thrown away
    """

    def __init__(self, tensors, centre, chi_max, cutoff=1e-12):
        """Store the tensors and the truncation settings."""
        self.a = tensors
        self.centre = centre
        self.chi_max = chi_max
        self.cutoff = cutoff
        self.discarded = 0.0
        # When True the state is NOT renormalised after a truncation: its norm
        # lives in the centre tensor. The adjoint gradient needs this for the
        # un-normalised state H|psi>.
        self.track_norm = False

    @property
    def n(self):
        """Number of sites."""
        return len(self.a)

    @classmethod
    def singlet_product(cls, n, chi_max):
        """Singlets on the pairs (0,1), (2,3), ... as an exact MPS."""
        pair = np.array([[0, 1], [-1, 0]], dtype=complex) / math.sqrt(2)
        u, s, vh = np.linalg.svd(pair)
        tensors = []
        for _ in range(n // 2):
            tensors.append(u.reshape(1, 2, 2))
            tensors.append((s[:, None] * vh).reshape(2, 2, 1))
        # Every tensor is left-orthogonal, so the centre is the last site.
        return cls(tensors, n - 1, chi_max)

    @classmethod
    def neel(cls, n, chi_max):
        """Product state up, down, up, down, ... (bond dimension 1)."""
        tensors = []
        for i in range(n):
            t = np.zeros((1, 2, 1), dtype=complex)
            t[0, i % 2, 0] = 1.0
            tensors.append(t)
        return cls(tensors, n - 1, chi_max)

    def copy(self):
        """Deep copy (gates modify the tensors in place)."""
        new = MPS([t.copy() for t in self.a], self.centre, self.chi_max, self.cutoff)
        new.discarded = self.discarded
        new.track_norm = self.track_norm
        return new

    def move_centre(self, target):
        """Shift the orthogonality centre with QR (no approximation)."""
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
        s = s[:keep] if self.track_norm else s[:keep] / np.linalg.norm(s[:keep])
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

    def bond_energy(self, i):
        """Expectation of the bond term XX+YY+ZZ on sites (i, i+1)."""
        # With the centre on site i or i+1 the rest of the chain is an
        # isometry, so the two-site tensor alone gives the expectation value.
        if self.centre < i:
            self.move_centre(i)
        elif self.centre > i + 1:
            self.move_centre(i + 1)
        theta = np.tensordot(self.a[i], self.a[i + 1], axes=(2, 0))
        left, right = theta.shape[0], theta.shape[3]
        flat = theta.reshape(left, 4, right)
        h_theta = np.einsum("ab,lbr->lar", BOND, flat)
        return float(np.vdot(flat, h_theta).real / np.vdot(flat, flat).real)

    def bond_energies(self):
        """All n-1 bond energies, sweeping the centre once from the left."""
        self.move_centre(0)
        return [self.bond_energy(i) for i in range(self.n - 1)]

    def energy(self):
        """<H> / <psi|psi> as the sum of all bond energies."""
        return float(sum(self.bond_energies()))

    def entropy(self, i):
        """Von Neumann entropy (nats) across the cut between sites i and i+1."""
        self.move_centre(i)
        left, d, right = self.a[i].shape
        s = np.linalg.svd(self.a[i].reshape(left * d, right), compute_uv=False)
        p = s**2 / np.sum(s**2)
        p = p[p > 1e-16]
        return float(-np.sum(p * np.log(p)))

    def bond_dims(self):
        """Bond dimension on each of the n-1 internal cuts."""
        return [t.shape[2] for t in self.a[:-1]]

    def norm(self):
        """Euclidean norm of the (possibly un-normalised) state."""
        return math.sqrt(overlap(self, self).real)

    def compress(self, chi_max=None, cutoff=None):
        """Re-canonicalise arbitrary tensors and truncate them (norm kept).

        Used after applying an MPO, which multiplies every bond dimension by
        the MPO's. First a QR sweep makes everything left-orthogonal (exact),
        then an SVD sweep from the right drops small singular values. The
        centre ends on site 0 and carries the norm.
        """
        chi_max = self.chi_max if chi_max is None else chi_max
        cutoff = self.cutoff if cutoff is None else cutoff
        self.centre = 0
        self.move_centre(self.n - 1)
        for i in range(self.n - 1, 0, -1):
            left, d, right = self.a[i].shape
            u, s, vh = np.linalg.svd(
                self.a[i].reshape(left, d * right), full_matrices=False
            )
            keep = max(1, min(int(np.sum(s > cutoff * s[0])), chi_max))
            self.discarded += float((s[keep:] ** 2).sum() / (s**2).sum())
            self.a[i] = vh[:keep].reshape(keep, d, right)
            carry = u[:, :keep] * s[:keep]
            self.a[i - 1] = np.tensordot(self.a[i - 1], carry, axes=(2, 0))
            self.centre = i - 1

    def to_vector(self):
        """Contract to the full 2^n vector (site q = bit q, little endian)."""
        t = self.a[0][0]
        for tensor in self.a[1:]:
            t = np.tensordot(t, tensor, axes=(-1, 0))
        t = t.reshape((2,) * self.n)
        return t.transpose(range(self.n - 1, -1, -1)).reshape(-1)


def overlap(a, b):
    """<a|b> for two MPS of equal length (transfer-matrix contraction)."""
    env = np.ones((1, 1), dtype=complex)
    for ta, tb in zip(a.a, b.a):
        # env[x, y]: x runs over a's bond (bra), y over b's bond (ket).
        env = np.einsum("xy,xsz,ysw->zw", env, ta.conj(), tb)
    return complex(env[0, 0])


def apply_mpo(mps, mpo, chi_max, cutoff=1e-12):
    """Return the MPS for (MPO)|mps>, compressed to bond dimension chi_max.

    Each new tensor B[(p,a), t, (q,b)] = sum_s W[p,q,t,s] A[a,s,b], so every
    bond dimension is multiplied by the MPO bond (5 for the Heisenberg MPO).
    The result is NOT normalised: its norm is that of H|psi>.
    """
    tensors = []
    for a, w in zip(mps.a, mpo):
        # Output order (p, a, t, q, b) puts the physical index in the middle
        # so the two bond pairs merge into single (MPO x MPS) bonds.
        b = np.einsum("pqts,asb->patqb", w, a)
        wl, al, d, wr, ar = b.shape
        tensors.append(b.reshape(wl * al, d, wr * ar))
    out = MPS(tensors, 0, chi_max, cutoff)
    out.track_norm = True
    out.compress()
    return out


def bond_overlaps(phi, psi, bonds, mat):
    """List of <phi| mat_(q,q+1) |psi> for each bond q (no normalisation).

    Left and right environments are built once (cost n chi^3); every bond then
    needs only a two-site contraction. Used by the adjoint gradient.
    """
    n = psi.n
    left = [np.ones((1, 1), dtype=complex)]
    for ta, tb in zip(phi.a, psi.a):
        # left[i][x, y]: x = phi (bra) bond, y = psi (ket) bond.
        left.append(np.einsum("xy,xsz,ysw->zw", left[-1], ta.conj(), tb, optimize=True))
    right = [None] * (n + 1)
    right[n] = np.ones((1, 1), dtype=complex)
    for i in range(n - 1, -1, -1):
        right[i] = np.einsum(
            "zw,xsz,ysw->xy", right[i + 1], phi.a[i].conj(), psi.a[i], optimize=True
        )
    values = []
    for q in bonds:
        th_psi = np.tensordot(psi.a[q], psi.a[q + 1], axes=(2, 0))
        th_phi = np.tensordot(phi.a[q], phi.a[q + 1], axes=(2, 0))
        y, _, _, w = th_psi.shape
        flat = np.einsum("ab,ybw->yaw", mat, th_psi.reshape(y, 4, w))
        t = np.tensordot(left[q], flat, axes=(1, 0))  # x a w
        t = np.tensordot(t, right[q + 2], axes=(2, 1))  # x a z
        values.append(complex(np.vdot(th_phi.reshape(t.shape), t)))
    return values


def mps_energy_and_adjoint_grad(params, n, chi, chi_phi=None, cutoff=1e-10):
    """HVA energy on an MPS and its gradient by the ADJOINT method.

    dE/dtheta_k = 2 Im <phi_k| G_k |psi_k>, with psi_k the state after step k,
    G_k the step's group Hamiltonian and phi_k = (later steps)^dagger H psi_f.
    One forward pass (states kept), one MPO application, one backward pass: a
    cost independent of the number of parameters, unlike finite differences.
    """
    # H|psi> needs more room than |psi> itself, hence the larger default.
    chi_phi = chi_phi or 2 * chi
    steps = layer_plan(params)
    start = MPS.singlet_product(n, chi)
    start.cutoff = cutoff
    states = [start]
    for parity, angle in steps:
        nxt = states[-1].copy()
        nxt.apply_group(parity, angle)
        states.append(nxt)
    # energy() moves the centre, so use a copy and keep the cached state.
    final = states[-1].copy()
    energy = final.energy()
    hpsi = apply_mpo(states[-1].copy(), heisenberg_mpo(n), chi_phi, cutoff)
    # Backward pass: phi starts as H|psi_final> and is un-rotated step by step.
    phi = hpsi
    phi.chi_max = chi_phi
    grad = np.zeros(len(steps))
    for k in range(len(steps) - 1, -1, -1):
        parity, angle = steps[k]
        vals = bond_overlaps(phi, states[k + 1], group_terms(n, parity), BOND)
        grad[k] = 2 * float(np.imag(sum(vals)))
        phi.apply_group(parity, -angle)
    return energy, grad


def run_hva(n, params, chi):
    """Build the HVA state on n sites with bond-dimension limit chi."""
    mps = MPS.singlet_product(n, chi)
    for parity, angle in layer_plan(params):
        mps.apply_group(parity, angle)
    return mps


def lanczos_ground(matvec, v0, max_steps=20, tol=1e-10):
    """Lowest eigenpair of a Hermitian operator by Lanczos with reorthogonalising.

    DMRG calls this thousands of times with a start vector that is already
    close to the answer, so a short Krylov space is enough. (ARPACK, the engine
    behind scipy's eigsh, can stall when the start vector is almost an exact
    eigenvector, which is the normal situation late in a DMRG sweep.)
    """
    v = v0 / np.linalg.norm(v0)
    basis, alphas, betas = [v], [], []
    energy, vec = 0.0, v
    for j in range(max_steps):
        w = matvec(basis[j])
        alphas.append(float(np.vdot(basis[j], w).real))
        # Full reorthogonalisation keeps the Krylov basis honest in floating point.
        # Two passes ("twice is enough"): after the first pass cancellation
        # can leave an error as large as the surviving vector itself.
        for _ in range(2):
            for u in basis:
                w = w - np.vdot(u, w) * u
        beta = float(np.linalg.norm(w))
        # Ritz pair of the tridiagonal matrix built so far.
        tri = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
        vals, vecs = np.linalg.eigh(tri)
        energy = float(vals[0])
        vec = sum(c * u for c, u in zip(vecs[:, 0], basis))
        # The residual norm of a Ritz pair is |beta_j * last component|.
        if beta < 1e-13 or abs(beta * vecs[-1, 0]) < tol or j == max_steps - 1:
            break
        betas.append(beta)
        basis.append(w / beta)
    return energy, vec / np.linalg.norm(vec)


# ----------------------------------------------------------------------------
# Hamiltonian as an MPO, and DMRG
# ----------------------------------------------------------------------------
def heisenberg_mpo(n):
    """MPO tensors W[i] with shape (wl, wr, 2, 2), element W[p, q, bra, ket].

    The 5x5 operator-valued matrix is the usual finite-state machine: row 4 is
    "nothing placed yet" (start), columns 1-3 hold a pending X, Y or Z, and
    column 0 is "finished". The boundary tensors keep only row 4 (left end) and
    column 0 (right end), so a chain of n sites sums every neighbour pair once.
    """
    w = np.zeros((5, 5, 2, 2), dtype=complex)
    w[0, 0] = PAULI["I"]
    w[4, 4] = PAULI["I"]
    for k, name in enumerate("XYZ", 1):
        w[k, 0] = PAULI[name]
        w[4, k] = PAULI[name]
    tensors = [w.copy() for _ in range(n)]
    tensors[0] = w[4:5].copy()
    tensors[-1] = w[:, 0:1].copy()
    if n == 1:
        tensors[0] = w[4:5, 0:1].copy()
    return tensors


def mpo_to_dense(tensors):
    """Contract an MPO to a dense matrix (site 0 = most significant bit)."""
    t = tensors[0][0]  # (wr, bra, ket)
    t = np.transpose(t, (1, 2, 0))  # (bra, ket, wr)
    for w in tensors[1:]:
        # t: (bra..., ket..., wl); append the new site's bra and ket indices.
        t = np.einsum(
            "bkp,pqxy->bxkyq", t.reshape(t.shape[0], t.shape[1], t.shape[2]), w
        )
        t = t.reshape(t.shape[0] * 2, t.shape[2] * 2, t.shape[4])
    return t[:, :, 0]


class DMRG:
    """Two-site DMRG for an MPO Hamiltonian, starting from the Neel state.

    One sweep optimises every neighbouring pair of sites in turn: it builds the
    Hamiltonian restricted to the pair (left environment, two MPO tensors,
    right environment), finds its lowest eigenvector with Lanczos, and splits
    the result with an SVD that keeps at most chi_max singular values.
    """

    def __init__(self, mpo, chi_max, cutoff=1e-10):
        """Set up the Neel start state and the right-hand environments."""
        self.w = mpo
        self.n = len(mpo)
        self.chi_max = chi_max
        self.cutoff = cutoff
        self.mps = MPS.neel(self.n, chi_max)
        self.a = self.mps.a
        self.discarded = 0.0
        self.mid_entropy = 0.0
        # Environments: left[i] covers sites < i; right[i] covers sites >= i.
        self.left = [None] * (self.n + 1)
        self.right = [None] * (self.n + 1)
        self.left[0] = np.ones((1, 1, 1), dtype=complex)
        self.right[self.n] = np.ones((1, 1, 1), dtype=complex)
        for i in range(self.n - 1, 1, -1):
            self.right[i] = self._grow_right(self.right[i + 1], self.a[i], self.w[i])

    @staticmethod
    def _grow_left(env, a, w):
        """Absorb one site into a left environment (bra, mpo, ket).

        Index letters: env (a,p,b), ket A (b,s,d), MPO W (p,q,t,s) with t the
        bra index, bra conj(A) (a,t,c). Result (c,q,d). The contraction is done
        as three tensordots; a single np.einsum over four operands was 1000x
        slower here because it does not use BLAS.
        """
        t = np.tensordot(env, a, axes=(2, 0))  # a p s d
        t = np.tensordot(t, w, axes=([1, 2], [0, 3]))  # a d q t
        t = np.tensordot(a.conj(), t, axes=([0, 1], [0, 3]))  # c d q
        return t.transpose(0, 2, 1)

    @staticmethod
    def _grow_right(env, a, w):
        """Absorb one site into a right environment (bra, mpo, ket).

        env (c,q,d), ket A (b,s,d), W (p,q,t,s), bra conj(A) (a,t,c); the
        result has indices (a,p,b).
        """
        t = np.tensordot(env, a, axes=(2, 2))  # c q b s
        t = np.tensordot(t, w, axes=([1, 3], [1, 3]))  # c b p t
        t = np.tensordot(a.conj(), t, axes=([1, 2], [3, 0]))  # a b p
        return t.transpose(0, 2, 1)

    def _solve_pair(self, i, theta0):
        """Lowest eigenpair of the two-site effective Hamiltonian at (i, i+1)."""
        lenv, renv = self.left[i], self.right[i + 2]
        w1, w2 = self.w[i], self.w[i + 1]
        shape = theta0.shape
        dim = int(np.prod(shape))

        def matvec(vec):
            """Apply the effective Hamiltonian to a flattened two-site tensor."""
            t = vec.reshape(shape)
            # Each tensordot is a BLAS matrix product; the axis lists name the
            # shared indices (L: a,p,b  theta: b,s,u,e  W1: p,q,x,s ...).
            t = np.tensordot(lenv, t, axes=(2, 0))  # a p s u e
            t = np.tensordot(t, w1, axes=([1, 2], [0, 3]))  # a u e q x
            t = np.tensordot(t, w2, axes=([1, 3], [3, 0]))  # a e x r y
            t = np.tensordot(t, renv, axes=([1, 3], [2, 1]))  # a x y d
            return t.reshape(-1)

        if dim <= 40:
            # Tiny blocks: build the dense matrix and use a full eigensolver.
            h = np.zeros((dim, dim), dtype=complex)
            for k in range(dim):
                e = np.zeros(dim, dtype=complex)
                e[k] = 1.0
                h[:, k] = matvec(e)
            vals, vecs = np.linalg.eigh((h + h.conj().T) / 2)
            return float(vals[0]), vecs[:, 0].reshape(shape)
        energy, vec = lanczos_ground(matvec, theta0.reshape(-1))
        return energy, vec.reshape(shape)

    def _split(self, theta, move):
        """SVD a two-site tensor and truncate; return (u, s, vh)."""
        left, right = theta.shape[0], theta.shape[3]
        u, s, vh = np.linalg.svd(
            theta.reshape(left * 2, 2 * right), full_matrices=False
        )
        keep = max(1, min(int(np.sum(s > self.cutoff * s[0])), self.chi_max))
        weights = s**2
        self.discarded = max(
            self.discarded, float(weights[keep:].sum() / weights.sum())
        )
        s = s[:keep] / np.linalg.norm(s[:keep])
        return u[:, :keep].reshape(left, 2, keep), s, vh[:keep].reshape(keep, 2, right)

    def sweep(self):
        """One full sweep (left to right, then right to left); return energy."""
        n = self.n
        energy = 0.0
        self.discarded = 0.0
        for i in range(n - 1):
            theta0 = np.tensordot(self.a[i], self.a[i + 1], axes=(2, 0))
            energy, theta = self._solve_pair(i, theta0)
            u, s, vh = self._split(theta, "right")
            self.a[i] = u
            self.a[i + 1] = s[:, None, None] * vh
            self.left[i + 1] = self._grow_left(self.left[i], self.a[i], self.w[i])
        for i in range(n - 2, -1, -1):
            theta0 = np.tensordot(self.a[i], self.a[i + 1], axes=(2, 0))
            energy, theta = self._solve_pair(i, theta0)
            u, s, vh = self._split(theta, "left")
            self.a[i] = u * s
            self.a[i + 1] = vh
            self.right[i + 1] = self._grow_right(
                self.right[i + 2], self.a[i + 1], self.w[i + 1]
            )
            if i == n // 2 - 1:
                p = s**2
                p = p[p > 1e-16]
                self.mid_entropy = float(-np.sum(p * np.log(p)))
        # The state is now mixed-canonical with the centre on site 0.
        self.mps.centre = 0
        return energy

    def run(self, max_sweeps=20, tol=1e-9):
        """Sweep until the energy changes by less than tol; return the history."""
        history = []
        for _ in range(max_sweeps):
            history.append(self.sweep())
            if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
                break
        return history


# ----------------------------------------------------------------------------
# State-vector tools (exact reference and angle training, n up to about 14)
# ----------------------------------------------------------------------------
def sparse_hamiltonian(n):
    """H = sum over bonds of XX+YY+ZZ as a scipy sparse matrix."""
    h = sp.csr_matrix((2**n, 2**n), dtype=complex)
    for q in range(n - 1):
        left = sp.identity(2 ** (n - q - 2), format="csr")
        right = sp.identity(2**q, format="csr")
        h = h + sp.kron(sp.kron(left, sp.csr_matrix(BOND)), right, format="csr")
    return h


def sv_apply_group(psi, n, parity, angle):
    """State-vector version of MPS.apply_group (qubit q = bit q)."""
    u = bond_unitary(angle)
    for q in group_terms(n, parity):
        t = psi.reshape(2 ** (n - q - 2), 4, 2**q)
        psi = np.einsum("ab,ibj->iaj", u, t).reshape(-1)
    return psi


def sv_apply_group_h(psi, n, parity):
    """Apply the group Hamiltonian (sum of its bond terms) to a vector."""
    out = np.zeros_like(psi)
    for q in group_terms(n, parity):
        t = psi.reshape(2 ** (n - q - 2), 4, 2**q)
        out += np.einsum("ab,ibj->iaj", BOND, t).reshape(-1)
    return out


def singlet_vector(n):
    """Singlet-product start state as a full vector."""
    pair = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2)
    psi = np.array([1.0 + 0j])
    for _ in range(n // 2):
        psi = np.kron(pair, psi)
    return psi


def sv_energy_and_grad(params, n, h, psi0):
    """HVA energy and its exact adjoint gradient (see Lab 23 for the proof)."""
    steps = layer_plan(params)
    states = [psi0]
    for parity, angle in steps:
        states.append(sv_apply_group(states[-1], n, parity, angle))
    hpsi = h @ states[-1]
    energy = float(np.vdot(states[-1], hpsi).real)
    grad = np.zeros(len(steps))
    phi = hpsi
    for k in range(len(steps) - 1, -1, -1):
        parity, angle = steps[k]
        grad[k] = 2 * np.vdot(phi, sv_apply_group_h(states[k + 1], n, parity)).imag
        # Undo this step on the adjoint state (apply U^dagger).
        phi = sv_apply_group(phi, n, parity, -angle)
    return energy, grad


def train_angles(n, e0, max_layers, rng, extra_starts=2):
    """Warm-started L-BFGS training for depths 1..max_layers on n spins.

    Returns a list of (error above e0, angles) for each depth. Each depth is
    started from the previous solution (plus a new zero layer = identity), a
    jiggled copy of it, and `extra_starts` random points, because the
    landscape has plateaus that a single start often cannot leave.
    """
    h = sparse_hamiltonian(n)
    psi0 = singlet_vector(n)
    prev, out = np.zeros(0), []
    for p in range(1, max_layers + 1):
        warm = np.concatenate([prev, np.zeros(2)])
        starts = [warm, warm + rng.normal(0, 0.1, 2 * p)]
        starts += [rng.uniform(-0.8, 0.8, 2 * p) for _ in range(extra_starts)]
        best = None
        for x0 in starts:
            res = minimize(
                sv_energy_and_grad,
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
