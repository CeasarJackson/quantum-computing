#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/dmrgx.py

Purpose
-------
Extensions of the two-site DMRG of Lab 25 (mpslib.DMRG) used by Lab 42:

  * SYMMETRY SECTORS. The Heisenberg Hamiltonian conserves total S_z, but a
    plain two-site DMRG does not stay in the sector it starts in: the two
    centre sites span all four spin states, so the lowest eigenvector may
    belong to a different S_z and the run drifts to the global ground state.
    The fix used here is the standard one for a U(1) symmetry: every bond index
    carries a charge (the S_z of everything to its left), a two-site tensor is
    allowed only where the charges add up to the target, and the SVD is done
    block by block so the new bond indices keep a definite charge.
    `sector_bits` builds a product start state in the chosen sector.
  * EXCITED STATES. To find the next level inside a sector, add a penalty
    w |phi><phi| to the effective Hamiltonian for each earlier state, where
    |phi> is that state projected onto the current two-site basis. Because the
    basis is orthonormal (mixed-canonical form) the projection only needs
    "overlap environments" between the state being optimised and each earlier
    one. With w larger than the gap, the lowest eigenvector of the penalised
    problem is the lowest state orthogonal to all earlier ones.
  * `mpo_energy` (energy of any MPS from the MPO) and `reflect` (the mirror
    image of an MPS, to read off the reflection parity of a state).

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    from dmrgx import DMRGX, sector_bits, mpo_energy, reflect

Log files
---------
None. This is a library module. (Logging is the caller's job, through
`from applog import get_logger`; this library only returns numbers.)

Exit Codes
----------
Library module; it never exits the process.
===============================================================================
"""

import numpy as np

from mpslib import DMRG, MPS, lanczos_ground, overlap


def sector_bits(n, sz_twice):
    """Product-state bits (0 = up, 1 = down) with total S_z = sz_twice / 2.

    Starts from the Neel state (up on even sites, S_z = 0) and flips down
    spins up (or up spins down) from the right end until S_z is as requested.
    `sz_twice` must have the parity of n and satisfy |sz_twice| <= n.
    """
    if (sz_twice - n) % 2 or abs(sz_twice) > n:
        raise ValueError(f"S_z = {sz_twice}/2 is impossible on {n} spins")
    bits = [i % 2 for i in range(n)]
    # Each flip changes 2 S_z by 2; Neel has 2 S_z = n % 2 (0 for even n).
    current = sum(1 - 2 * b for b in bits)
    site = n - 1
    while current != sz_twice:
        want = 0 if current < sz_twice else 1
        # Move left until a spin of the opposite kind is found.
        while bits[site] == want:
            site -= 1
        bits[site] = want
        current += 2 if want == 0 else -2
    return bits


def product_mps(bits, chi_max):
    """MPS of the product state with the given bits (0 = up, 1 = down)."""
    # One (1, 2, 1) tensor per site: a bond dimension of 1 is a product state.
    tensors = []
    for b in bits:
        t = np.zeros((1, 2, 1), dtype=complex)
        t[0, b, 0] = 1.0
        tensors.append(t)
    return MPS(tensors, len(bits) - 1, chi_max)


def mpo_energy(mpo, mps):
    """<psi|H|psi> / <psi|psi> of an MPS from the MPO, by environment growth."""
    # Sweep once left to right; the dummy boundary bonds have dimension 1.
    env = np.ones((1, 1, 1), dtype=complex)
    for a, w in zip(mps.a, mpo):
        env = DMRG._grow_left(env, a, w)
    norm = overlap(mps, mps).real
    return float(env[0, 0, 0].real / norm)


def reflect(mps):
    """The mirror image of an MPS: site order reversed, bond indices swapped."""
    # Reversing the chain turns every right bond into a left bond, hence the
    # transpose of the bond indices.
    tensors = [t.transpose(2, 1, 0).copy() for t in reversed(mps.a)]
    return MPS(tensors, 0, mps.chi_max)


def _grow_overlap_left(env, a, prev):
    """Absorb one site into a left overlap environment <a|prev>."""
    return np.einsum("xy,xsz,ysw->zw", env, a.conj(), prev)


def _grow_overlap_right(env, a, prev):
    """Absorb one site into a right overlap environment <a|prev>."""
    return np.einsum("xsz,ysw,zw->xy", a.conj(), prev, env)


class DMRGX(DMRG):
    """Two-site DMRG in a chosen S_z sector, optionally orthogonal to others.

    Parameters
    ----------
    mpo        Hamiltonian as an MPO (mpslib.heisenberg_mpo)
    chi_max    largest bond dimension
    start_bits product-state bits fixing the S_z sector (default: Neel)
    previous   earlier states (MPS) the new state must be orthogonal to
    weight     penalty strength; it must exceed the energy gap to the state
               sought (the lowest levels here are separated by far less than 20)

    Charges are stored as 2 S_z (an integer): +1 for an up spin, -1 for down.
    """

    def __init__(
        self,
        mpo,
        chi_max,
        start_bits=None,
        previous=(),
        weight=20.0,
        cutoff=1e-10,
    ):
        """Set up the start state, the charges and the overlap environments."""
        super().__init__(mpo, chi_max, cutoff)
        n = self.n
        if start_bits is None:
            start_bits = [i % 2 for i in range(n)]
        # Replace the Neel start by the requested product state and rebuild
        # the right-hand environments for it.
        self.mps = product_mps(start_bits, chi_max)
        self.a = self.mps.a
        for i in range(n - 1, 1, -1):
            self.right[i] = self._grow_right(self.right[i + 1], self.a[i], self.w[i])
        # q[i][x] = 2 S_z of everything left of bond i, for bond index x.
        self.q = [np.zeros(1, dtype=int)]
        for b in start_bits:
            self.q.append(self.q[-1] + (1 - 2 * b))
        self.target = int(self.q[-1][0])
        # The penalty strength must exceed the gap to the level being sought.
        self.weight = weight
        # Keep private, normalised copies: later sweeps must never see them
        # change, and the projector only works for unit vectors.
        self.previous = []
        for p in previous:
            tensors = [t.copy() for t in p.a]
            tensors[0] = tensors[0] / np.sqrt(overlap(p, p).real)
            self.previous.append(tensors)
        # xl[k][i] / yr[k][i]: overlap of the current state with earlier state k,
        # over the sites left of bond i / from bond i on.
        self.xl = [[None] * (n + 1) for _ in self.previous]
        self.yr = [[None] * (n + 1) for _ in self.previous]
        for k, prev in enumerate(self.previous):
            self.xl[k][0] = np.ones((1, 1), dtype=complex)
            self.yr[k][n] = np.ones((1, 1), dtype=complex)
            for i in range(n - 1, 1, -1):
                self.yr[k][i] = _grow_overlap_right(
                    self.yr[k][i + 1], self.a[i], prev[i]
                )

    def _mask(self, i, shape):
        """Boolean mask of the two-site entries whose charges add to the target."""
        up = np.array([1, -1])
        # Charge of (left block + two sites) at every index combination.
        total = (
            self.q[i][:, None, None, None]
            + up[None, :, None, None]
            + up[None, None, :, None]
        )
        return total == self.q[i + 2][None, None, None, :]

    def _penalty_vectors(self, i, mask):
        """Earlier states projected onto the two-site basis at (i, i + 1)."""
        out = []
        for k, prev in enumerate(self.previous):
            # phi[l, s, t, r] = <basis(l, s, t, r) | earlier state>
            phi = np.einsum(
                "xy,ysm,mtz,wz->xstw",
                self.xl[k][i],
                prev[i],
                prev[i + 1],
                self.yr[k][i + 2],
            )
            out.append((phi * mask).reshape(-1))
        return out

    def _solve_pair(self, i, theta0):
        """Lowest eigenpair of the penalised, charge-restricted pair problem."""
        lenv, renv = self.left[i], self.right[i + 2]
        w1, w2 = self.w[i], self.w[i + 1]
        shape = theta0.shape
        mask = self._mask(i, shape)
        flat_mask = mask.reshape(-1)
        phis = self._penalty_vectors(i, mask)

        def matvec(vec):
            """Effective Hamiltonian plus w |phi><phi| inside the allowed block."""
            t = (vec.reshape(shape)) * mask
            t = np.tensordot(lenv, t, axes=(2, 0))
            t = np.tensordot(t, w1, axes=([1, 2], [0, 3]))
            t = np.tensordot(t, w2, axes=([1, 3], [3, 0]))
            t = np.tensordot(t, renv, axes=([1, 3], [2, 1]))
            out = t.reshape(-1) * flat_mask
            for phi in phis:
                out = out + self.weight * phi * np.vdot(phi, vec)
            return out

        # Indices of the allowed (charge-conserving) entries of the pair tensor.
        allowed = np.flatnonzero(flat_mask)
        if len(allowed) <= 40:
            # Small block: build it densely over the allowed entries only.
            h = np.zeros((len(allowed), len(allowed)), dtype=complex)
            for col, k in enumerate(allowed):
                e = np.zeros(flat_mask.size, dtype=complex)
                e[k] = 1.0
                h[:, col] = matvec(e)[allowed]
            vals, vecs = np.linalg.eigh((h + h.conj().T) / 2)
            full = np.zeros(flat_mask.size, dtype=complex)
            full[allowed] = vecs[:, 0]
            return float(vals[0]), full.reshape(shape)
        start = theta0.reshape(-1) * flat_mask
        energy, vec = lanczos_ground(matvec, start)
        return energy, vec.reshape(shape)

    def _split(self, theta, move):
        """Charge-blocked SVD of a two-site tensor; return (u, s, vh).

        The matrix (l, s | t, r) is block diagonal in the charge flowing
        through the cut. Each block is decomposed on its own; the singular
        values of all blocks are then ranked together and the largest chi_max
        kept, so the new bond keeps a definite charge on every index.
        """
        left, right = theta.shape[0], theta.shape[3]
        i = self._pair
        up = np.array([1, -1])
        row_q = (self.q[i][:, None] + up[None, :]).reshape(-1)
        col_q = (self.q[i + 2][:, None] - up[None, :]).T.reshape(-1)
        mat = theta.reshape(left * 2, 2 * right)
        # Each block SVD yields singular triplets tagged with their charge.
        pieces = []
        for c in np.unique(row_q):
            rows, cols = np.flatnonzero(row_q == c), np.flatnonzero(col_q == c)
            if len(cols) == 0:
                continue
            u, s, vh = np.linalg.svd(mat[np.ix_(rows, cols)], full_matrices=False)
            for k in range(len(s)):
                pieces.append((s[k], c, rows, cols, u[:, k], vh[k]))
        # Rank all blocks together so the truncation keeps the globally largest.
        pieces.sort(key=lambda p: -p[0])
        top = pieces[0][0]
        keep = max(1, min(sum(p[0] > self.cutoff * top for p in pieces), self.chi_max))
        weights = np.array([p[0] ** 2 for p in pieces])
        self.discarded = max(
            self.discarded, float(weights[keep:].sum() / weights.sum())
        )
        u_full = np.zeros((left * 2, keep), dtype=complex)
        vh_full = np.zeros((keep, 2 * right), dtype=complex)
        s_kept = np.array([p[0] for p in pieces[:keep]])
        for k, (_, c, rows, cols, uvec, vvec) in enumerate(pieces[:keep]):
            u_full[rows, k] = uvec
            vh_full[k, cols] = vvec
        self._new_q = np.array([p[1] for p in pieces[:keep]], dtype=int)
        s_kept = s_kept / np.linalg.norm(s_kept)
        return u_full.reshape(left, 2, keep), s_kept, vh_full.reshape(keep, 2, right)

    def _optimise_pair(self, i, move):
        """Solve and split the pair (i, i + 1); record the new bond charges."""
        # _split needs the pair position to look up the bond charges.
        self._pair = i
        theta0 = np.tensordot(self.a[i], self.a[i + 1], axes=(2, 0))
        energy, theta = self._solve_pair(i, theta0)
        u, s, vh = self._split(theta, move)
        self.q[i + 1] = self._new_q
        return energy, u, s, vh

    def sweep(self):
        """One full sweep with the overlap environments kept in step."""
        n = self.n
        energy = 0.0
        self.discarded = 0.0
        for i in range(n - 1):
            energy, u, s, vh = self._optimise_pair(i, "right")
            self.a[i] = u
            self.a[i + 1] = s[:, None, None] * vh
            # Left environments are valid again once site i is left-isometric.
            self.left[i + 1] = self._grow_left(self.left[i], self.a[i], self.w[i])
            for k, prev in enumerate(self.previous):
                self.xl[k][i + 1] = _grow_overlap_left(
                    self.xl[k][i], self.a[i], prev[i]
                )
        for i in range(n - 2, -1, -1):
            energy, u, s, vh = self._optimise_pair(i, "left")
            self.a[i] = u * s
            self.a[i + 1] = vh
            self.right[i + 1] = self._grow_right(
                self.right[i + 2], self.a[i + 1], self.w[i + 1]
            )
            for k, prev in enumerate(self.previous):
                self.yr[k][i + 1] = _grow_overlap_right(
                    self.yr[k][i + 2], self.a[i + 1], prev[i + 1]
                )
            if i == n // 2 - 1:
                p = s**2
                p = p[p > 1e-16]
                self.mid_entropy = float(-np.sum(p * np.log(p)))
        # The centre of orthogonality is back on site 0 after the right-to-left pass.
        self.mps.centre = 0
        return energy
