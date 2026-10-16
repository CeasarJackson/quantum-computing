#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/noiselib.py

Purpose
-------
Shared noise toolkit for Labs 29 and later: noisy HVA circuits on a matrix
product state by quantum trajectories, plus an exact density-matrix reference
for small chains.

  * noisy_energy   : one trajectory (random Pauli errors after each gate layer).
  * sample         : many trajectories; mean_se gives mean and standard error.
  * fold_steps     : unitary folding U -> U U^dagger U, which multiplies the
                     number of gates, and so the noise, by an odd factor while
                     leaving the ideal circuit unchanged (used for zero-noise
                     extrapolation).
  * dm_energy      : the same channel on the full density matrix (n <= 8).
  * weights        : biased Pauli noise (relative X, Y, Z probabilities).
  * sz_verified    : symmetry verification: energy and acceptance after
                     projecting onto total S_z = 0 (MPS and density matrix).
  * damped_mps     : amplitude damping (T1 decay) by quantum jumps, a
                     NON-unitary channel, on the MPS (Lab 33).
  * coherent_mps   : coherent errors (one-site over-rotations, mis-scaled
                     gate angles) as a single deterministic MPS run (Lab 33).
  * dm_kraus_state : exact density matrix for any single-qubit Kraus channel.
  * twirled_mps    : coherent errors with random Pauli frames P (x) P around
                     each bond gate (free, because the gate commutes with
                     them), which turns them into random errors (Lab 34).
  * dm_twirled_state : the exact average over those frames.
  * pec_mps        : probabilistic error cancellation (Lab 35): sample Pauli
                     corrections from the quasi-probability form of the
                     inverse depolarizing channel; each run carries a signed
                     weight and the weighted mean is unbiased.
  * dm_pec_state   : the exact noisy-then-corrected density matrix.
  * s2_matrix, singlet_projector, dm_s2_verified : total-spin (S^2) symmetry
                     verification on the exact density matrix (Lab 36).
  * dm_with_errors : the exact state after chosen, deterministic Pauli errors.
  * general_gate, aniso_steps, free_steps, general_dm_state, general_sv,
    general_energy, train_aniso, perbond_steps, train_steps, rho_projected :
                     ansatz families (Lab 40): bond gates exp(-i(a XX + b YY +
                     c ZZ)) that break total spin, and symmetry-adapted ones
                     with one angle PER BOND, plus their exact noisy state and
                     their S_z / S^2 projections.
  * bond_clifford, clifford_bond_steps, compiled_sv, tensor_energy, tensor_s2 :
                     the bond gate as Clifford . controlled phase . Clifford
                     (Lab 41), whole compiled circuits and their energy / spin.

Noise model: after each gate layer, every qubit the layer acted on suffers the
depolarizing channel rho -> (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z).

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    from noiselib import noisy_energy, sample, mean_se, dm_energy, fold_steps
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

import numpy as np
from scipy.optimize import minimize

from mpslib import (
    BOND,
    MPS,
    PAULI,
    bond_unitary,
    bond_overlaps,
    group_terms,
    layer_plan,
    overlap,
    singlet_vector,
)


# ----------------------------------------------------------------------------
# Circuit steps with optional folding
# ----------------------------------------------------------------------------
def fold_steps(steps, factor):
    """Repeat every step as U (U^dagger U)^k so the gate count grows `factor`x.

    `factor` must be odd. The ideal circuit is unchanged because
    U^dagger U = 1, but every extra gate layer brings its own noise.
    """
    if factor < 1 or factor % 2 == 0:
        raise ValueError(f"fold factor must be a positive odd integer, got {factor}")
    out = []
    for parity, angle in steps:
        for k in range(factor):
            out.append((parity, angle if k % 2 == 0 else -angle))
    return out


def touched_qubits(n, parity):
    """Qubits that one gate group acts on (both ends of every bond)."""
    return [q for b in group_terms(n, parity) for q in (b, b + 1)]


def apply_pauli(mps, site, name):
    """Apply a single-qubit Pauli to one site of an MPS, in place.

    A unitary on the physical index keeps every tensor's canonical form, so
    neither the orthogonality centre nor the bond dimensions change.
    """
    mps.a[site] = np.einsum("ab,lbr->lar", PAULI[name], mps.a[site])


def pick_pauli(rng, weights):
    """Name of the Pauli error drawn with relative probabilities `weights`."""
    if weights is None:
        return "XYZ"[rng.integers(3)]
    w = np.asarray(weights, dtype=float)
    return "XYZ"[rng.choice(3, p=w / w.sum())]


def noisy_mps(n, params, chi, p, rng, fold=1, weights=None, skip_clean=False):
    """One noisy trajectory of the HVA circuit as an MPS.

    All random draws for the whole circuit are made first. With skip_clean the
    function returns None, without simulating, when no error was drawn (the
    caller then substitutes the noiseless result).
    """
    steps = fold_steps(layer_plan(params), fold)
    hits = [rng.random(len(touched_qubits(n, par))) < p for par, _ in steps]
    if skip_clean and not any(h.any() for h in hits):
        return None
    mps = MPS.singlet_product(n, chi)
    mps.cutoff = 1e-10
    # Each X or Y error moves total S_z by one unit; Z errors do not. The count
    # bounds the sectors the state can occupy (used by sz_verified).
    mps.flips = 0
    # Gates first, then the errors of that layer, as on hardware.
    for (parity, angle), hit in zip(steps, hits):
        mps.apply_group(parity, angle)
        qubits = touched_qubits(n, parity)
        for q in np.flatnonzero(hit):
            name = pick_pauli(rng, weights)
            mps.flips += name != "Z"
            apply_pauli(mps, qubits[q], name)
    return mps


def noisy_energy(n, params, chi, p, rng, e_clean=None, fold=1, weights=None):
    """Energy of one noisy trajectory (see noisy_mps)."""
    mps = noisy_mps(n, params, chi, p, rng, fold, weights, e_clean is not None)
    return e_clean if mps is None else mps.energy()


def sample(n, params, chi, p, count, rng, e_clean=None, fold=1, weights=None):
    """Energies of `count` independent trajectories."""
    return np.array(
        [
            noisy_energy(n, params, chi, p, rng, e_clean, fold, weights)
            for _ in range(count)
        ]
    )


def sz_verified(mps):
    """Numerator <psi|H P0|psi> and acceptance <psi|P0|psi> for an MPS.

    P0 projects onto total S_z = 0. Because the Hamiltonian commutes with S_z,
    the energy of the accepted (post-selected) state is numerator / acceptance.
    P0 is built as an average over rotations exp(i phi S_z), phi = 2 pi k / M,
    with M points (M = n + 1 always works, fewer when the number of S_z-changing
    errors is known): S_z takes integer values between -n/2 and n/2 (n
    even), and the average of exp(i phi m) over M points is 1 for m = 0 and 0
    for every other integer in range. Each rotation is a product of
    single-site gates, so it costs one pass over the tensors.
    """
    n = mps.n
    # With f flips the state lives in sectors |m| <= f, so 2f + 1 points are
    # enough; without that information use the safe bound n + 1.
    m = min(n + 1, 2 * getattr(mps, "flips", n) + 1)
    bonds = list(range(n - 1))
    num, acc = 0.0, 0.0
    for k in range(m):
        phi = 2 * np.pi * k / m
        rot = mps.copy()
        phase = np.array([np.exp(1j * phi / 2), np.exp(-1j * phi / 2)])
        for i in range(n):
            rot.a[i] = rot.a[i] * phase[None, :, None]
        num += sum(bond_overlaps(mps, rot, bonds, BOND)) / m
        acc += overlap(mps, rot) / m
    return float(np.real(num)), float(np.real(acc))


def sz_ratio(nums, accs):
    """Post-selected energy sum(num)/sum(acc) and its standard error.

    A ratio of two sample means: the error uses the standard first-order
    (delta-method) formula, var = var(num - E acc) / (N mean(acc)^2).
    """
    nums, accs = np.asarray(nums), np.asarray(accs)
    # First-order propagation for a ratio of two sample means.
    energy = nums.sum() / accs.sum()
    resid = nums - energy * accs
    se = float(np.std(resid, ddof=1) / (len(nums) ** 0.5 * accs.mean()))
    return float(energy), se


def richardson_weights(scales):
    """Lagrange weights w_i with sum_i w_i E(scale_i) = fitted E at scale 0.

    Fitting a polynomial of degree len(scales) - 1 through the points and
    evaluating it at zero is a fixed linear combination of the measurements,
    which also makes the error propagation exact.
    """
    scales = np.asarray(scales, dtype=float)
    weights = []
    for i, xi in enumerate(scales):
        others = np.delete(scales, i)
        weights.append(np.prod(-others / (xi - others)))
    return np.array(weights)


def mean_se(values):
    """Sample mean and its standard error."""
    return float(np.mean(values)), float(np.std(values, ddof=1) / len(values) ** 0.5)


# ----------------------------------------------------------------------------
# Exact density-matrix reference (small n only)
# ----------------------------------------------------------------------------
def dm_apply(rho, axes, op):
    """Apply a k-qubit operator (as a 2^k x 2^k matrix) to the given axes."""
    k = len(axes)
    t = np.tensordot(
        op.reshape((2,) * (2 * k)), rho, axes=(list(range(k, 2 * k)), axes)
    )
    return np.moveaxis(t, list(range(k)), axes)


def dm_state(n, params, p, fold=1, weights=None):
    """Density matrix after the noisy circuit (axes 0..n-1 ket, n..2n-1 bra)."""
    w = np.ones(3) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    psi = singlet_vector(n)
    # singlet_vector is little endian (qubit q = bit q); flip to site order.
    psi = psi.reshape((2,) * n).transpose(list(range(n))[::-1])
    rho = np.multiply.outer(psi, psi.conj())
    # Same circuit and same channel as the trajectories, but on the full rho.
    for parity, angle in fold_steps(layer_plan(params), fold):
        u = bond_unitary(angle)
        for b in group_terms(n, parity):
            rho = dm_apply(rho, [b, b + 1], u)
            rho = dm_apply(rho, [n + b, n + b + 1], u.conj())
        for q in touched_qubits(n, parity):
            mixed = (1 - p) * rho
            for name, weight in zip("XYZ", w):
                s = dm_apply(rho, [q], PAULI[name])
                s = dm_apply(s, [n + q], PAULI[name].conj())
                mixed = mixed + (p * weight) * s
            rho = mixed
    return rho


def dm_expectation(rho, n):
    """Energy Tr(H rho) as the sum of bond expectation values."""
    total = 0.0
    for b in range(n - 1):
        h_rho = dm_apply(rho, [b, b + 1], BOND)
        total += np.trace(h_rho.reshape(2**n, 2**n)).real
    return float(total)


def dm_energy(n, params, p, fold=1, weights=None):
    """Exact noisy energy: density matrix with the same channel as above."""
    return dm_expectation(dm_state(n, params, p, fold, weights), n)


def dm_sz_verified(n, params, p, fold=1, weights=None):
    """Exact post-selected energy and acceptance for total S_z = 0."""
    rho = dm_state(n, params, p, fold, weights).reshape(2**n, 2**n)
    # Site order: axis 0 is the most significant bit; bit 1 means spin down.
    # Site-order bits: popcount equals n/2 means as many up spins as down.
    bits = (np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1
    keep = (bits.sum(axis=1) == n // 2).astype(float)
    rho0 = rho * keep[:, None] * keep[None, :]
    # Trace of the projected matrix is the probability that S_z = 0.
    acceptance = float(np.trace(rho0).real)
    return dm_expectation(rho0.reshape((2,) * (2 * n)), n) / acceptance, acceptance


# ----------------------------------------------------------------------------
# Amplitude damping and coherent errors (Lab 33)
# ----------------------------------------------------------------------------
def apply_site(mps, site, mat):
    """Apply any 2x2 matrix to one site of an MPS, in place.

    For a unitary this keeps the canonical form (like apply_pauli). Callers
    that pass a non-unitary matrix must renormalise the centre tensor.
    """
    mps.a[site] = np.einsum("ab,lbr->lar", mat, mps.a[site])


def rotation(axis, angle):
    """Single-qubit rotation exp(-i angle sigma_axis / 2) as a 2x2 matrix."""
    return np.cos(angle / 2) * PAULI["I"] - 1j * np.sin(angle / 2) * PAULI[axis]


def amplitude_damping_kraus(gamma):
    """Kraus pair of amplitude damping: K0 keeps |0>, K1 drops |1> to |0>."""
    k0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    k1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [k0, k1]


def damp_site(mps, site, gamma, rng):
    """One quantum-jump step of amplitude damping on a site; True if it jumped.

    The channel is rho -> K0 rho K0^+ + K1 rho K1^+. Unravelled into pure
    states: with probability gamma * P1 (P1 = population of |1> on this site)
    the state jumps, K1|psi>, otherwise it evolves with K0|psi>, which is not
    unitary: the absence of a jump still carries information (the |1> branch
    is slightly less likely). Either way the result is renormalised.
    """
    # With the orthogonality centre on the site, local norms are exact.
    # Moving the centre costs one QR per step because sites are visited in order.
    mps.move_centre(site)
    t = mps.a[site]
    p1 = float(np.sum(np.abs(t[:, 1, :]) ** 2) / np.sum(np.abs(t) ** 2))
    jumped = rng.random() < gamma * p1
    if jumped:
        new = np.zeros_like(t)
        new[:, 0, :] = t[:, 1, :]
    else:
        new = t.copy()
        new[:, 1, :] *= np.sqrt(1 - gamma)
    mps.a[site] = new / np.linalg.norm(new)
    return jumped


def damped_mps(n, params, chi, gamma, rng, fold=1):
    """One amplitude-damping trajectory of the HVA circuit as an MPS.

    After every gate layer each touched qubit takes one damp_site step.
    `mps.flips` counts the jumps: each moves total S_z by exactly one unit
    (always the same way), which is what sz_verified needs.
    """
    # Jumps follow the gate layer, as on hardware where decay happens during idling.
    mps = MPS.singlet_product(n, chi)
    mps.cutoff = 1e-10
    mps.flips = 0
    for parity, angle in fold_steps(layer_plan(params), fold):
        mps.apply_group(parity, angle)
        for q in touched_qubits(n, parity):
            mps.flips += damp_site(mps, q, gamma, rng)
    return mps


def coherent_mps(n, params, chi, errs=None, axis="Z", fold=1, delta=0.0):
    """Deterministic MPS run with coherent errors; one run, no sampling.

    errs   per-qubit rotation angles (radians): after every gate layer each
           touched qubit gets exp(-i errs[q] sigma_axis / 2), the same each time
           (a frozen miscalibration or detuning). None means no such error.
    delta  relative over-rotation of every bond gate, angle -> angle (1 + delta).
    Folded copies use the SAME errors, so coherent errors add in amplitude.
    """
    mps = MPS.singlet_product(n, chi)
    mps.cutoff = 1e-10
    mats = None if errs is None else [rotation(axis, e) for e in errs]
    for parity, angle in fold_steps(layer_plan(params), fold):
        mps.apply_group(parity, angle * (1 + delta))
        if mats is not None:
            for q in touched_qubits(n, parity):
                apply_site(mps, q, mats[q])
    return mps


def dm_kraus_state(n, params, kraus_of, fold=1, delta=0.0):
    """Exact density matrix when qubit q suffers the channel kraus_of(q).

    kraus_of(q) returns a list of 2x2 Kraus matrices (or None for no error);
    it is applied to every touched qubit after each gate layer. Axes 0..n-1
    are the ket, n..2n-1 the bra, as in dm_state.
    """
    psi = singlet_vector(n).reshape((2,) * n).transpose(list(range(n))[::-1])
    rho = np.multiply.outer(psi, psi.conj())
    for parity, angle in fold_steps(layer_plan(params), fold):
        u = bond_unitary(angle * (1 + delta))
        for b in group_terms(n, parity):
            rho = dm_apply(rho, [b, b + 1], u)
            rho = dm_apply(rho, [n + b, n + b + 1], u.conj())
        for q in touched_qubits(n, parity):
            ops = kraus_of(q)
            if ops is None:
                continue
            total = 0
            for k in ops:
                term = dm_apply(rho, [q], k)
                total = total + dm_apply(term, [n + q], k.conj())
            rho = total
    return rho


def dm_sz_from_rho(rho, n):
    """Post-selected S_z = 0 energy and acceptance of a density matrix."""
    flat = rho.reshape(2**n, 2**n)
    # Site-order bits: popcount n/2 means as many up spins as down.
    bits = (np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1
    keep = (bits.sum(axis=1) == n // 2).astype(float)
    rho0 = flat * keep[:, None] * keep[None, :]
    acceptance = float(np.trace(rho0).real)
    return dm_expectation(rho0.reshape((2,) * (2 * n)), n) / acceptance, acceptance


# ----------------------------------------------------------------------------
# Random Pauli frames on the SU(2)-symmetric bond gate (Lab 34)
# ----------------------------------------------------------------------------
FRAMES = "IXYZ"


def zz_rotation(angle):
    """Two-qubit rotation exp(-i angle Z (x) Z / 2) as a 4x4 matrix."""
    # Z (x) Z squares to the identity, so the exponential is cos + sin form.
    zz = np.kron(PAULI["Z"], PAULI["Z"])
    return np.cos(angle / 2) * np.eye(4) - 1j * np.sin(angle / 2) * zz


def apply_bond_matrix(mps, bond, mat):
    """Apply a 4x4 unitary to sites (bond, bond + 1) like any bond gate.

    The pair is contracted, rotated and split again by SVD, so the usual
    truncation (chi_max, cutoff) applies; the orthogonality centre follows.
    """
    mps.apply_bond(bond, mat, "right")


def twirled_mps(
    n, params, chi, errs, rng, axis="Z", fold=1, delta=0.0, crosstalk=0.0, frames=True
):
    """One frame-randomised run of the circuit with frozen coherent errors.

    The bond gate exp(-i a (XX+YY+ZZ)) commutes with P (x) P for every Pauli P
    (the same one on both qubits). So a random frame P (x) P can be applied
    before the gate and its inverse (the same matrix) after the gate's error
    without changing the ideal circuit. The error E is then seen as
    (P x P) E (P x P): a rotation by +eps or -eps depending on P, which makes
    its first-order effects average out across frames. Every bond of every
    layer, including folded copies, draws its own frame.

    crosstalk  extra coherent two-qubit error exp(-i crosstalk ZZ / 2) on every
               bond (it commutes with every frame, so frames cannot help).
    frames     False gives the identity frame always (the un-twirled circuit).
    """
    mps = MPS.singlet_product(n, chi)
    mps.cutoff = 1e-10
    # One fixed matrix per qubit: the same detuning is applied every layer.
    mats = [rotation(axis, e) for e in errs]
    for parity, angle in fold_steps(layer_plan(params), fold):
        bonds = group_terms(n, parity)
        # A fresh frame per bond and layer; the draws are what the average runs over.
        drawn = [FRAMES[rng.integers(4)] if frames else "I" for _ in bonds]
        for b, name in zip(bonds, drawn):
            for q in (b, b + 1):
                apply_site(mps, q, PAULI[name])
        mps.apply_group(parity, angle * (1 + delta))
        for q in touched_qubits(n, parity):
            apply_site(mps, q, mats[q])
        # Crosstalk acts on the pair as a whole and commutes with every frame.
        if crosstalk:
            for b in bonds:
                apply_bond_matrix(mps, b, zz_rotation(crosstalk))
        for b, name in zip(bonds, drawn):
            for q in (b, b + 1):
                apply_site(mps, q, PAULI[name])
    return mps


def dm_twirled_state(
    n, params, errs, axis="Z", fold=1, delta=0.0, crosstalk=0.0, frames=True
):
    """Exact frame average: after each bond gate, the twirled error channel.

    Averaging (P x P) E (P x P) over the four frames gives the channel with
    Kraus operators K_P = (P x P) E (P x P) / 2, with E the two-qubit error
    rotation(axis, eps_b) (x) rotation(axis, eps_b+1).
    """
    psi = singlet_vector(n).reshape((2,) * n).transpose(list(range(n))[::-1])
    rho = np.multiply.outer(psi, psi.conj())
    for parity, angle in fold_steps(layer_plan(params), fold):
        u = bond_unitary(angle * (1 + delta))
        for b in group_terms(n, parity):
            rho = dm_apply(rho, [b, b + 1], u)
            rho = dm_apply(rho, [n + b, n + b + 1], u.conj())
            # The two-qubit error is the product of the two one-qubit rotations.
            err = np.kron(rotation(axis, errs[b]), rotation(axis, errs[b + 1]))
            if crosstalk:
                err = zz_rotation(crosstalk) @ err
            total = 0
            names = FRAMES if frames else "I"
            for name in names:
                frame = np.kron(PAULI[name], PAULI[name])
                k = frame @ err @ frame / np.sqrt(len(names))
                term = dm_apply(rho, [b, b + 1], k)
                total = total + dm_apply(term, [n + b, n + b + 1], k.conj())
            rho = total
    return rho


# ----------------------------------------------------------------------------
# Probabilistic error cancellation (Lab 35)
# ----------------------------------------------------------------------------
def inverse_depolarizing(p):
    """Quasi-probabilities of the inverse of depolarizing noise with rate p.

    The channel (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z) multiplies every
    non-identity Pauli by lam = 1 - 4p/3. Its inverse multiplies them by
    1/lam, which is the Pauli "channel" with weights q_I (identity) and q_P
    (each of X, Y, Z). The q_P are negative: not a probability, hence
    "quasi". Returns (q_I, q_P, gamma) with gamma = |q_I| + 3 |q_P| >= 1 the
    sampling overhead of this one location.
    """
    lam = 1 - 4 * p / 3
    q_i = (1 + 3 / lam) / 4
    q_p = (1 - 1 / lam) / 4
    return q_i, q_p, abs(q_i) + 3 * abs(q_p)


def pec_mps(n, params, chi, p, rng, p_model=None, fold=1):
    """One probabilistic-error-cancellation run: (MPS or None, signed weight).

    Each touched qubit suffers the real depolarizing error (rate p) and then a
    correction drawn from the inverse channel built for the ASSUMED rate
    p_model (default p, a perfectly known noise). A correction other than the
    identity is drawn with probability 3|q_P| / gamma and flips the sign of the
    weight; the overall weight is gamma_total * product of signs. The state is
    None when nothing at all was applied (the ideal circuit), so callers can
    reuse the noiseless result.
    """
    p_model = p if p_model is None else p_model
    q_i, q_p, gamma = inverse_depolarizing(p_model)
    # Probability that a location gets a (negative-weight) corrective Pauli.
    p_fix = 3 * abs(q_p) / gamma
    steps = fold_steps(layer_plan(params), fold)
    events, sign = [], 1
    for parity, _ in steps:
        qubits = touched_qubits(n, parity)
        hit = rng.random(len(qubits)) < p
        fix = rng.random(len(qubits)) < p_fix
        sign *= (-1) ** int(fix.sum())
        events.append((hit, fix))
    weight = sign * gamma ** sum(len(touched_qubits(n, par)) for par, _ in steps)
    if not any(h.any() or f.any() for h, f in events):
        return None, weight
    mps = MPS.singlet_product(n, chi)
    mps.cutoff = 1e-10
    for (parity, angle), (hit, fix) in zip(steps, events):
        mps.apply_group(parity, angle)
        qubits = touched_qubits(n, parity)
        for k in np.flatnonzero(hit | fix):
            # The real error first, then the sampled correction.
            if hit[k]:
                apply_pauli(mps, qubits[k], pick_pauli(rng, None))
            if fix[k]:
                apply_pauli(mps, qubits[k], pick_pauli(rng, None))
    return mps, weight


def dm_pec_state(n, params, p, p_model=None, fold=1, weights=None):
    """Exact density matrix after noise followed by the quasi-probability inverse.

    With p_model equal to p the result is the ideal state (up to rounding);
    with a different p_model it shows the bias left by a mis-estimated noise.
    `weights` gives the relative X, Y, Z probabilities of the REAL noise (None:
    depolarizing); the correction always assumes depolarizing, so a different
    `weights` shows what a wrongly modelled noise type does.
    """
    w = np.ones(3) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    q_i, q_p, _ = inverse_depolarizing(p if p_model is None else p_model)
    psi = singlet_vector(n).reshape((2,) * n).transpose(list(range(n))[::-1])
    rho = np.multiply.outer(psi, psi.conj())
    for parity, angle in fold_steps(layer_plan(params), fold):
        u = bond_unitary(angle)
        for b in group_terms(n, parity):
            rho = dm_apply(rho, [b, b + 1], u)
            rho = dm_apply(rho, [n + b, n + b + 1], u.conj())
        for q in touched_qubits(n, parity):
            noisy = (1 - p) * rho
            for name, weight in zip("XYZ", w):
                term = dm_apply(rho, [q], PAULI[name])
                term = dm_apply(term, [n + q], PAULI[name].conj())
                noisy = noisy + (p * weight) * term
            fixed = q_i * noisy
            for name in "XYZ":
                term = dm_apply(noisy, [q], PAULI[name])
                term = dm_apply(term, [n + q], PAULI[name].conj())
                fixed = fixed + q_p * term
            rho = fixed
    return rho


# ----------------------------------------------------------------------------
# Total-spin (S^2) verification (Lab 36)
# ----------------------------------------------------------------------------
def s2_matrix(n):
    """Total spin operator S^2 as a dense 2^n x 2^n matrix (site-order basis)."""
    dim = 2**n
    idx = np.arange(dim)
    # sigma_i . sigma_j = 2 SWAP_ij - 1, so
    # S^2 = 3n/4 + sum_{i<j} (SWAP_ij - 1/2) = 3n/4 - n(n-1)/4 + sum SWAP_ij.
    mat = np.zeros((dim, dim))
    for i in range(n):
        for j in range(i + 1, n):
            bi = (idx >> i) & 1
            bj = (idx >> j) & 1
            # A SWAP exchanges bits i and j: flip both when they differ.
            swapped = np.where(bi != bj, idx ^ ((1 << i) | (1 << j)), idx)
            mat[swapped, idx] += 1.0
    return mat + (3 * n / 4 - n * (n - 1) / 4) * np.eye(dim)


def singlet_projector(n):
    """Projector onto the S = 0 subspace (dimension = Catalan(n/2))."""
    w, v = np.linalg.eigh(s2_matrix(n))
    # Eigenvalues are s(s+1); anything below 0.5 is the singlet sector.
    v0 = v[:, w < 0.5]
    return v0 @ v0.T


def dm_s2_verified(n, params, p, fold=1, weights=None, proj=None):
    """Exact post-selected energy and acceptance for total spin S = 0."""
    rho = dm_state(n, params, p, fold, weights).reshape(2**n, 2**n)
    pr = singlet_projector(n) if proj is None else proj
    rho0 = pr @ rho @ pr
    # The trace of the projected matrix is the probability of S = 0.
    acceptance = float(np.trace(rho0).real)
    return dm_expectation(rho0.reshape((2,) * (2 * n)), n) / acceptance, acceptance


def dm_with_errors(n, params, errors):
    """Density matrix after the circuit with chosen Pauli errors inserted.

    `errors` maps (step, qubit) to "X", "Y" or "Z"; step counts the gate
    layers (group steps) from 0, and the error acts right after that step.
    """
    psi = singlet_vector(n)
    psi = psi.reshape((2,) * n).transpose(list(range(n))[::-1])
    rho = np.multiply.outer(psi, psi.conj())
    for step, (parity, angle) in enumerate(layer_plan(params)):
        u = bond_unitary(angle)
        for b in group_terms(n, parity):
            rho = dm_apply(rho, [b, b + 1], u)
            rho = dm_apply(rho, [n + b, n + b + 1], u.conj())
        for q in touched_qubits(n, parity):
            name = errors.get((step, q))
            if name is not None:
                rho = dm_apply(rho, [q], PAULI[name])
                rho = dm_apply(rho, [n + q], PAULI[name].conj())
    return rho


def dm_damping_state(n, params, gamma, fold=1, frames=False):
    """Exact density matrix with amplitude damping after every bond gate.

    frames=False is the plain channel on both qubits of each bond (identical to
    dm_kraus_state). frames=True averages over the Lab 34 Pauli frames
    P (x) P around each gate, i.e. uses Kraus operators
    (P (x) P) (K_a (x) K_b) (P (x) P) / 2.
    """
    psi = singlet_vector(n).reshape((2,) * n).transpose(list(range(n))[::-1])
    rho = np.multiply.outer(psi, psi.conj())
    ks = amplitude_damping_kraus(gamma)
    # Four two-qubit Kraus operators: the channel acts on both qubits of the bond.
    pair = [np.kron(a, b) for a in ks for b in ks]
    names = FRAMES if frames else "I"
    for parity, angle in fold_steps(layer_plan(params), fold):
        u = bond_unitary(angle)
        for b in group_terms(n, parity):
            rho = dm_apply(rho, [b, b + 1], u)
            rho = dm_apply(rho, [n + b, n + b + 1], u.conj())
            total = 0
            for name in names:
                frame = np.kron(PAULI[name], PAULI[name])
                for k in pair:
                    kk = frame @ k @ frame / np.sqrt(len(names))
                    term = dm_apply(rho, [b, b + 1], kk)
                    total = total + dm_apply(term, [n + b, n + b + 1], kk.conj())
            rho = total
    return rho


# ----------------------------------------------------------------------------
# CNOT-level randomised compiling (Lab 38)
# ----------------------------------------------------------------------------
_CNOT01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)
_CNOT10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], complex)


def _local(first, second):
    """Tensor product of two one-qubit matrices (first acts on the left qubit)."""
    return np.kron(first, second)


def compiled_bond_steps(angle, amp=1.0):
    """exp(-i angle (XX+YY+ZZ)) as three CNOTs plus one-qubit rotations.

    Returns [(kind, 4x4 matrix)] in time order, kind "C" for a CNOT and "1" for
    a layer of one-qubit gates. The construction is the three-CNOT circuit for
    exp(-i(a XX + b YY + c ZZ)) with a = b = c = angle (up to a global phase).
    `amp` scales the three angle-dependent rotations (an amplitude
    miscalibration): one factor for all, or three (one per rotation); amp = 1 is
    exact.
    """
    ident = PAULI["I"]
    half = np.pi / 2

    def rz(t):
        """Rotation about Z by angle t."""
        return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])

    def ry(t):
        """Rotation about Y by angle t."""
        return rotation("Y", t)

    # Angles of the three middle rotations (Vatan and Williams, 2004).
    # One amplitude factor, or one per rotation (theta, phi, psi).
    amp_t, amp_p, amp_s = np.broadcast_to(np.asarray(amp, dtype=float), (3,))
    theta = -(half - 2 * angle) * amp_t
    phi = (2 * angle - half) * amp_p
    psi = -(2 * angle - half) * amp_s
    return [
        ("1", _local(ident, rz(half))),
        ("C", _CNOT10),
        ("1", _local(rz(theta), ry(phi))),
        ("C", _CNOT01),
        ("1", _local(ident, ry(psi))),
        ("C", _CNOT10),
        ("1", _local(rz(-half), ident)),
    ]


def cnot_error(zz, xi):
    """Coherent two-qubit error after a CNOT: exp(-i zz ZZ/2) exp(-i xi X(x)I/2)."""
    zz_op = np.kron(PAULI["Z"], PAULI["Z"])
    xi_op = np.kron(PAULI["X"], PAULI["I"])
    one = np.eye(4)
    return (np.cos(zz / 2) * one - 1j * np.sin(zz / 2) * zz_op) @ (
        np.cos(xi / 2) * one - 1j * np.sin(xi / 2) * xi_op
    )


def _pauli_pairs():
    """All sixteen two-qubit Pauli matrices, as (A (x) B)."""
    return [np.kron(PAULI[a], PAULI[b]) for a in "IXYZ" for b in "IXYZ"]


def dm_compiled_state(n, params, zz, xi, mode="none", fold=1):
    """Exact density matrix of the CNOT-compiled circuit with coherent errors.

    mode  "none": the same error E after every CNOT, every time;
          "sym" : Lab 34 frames P (x) P around each bond (4 frames);
          "full": randomised compiling, a random Pauli pair around every CNOT,
                  i.e. the error is averaged over B E B^+ for all 16 Paulis B.
    """
    psi = singlet_vector(n).reshape((2,) * n).transpose(list(range(n))[::-1])
    rho = np.multiply.outer(psi, psi.conj())
    # Same circuit for all three modes; only the random Paulis differ.
    err = cnot_error(zz, xi)
    pairs = _pauli_pairs()
    frames = [np.kron(PAULI[k], PAULI[k]) for k in FRAMES]

    def app(state, bond, mat):
        """Apply a two-qubit unitary to the ket and its conjugate to the bra."""
        state = dm_apply(state, [bond, bond + 1], mat)
        return dm_apply(state, [n + bond, n + bond + 1], mat.conj())

    # Each bond gate is expanded into its three CNOTs and the one-qubit rotations.
    for parity, angle in fold_steps(layer_plan(params), fold):
        steps = compiled_bond_steps(angle)
        for b in group_terms(n, parity):
            if mode == "sym":
                # The exact average over the four frames, as a mixture of unitaries.
                total = 0
                for frame in frames:
                    r = app(rho, b, frame)
                    for kind, mat in steps:
                        r = app(r, b, mat)
                        if kind == "C":
                            r = app(r, b, err)
                    total = total + app(r, b, frame)
                rho = total / len(frames)
                continue
            for kind, mat in steps:
                rho = app(rho, b, mat)
                if kind != "C":
                    continue
                if mode == "none":
                    rho = app(rho, b, err)
                else:
                    # Conjugating E by every Pauli pair B gives a Pauli channel.
                    total = 0
                    for pauli in pairs:
                        total = total + app(rho, b, pauli @ err @ pauli.conj().T)
                    rho = total / len(pairs)
    return rho


def compiled_mps(n, params, chi, zz, xi, rng, mode="full", fold=1):
    """One trajectory of the CNOT-compiled circuit with random Pauli pairs.

    mode "full" draws a Pauli pair A before every CNOT and the compensating
    B = CNOT A^+ CNOT^+ after the error, so that the ideal circuit is unchanged;
    mode "sym" draws one P (x) P frame per bond; "none" draws nothing.
    """
    mps = MPS.singlet_product(n, chi)
    # One random branch per call; averaging many calls gives the channel.
    mps.cutoff = 1e-10
    err = cnot_error(zz, xi)
    pairs = _pauli_pairs()
    frames = [np.kron(PAULI[k], PAULI[k]) for k in FRAMES]
    for parity, angle in fold_steps(layer_plan(params), fold):
        steps = compiled_bond_steps(angle)
        for b in group_terms(n, parity):
            # A fresh frame per bond and layer, including folded copies.
            frame = frames[rng.integers(4)] if mode == "sym" else None
            if frame is not None:
                apply_bond_matrix(mps, b, frame)
            for kind, mat in steps:
                if kind == "C" and mode == "full":
                    # A fresh random Pauli pair before every CNOT.
                    pre = pairs[rng.integers(16)]
                    apply_bond_matrix(mps, b, pre)
                    apply_bond_matrix(mps, b, mat)
                    apply_bond_matrix(mps, b, err)
                    # Compensation: B CNOT A = CNOT, so B = CNOT A^+ CNOT^+.
                    apply_bond_matrix(mps, b, mat @ pre.conj().T @ mat.conj().T)
                else:
                    apply_bond_matrix(mps, b, mat)
                    if kind == "C":
                        apply_bond_matrix(mps, b, err)
            if frame is not None:
                apply_bond_matrix(mps, b, frame)
    return mps


# ----------------------------------------------------------------------------
# Total spin from product-basis measurements (Lab 39)
# ----------------------------------------------------------------------------
def dm_basis_probs(rho, n):
    """Outcome probabilities of measuring every qubit in the X, Y and Z bases.

    Returns {"X": p, "Y": p, "Z": p}, each an array over the 2^n bit strings in
    site order (site 0 is the most significant bit).
    """
    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    # Rotations that map the X and Y eigenbases onto the computational basis.
    to_z = {
        "Z": np.eye(2, dtype=complex),
        "X": hadamard,
        "Y": hadamard @ np.diag([1, -1j]).conj().T,
    }
    out = {}
    for name, rot in to_z.items():
        state = rho
        for q in range(n):
            state = dm_apply(state, [q], rot)
            state = dm_apply(state, [n + q], rot.conj())
        diag = np.real(np.diag(state.reshape(2**n, 2**n)))
        out[name] = diag / diag.sum()
    return out


def basis_shot_values(n):
    """Per-outcome energy and S^2 contributions of one measurement setting.

    For a bit string with spins s_i = +-1: the bond sum is sum s_i s_(i+1)
    (the XX, YY or ZZ part of H) and (sum s_i)^2 / 4 is that axis' (S^a)^2.
    Summing the three settings gives the full energy and the full S^2.
    """
    idx = np.arange(2**n)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    spins = 1 - 2 * bits
    energy = (spins[:, :-1] * spins[:, 1:]).sum(axis=1).astype(float)
    return energy, (spins.sum(axis=1) ** 2) / 4.0


def dm_energy_and_s2(rho, n):
    """Exact energy and <S^2> of a density matrix from the three settings."""
    probs = dm_basis_probs(rho, n)
    energy_vals, s2_vals = basis_shot_values(n)
    return (
        float(sum(probs[a] @ energy_vals for a in "XYZ")),
        float(sum(probs[a] @ s2_vals for a in "XYZ")),
    )


# ----------------------------------------------------------------------------
# Ansatz families that break the symmetry (Lab 40)
# ----------------------------------------------------------------------------
def general_gate(a, b, c):
    """exp(-i (a XX + b YY + c ZZ)) as a 4x4 matrix.

    The three terms commute, so the exponential is an exact product. With
    a = b = c this is `bond_unitary(a)`, which conserves total spin. Any other
    choice breaks S^2; a != b also breaks S_z.
    """
    out = np.eye(4, dtype=complex)
    for angle, name in ((a, "X"), (b, "Y"), (c, "Z")):
        pp = np.kron(PAULI[name], PAULI[name])
        out = out @ (np.cos(angle) * np.eye(4) - 1j * np.sin(angle) * pp)
    return out


def aniso_steps(angles, xi=0.0, zeta=0.0):
    """HVA angles as (parity, (a, b, c)) steps with a controlled anisotropy.

    xi   : XY anisotropy, a = t (1 + xi), b = t (1 - xi); breaks S_z and S^2.
    zeta : ZZ anisotropy, c = t (1 + zeta); keeps S_z, breaks S^2.
    With xi = zeta = 0 the circuit is the ordinary HVA of Labs 28-39.
    """
    return [
        (parity, (t * (1 + xi), t * (1 - xi), t * (1 + zeta)))
        for parity, t in layer_plan(angles)
    ]


def free_steps(x):
    """Steps from a flat vector holding three free angles per half layer."""
    rows = np.asarray(x, dtype=float).reshape(-1, 3)
    # Same order as layer_plan: the odd group first, then the even group.
    return [(1 - k % 2, tuple(row)) for k, row in enumerate(rows)]


def _fold_general(steps, fold):
    """Repeat each step as U (U^dagger U)^k; U^dagger flips all three angles."""
    if fold < 1 or fold % 2 == 0:
        raise ValueError(f"fold factor must be a positive odd integer, got {fold}")
    out = []
    for parity, abc in steps:
        for k in range(fold):
            sign = 1.0 if k % 2 == 0 else -1.0
            out.append((parity, sign * np.asarray(abc, dtype=float)))
    return out


def _site_order_singlets(n):
    """The singlet-product start state as a tensor with one axis per site."""
    psi = singlet_vector(n).reshape((2,) * n)
    # singlet_vector is little endian; flip the axes to site order.
    return psi.transpose(list(range(n))[::-1])


def _step_gates(n, parity, abc):
    """The 4x4 gate of every bond in one group.

    `abc` is either one (a, b, c) triple shared by all bonds, or a list with
    one triple per bond (the per-bond ansatz).
    """
    count = len(group_terms(n, parity))
    arr = np.asarray(abc, dtype=float)
    if arr.ndim == 1:
        return [general_gate(*arr)] * count
    if len(arr) != count:
        raise ValueError(f"expected {count} bond triples, got {len(arr)}")
    return [general_gate(*row) for row in arr]


def general_sv(n, steps):
    """Pure state after the noiseless circuit, as a tensor (one axis per site)."""
    psi = _site_order_singlets(n)
    for parity, abc in steps:
        for b, u in zip(group_terms(n, parity), _step_gates(n, parity, abc)):
            psi = dm_apply(psi, [b, b + 1], u)
    return psi


def general_energy(n, steps):
    """Noiseless energy <H> of the circuit, from the state tensor."""
    psi = general_sv(n, steps)
    total = 0.0
    for b in range(n - 1):
        total += np.vdot(psi, dm_apply(psi, [b, b + 1], BOND)).real
    return float(total)


def general_dm_state(n, steps, p, fold=1, weights=None):
    """Density matrix after the noisy circuit for any (a, b, c) steps.

    Same channel as `dm_state`: depolarizing (relative X, Y, Z weights) on every
    qubit a gate layer touched. With a = b = c it reproduces `dm_state`.
    """
    w = np.ones(3) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    psi = _site_order_singlets(n)
    rho = np.multiply.outer(psi, psi.conj())
    for parity, abc in _fold_general(steps, fold):
        for b, u in zip(group_terms(n, parity), _step_gates(n, parity, abc)):
            rho = dm_apply(rho, [b, b + 1], u)
            rho = dm_apply(rho, [n + b, n + b + 1], u.conj())
        for q in touched_qubits(n, parity):
            mixed = (1 - p) * rho
            for name, weight in zip("XYZ", w):
                s = dm_apply(rho, [q], PAULI[name])
                s = dm_apply(s, [n + q], PAULI[name].conj())
                mixed = mixed + (p * weight) * s
            rho = mixed
    return rho


def perbond_steps(n, x):
    """Symmetry-adapted steps with ONE ANGLE PER BOND (a = b = c on each bond).

    `x` lists the bond angles group by group in circuit order (odd group first,
    then even, repeated per layer). Every gate is exp(-i t (XX + YY + ZZ)), so
    total spin is conserved exactly however the angles differ.
    """
    steps, i = [], 0
    while i < len(x):
        parity = 1 - len(steps) % 2
        count = len(group_terms(n, parity))
        steps.append((parity, [(t, t, t) for t in x[i : i + count]]))
        i += count
    return steps


def perbond_from_hva(n, angles):
    """Per-bond angles equal to an ordinary HVA's one angle per group."""
    out = []
    for parity, t in layer_plan(angles):
        out += [t] * len(group_terms(n, parity))
    return np.array(out)


def train_steps(n, x0, build):
    """Minimize the noiseless energy of build(x) by L-BFGS from x0."""
    res = minimize(
        lambda x: general_energy(n, build(x)),
        np.asarray(x0, dtype=float),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-14, "gtol": 1e-10},
    )
    return res.x, float(res.fun)


def train_aniso(n, x0, xi, zeta):
    """Retrain the half-layer angles at fixed anisotropy; return (angles, E)."""
    return train_steps(n, x0, lambda x: aniso_steps(x, xi, zeta))


def rho_projected(rho, n, kind, proj=None):
    """Energy and acceptance of rho after projecting onto S_z = 0 or S = 0.

    kind is "Sz" or "S2". `rho` is the tensor from `general_dm_state` or
    `dm_state`. Returns (energy of the projected state, probability to pass).
    """
    mat = rho.reshape(2**n, 2**n)
    if kind == "S2":
        pr = singlet_projector(n) if proj is None else proj
        mat0 = pr @ mat @ pr
    else:
        # Site order: bit 1 is spin down, so S_z = 0 means n/2 ones.
        bits = (np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1
        keep = (bits.sum(axis=1) == n // 2).astype(float)
        mat0 = mat * keep[:, None] * keep[None, :]
    acceptance = float(np.trace(mat0).real)
    return dm_expectation(mat0.reshape((2,) * (2 * n)), n) / acceptance, acceptance


# ----------------------------------------------------------------------------
# The Clifford-conjugated bond gate (Lab 41)
# ----------------------------------------------------------------------------
_HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def bond_clifford():
    """The fixed Clifford C = (H on the left qubit) . CNOT(left -> right).

    C maps the two-qubit singlet to |11> and the triplet onto the span of
    |00>, |01> and |10>, so the bond gate is a phase on |11> in C's frame.
    """
    return np.kron(_HADAMARD, PAULI["I"]) @ _CNOT01


def clifford_bond_steps(angle, amp=1.0):
    """exp(-i angle (XX+YY+ZZ)) as Clifford . controlled phase . Clifford.

    exp(-i t sigma.sigma) = e^(-it) P_triplet + e^(3it) P_singlet, a phase of
    4t on the singlet. Returns [(kind, 4x4 matrix)] in time order (kind "C" for
    a two-qubit gate, "1" for one-qubit gates): CNOT, H, CPHASE(4 t amp), H,
    CNOT. The ONLY angle dependence is in the controlled phase; `amp` scales
    its phase (an amplitude miscalibration of that one gate).
    """
    had = np.kron(_HADAMARD, PAULI["I"])
    cphase = np.diag([1, 1, 1, np.exp(4j * angle * amp)])
    return [("C", _CNOT01), ("1", had), ("C", cphase), ("1", had), ("C", _CNOT01)]


def compiled_sv(n, steps, bond_seq, **kwargs):
    """Pure state of a circuit whose every bond is compiled by `bond_seq`.

    `steps` is a list of (parity, angles) as for `general_sv`, with one angle
    per bond or one shared angle (only the first of a triple is used, since
    a = b = c). `bond_seq(angle, **kwargs)` returns the [(kind, matrix)] list.
    """
    psi = _site_order_singlets(n)
    for parity, abc in steps:
        bonds = group_terms(n, parity)
        arr = np.asarray(abc, dtype=float)
        angles = [arr[0]] * len(bonds) if arr.ndim == 1 else arr[:, 0]
        for b, t in zip(bonds, angles):
            for _, mat in bond_seq(t, **kwargs):
                psi = dm_apply(psi, [b, b + 1], mat)
    return psi


def tensor_energy(n, psi):
    """<H> of a pure state tensor with one axis per site."""
    total = 0.0
    for b in range(n - 1):
        total += np.vdot(psi, dm_apply(psi, [b, b + 1], BOND)).real
    return float(total)


def tensor_s2(n, psi):
    """<S^2> of a pure state tensor with one axis per site."""
    vec = psi.reshape(-1)
    return float(np.vdot(vec, s2_matrix(n) @ vec).real)
