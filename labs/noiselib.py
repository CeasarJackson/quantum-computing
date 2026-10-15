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
