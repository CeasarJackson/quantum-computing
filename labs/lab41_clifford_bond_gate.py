#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab41_clifford_bond_gate.py

Purpose
-------
The Heisenberg bond gate exp(-i t (XX + YY + ZZ)) is a PHASE on the two-qubit
singlet: e^(-it) on the triplet and e^(3it) on the singlet, i.e. a relative
phase of 4t. A fixed Clifford C = H(left) . CNOT(left -> right) maps the
singlet to |11> and the triplet onto {|00>, |01>, |10>}, so

    exp(-i t (XX + YY + ZZ)) = e^(-it) C^+ CPHASE(4t) C.

Everything that depends on the angle is ONE controlled-phase gate; the rest is
two fixed Clifford blocks. The usual three-CNOT compilation of Labs 38 (Vatan
and Williams) also uses three CNOTs, but spreads the angle over three
one-qubit rotations. This lab finds out what the difference is worth.

  * Under uniform two-qubit depolarizing noise after every two-qubit gate the
    two compilations are IDENTICAL at first order (exact fault table); the
    benefit is not there.
  * It is in AMPLITUDE (calibration) errors of the angle-dependent gates. Any
    error confined to the one controlled phase just changes the bond angle, so
    it conserves spin exactly and a retrained circuit absorbs it completely.
    Independent errors on the three rotations of the other compilation leak
    spin, are about 10x larger and can only be partly retrained away.
  * A common-mode error (one factor on all three rotations) stays symmetric in
    both compilations: bond angle a becomes (1 + e) a - e pi/4 in the
    three-CNOT form but (1 + e) a for the controlled phase, so the latter is
    smaller by the factor ((a - pi/4)/a)^2 whenever |a| < pi/4.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/lab41_clifford_bond_gate.py
    python3 labs/lab41_clifford_bond_gate.py --sigma 0.03 --draws 8
    python3 labs/lab41_clifford_bond_gate.py --quick

Log files
---------
logs/lab41/lab41_<YYYYmmdd_HHMMSS>.log

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

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

from applog import get_logger
from mpslib import PAULI, bond_unitary, group_terms, sparse_hamiltonian, train_angles
from noiselib import (
    _site_order_singlets,
    bond_clifford,
    clifford_bond_steps,
    compiled_bond_steps,
    compiled_sv,
    dm_apply,
    general_energy,
    general_sv,
    perbond_from_hva,
    perbond_steps,
    singlet_projector,
    tensor_energy,
    tensor_s2,
    train_steps,
)

log = get_logger("lab41")

# The two-qubit Paulis, as (name, matrix) pairs without the identity.
PAULI_PAIRS = [
    (a + b, np.kron(PAULI[a], PAULI[b]))
    for a in "IXYZ"
    for b in "IXYZ"
    if a + b != "II"
]


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def ground_energy(n):
    """Exact ground-state energy of the open Heisenberg chain."""
    return float(
        eigsh(sparse_hamiltonian(n), k=1, which="SA", return_eigenvectors=False)[0].real
    )


def sequence_matrix(seq):
    """Product of a [(kind, matrix)] gate list, later gates on the left."""
    total = np.eye(4, dtype=complex)
    for _, mat in seq:
        total = mat @ total
    return total


def up_to_phase(a, b):
    """Largest entry of |a - e^(i phi) b| with the best global phase phi."""
    overlap = np.vdot(b.reshape(-1), a.reshape(-1))
    phase = overlap / abs(overlap)
    return float(np.abs(a - phase * b).max())


def amp_state(n, x, seq, amps):
    """State of the compiled circuit with one amplitude factor set per bond.

    `amps` holds one entry per bond in circuit order: a scalar for the
    controlled-phase form, or a triple (one factor per rotation) for the
    three-CNOT form.
    """
    # A fresh iterator per call, so the k-th bond always gets the k-th factor.
    it = iter(amps)

    def bond(angle):
        """Compile one bond with its own amplitude factor."""
        return seq(angle, amp=next(it))

    return compiled_sv(n, perbond_steps(n, x), bond)


def retrain(n, x, seq, amps):
    """Re-optimize the bond angles of the miscalibrated circuit by L-BFGS."""
    res = minimize(
        lambda w: tensor_energy(n, amp_state(n, w, seq, amps)),
        x,
        method="L-BFGS-B",
        options={"maxiter": 1500, "ftol": 1e-14, "gtol": 1e-10},
    )
    return res.x


def projected_energy(n, psi, proj):
    """Energy of a pure state after projecting onto total spin S = 0."""
    vec = proj @ psi.reshape(-1)
    norm = float(np.vdot(vec, vec).real)
    return tensor_energy(n, vec.reshape((2,) * n) / np.sqrt(norm)), norm


def fault_table(n, x, seq):
    """Energy and <S^2> change of every single two-qubit Pauli fault.

    A fault is one of the 15 two-qubit Paulis applied right after one
    two-qubit gate of the compiled circuit. Returns arrays (dE, dS2) over all
    faults, relative to the fault-free circuit.
    """
    ops = []
    for parity, gates in perbond_steps(n, x):
        for b, (t, _, _) in zip(group_terms(n, parity), gates):
            ops += [(b, mat, kind) for kind, mat in seq(t)]

    def run(fault):
        """Energy and <S^2> with an optional (op index, pauli pair) fault."""
        psi = _site_order_singlets(n)
        for k, (b, mat, _) in enumerate(ops):
            psi = dm_apply(psi, [b, b + 1], mat)
            if fault is not None and fault[0] == k:
                # The fault is a two-qubit Pauli on the pair the gate just used.
                psi = dm_apply(psi, [b, b + 1], fault[1])
        return tensor_energy(n, psi), tensor_s2(n, psi)

    e_ref, s_ref = run(None)
    d_e, d_s = [], []
    for k, (_, _, kind) in enumerate(ops):
        if kind != "C":
            continue
        for _, pauli in PAULI_PAIRS:
            e, s = run((k, pauli))
            d_e.append(e - e_ref)
            d_s.append(s - s_ref)
    return np.array(d_e), np.array(d_s)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_identity(r, ctx):
    """Check 1: the bond gate is a phase on the singlet, in a Clifford frame."""
    log.info("=== Check 1: Clifford . controlled phase . Clifford ===")
    rng = np.random.default_rng(ctx["seed"])
    cliff = bond_clifford()
    worst = 0.0
    # Random angles, so the identity is not tested at a few special values only.
    for t in rng.uniform(-1.5, 1.5, 20):
        cphase = np.diag([1, 1, 1, np.exp(4j * t)])
        lhs = np.exp(-1j * t) * cliff.conj().T @ cphase @ cliff
        worst = max(worst, float(np.abs(lhs - bond_unitary(t)).max()))
    log.info("largest deviation over 20 random angles: %.1e", worst)
    r.append(check(worst < 1e-13, "bond(t) = e^(-it) C^+ CPHASE(4t) C for random t"))
    # A Clifford maps every Pauli to a Pauli (up to sign), so errors stay Paulis.
    mapped = 0
    for _, pauli in PAULI_PAIRS:
        image = cliff @ pauli @ cliff.conj().T
        mapped += any(
            np.abs(image - s * q).max() < 1e-12 for _, q in PAULI_PAIRS for s in (1, -1)
        )
    r.append(check(mapped == 15, "C maps all 15 two-qubit Paulis to Paulis"))
    # The two-qubit singlet (|01> - |10>)/sqrt(2): the state the bond gate phases.
    singlet = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    r.append(
        check(
            np.abs(cliff @ singlet - np.array([0, 0, 0, 1])).max() < 1e-13,
            "C sends the singlet to |11>",
        )
    )
    # Its projector; the triplet is everything orthogonal to it.
    proj_s = np.outer(singlet, singlet.conj())
    r.append(
        check(
            np.abs(cliff @ proj_s @ cliff.conj().T - np.diag([0, 0, 0, 1])).max()
            < 1e-13,
            "and the whole singlet projector to |11><11|",
        )
    )
    # t = pi/4 gives CPHASE(pi) = CZ, a Clifford; the bond gate is a SWAP.
    # SWAP, the one gate that is both a bond gate and a Clifford.
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    r.append(
        check(
            up_to_phase(bond_unitary(np.pi / 4), swap.astype(complex)) < 1e-12,
            "t = pi/4: the controlled phase is CZ and the bond gate is SWAP",
        )
    )
    # The compiled gate lists must reproduce the bond gate (global phase aside).
    for name, seq in (
        ("controlled-phase form", clifford_bond_steps),
        ("three-CNOT form", compiled_bond_steps),
    ):
        err = max(
            up_to_phase(sequence_matrix(seq(t)), bond_unitary(t)) for t in (0.3, -0.7)
        )
        r.append(check(err < 1e-12, f"{name} reproduces the bond gate"))
    # Where does the angle live? Compare two angles gate by gate.
    # Two different angles: any gate that differs between them carries the angle.
    cp_a, cp_b = clifford_bond_steps(0.2), clifford_bond_steps(0.5)
    vw_a, vw_b = compiled_bond_steps(0.2), compiled_bond_steps(0.5)
    differ_cp = [
        i
        for i, (u, v) in enumerate(zip(cp_a, cp_b))
        if np.abs(u[1] - v[1]).max() > 1e-12
    ]
    differ_vw = [
        i
        for i, (u, v) in enumerate(zip(vw_a, vw_b))
        if np.abs(u[1] - v[1]).max() > 1e-12
    ]
    log.info(
        "gates that change with the angle: controlled-phase form %s of %d; "
        "three-CNOT form %s of %d",
        differ_cp,
        len(cp_a),
        differ_vw,
        len(vw_a),
    )
    r.append(
        check(differ_cp == [2], "controlled-phase form: only the CPHASE depends on t")
    )
    r.append(
        check(
            differ_vw == [2, 4],
            "three-CNOT form: three rotations, in two layers, depend on t",
        )
    )
    # Folding U -> U U^+ U needs U^+ = bond(-t): same Cliffords, CPHASE sign flipped.
    prod = sequence_matrix(clifford_bond_steps(-0.4)) @ sequence_matrix(
        clifford_bond_steps(0.4)
    )
    r.append(
        check(
            up_to_phase(prod, np.eye(4, dtype=complex)) < 1e-12,
            "the inverse gate only flips the sign of the CPHASE angle",
        )
    )


def check_circuits(r, ctx):
    """Check 2: whole compiled circuits equal the ideal per-bond circuit."""
    log.info("=== Check 2: compiled circuits reproduce the ideal state ===")
    for n in (6, 8):
        x = ctx[f"x{n}"]
        # The reference: the ideal per-bond circuit, built from exact bond gates.
        ideal = general_sv(n, perbond_steps(n, x))
        for name, seq in (
            ("controlled-phase form", clifford_bond_steps),
            ("three-CNOT form", compiled_bond_steps),
        ):
            psi = compiled_sv(n, perbond_steps(n, x), seq)
            # |<ideal|compiled>| = 1 means equal states up to a global phase.
            fidelity = abs(np.vdot(ideal.reshape(-1), psi.reshape(-1)))
            r.append(check(abs(fidelity - 1) < 1e-12, f"n={n}: {name}, fidelity 1"))


def check_independent(r, ctx):
    """Check 3 and 4: independent amplitude errors on the angle-dependent gates."""
    n, sigma = 6, ctx["sigma"]
    x = ctx["x6"]
    nb = len(x)
    e_ideal = ctx["e_ideal6"]
    proj = singlet_projector(n)
    log.info("=== Check 3: independent amplitude errors (sigma %.3f) ===", sigma)
    # A separate stream for the error draws, so the HVA starts do not move them.
    rng = np.random.default_rng(ctx["seed"] + 1)
    # A is the three-CNOT form, B the controlled-phase form; one record per draw.
    rows = {"A": [], "B": []}
    for draw in range(ctx["draws"]):
        # Standard-normal errors; the rotation errors are independent, one per rotation.
        z = rng.normal(size=(nb, 3))
        # B has one angle-dependent gate per bond, so it gets one factor per bond.
        amps = {"A": 1 + sigma * z, "B": 1 + sigma * z[:, 0]}
        for key, seq in (("A", compiled_bond_steps), ("B", clifford_bond_steps)):
            psi = amp_state(n, x, seq, amps[key])
            err = tensor_energy(n, psi) - e_ideal
            # Projecting onto S = 0 is the symmetry check of Labs 36-40.
            e_proj, _ = projected_energy(n, psi, proj)
            # Re-optimize the angles ON the miscalibrated circuit, as the noise-aware
            # training of Lab 21 does.
            w = retrain(n, x, seq, amps[key])
            psi_r = amp_state(n, w, seq, amps[key])
            rows[key].append(
                dict(
                    err=err,
                    s2=tensor_s2(n, psi),
                    proj=e_proj - e_ideal,
                    retrained=tensor_energy(n, psi_r) - e_ideal,
                    s2_retrained=tensor_s2(n, psi_r),
                )
            )
        log.info(
            "draw %d: controlled phase err %.2e <S^2> %.0e retrained %.1e | "
            "three-CNOT err %.2e <S^2> %.1e retrained %.1e",
            draw,
            rows["B"][-1]["err"],
            rows["B"][-1]["s2"],
            rows["B"][-1]["retrained"],
            rows["A"][-1]["err"],
            rows["A"][-1]["s2"],
            rows["A"][-1]["retrained"],
        )
    # Average over draws, because a single draw can be lucky.
    mean = {
        k: {f: float(np.mean([d[f] for d in v])) for f in v[0]} for k, v in rows.items()
    }
    log.info(
        "mean controlled-phase form: %s", {k: f"{v:.2e}" for k, v in mean["B"].items()}
    )
    log.info(
        "mean three-CNOT form:       %s", {k: f"{v:.2e}" for k, v in mean["A"].items()}
    )
    r.append(
        check(
            max(abs(d["s2"]) for d in rows["B"]) < 1e-12,
            "controlled phase: every error draw keeps <S^2> = 0 exactly",
        )
    )
    r.append(
        check(
            min(d["s2"] for d in rows["A"]) > 1e-4,
            "three-CNOT: every error draw leaks spin (<S^2> > 1e-4)",
        )
    )
    r.append(
        check(
            mean["B"]["err"] < 0.25 * mean["A"]["err"],
            "the controlled-phase error is under a quarter of the three-CNOT error",
        )
    )
    r.append(
        check(
            max(abs(d["proj"] - d["err"]) for d in rows["B"]) < 1e-10,
            "S = 0 projection changes nothing for the controlled phase",
        )
    )
    r.append(
        check(
            mean["A"]["proj"] < 0.8 * mean["A"]["err"],
            "but it helps the three-CNOT form (partly)",
        )
    )
    r.append(
        check(
            max(abs(d["retrained"]) for d in rows["B"]) < 1e-6,
            "retraining ABSORBS every controlled-phase error (back to the ideal)",
        )
    )
    r.append(
        check(
            min(d["retrained"] for d in rows["A"]) > 1e-4,
            "retraining cannot absorb the three-CNOT errors (>1e-4 left)",
        )
    )
    # An error E(sigma) = 1/2 dx^T H dx is quadratic: doubling sigma quadruples it.
    # One fixed draw, scaled by sigma/2 and sigma, so only the size changes.
    z = np.random.default_rng(ctx["seed"] + 2).normal(size=(nb, 3))
    log.info("=== Check 4: the error is second order in the miscalibration ===")
    for key, seq in (
        ("controlled phase", clifford_bond_steps),
        ("three-CNOT", compiled_bond_steps),
    ):
        errs = []
        for s in (sigma / 2, sigma):
            amp = 1 + s * (z[:, 0] if key == "controlled phase" else z)
            errs.append(tensor_energy(n, amp_state(n, x, seq, amp)) - e_ideal)
        # A pure second-order law gives exactly 4; allow 10% for higher orders.
        ratio = errs[1] / errs[0]
        log.info("%s: error(sigma)/error(sigma/2) = %.2f", key, ratio)
        r.append(check(3.5 < ratio < 4.5, f"{key}: doubling the error quadruples it"))


def check_common_mode(r, ctx):
    """Check 5: one factor on every angle-dependent gate is a re-parameterization."""
    log.info("=== Check 5: a common-mode amplitude error ===")
    # A 2% common-mode miscalibration, large enough to see, small enough for 2nd order.
    n, eps = 6, 0.02
    x = ctx["x6"]
    e_ideal = ctx["e_ideal6"]
    nb = len(x)
    # B: phase 4 t -> 4 t (1 + e), so t -> t (1 + e).
    # Controlled-phase form: the phase 4t becomes 4t(1+e), so t becomes t(1+e).
    psi_b = amp_state(n, x, clifford_bond_steps, np.full(nb, 1 + eps))
    x_b = x * (1 + eps)
    ref_b = general_sv(n, perbond_steps(n, x_b))
    # A: each rotation angle 2t - pi/2 scales by (1 + e), so t -> t(1+e) - e pi/4.
    # Three-CNOT form: the rotations carry an offset pi/2 that is scaled too.
    psi_a = amp_state(n, x, compiled_bond_steps, np.full((nb, 3), 1 + eps))
    x_a = x * (1 + eps) - eps * np.pi / 4
    ref_a = general_sv(n, perbond_steps(n, x_a))
    for name, psi, ref in (
        ("controlled phase", psi_b, ref_b),
        ("three-CNOT", psi_a, ref_a),
    ):
        fid = abs(np.vdot(psi.reshape(-1), ref.reshape(-1)))
        r.append(
            check(
                abs(fid - 1) < 1e-10 and abs(tensor_s2(n, psi)) < 1e-12,
                f"{name}: a common-mode error is a pure change of bond angle",
            )
        )
    # Excess energy over the ideal circuit isolates the miscalibration.
    err_b = tensor_energy(n, psi_b) - e_ideal
    err_a = tensor_energy(n, psi_a) - e_ideal
    # Predicted ratio of the leading terms: the angle changes are
    # dx_B = eps x and dx_A = eps (x - pi/4); here the energy is quadratic in dx.
    log.info(
        "common-mode eps %.2f: controlled phase err %.2e, three-CNOT err %.2e "
        "(ratio %.1f)",
        eps,
        err_b,
        err_a,
        err_a / err_b,
    )
    r.append(
        check(
            err_a > 4 * err_b,
            "and the controlled phase is more than 4x smaller (|t| < pi/4)",
        )
    )
    for name, seq in (
        ("controlled phase", clifford_bond_steps),
        ("three-CNOT", compiled_bond_steps),
    ):
        amps = (
            np.full(nb, 1 + eps)
            if name == "controlled phase"
            else np.full((nb, 3), 1 + eps)
        )
        w = retrain(n, x, seq, amps)
        left = tensor_energy(n, amp_state(n, w, seq, amps)) - e_ideal
        r.append(check(abs(left) < 1e-6, f"{name}: retraining absorbs it completely"))


def check_faults(r, ctx):
    """Check 6: under two-qubit depolarizing noise the two forms are equivalent."""
    log.info("=== Check 6: exact fault table (every 2-qubit Pauli fault) ===")
    n, x = 6, ctx["x6"]
    # One (dE, dS2) table per compilation, over all 450 single faults.
    tables = {}
    for name, seq in (
        ("controlled phase", clifford_bond_steps),
        ("three-CNOT", compiled_bond_steps),
    ):
        d_e, d_s = fault_table(n, x, seq)
        tables[name] = (d_e, d_s)
        log.info(
            "%s: %d faults, mean dE %.4f, mean d<S^2> %.4f, kappa %.3f, "
            "fraction leaking spin %.3f",
            name,
            len(d_e),
            d_e.mean(),
            d_s.mean(),
            d_e.sum() / d_s.sum(),
            float(np.mean(d_s > 1e-9)),
        )
    (e_b, s_b), (e_a, s_a) = tables["controlled phase"], tables["three-CNOT"]
    r.append(
        check(
            len(e_b) == len(e_a) == 450, "30 two-qubit gates x 15 Paulis = 450 faults"
        )
    )
    # Uniform noise on a pair erases it, and every bond gate acts on that same
    # pair afterwards, so the average over the 15 Paulis cannot tell the forms apart.
    r.append(
        check(
            abs(e_b.mean() - e_a.mean()) < 1e-9 and abs(s_b.mean() - s_a.mean()) < 1e-9,
            "the mean fault effect is identical for both compilations",
        )
    )
    r.append(
        check(
            float(np.mean(s_b > 1e-9)) > 0.99 and float(np.mean(s_a > 1e-9)) > 0.99,
            "almost every two-qubit fault leaks spin (S^2 verification sees it)",
        )
    )
    # Ratio of total energy shift to total spin shift, as in Labs 36 and 39.
    kappa = e_b.sum() / s_b.sum()
    r.append(check(2.0 < kappa < 2.6, "first-order kappa = dE/d<S^2> is about 2.3"))


def check_longer(r, ctx):
    """Check 7: the same on 8 sites."""
    n, sigma = 8, ctx["sigma"]
    x, e_ideal = ctx["x8"], ctx["e_ideal8"]
    nb = len(x)
    log.info("=== Check 7: n=%d, independent errors (sigma %.3f) ===", n, sigma)
    rng = np.random.default_rng(ctx["seed"] + 3)
    # Per-draw (excess, <S^2>, retrained excess) for each compilation.
    res = {"A": [], "B": []}
    for _ in range(ctx["draws8"]):
        z = rng.normal(size=(nb, 3))
        for key, seq, amps in (
            ("A", compiled_bond_steps, 1 + sigma * z),
            ("B", clifford_bond_steps, 1 + sigma * z[:, 0]),
        ):
            psi = amp_state(n, x, seq, amps)
            w = retrain(n, x, seq, amps)
            psi_r = amp_state(n, w, seq, amps)
            res[key].append(
                (
                    tensor_energy(n, psi) - e_ideal,
                    tensor_s2(n, psi),
                    tensor_energy(n, psi_r) - e_ideal,
                )
            )
    a, b = np.array(res["A"]), np.array(res["B"])
    log.info("three-CNOT mean (err, <S^2>, retrained): %s", np.round(a.mean(axis=0), 6))
    log.info(
        "controlled phase mean (err, <S^2>, retrained): %s", np.round(b.mean(axis=0), 8)
    )
    r.append(
        check(np.abs(b[:, 1]).max() < 1e-12, f"n={n}: controlled phase keeps <S^2> = 0")
    )
    r.append(check(a[:, 1].min() > 1e-4, f"n={n}: three-CNOT leaks spin"))
    r.append(
        check(
            b[:, 0].mean() < 0.25 * a[:, 0].mean(),
            f"n={n}: controlled-phase error is smaller",
        )
    )
    r.append(
        check(np.abs(b[:, 2]).max() < 1e-6, f"n={n}: retraining absorbs it completely")
    )
    r.append(check(a[:, 2].min() > 1e-5, f"n={n}: retraining leaves three-CNOT errors"))


def build_context(seed, sigma, quick):
    """Train the per-bond circuits once (best of 12 HVA starts, then per-bond)."""
    ctx = {
        "seed": seed,
        "sigma": sigma,
        "draws": 3 if quick else 6,
        "draws8": 2 if quick else 3,
    }
    for n in (6, 8):
        e0 = ground_energy(n)
        best = None
        # The HVA landscape has plateaus (Lab 28); keep the best of several runs.
        for offset in range(12):
            trained = train_angles(
                n, e0, 2, np.random.default_rng(seed + offset), extra_starts=3
            )
            if best is None or trained[1][0] < best[0]:
                best = trained[1]
        x, _ = train_steps(
            n,
            perbond_from_hva(n, best[1]),
            lambda v, n=n: perbond_steps(n, v),
        )
        # t and t + pi/2 are the same gate up to a global phase, so the optimizer
        # may return a huge representative (9.2 for one seed). A calibration error
        # is a FRACTION of the physical rotation, so use the smallest one,
        # in (-pi/4, pi/4].
        x = (x + np.pi / 4) % (np.pi / 2) - np.pi / 4
        ctx[f"x{n}"] = x
        # Errors are measured against this circuit's own ideal energy.
        ctx[f"e_ideal{n}"] = general_energy(n, perbond_steps(n, x))
    return ctx


def main(argv=None):
    """Run every Lab 41 check; return the process exit code."""
    ap = argparse.ArgumentParser(
        description="Lab 41: the Clifford-conjugated bond gate"
    )
    ap.add_argument("--sigma", type=float, default=0.02, help="amplitude error")
    # The seed fixes the HVA starts and every random error draw.
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="fewer error draws")
    args = ap.parse_args(argv)
    # Beyond about 3% the 8-site errors leave the near-quadratic regime that the
    # checks describe, so the range stops there.
    if not 0.005 <= args.sigma <= 0.03:
        ap.error(f"--sigma must be between 0.005 and 0.03. Got {args.sigma}.")
    ctx = build_context(args.seed, args.sigma, args.quick)
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_identity(r, ctx)
    check_circuits(r, ctx)
    check_independent(r, ctx)
    check_common_mode(r, ctx)
    check_faults(r, ctx)
    check_longer(r, ctx)
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
