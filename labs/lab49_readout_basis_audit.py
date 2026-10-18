#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab49_readout_basis_audit.py

Purpose
-------
Labs 46 and 48 found that the ibm_fez Heisenberg energies lose about 0.28
before any bond gate, and that the loss is not the same in the three
measurement bases: at depth 0 the Z basis loses less than X and Y (A = -0.076
+/- 0.022), and from the first bond gate on Y loses 0.10 +/- 0.013 more than
X. No simulated gate mechanism (depolarizing, T1, dephasing, coherent
over-rotation) makes either pattern, so the MEASUREMENT is the suspect: the
readout correction of every lab so far was calibrated in the Z basis only
(|000000> and |111111>) and then reused for X and Y, which need an extra
single-qubit rotation before the readout, and it treats the qubits as
independent readers.

This lab tests that suspicion directly, with NO two-qubit gate at all. For each
basis b in Z, X, Y it prepares four product states that are eigenstates of
that basis,

    all0   000000       all1   111111       alt0   010101       alt1   101010

(bit v of a qubit means the eigenvalue (-1)^v of sigma_b), applies the SAME
basis-change rotations as Labs 44 to 48 (nothing for Z, H for X, S-dagger then
H for Y) and measures. The ideal outcome is the prepared bit string every time,
so every deviation is prepare / rotate / read error and nothing else. A
barrier keeps the transpiler from cancelling the preparation against the
rotation. 12 circuits, 4000 shots each.

WHAT IT MEASURES (all offline-checkable; see the checks):
  1. The readout attenuation a_b,q of every qubit IN EACH BASIS,
     1 - P(1|0) - P(0|1), compared with the Z-basis value that Labs 44 to 48
     used for all three bases.
  2. The attenuation of every neighbouring PAIR, compared with the product of
     the two single-qubit values. They are equal if the qubits read out
     independently; a difference is correlated readout error (a flip of both
     bits together leaves the pair product unchanged, so pair > product).
     Asymmetric single-qubit errors, P(1|0) not equal to P(0|1), leave small
     residuals in single ordered states; they cancel in the average over the
     four patterns, which is why the lab reports that average.
  3. The residual pair-sum loss of the four ordered states after the usual
     Z-based correction, per basis (and Y minus X).
  4. A PREDICTION of the Lab 48 depth-0 loss in each basis from these numbers
     alone. A depth-0 state is three singlets; each singlet bond reads out
     like the alternating product states (bits that differ), so the corrected
     singlet-bond value is -att_alt,b / (a_z,q a_z,q+1) instead of -1, and the
     predicted loss in basis b is the sum over the three singlet bonds of
     1 - att_alt,b / (a_z,q a_z,q+1), plus the noise-model loss of those gates
     (the Lab 21 value, 0.0334 / 3 per basis). It is compared with the
     measured depth-0 losses stored in fezdata.LAB48_BASIS.

THE RULE, fixed here before any run: if all three predicted depth-0 losses are
within 3 sigma of the measured ones, readout and basis rotation account for
the depth-0 pattern; if the prediction covers more than half of the measured
loss but misses in some basis, they account for part; otherwise they do not,
and the cause is on the gate side (the singlet-preparation gates), which is
the next experiment. The audit cannot see gate error, by construction.

FIRST LIVE RUN (ibm_fez, 2026-10-17 22:14, same six qubits, 26 s from queue to
done, 12 circuits x 4000 shots): the verdict is that readout and basis rotation
do NOT account for the depth-0 loss.
  * Readout is independent: the pair attenuation equals the product of the
    single-qubit values to within 0.0005 on every bond in every basis (all
    within 0.2 sigma). The per-qubit attenuation in X and Y is 0.000 to 0.007
    below the Z value the labs used (mean about 0.004).
  * The Z-based correction leaves 0.027 +/- 0.006 (X) and 0.037 +/- 0.006 (Y)
    of the 5.0 ordered pair-sum, and -0.004 +/- 0.005 (Z). Y minus X is only
    +0.010 +/- 0.009, against +0.103 +/- 0.013 seen at k >= 1 in Lab 48, so the
    Y > X gap is not a readout effect.
  * Predicted depth-0 loss from readout plus the gate model: X 0.030, Y 0.025,
    Z 0.007 (each +/- 0.007), against measured 0.111, 0.148, 0.054 (+/- 0.018):
    4.2, 6.5 and 2.4 sigma short. In total 0.062 of the measured 0.312.
  The remaining depth-0 loss, about 0.25 over three singlets (3% of each
  singlet's correlation, against the 0.4% per two-qubit gate of the Lab 17
  mirror benchmark), therefore comes from the singlet-preparation gates. Its
  basis pattern (Z loses less, A = -0.076 +/- 0.022) points to phase-type
  error (dephasing or a Z-axis error in the two-qubit gate) rather than
  amplitude damping, which would make Z lose MORE.

LIMITATIONS: one device, one chain, one job; the preparation needs one or two
more single-qubit gates than the Lab 48 circuits (single-qubit errors are about
0.03% on this device, small next to the effects searched for); readout error is
assumed to depend only on the prepared computational state, and the prediction
for singlets ignores second-order bias on the two unpaired bonds (about 1e-4);
shot-noise error bars only; the audit and the Lab 48 runs were hours apart, so
drift between them is not in the error bars.

Modes
-----
  (default)  DRY RUN on a simulated ibm_fez (no account, no network).
  --live     Real hardware through Qiskit Runtime (needs an IBM account).
  --retrieve JOB_ID   Fetch and analyse a job submitted earlier.

Security
--------
The API key is read ONLY from an environment variable (default name
IBM_QUANTUM_API_KEY; change the NAME with --token-env). There is no --token
flag because command-line arguments leak into shell history. The key is never
logged, never written to a file, and is scrubbed from error text. The optional
instance CRN comes from IBM_QUANTUM_INSTANCE. Nothing secret touches disk. The
connection code is shared with Lab 17 (labs/lab17_mirror_benchmark.py).

Usage
-----
    python3 labs/lab49_readout_basis_audit.py                       # dry run
    read -rs IBM_QUANTUM_API_KEY; export IBM_QUANTUM_API_KEY     # hidden prompt
    python3 labs/lab49_readout_basis_audit.py --live --backend ibm_fez
    python3 labs/lab49_readout_basis_audit.py --retrieve <job id>

Log files
---------
logs/lab49/lab49_<YYYYmmdd_HHMMSS>.log (plain text, never contains the key)
logs/lab49/last_job.json (job id, backend, shots, layout; no secrets)

Exit Codes
----------
0   Success (all checks passed; a --live job still queued is reported, not
    failed, and can be fetched later with --retrieve).
1   One or more checks failed.
2   Invalid usage or missing prerequisite (missing key, package, bad flag).
130 User interrupted execution.
===============================================================================
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

from applog import get_logger

log = get_logger("lab49")

# Cloud plumbing, the estimator and the recorded data are reused, not copied:
# the security-critical code then exists in one place.
try:
    from fezdata import BASIS_SE, LAB48_BASIS, LAYOUT, RUNS
    from lab17_mirror_benchmark import (
        TERMINAL_BAD,
        connect,
        make_sampler,
        scrub,
        wait_for,
    )
    from lab21_noise_aware_training import N, NoisyChain
    from lab44_noise_aware_on_qpu import basis_value, run_ideal
    from lab46_hardware_noise_budget import MAX_SHOTS, pick_backend, transpile_all
    from lab48_first_layer_scan import ORDERS, model_table
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

JOB_FILE = Path("logs/lab49/last_job.json")
AUDIT_BASES = ("Z", "X", "Y")
# Prepared bit of every qubit (index = qubit) for the four product states.
PATTERNS = {
    "all0": tuple(0 for q in range(N)),
    "all1": tuple(1 for q in range(N)),
    "alt0": tuple(q % 2 for q in range(N)),
    "alt1": tuple(1 - q % 2 for q in range(N)),
}
SINGLET_BONDS = (0, 2, 4)  # the bonds that hold a singlet at depth 0
Z_SIGMA = 3.0  # agreement limit of the pre-registered rule


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


# ----------------------------------------------------------------------------
# Circuits
# ----------------------------------------------------------------------------
def audit_circuit(basis, bits):
    """Product eigenstate of `basis` with the given bits, then the Lab 48 readout.

    The preparation is the inverse of the measurement rotation, so the ideal
    result is exactly `bits`. The barrier keeps the two apart after
    transpilation (otherwise the optimiser would cancel the very gates tested).
    """
    qc = QuantumCircuit(N)
    for q in range(N):
        if bits[q]:
            qc.x(q)
        if basis == "X":
            qc.h(q)
        if basis == "Y":
            qc.h(q)
            qc.s(q)
    qc.barrier()
    # Same rotations as Lab 44: X needs H, Y needs S-dagger then H.
    for q in range(N):
        if basis == "Y":
            qc.sdg(q)
        if basis in ("X", "Y"):
            qc.h(q)
    qc.measure_all()
    return qc


def build_jobs():
    """Keys (basis, pattern) and circuits, in a fixed order."""
    keys, circuits = [], []
    for basis in AUDIT_BASES:
        for name, bits in PATTERNS.items():
            keys.append((basis, name))
            circuits.append(audit_circuit(basis, bits))
    return keys, circuits


def ideal_string(bits):
    """The outcome string of the ideal circuit (qubit 0 is the rightmost bit)."""
    return "".join(str(bits[q]) for q in reversed(range(N)))


# ----------------------------------------------------------------------------
# Analysis (pure functions of counts, tested on synthetic readout models)
# ----------------------------------------------------------------------------
def bit_array(counts):
    """(bit matrix [outcome, qubit], probabilities, shots) of a counts dict."""
    total = float(sum(counts.values()))
    strings = list(counts)
    bits = np.array([[int(s[N - 1 - q]) for q in range(N)] for s in strings])
    prob = np.array([counts[s] for s in strings], dtype=float) / total
    return bits, prob, total


def basis_attenuation(by_key, basis):
    """Per-qubit attenuation 1 - P(1|0) - P(0|1) in one basis, with errors.

    All four prepared patterns are pooled: for every qubit two patterns prepare
    a 0 and two prepare a 1.
    """
    # n0/n1: shots that prepared a 0/1 on each qubit; k0: of those, reads of 1;
    # k1: of the prepared-1 shots, reads of 0 (the two flip counts).
    n0, k0 = np.zeros(N), np.zeros(N)
    n1, k1 = np.zeros(N), np.zeros(N)
    for name, prep in PATTERNS.items():
        bits, prob, total = bit_array(by_key[(basis, name)])
        p_read1 = prob @ bits  # probability that each qubit reads 1
        for q in range(N):
            if prep[q] == 0:
                n0[q] += total
                k0[q] += total * p_read1[q]
            else:
                n1[q] += total
                k1[q] += total * (1 - p_read1[q])
    p10, p01 = k0 / n0, k1 / n1
    att = 1 - p10 - p01
    err = np.sqrt(p10 * (1 - p10) / n0 + p01 * (1 - p01) / n1)
    return att, err


def pair_attenuation(by_key, basis, names):
    """Per-bond attenuation of the correlator, averaged over the named patterns.

    For a pattern whose ideal bond value is s (+1 or -1), the attenuation is
    s times the measured mean of (-1)^(b_q + b_q+1). Independent readout gives
    the product of the two single-qubit attenuations.
    """
    # Accumulate the attenuation and its variance pattern by pattern.
    att = np.zeros(N - 1)
    var = np.zeros(N - 1)
    for name in names:
        prep = PATTERNS[name]
        bits, prob, total = bit_array(by_key[(basis, name)])
        sign = (-1.0) ** (bits[:, :-1] + bits[:, 1:])  # [outcome, bond]
        mean = prob @ sign
        ideal = np.array([(-1.0) ** (prep[q] + prep[q + 1]) for q in range(N - 1)])
        att += ideal * mean
        var += (1 - mean**2) / total
    return att / len(names), np.sqrt(var) / len(names)


def z_attenuation(by_key):
    """The Z-basis attenuation of Labs 44 to 48 (from 000000 and 111111 only)."""
    p0, p1 = by_key[("Z", "all0")], by_key[("Z", "all1")]
    a = np.zeros(N)
    err = np.zeros(N)
    for q in range(N):
        pos = N - 1 - q
        n0, n1 = sum(p0.values()), sum(p1.values())
        f0 = sum(c for s, c in p0.items() if s[pos] == "1") / n0
        f1 = sum(c for s, c in p1.items() if s[pos] == "0") / n1
        a[q] = 1 - f0 - f1
        err[q] = math.sqrt(f0 * (1 - f0) / n0 + f1 * (1 - f1) / n1)
    return a, err


def predicted_singlet_loss(by_key, basis, az, az_err):
    """(loss, error) the readout alone predicts for three singlets in `basis`."""
    att, err = pair_attenuation(by_key, basis, ("alt0", "alt1"))
    # Sum over the three singlet bonds; errors of different bonds are independent.
    loss, var = 0.0, 0.0
    for q in SINGLET_BONDS:
        denom = az[q] * az[q + 1]
        ratio = att[q] / denom
        loss += 1 - ratio
        rel = (az_err[q] / az[q]) ** 2 + (az_err[q + 1] / az[q + 1]) ** 2
        var += (err[q] / denom) ** 2 + ratio**2 * rel
    return loss, math.sqrt(var)


def ordered_residual(by_key, basis, az):
    """Pair-sum magnitude lost by each ordered state after the Z-based correction.

    Returns {pattern: (residual, error)}; zero means the correction is exact.
    """
    out = {}
    for name, prep in PATTERNS.items():
        ideal = sum((-1.0) ** (prep[q] + prep[q + 1]) for q in range(N - 1))
        value, err = basis_value(by_key[(basis, name)], az)
        out[name] = (abs(ideal) - math.copysign(1.0, ideal) * value, err)
    return out


def explain_verdict(z, predicted, measured):
    """Apply the pre-registered rule to the per-basis z values."""
    if all(abs(x) < Z_SIGMA for x in z):
        return "account for the depth-0 pattern"
    if sum(predicted) > 0.5 * sum(measured):
        return "account for part of the depth-0 loss"
    return "do NOT account for the depth-0 loss: look at the gates"


# ----------------------------------------------------------------------------
# Synthetic readout models (to test the analysis against known answers)
# ----------------------------------------------------------------------------
def readout_distribution(prep, e0, e1, corr=0.0):
    """Exact outcome probabilities (64) for a prepared bit pattern.

    e0[q] = P(read 1 | prepared 0), e1[q] = P(read 0 | prepared 1); corr is the
    probability, per neighbouring pair, that both bits are flipped together
    (correlated readout error). prep is a probability vector over 64 states.
    """
    prob = prep.astype(float)
    idx = np.arange(2**N)
    for q in range(N):
        bit = (idx >> q) & 1
        new = np.zeros_like(prob)
        flipped = idx ^ (1 << q)
        # A prepared 0 flips with e0, a prepared 1 with e1.
        flip = np.where(bit == 0, e0[q], e1[q])
        new += prob * (1 - flip)
        np.add.at(new, flipped, prob * flip)
        prob = new
    for q in range(N - 1):
        both = idx ^ (1 << q) ^ (1 << (q + 1))
        new = (1 - corr) * prob
        np.add.at(new, both, corr * prob)
        prob = new
    return prob


def to_counts(prob, shots=10**9):
    """Counts dict (floats, so no sampling noise) from a probability vector."""
    return {
        format(i, f"0{N}b"): float(p) * shots for i, p in enumerate(prob) if p > 1e-15
    }


def pattern_state(bits):
    """Probability vector (64) of a computational-basis pattern."""
    v = np.zeros(2**N)
    v[int(ideal_string(bits), 2)] = 1.0
    return v


def synthetic_audit(errors, corr=0.0):
    """by_key for a synthetic device; errors[basis] = (e0, e1) arrays of length N."""
    by_key = {}
    for basis in AUDIT_BASES:
        e0, e1 = errors[basis]
        for name, bits in PATTERNS.items():
            dist = readout_distribution(pattern_state(bits), e0, e1, corr)
            by_key[(basis, name)] = to_counts(dist)
    return by_key


def synthetic_singlet_loss(errors, corr, az, basis):
    """Exact depth-0 loss in `basis` on the synthetic device (three singlets)."""
    # Three singlets: each pair is 01 or 10 with probability 1/2.
    state = np.zeros(2**N)
    for choice in range(8):
        bits = []
        for pair in range(3):
            first = (choice >> pair) & 1
            bits += [first, 1 - first]
        state[int(ideal_string(bits), 2)] += 1 / 8
    e0, e1 = errors[basis]
    dist = readout_distribution(state, e0, e1, corr)
    value, _ = basis_value(to_counts(dist), az)
    return value - (-3.0)


def check_analysis(r):
    """The analysis recovers known readout errors and the singlet prediction."""
    log.info("=== Check 2: the analysis on synthetic readout models ===")
    rng = np.random.default_rng(49)
    errors = {
        b: (rng.uniform(0.002, 0.02, N), rng.uniform(0.005, 0.04, N))
        for b in AUDIT_BASES
    }
    by_key = synthetic_audit(errors)
    ok_att = True
    for b in AUDIT_BASES:
        att, _ = basis_attenuation(by_key, b)
        ok_att &= bool(np.allclose(att, 1 - errors[b][0] - errors[b][1], atol=1e-9))
    r.append(check(ok_att, "per-basis attenuation recovers the injected errors"))
    az, _ = z_attenuation(by_key)
    r.append(
        check(
            bool(np.allclose(az, 1 - errors["Z"][0] - errors["Z"][1], atol=1e-9)),
            "the Z calibration reproduces the Lab 44 attenuation",
        )
    )
    # Independent readout: pair attenuation is the product of the singles.
    att_pair, _ = pair_attenuation(by_key, "X", tuple(PATTERNS))
    att_x, _ = basis_attenuation(by_key, "X")
    r.append(
        check(
            bool(np.allclose(att_pair, att_x[:-1] * att_x[1:], atol=1e-9)),
            "independent readout: pair attenuation = product of the singles",
        )
    )
    # The singlet prediction against the exact classical calculation, with
    # basis-dependent errors. The only difference is the bias of the two
    # unpaired bonds, which is second order.
    worst = 0.0
    for b in AUDIT_BASES:
        pred, _ = predicted_singlet_loss(by_key, b, az, np.full(N, 1e-3))
        exact = synthetic_singlet_loss(errors, 0.0, az, b)
        worst = max(worst, abs(pred - exact))
        log.info("  %s: predicted %+.4f, exact %+.4f", b, pred, exact)
    r.append(
        check(worst < 3e-3, f"singlet prediction matches the exact value ({worst:.1e})")
    )
    # No error at all: everything is exactly zero.
    zero = {b: (np.zeros(N), np.zeros(N)) for b in AUDIT_BASES}
    clean = synthetic_audit(zero)
    az0, _ = z_attenuation(clean)
    pred0, _ = predicted_singlet_loss(clean, "Y", az0, np.full(N, 1e-3))
    res = ordered_residual(clean, "Y", az0)
    r.append(
        check(
            abs(pred0) < 1e-12 and all(abs(v[0]) < 1e-9 for v in res.values()),
            "no readout error: prediction and residuals are exactly zero",
        )
    )
    # Correlated readout error must show up as pair != product.
    corr = synthetic_audit(errors, corr=0.03)
    att_pair, pe = pair_attenuation(corr, "Z", tuple(PATTERNS))
    att_z, ze = basis_attenuation(corr, "Z")
    gap = att_pair - att_z[:-1] * att_z[1:]
    r.append(
        check(
            bool(np.all(gap > 0.01)),
            f"correlated readout (3% pair flips) shows as pair > product ({gap.mean():+.3f})",
        )
    )
    # A basis whose readout is worse than Z is seen by the prediction.
    worse = dict(errors)
    worse["Y"] = (errors["Z"][0] + 0.02, errors["Z"][1] + 0.02)
    wk = synthetic_audit(worse)
    azw, _ = z_attenuation(wk)
    py, _ = predicted_singlet_loss(wk, "Y", azw, np.full(N, 1e-3))
    pz, _ = predicted_singlet_loss(wk, "Z", azw, np.full(N, 1e-3))
    r.append(
        check(
            py > pz + 0.05,
            f"a worse Y readout predicts more Y loss ({py:.3f} vs {pz:.3f})",
        )
    )
    # The rule itself.
    r.append(
        check(
            explain_verdict([0.5, -1, 2], [0.1] * 3, [0.1] * 3).startswith(
                "account for the"
            ),
            "rule: all within 3 sigma",
        )
    )
    r.append(
        check("part" in explain_verdict([5, 0, 0], [0.2] * 3, [0.3] * 3), "rule: part")
    )
    r.append(
        check("NOT" in explain_verdict([5, 5, 5], [0.01] * 3, [0.3] * 3), "rule: not")
    )


def check_circuits(r):
    """The audit circuits return exactly the prepared bit strings when noiseless."""
    log.info("=== Check 1: audit circuits on an ideal simulator ===")
    keys, circuits = build_jobs()
    counts = run_ideal(circuits, 200, 49)
    bad = [k for k, c in zip(keys, counts) if set(c) != {ideal_string(PATTERNS[k[1]])}]
    r.append(
        check(not bad, f"all {len(keys)} circuits return the prepared string {bad}")
    )
    twoq = sum(
        sum(c.count_ops().get(g, 0) for g in ("cx", "cz", "ecr")) for c in circuits
    )
    r.append(check(twoq == 0, "no two-qubit gate in any circuit"))
    r.append(check(len(keys) == 12, "12 circuits: 3 bases x 4 patterns"))


# ----------------------------------------------------------------------------
# Report on a finished run
# ----------------------------------------------------------------------------
def analyse(r, ctx, keys, counts, backend_name, dry):
    """Attenuations per basis, pair vs product, ordered residuals, the prediction."""
    where = "simulated device" if dry else "REAL HARDWARE"
    by_key = dict(zip(keys, counts))
    if dry:
        log.info("(dry run: the numbers below describe the simulator, not ibm_fez)")
    az, az_err = z_attenuation(by_key)
    log.info("%s %s: Z-basis attenuation %s", backend_name, where, np.round(az, 3))
    r.append(check(bool(np.all((az > 0.8) & (az <= 1.0))), "attenuation in (0.8, 1]"))

    log.info("attenuation per basis (and minus the Z value the labs used):")
    for b in AUDIT_BASES:
        a, e = basis_attenuation(by_key, b)
        log.info(
            "  %s  a = %s   a - a_Z = %s   (+/- %.3f)",
            b,
            np.round(a, 3),
            np.round(a - az, 3),
            float(e.mean()),
        )

    log.info("pair attenuation minus product of singles (0 = independent readout):")
    flagged = 0
    for b in AUDIT_BASES:
        a, e = basis_attenuation(by_key, b)
        pair, pe = pair_attenuation(by_key, b, tuple(PATTERNS))
        prod = a[:-1] * a[1:]
        se = np.sqrt(pe**2 + (a[1:] * e[:-1]) ** 2 + (a[:-1] * e[1:]) ** 2)
        z = (pair - prod) / se
        flagged += int(np.sum(np.abs(z) > 5))
        log.info("  %s  %s   z = %s", b, np.round(pair - prod, 4), np.round(z, 1))
    log.info(
        "correlated readout error: %s",
        (
            "seen in %d bond(s) beyond 5 sigma" % flagged
            if flagged
            else "not seen (all within 5 sigma)"
        ),
    )

    log.info(
        "pair-sum magnitude lost by the ordered states after the Z-based correction:"
    )
    mean_res = {}
    for b in AUDIT_BASES:
        res = ordered_residual(by_key, b, az)
        vals = np.array([v[0] for v in res.values()])
        errs = np.array([v[1] for v in res.values()])
        mean_res[b] = (float(vals.mean()), float(np.sqrt((errs**2).sum()) / len(vals)))
        log.info(
            "  %s  %s   mean %+.3f +/- %.3f of 5",
            b,
            " ".join(f"{n}={v[0]:+.3f}" for n, v in res.items()),
            *mean_res[b],
        )
    d = mean_res["Y"][0] - mean_res["X"][0]
    de = math.hypot(mean_res["Y"][1], mean_res["X"][1])
    log.info(
        "  Y minus X: %+.3f +/- %.3f (Lab 48 saw +0.103 +/- 0.013 at k >= 1)", d, de
    )

    # The prediction of the depth-0 loss in each basis, against Lab 48.
    model0 = float(model_table(ctx["chain"], ORDERS["forward"])[1][0]) / 3
    f, rv = np.array(LAB48_BASIS["forward"][0]), np.array(LAB48_BASIS["reverse"][0])
    meas = (f + rv) / 2
    meas_err = BASIS_SE[0] / math.sqrt(2)
    log.info(
        "depth 0, per basis (X, Y, Z order as Lab 48): readout prediction + gate model "
        "(%.4f each) against the measured loss",
        model0,
    )
    pred, zs, perr = [], [], []
    for j, b in enumerate(("X", "Y", "Z")):
        p, pe = predicted_singlet_loss(by_key, b, az, az_err)
        total = p + model0
        z = (meas[j] - total) / math.hypot(meas_err, pe)
        pred.append(total)
        perr.append(pe)
        zs.append(z)
        log.info(
            "  %s  readout %+.3f +/- %.3f  total %+.3f  measured %+.3f +/- %.3f  z %+.1f",
            b,
            p,
            pe,
            total,
            meas[j],
            meas_err,
            z,
        )
    text = explain_verdict(zs, pred, list(meas))
    log.info(
        "verdict (pre-registered rule): readout and basis rotation %s "
        "(predicted %.3f of the measured %.3f in total)",
        text,
        sum(pred),
        float(meas.sum()),
    )
    if dry:
        log.info("(dry run: that verdict is about the simulator, not ibm_fez)")


def summarise(r):
    """Log the tally and return 0/1."""
    bad = r.count(False)
    if bad:
        log.fail("%d of %d checks failed", bad, len(r))
        return 1
    log.ok("All %d checks passed", len(r))
    return 0


def plan_time(shots, n_circuits):
    """Log the QPU time this job should take, from the recorded rates."""
    rates = [secs / (circ * sh) for circ, sh, secs in RUNS]
    total = n_circuits * shots
    log.info(
        "plan: %d circuits x %d shots = %d shots, about %.0f to %.0f s running "
        "(recorded rates %.2f and %.2f ms per shot)",
        n_circuits,
        shots,
        total,
        min(rates) * total,
        max(rates) * total,
        min(rates) * 1e3,
        max(rates) * 1e3,
    )


def finish_retrieve(args, service, secret, r, ctx):
    """--retrieve: fetch a finished job and analyse it."""
    try:
        meta = json.loads(JOB_FILE.read_text())
        keys = build_jobs()[0]
        job = service.job(args.retrieve)
        state = wait_for(job, args.wait, poll=10)
        if state != "DONE":
            log.warn_("job %s is %s; try again later", args.retrieve, state)
            return 0 if all(r) else 1
        res = job.result()
        counts = [res[i].data.meas.get_counts() for i in range(len(keys))]
    except Exception as exc:  # noqa: BLE001
        log.fail("retrieve failed: %s: %s", type(exc).__name__, scrub(exc, secret))
        return 1
    analyse(r, ctx, keys, counts, meta.get("backend", "device"), False)
    return summarise(r)


def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 49: readout and basis audit")
    ap.add_argument("--live", action="store_true", help="use real IBM hardware")
    ap.add_argument("--retrieve", metavar="JOB_ID", help="analyse an earlier job")
    ap.add_argument("--backend", help="backend name; default: least busy")
    ap.add_argument(
        "--layout",
        default=",".join(str(q) for q in LAYOUT),
        help="six physical qubits (default: the Lab 44 chain) or 'auto'",
    )
    ap.add_argument(
        "--shots", type=int, default=4000, help=f"shots per circuit, 100..{MAX_SHOTS}"
    )
    ap.add_argument("--max-seconds", type=int, default=100, help="QPU time cap")
    ap.add_argument("--wait", type=int, default=900, help="queue wait (0 = submit)")
    ap.add_argument(
        "--token-env",
        default="IBM_QUANTUM_API_KEY",
        help="NAME of the environment variable holding the key",
    )
    ap.add_argument("--sampler", choices=("v2", "client"), default="v2")
    args = ap.parse_args(argv)
    if not 100 <= args.shots <= MAX_SHOTS:
        ap.error(f"--shots must be 100..{MAX_SHOTS}. Got {args.shots}.")
    if args.max_seconds < 10:
        ap.error("--max-seconds must be >= 10.")
    if args.live and args.retrieve:
        ap.error("use --live OR --retrieve, not both.")
    layout = None
    if args.layout != "auto":
        layout = [int(x) for x in args.layout.split(",")]
        if len(layout) != N or len(set(layout)) != N:
            ap.error(f"--layout needs {N} different qubits or 'auto'")
    ctx = {"chain": NoisyChain(N)}
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_circuits(r)
    check_analysis(r)

    log.info("=== Check 3: connect and choose a backend ===")
    service, secret = connect(args)
    if isinstance(service, str):
        return 2
    if args.retrieve:
        return finish_retrieve(args, service, secret, r, ctx)
    backend = pick_backend(args, service, secret)
    if backend is None:
        return 1
    log.info("backend %s: %d qubits", backend.name, backend.num_qubits)

    log.info("=== Check 4: transpile onto one set of physical qubits ===")
    keys, circuits = build_jobs()
    isa, used = transpile_all(backend, circuits, layout)
    log.info("physical qubits used (same for every circuit): %s", used)
    same = all(c.layout.initial_index_layout(filter_ancillas=True) == used for c in isa)
    r.append(check(same, "all circuits run on identical physical qubits"))
    allowed = set(backend.operation_names) | {"measure", "barrier", "delay"}
    r.append(
        check(
            set().union(*(c.count_ops() for c in isa)) <= allowed,
            "only native gates after transpilation",
        )
    )
    twoq = [sum(c.count_ops().get(g, 0) for g in ("cz", "ecr", "cx")) for c in isa]
    r.append(check(not any(twoq), "no two-qubit gate after transpilation"))
    # The rotations under test must survive: X and Y keep their sx gates, Z has none.
    sx = {k: c.count_ops().get("sx", 0) for k, c in zip(keys, isa)}
    kept = all(sx[(b, p)] >= N for b in ("X", "Y") for p in PATTERNS)
    none_z = all(sx[("Z", p)] == 0 for p in PATTERNS)
    log.info("sx gates per circuit: %s", sx)
    r.append(check(kept and none_z, "the basis rotations survive transpilation"))

    log.info("=== Check 5: run %d circuits x %d shots ===", len(isa), args.shots)
    plan_time(args.shots, len(isa))
    if args.live:
        log.warn_("LIVE: this uses real QPU time. cap=%d s", args.max_seconds)
    job = make_sampler(args, backend).run(isa)
    job_id = job.job_id()
    # Save the id and the run parameters (never the key) for --retrieve.
    JOB_FILE.parent.mkdir(parents=True, exist_ok=True)
    JOB_FILE.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "backend": backend.name,
                "shots": args.shots,
                "layout": used,
            }
        )
    )
    log.info("job id: %s (saved to %s)", job_id, JOB_FILE)
    state = wait_for(job, args.wait if args.live else 600, poll=5)
    if state == "TIMEOUT":
        log.warn_(
            "still queued after %d s. Fetch it later with: "
            "python3 labs/lab49_readout_basis_audit.py --retrieve %s",
            args.wait,
            job_id,
        )
        return 0 if all(r) else 1
    if state in TERMINAL_BAD:
        log.fail("job ended with status %s", state)
        return 1
    result = job.result()

    log.info("=== Check 6: results ===")
    counts = [result[i].data.meas.get_counts() for i in range(len(isa))]
    r.append(
        check(
            all(sum(c.values()) == args.shots for c in counts),
            f"every circuit returned {args.shots} shots",
        )
    )
    analyse(r, ctx, keys, counts, backend.name, not args.live)
    return summarise(r)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
