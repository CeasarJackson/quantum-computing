#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab54_pair_tomography.py

Purpose
-------
Labs 51 to 53 left the depth-0 loss in the singlet-preparation gates, local to
each link, not idle dephasing (11%), not readout, not the initial state, not
crosstalk. The three correlators XX, YY, ZZ already fix how much of the lost
singlet weight went into each of the three triplets T_k = (sigma_k x I)|S>:
from the Lab 51 plain arm, p_x = 0.007, p_y = 0.006, p_z = 0.016 per singlet
(+/- 0.002), about 97% singlet fidelity, with the Z-type triplet twice the
others. What three correlators cannot say is whether those errors are
INCOHERENT (a mixture: random Pauli flips) or COHERENT (a definite small
rotation, which a recalibration or a compiler could undo). The two differ in
the off-diagonal elements of the density matrix:

    a rotation by e about axis k on one qubit turns |S> into
    cos(e/2)|S> - i sin(e/2)|T_k>, so  rho(S, T_k)  has size
    sin(e/2) cos(e/2) = sqrt(p_k) (1 - p_k)^(1/2),

    while a random flip with probability p_k leaves rho(S, T_k) = 0.

So this lab measures the full two-qubit state of each prepared singlet, by
quantum state tomography: 9 settings per link (qubit a in X, Y or Z times
qubit b in X, Y or Z), the same single-link circuit as Labs 52 and 53 (variant
A) but with the basis chosen qubit by qubit. Nine settings give all 15 Pauli
expectations (the single-qubit ones come as marginals). Linear inversion, then
eigenvalue clipping to the nearest physical state; error bars by a parametric
bootstrap (200 resamples of every setting's counts and of the calibration).

CIRCUITS: 9 settings x 3 links = 27, plus the two readout calibration circuits
= 29, at 4000 shots (116000 shots; Lab 52's 114000 took 42 s).

READOUT: each qubit's reading is  m = a z + b  with attenuation a = 1 - p10 - p01
and offset b = p01 - p10 from the calibration circuits; a two-qubit correlator
is corrected for both (Labs 44 to 53 used a alone, enough while the singles
were zero; the singles here are not).

THE RULES, fixed here before any live run (per link and pooled over the three
links; sigma is the bootstrap error of each number):

  DECOMP  singlet fidelity F and the triplet populations p_x, p_y, p_z.
  COHER   axis k is COHERENT when the real or imaginary part of rho(S, T_k)
          differs from 0 by more than 3 sigma. Its coherent weight is
          m_k^2, where m_k = |rho(S, T_k)|.
  SHARE   (sum of m_k^2 over the coherent axes) / (p_x + p_y + p_z).
          >= 0.5 : MOSTLY COHERENT (a rotation that could be calibrated away);
          < 0.2 : INCOHERENT (a mixture, no software fix); otherwise MIXED.
  ROTATION  for a coherent axis, the rotation angle e = 2 asin(m_k)
          (about the z, x or y axis of one qubit) is printed.

FIRST LIVE RUN (ibm_fez, 2026-10-18 00:07, same six qubits, 41 s from queue to
done, 29 circuits x 4000 shots): the error is INCOHERENT, a mixture of random
flips, so no recalibration or compiler change can remove it.
  * DECOMP, pooled over the three links: singlet fidelity F = 0.957 (0.964,
    0.944, 0.962 by link, each +/- 0.005); triplet populations p_x = 0.012,
    p_y = 0.007, p_z = 0.024 (each +/- 0.003). The Z-type triplet (a relative
    phase flip between |01> and |10>) is again about twice the others. The
    total non-singlet weight 0.043 is higher than the 0.029 the XX, YY, ZZ
    correlators of Lab 51 gave (about 2 sigma); a small positive bias of
    linear inversion with clipping, or drift between the jobs, may account
    for it.
  * COHER: |rho(S, T_k)| is 0.003 to 0.036, against sqrt(p_k) of 0.07 to 0.18
    that a fully coherent error of the same size would give. Axis z is flagged
    on links 0 and 1 (real parts 0.018 and 0.003, imaginary parts -0.020 and
    -0.036, each +/- 0.006); axis x on links 1 and 2.
  * SHARE: coherent weight over the non-singlet weight is 0.02 pooled (0.02,
    0.03, 0.01 by link): INCOHERENT. What is coherent is small but has the
    same sign on all three links: a z rotation of 0.05 to 0.07 rad (3 to 4
    degrees) between the two qubits, worth a population of about 0.001.
  So more than 95% of the preparation loss is random, with a Z-type excess.

LIMITATIONS: one device and one job; linear inversion with clipping (a bias
of order 1/sqrt(shots) in small populations; the bootstrap includes it only
partly); the measured state includes the basis-change gates (single-qubit,
errors about 3e-4 on this device) and readout; a common rotation of both qubits
(Z_a + Z_b) leaves the singlet unchanged and is invisible here by construction;
shot-noise error bars only.

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
    python3 labs/lab54_pair_tomography.py                            # dry run
    read -rs IBM_QUANTUM_TOKEN; export IBM_QUANTUM_TOKEN             # hidden
    python3 labs/lab54_pair_tomography.py --live --backend ibm_fez \\
        --token-env IBM_QUANTUM_TOKEN
    python3 labs/lab54_pair_tomography.py --retrieve <job id> --token-env IBM_QUANTUM_TOKEN

Log files
---------
logs/lab54/lab54_<YYYYmmdd_HHMMSS>.log (plain text, never contains the key)
logs/lab54/last_job.json (job id, backend, shots, layout; no secrets)

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
from qiskit.quantum_info import DensityMatrix, Statevector

from applog import get_logger

log = get_logger("lab54")

# Everything shared is imported so each piece of logic exists once.
try:
    from fezdata import LAYOUT
    from lab17_mirror_benchmark import (
        TERMINAL_BAD,
        connect,
        make_sampler,
        scrub,
        wait_for,
    )
    from lab21_noise_aware_training import N
    from lab44_noise_aware_on_qpu import calibration_circuits
    from lab46_hardware_noise_budget import MAX_SHOTS, pick_backend, transpile_all
    from lab51_herald_and_isolate import LINKS, SIGMA, check, rotate, singlet_pair
    from lab52_gate_count_orientation_idle import plan_time, summarise
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

JOB_FILE = Path("logs/lab54/last_job.json")
SETTINGS = [u + v for u in "XYZ" for v in "XYZ"]  # basis of qubit a, then qubit b
BOOTSTRAPS = 200
SHARE_HIGH, SHARE_LOW = 0.5, 0.2
PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


# ----------------------------------------------------------------------------
# Circuits
# ----------------------------------------------------------------------------
def setting_circuit(link, setting):
    """The singlet on `link`, then qubit a measured in setting[0], b in setting[1]."""
    a, b = 2 * link, 2 * link + 1
    qc = QuantumCircuit(N)
    singlet_pair(qc, link)
    # Barrier: the basis-change gates must not merge into the preparation.
    qc.barrier(a, b)
    rotate(qc, setting[0], (a,))
    rotate(qc, setting[1], (b,))
    qc.measure_all()
    return qc


def build_jobs():
    """Keys and circuits in a fixed order: calibration first, then settings."""
    keys, circuits = ["cal0", "cal1"], calibration_circuits()
    for j in LINKS:
        for s in SETTINGS:
            keys.append((j, s))
            circuits.append(setting_circuit(j, s))
    return keys, circuits


# ----------------------------------------------------------------------------
# Readout parameters and expectation values
# ----------------------------------------------------------------------------
def bit_of(counts, q):
    """P(reading 1) of physical-chain qubit q from a counts dictionary."""
    total = float(sum(counts.values()))
    return sum(c for s, c in counts.items() if s[N - 1 - q] == "1") / total


def readout_params(cal0, cal1):
    """(alpha, beta) per qubit: reading m = alpha z + beta, from |0..0> and |1..1>."""
    alpha, beta = np.ones(N), np.zeros(N)
    for q in range(N):
        p10 = bit_of(cal0, q)  # prepared 0, read 1
        p01 = 1.0 - bit_of(cal1, q)  # prepared 1, read 0
        alpha[q] = 1.0 - p10 - p01
        beta[q] = p01 - p10
    return alpha, beta


def raw_moments(counts, qa, qb):
    """(<z_a>, <z_b>, <z_a z_b>) of the raw readings, z = +1 for bit 0."""
    total = float(sum(counts.values()))
    ma = mb = mab = 0.0
    for s, c in counts.items():
        za = 1 - 2 * int(s[N - 1 - qa])
        zb = 1 - 2 * int(s[N - 1 - qb])
        ma += za * c
        mb += zb * c
        mab += za * zb * c
    return ma / total, mb / total, mab / total


def pauli_expectations(counts_by_setting, link, alpha, beta):
    """The 15 corrected Pauli expectations {(P, Q): value} of the pair.

    P acts on qubit a, Q on qubit b; 'I' means the identity. A single-qubit
    value is the average of the three settings that measure it.
    """
    qa, qb = 2 * link, 2 * link + 1
    aa, ab, ba, bb = alpha[qa], alpha[qb], beta[qa], beta[qb]
    raw = {s: raw_moments(counts_by_setting[s], qa, qb) for s in SETTINGS}
    single_a = {u: np.mean([(raw[u + v][0] - ba) / aa for v in "XYZ"]) for u in "XYZ"}
    single_b = {v: np.mean([(raw[u + v][1] - bb) / ab for u in "XYZ"]) for v in "XYZ"}
    c = {("I", "I"): 1.0}
    for u in "XYZ":
        c[(u, "I")] = single_a[u]
        c[("I", u)] = single_b[u]
    for s in SETTINGS:
        u, v = s
        # m_ab = aa ab <za zb> + aa bb <za> + ba ab <zb> + ba bb, solved for <za zb>.
        c[(u, v)] = (
            raw[s][2] - aa * bb * single_a[u] - ba * ab * single_b[v] - ba * bb
        ) / (aa * ab)
    return c


# ----------------------------------------------------------------------------
# Reconstruction and the singlet/triplet decomposition
# ----------------------------------------------------------------------------
def reconstruct(c):
    """Linear inversion, then clipping of negative eigenvalues; returns a 4x4 matrix.

    Basis order is the Qiskit one: qubit a is the rightmost tensor factor.
    """
    rho = np.zeros((4, 4), dtype=complex)
    for (p, q), val in c.items():
        rho += 0.25 * val * np.kron(PAULI[q], PAULI[p])
    rho = (rho + rho.conj().T) / 2
    w, v = np.linalg.eigh(rho)
    w = np.clip(w, 0.0, None)
    w = w / w.sum()
    return (v * w) @ v.conj().T


def singlet_states():
    """The singlet and the three triplets T_k = (sigma_k on qubit a)|S>."""
    prep = QuantumCircuit(2)
    prep.h(0)
    prep.cx(0, 1)
    prep.z(0)
    prep.x(1)
    s = np.asarray(Statevector(prep).data)
    out = {"S": s}
    for k in "XYZ":
        out[k] = np.kron(np.eye(2), PAULI[k]) @ s
    return out


def decompose(rho):
    """(F, {k: p_k}, {k: rho(S, T_k)}) for a 4x4 density matrix."""
    st = singlet_states()
    f = float(np.real(st["S"].conj() @ rho @ st["S"]))
    p = {k: float(np.real(st[k].conj() @ rho @ st[k])) for k in "XYZ"}
    coh = {k: complex(st["S"].conj() @ rho @ st[k]) for k in "XYZ"}
    return f, p, coh


def resample(counts, rng):
    """A multinomial resample of a counts dictionary with the same total."""
    keys = list(counts)
    probs = np.array([counts[k] for k in keys], dtype=float)
    shots = int(round(probs.sum()))
    new = rng.multinomial(shots, probs / probs.sum())
    return {k: int(n) for k, n in zip(keys, new) if n}


def analyse_link(counts_by_setting, cal0, cal1, link, rng, boots=BOOTSTRAPS):
    """Point estimate and bootstrap errors for one link's tomography."""
    alpha, beta = readout_params(cal0, cal1)
    rho = reconstruct(pauli_expectations(counts_by_setting, link, alpha, beta))
    f, p, coh = decompose(rho)
    samples = []
    for _ in range(boots):
        cs = {s: resample(counts_by_setting[s], rng) for s in SETTINGS}
        al, be = readout_params(resample(cal0, rng), resample(cal1, rng))
        r = reconstruct(pauli_expectations(cs, link, al, be))
        samples.append(decompose(r))
    err = {
        "F": float(np.std([x[0] for x in samples])),
        "p": {k: float(np.std([x[1][k] for x in samples])) for k in "XYZ"},
        "re": {k: float(np.std([x[2][k].real for x in samples])) for k in "XYZ"},
        "im": {k: float(np.std([x[2][k].imag for x in samples])) for k in "XYZ"},
    }
    return {"F": f, "p": p, "coh": coh, "err": err, "rho": rho}


def coherent_axes(res):
    """Axes whose rho(S, T_k) differs from zero by more than 3 sigma."""
    out = []
    for k in "XYZ":
        z_re = abs(res["coh"][k].real) / max(res["err"]["re"][k], 1e-12)
        z_im = abs(res["coh"][k].imag) / max(res["err"]["im"][k], 1e-12)
        if max(z_re, z_im) > SIGMA:
            out.append(k)
    return out


def share_coherent(res, axes):
    """Coherent weight over the total non-singlet weight."""
    total = sum(res["p"].values())
    if total <= 0:
        return 0.0
    return sum(abs(res["coh"][k]) ** 2 for k in axes) / total


def share_verdict(share):
    """The pre-registered SHARE rule."""
    if share >= SHARE_HIGH:
        return "MOSTLY COHERENT: a definite rotation, in principle calibrated away"
    if share < SHARE_LOW:
        return "INCOHERENT: a mixture of random flips, no software fix"
    return "MIXED coherent and incoherent"


def rotation_angle(m):
    """Rotation angle (rad) that gives rho(S, T_k) of size m."""
    return 2.0 * math.asin(min(1.0, m))


# ----------------------------------------------------------------------------
# Exact two-qubit model (to test the pipeline against known causes)
# ----------------------------------------------------------------------------
def model_rho(axis=None, eps=0.0, flips=None):
    """The singlet after a coherent rotation and/or random Pauli flips.

    axis, eps  rotation by eps about `axis` (X, Y or Z) of qubit a (coherent);
    flips      {axis: probability} of random sigma flips of qubit a (incoherent).
    """
    st = singlet_states()["S"]
    rho = np.outer(st, st.conj())
    if axis:
        u = math.cos(eps / 2) * np.eye(4) - 1j * math.sin(eps / 2) * np.kron(
            np.eye(2), PAULI[axis]
        )
        rho = u @ rho @ u.conj().T
    for k, pr in (flips or {}).items():
        op = np.kron(np.eye(2), PAULI[k])
        rho = (1 - pr) * rho + pr * op @ rho @ op.conj().T
    return rho


def model_counts(rho, link, setting, shots, rng):
    """Sampled counts of one setting for a two-qubit state rho on `link`."""
    rot = QuantumCircuit(2)
    rotate(rot, setting[0], (0,))
    rotate(rot, setting[1], (1,))
    probs = np.real(np.diag(DensityMatrix(rho).evolve(rot).data))
    probs = np.clip(probs, 0, None)
    probs = probs / probs.sum()
    draws = rng.multinomial(shots, probs)
    out = {}
    for i, n in enumerate(draws):
        if n:
            bits = ["0"] * N
            bits[N - 1 - 2 * link] = str(i & 1)
            bits[N - 2 - 2 * link] = str((i >> 1) & 1)
            key = "".join(bits)
            out[key] = out.get(key, 0) + int(n)
    return out


def model_run(rho, link, shots, rng, cal=(0.0, 0.0)):
    """(counts_by_setting, cal0, cal1) for a state with symmetric readout flips."""
    by = {s: model_counts(rho, link, s, shots, rng) for s in SETTINGS}
    zero, one = "0" * N, "1" * N
    return by, {zero: shots}, {one: shots}


# ----------------------------------------------------------------------------
# Checks with no device
# ----------------------------------------------------------------------------
def check_circuits(r):
    """The circuits are right: the ideal state, the settings, gate counts."""
    log.info("=== Check 1: the circuits on an exact simulator ===")
    keys, circuits = build_jobs()
    r.append(
        check(len(keys) == 29, "29 circuits: 2 calibration + 9 settings x 3 links")
    )
    twoq = [sum(c.count_ops().get(g, 0) for g in ("cx", "cz")) for c in circuits]
    r.append(
        check(twoq == [0, 0] + [1] * 27, "one two-qubit gate in every setting circuit")
    )
    rng = np.random.default_rng(1)
    rho = model_rho()
    c = pauli_expectations(model_run(rho, 1, 10**6, rng)[0], 1, np.ones(N), np.zeros(N))
    worst = max(abs(c[(u, v)] + (1 if u == v else 0)) for u in "XYZ" for v in "XYZ")
    r.append(
        check(
            worst < 0.01,
            f"singlet: <PP> = -1 and <PQ> = 0 within shot noise ({worst:.4f})",
        )
    )
    st = singlet_states()
    ov = [abs(st[a].conj() @ st[b]) for a in "SXYZ" for b in "SXYZ" if a < b]
    r.append(check(max(ov) < 1e-12, "S, T_x, T_y, T_z are orthonormal"))
    f, p, coh = decompose(rho)
    r.append(
        check(
            abs(f - 1) < 1e-12 and max(p.values()) < 1e-12,
            "the ideal singlet has F = 1",
        )
    )


def check_analysis(r):
    """The pipeline recovers known populations and tells coherent from incoherent."""
    log.info("=== Check 2: the analysis on exact models with known causes ===")
    shots = 4000
    rng = np.random.default_rng(7)
    ones, zeros = np.ones(N), np.zeros(N)
    # (a) coherent Z rotation of 0.20 rad: p_z = sin^2(0.1), |rho(S,Tz)| = sin cos.
    eps = 0.20
    res = analyse_link(*model_run(model_rho("Z", eps), 0, shots, rng), 0, rng, boots=60)
    want_p = math.sin(eps / 2) ** 2
    want_m = math.sin(eps / 2) * math.cos(eps / 2)
    log.info(
        "  coherent Z: p_z %.4f (exact %.4f), |rho| %.4f (exact %.4f)",
        res["p"]["Z"],
        want_p,
        abs(res["coh"]["Z"]),
        want_m,
    )
    r.append(
        check(
            abs(res["p"]["Z"] - want_p) < 0.006,
            "coherent Z: the triplet population is recovered",
        )
    )
    r.append(
        check(
            abs(abs(res["coh"]["Z"]) - want_m) < 0.02,
            "coherent Z: the coherence is recovered",
        )
    )
    axes = coherent_axes(res)
    r.append(check(axes == ["Z"], f"coherent Z: only the z axis is coherent {axes}"))
    r.append(
        check(
            share_verdict(share_coherent(res, axes)).startswith("MOSTLY COHERENT"),
            "rule: coherent rotation -> MOSTLY COHERENT",
        )
    )
    r.append(
        check(
            abs(rotation_angle(abs(res["coh"]["Z"])) - eps) < 0.05,
            "the rotation angle is recovered",
        )
    )
    # (b) incoherent flips with the same population: no coherence.
    res = analyse_link(
        *model_run(model_rho(flips={"Z": want_p}), 0, shots, rng), 0, rng, boots=60
    )
    axes = coherent_axes(res)
    log.info(
        "  incoherent Z: p_z %.4f, |rho| %.4f", res["p"]["Z"], abs(res["coh"]["Z"])
    )
    r.append(check(axes == [], f"incoherent flips: no axis is coherent {axes}"))
    r.append(
        check(
            share_verdict(share_coherent(res, axes)).startswith("INCOHERENT"),
            "rule: random flips -> INCOHERENT",
        )
    )
    # (c) a mixture: a coherent X rotation plus incoherent Z flips.
    mix = model_rho("X", 0.20, flips={"Z": 0.02})
    res = analyse_link(*model_run(mix, 2, shots, rng), 2, rng, boots=60)
    axes = coherent_axes(res)
    sh = share_coherent(res, axes)
    log.info("  mixture: axes %s, share %.2f", axes, sh)
    r.append(
        check("X" in axes and "Z" not in axes, f"mixture: X coherent, Z not {axes}")
    )
    r.append(check(0.2 <= sh < 0.9, f"mixture: share is intermediate ({sh:.2f})"))
    # (d) readout correction: an offset and an attenuation on every qubit.
    alpha = np.full(N, 0.96)
    beta = np.full(N, 0.01)
    c = {("I", "I"): 1.0}
    st = singlet_states()["S"]
    rho = np.outer(st, st.conj())
    for u in "XYZ":
        for v in "XYZ":
            c[(u, v)] = float(np.real(np.trace(rho @ np.kron(PAULI[v], PAULI[u]))))
    # Raw moments a perfect singlet would give under that readout, fed through the inverse.
    raw = {}
    for s in SETTINGS:
        u, v = s
        e_ab = c[(u, v)]
        raw[s] = (beta[0], beta[1], alpha[0] * alpha[1] * e_ab + beta[0] * beta[1])
    counts = {}
    for s in SETTINGS:
        ma, mb, mab = raw[s]
        probs = np.array(
            [
                (1 + ma + mb + mab) / 4,
                (1 - ma + mb - mab) / 4,
                (1 + ma - mb - mab) / 4,
                (1 - ma - mb + mab) / 4,
            ]
        )
        counts[s] = {}
        for i, pr in enumerate(probs):
            bits = ["0"] * N
            # Index i: bit of qubit a is the low bit, b the next one (0 = z +1).
            bits[N - 1] = str(i & 1)
            bits[N - 2] = str((i >> 1) & 1)
            counts[s]["".join(bits)] = pr * 1e9
    ex = pauli_expectations(counts, 0, alpha, beta)
    worst = max(abs(ex[(u, v)] - c[(u, v)]) for u in "XYZ" for v in "XYZ")
    r.append(
        check(worst < 1e-6, f"readout offset and attenuation are undone ({worst:.1e})")
    )
    del ones, zeros


# ----------------------------------------------------------------------------
# Report on a finished run
# ----------------------------------------------------------------------------
def report(per_link):
    """Print the decomposition, the coherences and the verdicts."""
    tot = {"F": 0.0, "p": {k: 0.0 for k in "XYZ"}, "m2": 0.0}
    for j, res in per_link.items():
        e = res["err"]
        log.info(
            "link %d: singlet fidelity F = %.4f +/- %.4f; triplets p_x %.4f, p_y %.4f, p_z %.4f (each +/- %.4f)",
            j,
            res["F"],
            e["F"],
            res["p"]["X"],
            res["p"]["Y"],
            res["p"]["Z"],
            float(np.mean(list(e["p"].values()))),
        )
        for k in "XYZ":
            c = res["coh"][k]
            log.info(
                "   rho(S,T_%s) = %+.4f %+.4fi  (+/- %.4f, %.4f)   size %.4f vs sqrt(p) %.4f",
                k.lower(),
                c.real,
                c.imag,
                e["re"][k],
                e["im"][k],
                abs(c),
                math.sqrt(max(res["p"][k], 0.0)),
            )
        axes = coherent_axes(res)
        sh = share_coherent(res, axes)
        log.info("   coherent axes: %s; coherent share %.2f", axes or "none", sh)
        for k in axes:
            log.info(
                "   axis %s: rotation of %.3f rad on one qubit",
                k,
                rotation_angle(abs(res["coh"][k])),
            )
        log.info("   verdict (SHARE rule): %s", share_verdict(sh))
        tot["F"] += res["F"] / len(per_link)
        for k in "XYZ":
            tot["p"][k] += res["p"][k] / len(per_link)
        tot["m2"] += sum(abs(res["coh"][k]) ** 2 for k in axes) / len(per_link)
    nonsinglet = sum(tot["p"].values())
    pooled = tot["m2"] / nonsinglet if nonsinglet > 0 else 0.0
    log.info(
        "pooled over links: F %.4f; p_x %.4f, p_y %.4f, p_z %.4f; coherent share %.2f",
        tot["F"],
        tot["p"]["X"],
        tot["p"]["Y"],
        tot["p"]["Z"],
        pooled,
    )
    log.info("verdict (SHARE rule, pooled): %s", share_verdict(pooled))


def analyse(r, keys, counts, dry, seed=11):
    """Collect the counts per link and run the tomography on each."""
    by_key = dict(zip(keys, counts))
    if dry:
        log.info("(dry run: the numbers below describe the simulator, not ibm_fez)")
    rng = np.random.default_rng(seed)
    per_link = {}
    for j in LINKS:
        per_link[j] = analyse_link(
            {s: by_key[(j, s)] for s in SETTINGS},
            by_key["cal0"],
            by_key["cal1"],
            j,
            rng,
        )
    r.append(
        check(
            all(0.5 < v["F"] <= 1.0 for v in per_link.values()),
            "singlet fidelity is between 0.5 and 1",
        )
    )
    report(per_link)
    if dry:
        log.info("(dry run: those verdicts are about the simulator, not ibm_fez)")


def finish_retrieve(args, service, secret, r):
    """--retrieve: fetch a finished job and analyse it."""
    try:
        meta = json.loads(JOB_FILE.read_text())
        keys = build_jobs()[0]
        job = service.job(args.retrieve)
        state = wait_for(job, args.wait, poll=10)
        if state != "DONE":
            log.warn_("job %s is %s; try again later", args.retrieve, state)
            return 0 if all(r) else 1
        result = job.result()
        counts = [result[i].data.meas.get_counts() for i in range(len(keys))]
    except Exception as exc:  # noqa: BLE001
        log.fail("retrieve failed: %s: %s", type(exc).__name__, scrub(exc, secret))
        return 1
    log.info("backend %s, layout %s", meta.get("backend"), meta.get("layout"))
    analyse(r, keys, counts, False)
    return summarise(r)


def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 54: pair tomography")
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
    ap.add_argument("--max-seconds", type=int, default=150, help="QPU time cap")
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
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_circuits(r)
    check_analysis(r)

    log.info("=== Check 3: connect and choose a backend ===")
    service, secret = connect(args)
    if isinstance(service, str):
        return 2
    if args.retrieve:
        return finish_retrieve(args, service, secret, r)
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
    r.append(
        check(twoq == [0, 0] + [1] * 27, "no gate cancelled and no routing overhead")
    )

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
            "python3 labs/lab54_pair_tomography.py --retrieve %s",
            args.wait,
            job_id,
        )
        return 0 if all(r) else 1
    if state in TERMINAL_BAD:
        log.fail("job ended with status %s", state)
        return 1

    log.info("=== Check 6: results ===")
    result = job.result()
    counts = [result[i].data.meas.get_counts() for i in range(len(keys))]
    r.append(
        check(
            all(sum(c.values()) == args.shots for c in counts),
            f"every circuit returned {args.shots} shots",
        )
    )
    analyse(r, keys, counts, not args.live)
    return summarise(r)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
