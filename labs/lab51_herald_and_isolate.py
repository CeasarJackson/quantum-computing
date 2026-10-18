#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab51_herald_and_isolate.py

Purpose
-------
After Labs 46 to 50 the ibm_fez depth-0 loss is cornered. Three singlets
(prepared by H, CZ and single-qubit gates on the links 129-118, 109-110 and
111-112) lose about 0.28 in energy, 3% of each singlet's correlation. Readout
is independent and explains 0.03 of it (Lab 49); the links are among the best on
the device and the calibration implies only 0.036 (Lab 50). What is left, with
the basis pattern "Z loses less than X and Y" (A = -0.076 +/- 0.022), has three
candidate causes this lab separates in ONE job:

  1. INITIAL-STATE EXCITATION. A qubit that starts in |1> instead of |0> is,
     after the singlet circuit, a Z error on the control qubit or an X error on
     the target. Exact bookkeeping (tested below): the loss then goes as X = Z
     = 6p and Y = 2X (1 - p) for three singlets, which has the Y > X gap seen
     on hardware but NOT its Z < X pattern. Expect it to explain part at most.
     Remedy under test: HERALDING. Measure
     all six qubits first and keep only the shots that read 000000. A qubit
     that was really in |1> reads 0 only with the small probability P(0|1), so
     almost all excited starts are removed (and about 94% of shots are kept).
  2. CROSSTALK between the three CZ gates run at the same time (links 109-110
     and 111-112 are neighbours through 110-111). Test: prepare ONE singlet at
     a time with the other qubits idle, and compare the sum of the three single
     losses with the loss of the three together.
  3. Whatever is left: a state-dependent or coherent error local to each gate.

CIRCUITS (17, 4000 shots each, the same six qubits as Labs 44 to 50):
  P   plain   three singlets, then the basis change (Z, X, Y)           3
  H   herald  the same, after a first measurement of all qubits into a
              separate register; offline, only the shots with that
              register at 000000 are kept                               3
  S   single  one singlet on link j = 0, 1, 2, nothing on the others,
              basis change on the pair only                             9
  cal 000000 and 111111, for the readout attenuation of Labs 44 to 50   2

THE RULES, fixed here before any live run (loss = value + 1 per singlet
bond, summed over the three singlet bonds and the three bases; positive = lost):

  CONTROL  P must reproduce the Lab 48 depth-0 loss in every basis (X 0.111,
           Y 0.148, Z 0.054, each +/- 0.018) within 3 sigma; otherwise the
           device has drifted and the comparisons below are weakened.
  HERALD   gain = loss(P) - loss(H). If gain > 3 sigma and at least 50% of
           loss(P): initial-state excitation is the MAIN cause; 10% to 50%:
           it explains PART; below 10% or within 3 sigma: it is NOT the cause.
  CROSS    diff = loss(P) - sum of the three loss(S). If diff > 3 sigma and at
           least 30% of loss(P): parallel-gate crosstalk contributes; if
           |diff| < 3 sigma: the loss is local to each gate, not crosstalk;
           diff < -3 sigma: a single link loses MORE, unexpected.
  The per-link table shows whether one link carries the loss.

FIRST LIVE RUN (ibm_fez, 2026-10-17 22:41, same six qubits, 32 s from queue to
done, 17 circuits x 4000 shots): the loss is local to each singlet-preparation
gate; it is neither initial-state excitation nor crosstalk.
  * CONTROL passed: the plain arm gave X 0.149, Y 0.131, Z 0.093 (+/- 0.026)
    against Lab 48 0.111, 0.148, 0.054: within 3 sigma in every basis.
  * HERALD: loss 0.343 +/- 0.018 plain, 0.285 +/- 0.017 heralded (96.5% of
    shots kept). Gain 0.058 +/- 0.025 (2.3 sigma, 17% of the plain loss):
    below the 3 sigma line, so the rule says NOT an initial-state effect. The
    gain is about the 0.04 the small-p bookkeeping predicted, so a minor
    initial-excitation share is possible, but it is not the bulk.
  * CROSS: the three singlets one at a time sum to 0.321 +/- 0.018 against
    0.343 plain: difference +0.023 +/- 0.025, so NOT crosstalk.
  * Asymmetry A = Z - (X+Y)/2: plain -0.055, herald -0.046, singles -0.056,
    the Lab 48 sign, and it survives heralding and isolation.
  * Per link (X+Y+Z): link 0 (qubits 129-118) 0.056 +/- 0.010, link 1
    (109-110) 0.140 +/- 0.011, link 2 (111-112) 0.125 +/- 0.010. Link 0 loses
    less than half of the others (about 5 sigma), which the reported CZ
    errors (0.0018 to 0.0021) do not explain.
  About 0.29 of the 0.34 therefore remains: state-dependent or coherent error
  inside each singlet-preparation gate sequence (or a difference between
  qubits), local to each link, invisible to mirror benchmarks.

LIMITATIONS: one device and one job; the heralding measurement adds a
mid-circuit measurement (about 1.5 us) before the gates, and a qubit measured
as 0 is not a perfect |0> (the measurement error is small, about 1%, so about
1% of the excited starts survive); the S circuits run each link alone but the
idle qubits still sit in |0>; shot-noise error bars only; readout is corrected
with the Z-basis attenuation as in the earlier labs (Lab 49 showed this is
good to about 0.004 per qubit).

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
    python3 labs/lab51_herald_and_isolate.py                       # dry run
    read -rs IBM_QUANTUM_API_KEY; export IBM_QUANTUM_API_KEY       # hidden prompt
    python3 labs/lab51_herald_and_isolate.py --live --backend ibm_fez
    python3 labs/lab51_herald_and_isolate.py --retrieve <job id>

Log files
---------
logs/lab51/lab51_<YYYYmmdd_HHMMSS>.log (plain text, never contains the key)
logs/lab51/last_job.json (job id, backend, shots, layout; no secrets)

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
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import DensityMatrix, Pauli

from applog import get_logger

log = get_logger("lab51")

# Cloud plumbing and the recorded data are reused, not copied.
try:
    from fezdata import LAB48_BASIS, LAYOUT, RUNS
    from lab17_mirror_benchmark import (
        TERMINAL_BAD,
        connect,
        make_sampler,
        scrub,
        wait_for,
    )
    from lab21_noise_aware_training import N, singlets
    from lab44_noise_aware_on_qpu import attenuation, calibration_circuits
    from lab46_hardware_noise_budget import MAX_SHOTS, pick_backend, transpile_all
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

JOB_FILE = Path("logs/lab51/last_job.json")
BASES = ("Z", "X", "Y")
LINKS = (0, 1, 2)  # singlet j sits on chain positions (2j, 2j+1)
SIGMA = 3.0  # significance limit of every rule
HERALD_MAIN, HERALD_PART = 0.5, 0.1  # shares of the plain loss
CROSS_SHARE = 0.3
BIGSHOTS = 10**9  # scale for exact-probability "counts" in the offline checks


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


# ----------------------------------------------------------------------------
# Circuits
# ----------------------------------------------------------------------------
def rotate(qc, basis, qubits):
    """Basis change before the readout: X needs H, Y needs S-dagger then H."""
    for q in qubits:
        if basis == "Y":
            qc.sdg(q)
        if basis in ("X", "Y"):
            qc.h(q)


def singlet_pair(qc, link):
    """The Lab 21 singlet (|01> - |10>)/sqrt2 on chain positions (2j, 2j+1)."""
    a = 2 * link
    qc.h(a)
    qc.cx(a, a + 1)
    qc.z(a)
    qc.x(a + 1)


def plain_circuit(basis):
    """Three singlets, basis change on all qubits, measure everything."""
    qc = singlets(N)
    rotate(qc, basis, range(N))
    qc.measure_all()
    return qc


def herald_circuit(basis):
    """As plain_circuit, after a first measurement into a separate register."""
    init = ClassicalRegister(N, "init")
    meas = ClassicalRegister(N, "meas")
    qc = QuantumCircuit(QuantumRegister(N), init, meas)
    qc.measure(range(N), init)
    qc.barrier()
    qc.compose(singlets(N), inplace=True)
    rotate(qc, basis, range(N))
    qc.measure(range(N), meas)
    return qc


def single_circuit(link, basis):
    """One singlet on `link`; the other qubits stay in |0>; basis change on the pair."""
    qc = QuantumCircuit(N)
    singlet_pair(qc, link)
    rotate(qc, basis, (2 * link, 2 * link + 1))
    qc.measure_all()
    return qc


def build_jobs():
    """Keys and circuits in a fixed order: cal, P, H, then S."""
    # Calibration first, as in Labs 44 to 50; then plain, herald, singles.
    keys, circuits = ["cal0", "cal1"], calibration_circuits()
    for b in BASES:
        keys.append(("P", b))
        circuits.append(plain_circuit(b))
    for b in BASES:
        keys.append(("H", b))
        circuits.append(herald_circuit(b))
    for j in LINKS:
        for b in BASES:
            keys.append(("S", j, b))
            circuits.append(single_circuit(j, b))
    return keys, circuits


def expected_twoq(key):
    """Two-qubit gates a circuit should keep after transpilation."""
    if key in ("cal0", "cal1"):
        return 0
    return 1 if key[0] == "S" else 3


# ----------------------------------------------------------------------------
# Analysis (pure functions of counts)
# ----------------------------------------------------------------------------
# Counts are keyed by bit strings with qubit 0 on the right (Qiskit order), so
# qubit q is character N-1-q. Dividing by both attenuations undoes readout.
def bond_sum(counts, atten, bonds):
    """(mean, standard error) of the corrected correlator summed over `bonds`.

    Each bond contributes (-1)^(b_q + b_q+1) divided by the attenuation of both
    qubits, as in Lab 44; the error follows from the spread over shots.
    """
    total = float(sum(counts.values()))
    values = np.empty(len(counts))
    weights = np.empty(len(counts))
    for k, (bits, c) in enumerate(counts.items()):
        v = 0.0
        for q in bonds:
            sign = (-1) ** (int(bits[N - 1 - q]) + int(bits[N - 2 - q]))
            v += sign / (atten[q] * atten[q + 1])
        values[k], weights[k] = v, c / total
    mean = float(values @ weights)
    var = float(((values - mean) ** 2) @ weights)
    return mean, math.sqrt(var / total)


SINGLET_BONDS = (0, 2, 4)


def loss_of(counts, atten, bonds=SINGLET_BONDS):
    """Loss (positive = lost) of singlets on `bonds`: value + number of bonds."""
    mean, err = bond_sum(counts, atten, bonds)
    return mean + len(bonds), err


# Heralding is done offline: both registers come from the same shot, so the
# shot is kept only when the first (initial) reading was all zeros.
def herald_counts(init_bits, meas_bits):
    """Counts of `meas` over the shots whose `init` reads all zeros; kept fraction."""
    # Initial state accepted for the post-selection: every qubit read 0.
    zero = "0" * N
    kept = {}
    for i, m in zip(init_bits, meas_bits):
        if i == zero:
            kept[m] = kept.get(m, 0) + 1
    return kept, sum(kept.values()) / len(init_bits)


# The three comparisons the rules need all come out of compare(); keeping them
# in one pure function lets the checks feed it exact models with known causes.
def compare(data, atten):
    """All the numbers the rules need, from plain / herald / single counts.

    data = {"P": {basis: counts}, "H": {basis: counts}, "S": {(link, basis): counts}}
    Returns a dict of per-basis losses (value, error) and the rule inputs.
    """
    # P, H: plain and herald; S: per link; Ssum: sum of links; P5: plain, 5 bonds.
    out = {"P": {}, "H": {}, "Ssum": {}, "S": {}, "P5": {}}
    for b in BASES:
        # Singlet bonds only (0, 2, 4): the two unpaired bonds carry no singlet.
        out["P"][b] = loss_of(data["P"][b], atten)
        out["H"][b] = loss_of(data["H"][b], atten)
        # All five bonds, as in Lab 48: value + 3 (the exact value is -3).
        m, e = bond_sum(data["P"][b], atten, range(N - 1))
        out["P5"][b] = (m + 3.0, e)
        parts = [loss_of(data["S"][(j, b)], atten, (2 * j,)) for j in LINKS]
        for j, p in zip(LINKS, parts):
            out["S"][(j, b)] = p
        out["Ssum"][b] = (
            sum(p[0] for p in parts),
            math.sqrt(sum(p[1] ** 2 for p in parts)),
        )

    def total(key):
        """Sum a per-basis table over X, Y, Z with errors added in quadrature."""
        return (
            sum(out[key][b][0] for b in BASES),
            math.sqrt(sum(out[key][b][1] ** 2 for b in BASES)),
        )

    out["P_total"], out["H_total"], out["S_total"] = (
        total("P"),
        total("H"),
        total("Ssum"),
    )
    gain = out["P_total"][0] - out["H_total"][0]
    gerr = math.hypot(out["P_total"][1], out["H_total"][1])
    diff = out["P_total"][0] - out["S_total"][0]
    derr = math.hypot(out["P_total"][1], out["S_total"][1])
    out["gain"], out["diff"] = (gain, gerr), (diff, derr)
    return out


def herald_verdict(gain, gerr, plain_total):
    """The pre-registered HERALD rule."""
    share = gain / plain_total if plain_total > 0 else 0.0
    if gain > SIGMA * gerr and share >= HERALD_MAIN:
        return "initial-state excitation is the MAIN cause"
    if gain > SIGMA * gerr and share >= HERALD_PART:
        return f"initial-state excitation explains PART ({100 * share:.0f}%)"
    return "heralding does not help: NOT an initial-state effect"


def cross_verdict(diff, derr, plain_total):
    """The pre-registered CROSS rule."""
    if diff > SIGMA * derr and diff >= CROSS_SHARE * plain_total:
        return "parallel-gate crosstalk contributes"
    if abs(diff) <= SIGMA * derr:
        return "same loss one link at a time: NOT crosstalk (local to each gate)"
    if diff < -SIGMA * derr:
        return "a single link loses MORE than in parallel: unexpected"
    return "a small, significant parallel excess (below 30% of the loss)"


def control_verdict(p5, lab48):
    """The CONTROL rule: every basis within 3 sigma of the Lab 48 depth-0 loss."""
    ok = True
    for b, (m, e) in p5.items():
        ref, rerr = lab48[b]
        ok &= abs(m - ref) <= SIGMA * math.hypot(e, rerr)
    return ok


def lab48_depth0():
    """Lab 48 depth-0 loss per basis {X, Y, Z: (mean, error)} from the two runs."""
    f, r = np.array(LAB48_BASIS["forward"][0]), np.array(LAB48_BASIS["reverse"][0])
    mean = (f + r) / 2
    err = 0.025 / math.sqrt(2)
    return {b: (float(mean[i]), err) for i, b in enumerate(("X", "Y", "Z"))}


# ----------------------------------------------------------------------------
# Exact-probability simulation (to test the analysis against known causes)
# ----------------------------------------------------------------------------
def pauli_mix(rho, qubits, lam):
    """Two-qubit depolarizing with strength lam via the 15 Pauli errors."""
    if lam <= 0:
        return rho
    # Pauli errors of total probability 15 lam / 16 make rho -> (1-lam) rho + lam I/4.
    lam_p = 15 * lam / 16
    acc = (1 - lam_p) * rho.data
    for a in "IXYZ":
        for c in "IXYZ":
            if a + c != "II":
                acc = acc + (lam_p / 15) * rho.evolve(Pauli(a + c), qargs=qubits).data
    return DensityMatrix(acc)


# The exact simulator below injects each candidate cause by itself, so the
# analysis and the rules are tested on devices where the answer is known.
def exact_counts(kind, basis, link=None, p_init=0.0, lam=0.0, lam_par=0.0):
    """Exact outcome 'counts' of one circuit with the named causes injected.

    p_init   probability that each qubit STARTS in |1> (removed by heralding);
    lam      depolarizing strength after every singlet, in every circuit;
    lam_par  extra depolarizing after every singlet, only when three singlets
             run together (crosstalk).
    kind is "P", "H" (the same state with no initial excitation) or "S".
    """
    weights = np.ones(2**N)
    for i in range(2**N):
        ones = bin(i).count("1")
        weights[i] = (p_init**ones) * ((1 - p_init) ** (N - ones))
    if kind == "H":
        weights = np.zeros(2**N)
        weights[0] = 1.0
    if kind == "S":
        # Only the pair of this link can start excited; the rest are |0>.
        weights = np.zeros(2**N)
        for a in (0, 1):
            for c in (0, 1):
                i = (a << (2 * link)) | (c << (2 * link + 1))
                weights[i] += (p_init if a else 1 - p_init) * (
                    p_init if c else 1 - p_init
                )
    rho = DensityMatrix(np.diag(weights))
    links = LINKS if kind != "S" else (link,)
    for j in links:
        prep = QuantumCircuit(N)
        singlet_pair(prep, j)
        rho = rho.evolve(prep)
        extra = lam + (lam_par if kind != "S" else 0.0)
        rho = pauli_mix(rho, [2 * j, 2 * j + 1], extra)
    rot = QuantumCircuit(N)
    rotate(rot, basis, range(N) if kind != "S" else (2 * link, 2 * link + 1))
    rho = rho.evolve(rot)
    probs = rho.probabilities()
    return {
        format(i, f"0{N}b"): float(p) * BIGSHOTS
        for i, p in enumerate(probs)
        if p > 1e-15
    }


def exact_data(**causes):
    """The `data` dict of compare() for a device with the given causes."""
    data = {"P": {}, "H": {}, "S": {}}
    for b in BASES:
        data["P"][b] = exact_counts("P", b, **causes)
        data["H"][b] = exact_counts("H", b, **causes)
        for j in LINKS:
            data["S"][(j, b)] = exact_counts("S", b, link=j, **causes)
    return data


# ----------------------------------------------------------------------------
# Checks with no device
# ----------------------------------------------------------------------------
# Checks 1 and 2 need no device: they show the circuits are right and that the
# rules recognise each cause when it is the only one present.
def check_circuits(r):
    """The circuits are right: ideal result, gate counts, register layout."""
    log.info("=== Check 1: the circuits on an exact simulator ===")
    keys, circuits = build_jobs()
    r.append(
        check(
            len(keys) == 17, "17 circuits: 2 calibration, 3 plain, 3 herald, 9 single"
        )
    )
    # Ideal state of each plain / single circuit: Z-basis outcomes are the
    # anticorrelated pairs only (01 or 10 on each singlet).
    probs = exact_counts("P", "Z")
    ok = all(all(s[N - 1 - 2 * j] != s[N - 2 - 2 * j] for j in LINKS) for s in probs)
    r.append(check(ok, "plain Z outcomes are anticorrelated on every singlet"))
    psingle = exact_counts("S", "Z", link=1)
    ok = all(s[N - 3] != s[N - 4] and s[:2] == "00" and s[-2:] == "00" for s in psingle)
    r.append(check(ok, "single Z outcomes: only the chosen pair is active"))
    # Singlet correlator -1 in every basis, for plain and single circuits.
    atten = np.ones(N)
    worst = 0.0
    for b in BASES:
        worst = max(worst, abs(loss_of(exact_counts("P", b), atten)[0]))
        for j in LINKS:
            worst = max(
                worst, abs(loss_of(exact_counts("S", b, link=j), atten, (2 * j,))[0])
            )
    r.append(
        check(
            worst < 1e-9,
            f"ideal singlets give correlator -1 in X, Y and Z ({worst:.1e})",
        )
    )
    twoq = [sum(c.count_ops().get(g, 0) for g in ("cx", "cz")) for c in circuits]
    r.append(check(twoq == [expected_twoq(k) for k in keys], f"two-qubit gates {twoq}"))
    herald = [c for k, c in zip(keys, circuits) if isinstance(k, tuple) and k[0] == "H"]
    r.append(
        check(
            all({cr.name for cr in c.cregs} == {"init", "meas"} for c in herald),
            "herald circuits have separate init and meas registers",
        )
    )
    # Counting only: herald_counts keeps the shots whose init is all zeros.
    kept, frac = herald_counts(["000000", "010000", "000000"], ["a", "b", "c"])
    r.append(
        check(kept == {"a": 1, "c": 1} and abs(frac - 2 / 3) < 1e-12, "herald_counts")
    )


def check_analysis(r):
    """The analysis recovers each injected cause and the rules respond."""
    log.info("=== Check 2: the analysis on exact models with known causes ===")
    atten = np.ones(N)
    # (a) a perfect device: nothing anywhere.
    clean = compare(exact_data(), atten)
    r.append(
        check(
            abs(clean["P_total"][0]) < 1e-9 and abs(clean["gain"][0]) < 1e-9,
            "perfect device: no loss, no gain",
        )
    )
    # (b) initial-state excitation only: heralding removes it; X and Y lose, Z not.
    p = 0.01
    init = compare(exact_data(p_init=p), atten)
    lx, ly, lz = (init["P"][b][0] for b in ("X", "Y", "Z"))
    log.info(
        "  init %.0f%%: plain loss X %.4f Y %.4f Z %.4f; herald total %.1e",
        100 * p,
        lx,
        ly,
        lz,
        init["H_total"][0],
    )
    r.append(
        check(
            abs(lx - lz) < 1e-9 and abs(ly - 2 * lx * (1 - p)) < 1e-9 and lx > 0,
            "excited starts: X = Z and Y = 2X(1-p)",
        )
    )
    r.append(check(abs(init["H_total"][0]) < 1e-9, "heralding removes all of it"))
    r.append(
        check(
            abs(init["P"]["X"][0] - 6 * p) < 2e-3,
            "size: about 6 p in X and Z over three singlets",
        )
    )
    # With fake small errors so significance is testable, scale the error bars.
    big = 1e-4
    r.append(
        check(
            herald_verdict(init["gain"][0], big, init["P_total"][0]).endswith(
                "MAIN cause"
            ),
            "rule: init-only -> MAIN cause",
        )
    )
    # (c) depolarizing singlets only: heralding does nothing, no crosstalk.
    dep = compare(exact_data(lam=0.03), atten)
    r.append(
        check(
            abs(dep["gain"][0]) < 1e-9, "depolarizing singlets: heralding gains nothing"
        )
    )
    r.append(
        check(
            abs(dep["diff"][0]) < 1e-9,
            "depolarizing singlets: parallel = sum of singles",
        )
    )
    r.append(
        check(
            abs(dep["P"]["X"][0] - dep["P"]["Z"][0]) < 1e-9,
            "depolarizing singlets: equal in X, Y, Z",
        )
    )
    r.append(
        check(
            herald_verdict(dep["gain"][0], big, dep["P_total"][0]).startswith(
                "heralding does not"
            ),
            "rule: depolarizing -> not initial state",
        )
    )
    r.append(
        check(
            cross_verdict(dep["diff"][0], big, dep["P_total"][0]).startswith(
                "same loss"
            ),
            "rule: depolarizing -> not crosstalk",
        )
    )
    # (d) crosstalk only: the parallel arms (P and H) lose more than the singles.
    cx = compare(exact_data(lam_par=0.03), atten)
    r.append(
        check(
            cx["diff"][0] > 0.2 and abs(cx["gain"][0]) < 1e-9,
            "crosstalk: parallel > singles, heralding no help",
        )
    )
    r.append(
        check(
            cross_verdict(cx["diff"][0], big, cx["P_total"][0]).startswith(
                "parallel-gate"
            ),
            "rule: crosstalk detected",
        )
    )
    # (e) a mixture is split correctly: init 1% + depolarizing 6%.
    mix = compare(exact_data(p_init=0.01, lam=0.06), atten)
    share = mix["gain"][0] / mix["P_total"][0]
    r.append(
        check(
            0.1 < share < 0.5, f"mixture: heralding explains part ({100 * share:.0f}%)"
        )
    )
    r.append(
        check(
            "PART" in herald_verdict(mix["gain"][0], big, mix["P_total"][0]),
            "rule: mixture -> PART",
        )
    )
    # (f) the control rule.
    ref = {b: (0.1, 0.02) for b in "XYZ"}
    r.append(
        check(
            control_verdict({b: (0.11, 0.02) for b in "XYZ"}, ref),
            "control: within 3 sigma passes",
        )
    )
    r.append(
        check(
            not control_verdict(
                {"X": (0.3, 0.02), "Y": (0.1, 0.02), "Z": (0.1, 0.02)}, ref
            ),
            "control: a drifted basis fails",
        )
    )


# ----------------------------------------------------------------------------
# Report on a finished run
# ----------------------------------------------------------------------------
def extract(result, keys):
    """Per circuit: counts (herald circuits: kept counts) and kept fractions."""
    counts, kept = [], {}
    for i, key in enumerate(keys):
        data = result[i].data
        if isinstance(key, tuple) and key[0] == "H":
            c, frac = herald_counts(
                data.init.get_bitstrings(), data.meas.get_bitstrings()
            )
            counts.append(c)
            kept[key] = frac
        else:
            counts.append(data.meas.get_counts())
    return counts, kept


def analyse(r, keys, counts, kept, backend_name, dry):
    """Print the three comparisons and the verdicts."""
    where = "simulated device" if dry else "REAL HARDWARE"
    by_key = dict(zip(keys, counts))
    if dry:
        log.info("(dry run: the numbers below describe the simulator, not ibm_fez)")
    atten = attenuation(by_key["cal0"], by_key["cal1"])
    log.info("%s %s: readout attenuation %s", backend_name, where, np.round(atten, 3))
    r.append(
        check(bool(np.all((atten > 0.8) & (atten <= 1.0))), "attenuation in (0.8, 1]")
    )
    data = {
        "P": {b: by_key[("P", b)] for b in BASES},
        "H": {b: by_key[("H", b)] for b in BASES},
        "S": {(j, b): by_key[("S", j, b)] for j in LINKS for b in BASES},
    }
    out = compare(data, atten)
    log.info(
        "herald kept fraction: %s",
        {b: round(kept[("H", b)], 3) for b in BASES},
    )
    log.info("loss per basis (positive = lost), singlet bonds only, 3 singlets:")
    log.info("  basis    plain P        herald H       singles summed S")
    for b in BASES:
        log.info(
            "  %s    %+.3f +/- %.3f  %+.3f +/- %.3f  %+.3f +/- %.3f",
            b,
            *out["P"][b],
            *out["H"][b],
            *out["Ssum"][b],
        )
    log.info(
        "  total %+.3f +/- %.3f  %+.3f +/- %.3f  %+.3f +/- %.3f",
        *out["P_total"],
        *out["H_total"],
        *out["S_total"],
    )
    asym = lambda tab: tab["Z"][0] - (tab["X"][0] + tab["Y"][0]) / 2  # noqa: E731
    log.info(
        "asymmetry A = Z - (X+Y)/2: plain %+.3f, herald %+.3f, singles %+.3f",
        asym(out["P"]),
        asym(out["H"]),
        asym(out["Ssum"]),
    )
    log.info("per link, X+Y+Z (positive = lost):")
    for j in LINKS:
        tot = sum(out["S"][(j, b)][0] for b in BASES)
        err = math.sqrt(sum(out["S"][(j, b)][1] ** 2 for b in BASES))
        log.info(
            "  link %d (qubits %d-%d): %+.3f +/- %.3f", j, 2 * j, 2 * j + 1, tot, err
        )

    ref = lab48_depth0()
    ok = control_verdict(out["P5"], ref)
    for b in ("X", "Y", "Z"):
        log.info(
            "control %s: plain (5 bonds) %+.3f +/- %.3f, Lab 48 %+.3f +/- %.3f",
            b,
            *out["P5"][b],
            *ref[b],
        )
    log.info(
        "CONTROL: the plain arm %s the Lab 48 depth-0 loss%s",
        "reproduces" if ok else "does NOT reproduce",
        "" if ok else " (the device may have drifted)",
    )
    gain, gerr = out["gain"]
    diff, derr = out["diff"]
    log.info(
        "HERALD gain %+.3f +/- %.3f (%.0f%% of the plain loss)",
        gain,
        gerr,
        100 * gain / max(out["P_total"][0], 1e-12),
    )
    log.info("verdict (HERALD rule): %s", herald_verdict(gain, gerr, out["P_total"][0]))
    log.info("CROSS diff %+.3f +/- %.3f", diff, derr)
    log.info("verdict (CROSS rule): %s", cross_verdict(diff, derr, out["P_total"][0]))
    if dry:
        log.info("(dry run: those verdicts are about the simulator, not ibm_fez)")


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
        "(recorded rates %.2f and %.2f ms per shot; the herald circuits add a "
        "mid-circuit measurement)",
        n_circuits,
        shots,
        total,
        min(rates) * total,
        max(rates) * total,
        min(rates) * 1e3,
        max(rates) * 1e3,
    )


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
        counts, kept = extract(job.result(), keys)
    except Exception as exc:  # noqa: BLE001
        log.fail("retrieve failed: %s: %s", type(exc).__name__, scrub(exc, secret))
        return 1
    analyse(r, keys, counts, kept, meta.get("backend", "device"), False)
    return summarise(r)


def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 51: herald and isolate")
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
    want = [expected_twoq(k) for k in keys]
    log.info("two-qubit gates: %s (expected %s)", twoq, want)
    r.append(check(twoq == want, "no gate cancelled and no routing overhead"))
    # The first measurement of the herald circuits must survive: 12 measures.
    meas = {k: c.count_ops().get("measure", 0) for k, c in zip(keys, isa)}
    r.append(
        check(
            all(meas[("H", b)] == 2 * N for b in BASES),
            "the herald circuits keep both measurements (12 each)",
        )
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
            "python3 labs/lab51_herald_and_isolate.py --retrieve %s",
            args.wait,
            job_id,
        )
        return 0 if all(r) else 1
    if state in TERMINAL_BAD:
        log.fail("job ended with status %s", state)
        return 1

    log.info("=== Check 6: results ===")
    counts, kept = extract(job.result(), keys)
    r.append(
        check(
            all(
                sum(c.values()) == args.shots
                for k, c in zip(keys, counts)
                if not (isinstance(k, tuple) and k[0] == "H")
            ),
            f"every ungated circuit returned {args.shots} shots",
        )
    )
    analyse(r, keys, counts, kept, backend.name, not args.live)
    return summarise(r)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
