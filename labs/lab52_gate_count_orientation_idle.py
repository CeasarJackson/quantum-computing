#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab52_gate_count_orientation_idle.py

Purpose
-------
Lab 51 ruled out initial-state excitation and crosstalk: about 0.29 of the
0.34 depth-0 loss is local to each singlet-preparation gate sequence, and link 0
(qubits 129-118) loses 0.056 against 0.140 and 0.125 on links 1 and 2 although
their reported CZ errors are alike. This lab asks WHAT inside one link's
sequence costs the loss, with three single-link experiments in ONE job.

On each of the three links, one singlet at a time (the others idle in |0>), in
the X, Y and Z bases:

  A  the plain singlet (control; reproduces Lab 51's single-link arm)
  B  the singlet followed by EXTRA_CZ = 8 extra CX gates (four identity pairs,
     separated by barriers so the transpiler cannot cancel them). Ideal state
     unchanged; the added loss is the cost of those gates ON THIS STATE.
  D  the singlet prepared the other way round (H and CX with the roles of the
     two qubits exchanged; the state is the same up to a global sign). Tests
     whether the loss belongs to a qubit or to a direction of the sequence.
  E  the singlet followed by an idle DELAY_NS = 1000 ns on the pair. Tests
     whether the pair dephases faster than the calibration T2 says.

CIRCUITS: 4 variants x 3 links x 3 bases = 36, plus the two readout
calibration circuits = 38, at 3000 shots each (114000 shots, about 45 to 120 s
of QPU time by the recorded rates).

THE RULES, fixed here before any live run. Every "loss" is summed over the
three bases for one singlet bond (positive = lost) and corrected for readout
as in Labs 44 to 51. s = (loss B - loss A) / EXTRA_CZ is the cost of one extra
CX on the singlet, averaged over the three links; a = the mean loss A.

  GATE COUNT  s_model = 3 lambda = 0.0111 (Lab 21's per-gate depolarizing
              strength, summed over three bases).
              s >= 0.5 a : each extra gate costs about as much as the whole
                           preparation: the two-qubit gate IS the cause.
              s <= 3 s_model and s < 0.25 a : extra gates are cheap; the loss
                           is paid ONCE, in the preparation (first-use or
                           single-qubit-gate effect), not per gate.
              otherwise  : intermediate; report s and a.
  ORIENT      d_j = loss D - loss A on link j. |d_j| > 3 sigma: the loss
              depends on which qubit carries the H (a qubit- or direction-
              specific error); otherwise it does not.
  IDLE        excess = (loss E - loss A) - predicted, where the prediction is
              the dephasing of the two qubits at their reported T2:
              2 (1 - exp(-t (1/T2a + 1/T2b))). excess > 3 sigma: the pair
              dephases faster than reported; |excess| <= 3 sigma: idle
              decoherence follows the calibration.
  LINKS       loss A on link 0 against the mean of links 1 and 2: a gap above
              3 sigma reproduces Lab 51's per-link difference.

Readout is corrected with the Z-basis attenuation of the same job (Lab 49
showed that is good to about 0.004 per qubit).

FIRST LIVE RUNS (ibm_fez, 2026-10-17 23:04 and 23:09, same six qubits, 42 s
each, 38 circuits x 3000 shots; two independent jobs of the same script):
  * GATE COUNT: s = +0.0189 and +0.0172 (+/- 0.0014) per extra CX, summed over
    X, Y, Z, against the model 3 lambda = 0.0111 and a mean preparation loss
    a = 0.094 and 0.112. s is 1.6 times the model but only 17% of a, so the
    rule says the loss is paid ONCE, in the preparation, not per gate: the
    whole singlet costs about as much as 6 extra CX gates.
  * ORIENT: D - A is within 2.1 sigma on every link in both jobs (largest
    +0.036 +/- 0.017): the loss does NOT depend on which qubit carries the H.
  * IDLE: the 1 us idle adds 0.11 to 0.16 per link against 0.03 to 0.06 predicted
    from the reported T2; the mean excess is +0.092 +/- 0.011 and +0.086 +/-
    0.011 (about 8 sigma each): the pair dephases 2.5 to 4 times faster than the
    calibration T2 says. A singlet is immune to dephasing that is the same on
    both qubits, so this is a difference between the two qubits (or a
    free-evolution T2* shorter than the echo T2 that the calibration reports).
  * LINKS: link 0 (129-118) loses less: gap to the mean of links 1 and 2
    0.036 +/- 0.015 and 0.074 +/- 0.015 (mean 0.055 +/- 0.011), with Lab 51's
    0.077: the link difference reproduces in sign in all three jobs.
  What the run does not say: how much of the preparation loss (a = 0.10) is
  idle-type dephasing during the sequence itself. The next test is a delay
  scan with and without a mid-delay echo.

LIMITATIONS: one device and one job; each variant isolates one change but the
extra CX gates also carry their single-qubit gates (two sx per CX), so B
measures the cost of a gate WITH its dressing; shot-noise error bars only (about
0.012 per link and variant); the T2 values come from the calibration at the
time of the run, not from the run.

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
    python3 labs/lab52_gate_count_orientation_idle.py                # dry run
    read -rs IBM_QUANTUM_TOKEN; export IBM_QUANTUM_TOKEN             # hidden
    python3 labs/lab52_gate_count_orientation_idle.py --live --backend ibm_fez \\
        --token-env IBM_QUANTUM_TOKEN
    python3 labs/lab52_gate_count_orientation_idle.py --retrieve <job id>

Log files
---------
logs/lab52/lab52_<YYYYmmdd_HHMMSS>.log (plain text, never contains the key)
logs/lab52/last_job.json (job id, backend, shots, layout; no secrets)

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
from qiskit.quantum_info import DensityMatrix, Pauli

from applog import get_logger

log = get_logger("lab52")

# Everything shared is imported, so the security-critical code and the
# estimators exist in one place each.
try:
    from fezdata import LAMBDA_FEZ, LAYOUT, RUNS
    from lab17_mirror_benchmark import (
        TERMINAL_BAD,
        connect,
        make_sampler,
        scrub,
        wait_for,
    )
    from lab21_noise_aware_training import N
    from lab44_noise_aware_on_qpu import attenuation, calibration_circuits
    from lab46_hardware_noise_budget import MAX_SHOTS, pick_backend, transpile_all
    from lab51_herald_and_isolate import (
        BASES,
        LINKS,
        SIGMA,
        check,
        loss_of,
        rotate,
        singlet_pair,
    )
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

JOB_FILE = Path("logs/lab52/last_job.json")
VARIANTS = ("A", "B", "D", "E")
EXTRA_CZ = 8  # extra CX gates in variant B (even: the ideal state is unchanged)
DELAY_NS = 1000  # idle time in variant E
S_MODEL = 3 * LAMBDA_FEZ  # one gate's depolarizing cost, summed over 3 bases
RULE_HIGH, RULE_LOW = 0.5, 0.25  # shares of the mean preparation loss a
RULE_MODEL = 3.0  # "cheap" means s <= 3 s_model


# ----------------------------------------------------------------------------
# Circuits
# ----------------------------------------------------------------------------
def swapped_singlet(qc, link):
    """The Lab 21 singlet with the roles of the two qubits exchanged.

    H and CX start on the second qubit, Z and X finish on the other way round;
    the state is the singlet up to a global sign.
    """
    a, b = 2 * link, 2 * link + 1
    qc.h(b)
    qc.cx(b, a)
    qc.z(b)
    qc.x(a)


def variant_circuit(variant, link, basis):
    """One single-link circuit; the other four qubits stay in |0>."""
    qc = QuantumCircuit(N)
    a, b = 2 * link, 2 * link + 1
    if variant == "D":
        swapped_singlet(qc, link)
    else:
        singlet_pair(qc, link)
    if variant == "B":
        # A barrier before each CX keeps the transpiler from cancelling pairs.
        for _ in range(EXTRA_CZ):
            qc.barrier(a, b)
            qc.cx(a, b)
    if variant == "E":
        qc.barrier(a, b)
        qc.delay(DELAY_NS, [a, b], unit="ns")
        qc.barrier(a, b)
    rotate(qc, basis, (a, b))
    qc.measure_all()
    return qc


def build_jobs():
    """Keys and circuits in a fixed order: calibration first, then variants."""
    keys, circuits = ["cal0", "cal1"], calibration_circuits()
    for v in VARIANTS:
        for j in LINKS:
            for b in BASES:
                keys.append((v, j, b))
                circuits.append(variant_circuit(v, j, b))
    return keys, circuits


def expected_twoq(key):
    """Two-qubit gates a circuit should keep after transpilation."""
    if key in ("cal0", "cal1"):
        return 0
    return 1 + EXTRA_CZ if key[0] == "B" else 1


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def link_loss(counts_by_basis, atten, link):
    """(loss, error) of one link's singlet, summed over the three bases."""
    vals = [loss_of(counts_by_basis[b], atten, (2 * link,)) for b in BASES]
    return sum(v[0] for v in vals), math.sqrt(sum(v[1] ** 2 for v in vals))


def tabulate(by_key, atten):
    """{(variant, link): (loss, error)} from the counts of every circuit."""
    table = {}
    for v in VARIANTS:
        for j in LINKS:
            table[(v, j)] = link_loss({b: by_key[(v, j, b)] for b in BASES}, atten, j)
    return table


def idle_prediction(t2a, t2b, delay_ns):
    """Predicted loss summed over three bases from dephasing during the delay.

    Each qubit's X and Y coherence decays as exp(-t / T2), so XX and YY of the
    singlet are multiplied by exp(-t (1/T2a + 1/T2b)); ZZ is untouched.
    """
    t = delay_ns * 1e-9
    return 2.0 * (1.0 - math.exp(-t * (1.0 / t2a + 1.0 / t2b)))


def gate_count_verdict(s, a):
    """The pre-registered GATE COUNT rule; s = cost per extra gate, a = prep loss."""
    if a > 0 and s >= RULE_HIGH * a:
        return "each extra gate costs as much as the preparation: the two-qubit gate IS the cause"
    if s <= RULE_MODEL * S_MODEL and s < RULE_LOW * a:
        return "extra gates are cheap: the loss is paid ONCE, in the preparation, not per gate"
    return "intermediate: the cost per gate is neither normal nor the whole preparation"


def orient_verdict(diffs):
    """The pre-registered ORIENT rule; diffs = [(d, err)] per link."""
    hits = [j for j, (d, e) in enumerate(diffs) if abs(d) > SIGMA * e]
    if hits:
        return f"the loss depends on the orientation on link(s) {hits}"
    return "the loss does NOT depend on which qubit carries the H"


def idle_verdict(excess, err):
    """The pre-registered IDLE rule."""
    if excess > SIGMA * err:
        return "the pair dephases FASTER than the reported T2"
    return "idle decoherence follows the calibration"


def links_verdict(a0, e0, a12, e12):
    """The LINKS rule: link 0 against the mean of links 1 and 2."""
    gap, err = a12 - a0, math.hypot(e0, e12)
    if gap > SIGMA * err:
        return f"link 0 loses less than links 1 and 2 (gap {gap:.3f} +/- {err:.3f}): reproduced"
    return "no significant link difference in this job"


def analyse_table(table, t2s, delay_ns=DELAY_NS):
    """All the numbers the rules need, from the per-(variant, link) table."""
    a_mean = float(np.mean([table[("A", j)][0] for j in LINKS]))
    slopes = [
        (
            (table[("B", j)][0] - table[("A", j)][0]) / EXTRA_CZ,
            math.hypot(table[("B", j)][1], table[("A", j)][1]) / EXTRA_CZ,
        )
        for j in LINKS
    ]
    s = float(np.mean([x[0] for x in slopes]))
    s_err = math.sqrt(sum(x[1] ** 2 for x in slopes)) / len(LINKS)
    diffs = [
        (
            table[("D", j)][0] - table[("A", j)][0],
            math.hypot(table[("D", j)][1], table[("A", j)][1]),
        )
        for j in LINKS
    ]
    idle = []
    for j in LINKS:
        e = (
            table[("E", j)][0] - table[("A", j)][0],
            math.hypot(table[("E", j)][1], table[("A", j)][1]),
        )
        pred = idle_prediction(t2s[2 * j], t2s[2 * j + 1], delay_ns) if t2s else None
        idle.append((e, pred))
    return {
        "a": a_mean,
        "s": (s, s_err),
        "slopes": slopes,
        "orient": diffs,
        "idle": idle,
    }


# ----------------------------------------------------------------------------
# Exact two-qubit model (to test the analysis against known causes)
# ----------------------------------------------------------------------------
def depolarize2(rho, lam):
    """Two-qubit depolarizing of strength lam through the 15 Pauli errors."""
    if lam <= 0:
        return rho
    # Equivalent to rho -> (1 - lam) rho + lam I/4: each of the 15 Pauli errors
    # has probability 15 lam / 16 / 15, so that the correlator shrinks by 1 - lam.
    lam_p = 15 * lam / 16
    acc = (1 - lam_p) * rho.data
    for p in "IXYZ":
        for q in "IXYZ":
            if p + q != "II":
                acc = acc + (lam_p / 15) * rho.evolve(Pauli(p + q)).data
    return DensityMatrix(acc)


def dephase1(rho, qubit, factor):
    """Phase damping that multiplies the X and Y coherence of `qubit` by factor."""
    p = (1 - factor) / 2
    return DensityMatrix(
        (1 - p) * rho.data + p * rho.evolve(Pauli("Z"), qargs=[qubit]).data
    )


def model_counts(
    variant, link, basis, lam_prep=0.0, lam_cz=0.0, lam_orient=0.0, t2=None
):
    """Exact outcome 'counts' of one link circuit with the named causes injected.

    lam_prep    depolarizing after the first CX (the preparation);
    lam_cz      depolarizing after each of the extra CX gates (variant B);
    lam_orient  extra depolarizing after the preparation in variant D only;
    t2          (T2a, T2b) in seconds for the delay of variant E.
    """
    # Work on the two qubits of the link as qubits 0 (a) and 1 (b).
    rho = DensityMatrix.from_label("00")
    prep = QuantumCircuit(2)
    if variant == "D":
        prep.h(1)
        prep.cx(1, 0)
        prep.z(1)
        prep.x(0)
    else:
        prep.h(0)
        prep.cx(0, 1)
        prep.z(0)
        prep.x(1)
    rho = depolarize2(
        rho.evolve(prep), lam_prep + (lam_orient if variant == "D" else 0)
    )
    if variant == "B":
        cx = QuantumCircuit(2)
        cx.cx(0, 1)
        for _ in range(EXTRA_CZ):
            rho = depolarize2(rho.evolve(cx), lam_cz)
    if variant == "E" and t2:
        for q in (0, 1):
            rho = dephase1(rho, q, math.exp(-DELAY_NS * 1e-9 / t2[q]))
    rot = QuantumCircuit(2)
    rotate(rot, basis, (0, 1))
    probs = rho.evolve(rot).probabilities()
    out = {}
    for i, p in enumerate(probs):
        if p > 1e-15:
            # Place the two bits at the link's qubits in a six-bit string.
            bits = ["0"] * N
            bits[N - 1 - 2 * link] = str(i & 1)
            bits[N - 2 - 2 * link] = str((i >> 1) & 1)
            key = "".join(bits)
            out[key] = out.get(key, 0.0) + float(p) * 1e9
    return out


def model_data(**causes):
    """by_key-style counts for the three bases of every variant and link."""
    by_key = {}
    for v in VARIANTS:
        for j in LINKS:
            for b in BASES:
                by_key[(v, j, b)] = model_counts(v, j, b, **causes)
    return by_key


# ----------------------------------------------------------------------------
# Checks with no device
# ----------------------------------------------------------------------------
def check_circuits(r):
    """The circuits are right: ideal result, gate counts, no cancellation."""
    log.info("=== Check 1: the circuits on an exact simulator ===")
    keys, circuits = build_jobs()
    r.append(
        check(
            len(keys) == 38,
            "38 circuits: 2 calibration + 4 variants x 3 links x 3 bases",
        )
    )
    ones = np.ones(N)
    worst = 0.0
    for v in VARIANTS:
        for j in LINKS:
            by = {b: model_counts(v, j, b) for b in BASES}
            worst = max(worst, abs(link_loss(by, ones, j)[0]))
    r.append(
        check(worst < 1e-9, f"ideal singlet loss is 0 in every variant ({worst:.1e})")
    )
    # The circuits themselves (not only the model): run the state of each.
    from qiskit.quantum_info import Statevector  # noqa: PLC0415

    sv = Statevector(
        variant_circuit("A", 1, "Z").remove_final_measurements(inplace=False)
    )
    sv_d = Statevector(
        variant_circuit("D", 1, "Z").remove_final_measurements(inplace=False)
    )
    r.append(
        check(
            abs(abs(sv.inner(sv_d)) - 1) < 1e-9,
            "A and D prepare the same state up to a phase",
        )
    )
    sv_b = Statevector(
        variant_circuit("B", 1, "Z").remove_final_measurements(inplace=False)
    )
    r.append(
        check(
            abs(abs(sv.inner(sv_b)) - 1) < 1e-9,
            "B (8 extra CX) leaves the state unchanged",
        )
    )
    twoq = [sum(c.count_ops().get(g, 0) for g in ("cx", "cz")) for c in circuits]
    r.append(
        check(
            twoq == [expected_twoq(k) for k in keys],
            "two-qubit gate counts per circuit",
        )
    )
    delays = [
        c.count_ops().get("delay", 0)
        for k, c in zip(keys, circuits)
        if isinstance(k, tuple) and k[0] == "E"
    ]
    r.append(
        check(
            set(delays) == {2}, "variant E carries one delay on each qubit of the pair"
        )
    )
    r.append(
        check(
            abs(idle_prediction(100e-6, 100e-6, 1000) - 2 * (1 - math.exp(-0.02)))
            < 1e-12,
            "idle_prediction formula",
        )
    )


def check_analysis(r):
    """The analysis and the rules recognise each cause when it is alone."""
    log.info("=== Check 2: the analysis on exact models with known causes ===")
    ones = np.ones(N)
    t2 = [100e-6] * N
    # (a) preparation-only error: B adds almost nothing per gate.
    prep = analyse_table(tabulate(model_data(lam_prep=0.03), ones), None)
    log.info("  prep-only: a %.4f, s %.5f", prep["a"], prep["s"][0])
    r.append(
        check(
            abs(prep["a"] - 3 * 0.03) < 5e-3 and abs(prep["s"][0]) < 1e-9,
            "prep-only: a = 3 lam, s = 0",
        )
    )
    r.append(
        check(
            gate_count_verdict(prep["s"][0], prep["a"]).startswith(
                "extra gates are cheap"
            ),
            "rule: prep-only -> paid ONCE",
        )
    )
    # (b) gate error only: every CX costs 3 lam.
    gate = analyse_table(tabulate(model_data(lam_prep=0.03, lam_cz=0.03), ones), None)
    log.info("  gate-only: a %.4f, s %.5f", gate["a"], gate["s"][0])
    r.append(
        check(
            abs(gate["s"][0] - 3 * ((1 - 0.97**9) - 0.03) / EXTRA_CZ) < 1e-6,
            "gate error: s = exact compounded cost",
        )
    )
    r.append(
        check(
            gate_count_verdict(gate["s"][0], gate["a"]).startswith("each extra gate"),
            "rule: gate error -> the gate IS the cause",
        )
    )
    # (c) normal gate error (the Lab 21 value): the verdict is 'paid once'.
    norm = analyse_table(
        tabulate(model_data(lam_prep=0.03, lam_cz=LAMBDA_FEZ), ones), None
    )
    r.append(
        check(abs(norm["s"][0] - S_MODEL) < 2e-3, "normal gate error: s = 3 lambda")
    )
    r.append(
        check(
            gate_count_verdict(norm["s"][0], norm["a"]).startswith(
                "extra gates are cheap"
            ),
            "rule: normal gates -> paid ONCE",
        )
    )
    # (d) orientation: lam_orient only changes variant D.
    ori = analyse_table(
        tabulate(model_data(lam_prep=0.02, lam_orient=0.03), ones), None
    )
    r.append(
        check(
            all(abs(d[0] - 0.09) < 5e-3 for d in ori["orient"]),
            "orientation error: D - A = 3 lam_orient",
        )
    )
    small = [(d[0], 1e-4) for d in ori["orient"]]
    r.append(
        check(
            orient_verdict(small).startswith("the loss depends"),
            "rule: orientation detected",
        )
    )
    r.append(
        check(
            orient_verdict([(0.0, 0.01)] * 3).startswith("the loss does NOT"),
            "rule: no orientation effect",
        )
    )
    # (e) idle dephasing at the calibration T2: excess is zero.
    idle = analyse_table(tabulate(model_data(t2=(100e-6, 100e-6)), ones), t2)
    (d_idle, _), pred = idle["idle"][0]
    r.append(check(abs(d_idle - pred) < 5e-4, "idle model matches idle_prediction"))
    r.append(
        check(
            idle_verdict(0.0, 0.01).startswith("idle decoherence follows"),
            "rule: idle consistent",
        )
    )
    r.append(
        check(
            idle_verdict(0.1, 0.01).startswith("the pair dephases FASTER"),
            "rule: idle too fast",
        )
    )
    r.append(
        check(
            links_verdict(0.05, 0.01, 0.13, 0.01).endswith("reproduced"),
            "rule: link gap reproduced",
        )
    )
    r.append(
        check(
            links_verdict(0.1, 0.01, 0.11, 0.01).startswith("no significant"),
            "rule: no link gap",
        )
    )


# ----------------------------------------------------------------------------
# Report on a finished run
# ----------------------------------------------------------------------------
def t2_list(backend, physical):
    """Reported T2 (seconds) of the six physical qubits, or None if unavailable."""
    try:
        vals = [backend.qubit_properties(q).t2 for q in physical]
    except Exception:  # noqa: BLE001
        return None
    return vals if all(v for v in vals) else None


def report(r, table, res, t2s, dry):
    """Print the tables and the four verdicts."""
    log.info("loss per link, summed over X, Y, Z (positive = lost):")
    log.info(
        "  link   A plain        B +%d CX      D swapped      E +%d ns idle",
        EXTRA_CZ,
        DELAY_NS,
    )
    for j in LINKS:
        log.info(
            "  %d    %+.3f +/- %.3f  %+.3f +/- %.3f  %+.3f +/- %.3f  %+.3f +/- %.3f",
            j,
            *table[("A", j)],
            *table[("B", j)],
            *table[("D", j)],
            *table[("E", j)],
        )
    s, s_err = res["s"]
    log.info(
        "mean preparation loss a = %.3f; cost per extra CX s = %+.4f +/- %.4f (model 3 lambda = %.4f)",
        res["a"],
        s,
        s_err,
        S_MODEL,
    )
    log.info("per link s: %s", ", ".join(f"{x[0]:+.4f}" for x in res["slopes"]))
    log.info("verdict (GATE COUNT rule): %s", gate_count_verdict(s, res["a"]))
    log.info(
        "orientation, D - A per link: %s",
        ", ".join(f"{d:+.3f} +/- {e:.3f}" for d, e in res["orient"]),
    )
    log.info("verdict (ORIENT rule): %s", orient_verdict(res["orient"]))
    if t2s:
        excess = []
        for j, ((d, e), pred) in zip(LINKS, res["idle"]):
            log.info(
                "idle link %d: E - A %+.3f +/- %.3f, predicted from T2 %+.3f",
                j,
                d,
                e,
                pred,
            )
            excess.append((d - pred, e))
        tot = sum(x[0] for x in excess) / len(excess)
        terr = math.sqrt(sum(x[1] ** 2 for x in excess)) / len(excess)
        log.info("mean excess over the T2 prediction: %+.3f +/- %.3f", tot, terr)
        log.info("verdict (IDLE rule): %s", idle_verdict(tot, terr))
    else:
        log.warn_("no T2 values for this backend: IDLE rule not evaluated")
    a0, a12 = table[("A", 0)], [table[("A", 1)], table[("A", 2)]]
    log.info(
        "verdict (LINKS rule): %s",
        links_verdict(
            a0[0],
            a0[1],
            sum(x[0] for x in a12) / 2,
            math.sqrt(sum(x[1] ** 2 for x in a12)) / 2,
        ),
    )
    if dry:
        log.info("(dry run: those verdicts are about the simulator, not ibm_fez)")


def analyse(r, keys, counts, backend, used, dry):
    """Readout correction, the tables, and the report."""
    by_key = dict(zip(keys, counts))
    where = "simulated device" if dry else "REAL HARDWARE"
    if dry:
        log.info("(dry run: the numbers below describe the simulator, not ibm_fez)")
    atten = attenuation(by_key["cal0"], by_key["cal1"])
    log.info("%s %s: readout attenuation %s", backend.name, where, np.round(atten, 3))
    r.append(
        check(bool(np.all((atten > 0.8) & (atten <= 1.0))), "attenuation in (0.8, 1]")
    )
    table = tabulate(by_key, atten)
    t2s = t2_list(backend, used)
    report(r, table, analyse_table(table, t2s), t2s, dry)


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
        backend = service.backend(meta["backend"])
    except Exception as exc:  # noqa: BLE001
        log.fail("retrieve failed: %s: %s", type(exc).__name__, scrub(exc, secret))
        return 1
    analyse(r, keys, counts, backend, meta["layout"], False)
    return summarise(r)


def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 52: gate count, orientation, idle")
    ap.add_argument("--live", action="store_true", help="use real IBM hardware")
    ap.add_argument("--retrieve", metavar="JOB_ID", help="analyse an earlier job")
    ap.add_argument("--backend", help="backend name; default: least busy")
    ap.add_argument(
        "--layout",
        default=",".join(str(q) for q in LAYOUT),
        help="six physical qubits (default: the Lab 44 chain) or 'auto'",
    )
    ap.add_argument(
        "--shots", type=int, default=3000, help=f"shots per circuit, 100..{MAX_SHOTS}"
    )
    ap.add_argument("--max-seconds", type=int, default=200, help="QPU time cap")
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
    delays = [
        c.count_ops().get("delay", 0)
        for k, c in zip(keys, isa)
        if isinstance(k, tuple) and k[0] == "E"
    ]
    r.append(
        check(all(d >= 1 for d in delays), "every variant E circuit keeps its delay")
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
            "python3 labs/lab52_gate_count_orientation_idle.py --retrieve %s",
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
    analyse(r, keys, counts, backend, used, not args.live)
    return summarise(r)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
