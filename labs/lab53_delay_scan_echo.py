#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab53_delay_scan_echo.py

Purpose
-------
Lab 52 found that the singlet-preparation loss is paid once (about 0.10 per
link, summed over X, Y, Z), and that a 1 us idle after the singlet costs 0.11 to
0.16, 2.5 to 4 times what the reported T2 predicts. A singlet is immune to
dephasing that is the same on both qubits, so the idle loss is a difference
between the two qubits. This lab asks what kind of difference it is, and
whether it can account for the preparation loss. On each of the three links,
one singlet at a time, in X, Y and Z:

  F0 / F1000 / F2000   the singlet, then a FREE idle of 0, 1000, 2000 ns
  E0 / E1000 / E2000   the singlet, then an ECHOED idle of 0, 1000, 2000 ns:
                       half the idle, an X gate on both qubits, the other half.
                       X on both qubits maps the singlet to itself (up to a
                       sign), so the ideal state is unchanged, but a constant
                       difference of the two qubits' frequencies cancels.
                       E0 is the two X gates alone: the cost of the pulses.

CIRCUITS: 6 variants x 3 links x 3 bases = 54, plus the two readout calibration
circuits = 56, at 2000 shots (112000 shots; Lab 52's 114000 took 42 s).

THE RULES, fixed here before any live run. L(v) is the loss of variant v
(summed over X, Y, Z, averaged over the three links, positive = lost, readout
corrected as in Labs 44 to 52). f1 = L(F1000) - L(F0), f2 = L(F2000) - L(F0),
e1 = L(E1000) - L(E0), e2 = L(E2000) - L(E0).

  SHAPE   r = f2 / f1. A white (Markovian) dephasing grows linearly: r = 2.
          A constant frequency difference grows as t squared: r = 4 (for small
          loss). 1.4 <= r <= 2.6 : LINEAR, white dephasing. r >= 3.2 :
          QUADRATIC, a quasi-static frequency difference. Otherwise
          intermediate.
  ECHO    q = e2 / f2. q < 0.3 : the echo removes it, so the idle loss is a
          constant frequency difference that can be refocused. q > 0.7 : the
          echo does not help, so it is fast noise that cannot be refocused.
          Otherwise partial. The echoed idle is also compared with the
          dephasing the reported (echo) T2 predicts: e1 minus the prediction
          within 3 sigma means the echoed idle follows the calibration.
  PULSE   p = L(E0) - L(F0), the cost of the two X gates: significant above
          3 sigma.
  PREP    The idle loss rate extrapolated to the duration of the preparation
          itself (the singlet circuit's scheduled length on the chosen
          qubits): linear k1 = f1 / 1000 ns, or quadratic k2 = f1 / 1000 ns
          squared, as SHAPE says. Share = predicted loss / L(F0). >= 0.5 :
          idle-type dephasing during the sequence explains most of the
          preparation loss; 0.15 to 0.5 : part; < 0.15 : not the main cause
          (the preparation loss is then in the gates themselves).

FIRST LIVE RUN (ibm_fez, 2026-10-17 23:56, same six qubits, 42 s from queue to
done, 56 circuits x 2000 shots): the idle loss is real but it is NOT the cause
of the preparation loss.
  * SHAPE: f1 = 0.100 +/- 0.014, f2 = 0.204 +/- 0.015, r = 2.04 +/- 0.32:
    LINEAR in t, white (Markovian) dephasing, not a constant frequency
    difference.
  * ECHO: e1 = 0.071 +/- 0.014, e2 = 0.121 +/- 0.014, q = 0.60 +/- 0.08: the
    echo helps PARTLY (removes about 40% of the free-idle loss). The echoed idle
    is within 3 sigma of the dephasing the reported echo T2 predicts (0.071
    against 0.049), so the part the echo cannot remove follows the calibration;
    the part it removes is the excess that Lab 52 found over the reported T2.
  * PULSE: the two X gates cost -0.005 +/- 0.013: nothing.
  * PREP: the singlet is scheduled for 116 ns; the free-idle rate (linear)
    gives 0.012 over that time, 11% of the preparation loss L(F0) = 0.107 +/-
    0.009: idle-type dephasing is NOT the main cause, so about 0.095 of the
    preparation loss is in the gates themselves.
  Per-link numbers are in the log; link 0 again has the smallest F0 (0.080
  against 0.124 and 0.118).

LIMITATIONS: one device and one job; the free-idle rate is a proxy for what the
qubits do while being gated (a driven qubit is not an idling one), so PREP is an
estimate, not a measurement; the echo cancels only frequency differences that
are constant over the idle; shot-noise error bars only; the reported T2 is
read from the calibration at the time of the run.

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
    python3 labs/lab53_delay_scan_echo.py                            # dry run
    read -rs IBM_QUANTUM_TOKEN; export IBM_QUANTUM_TOKEN             # hidden
    python3 labs/lab53_delay_scan_echo.py --live --backend ibm_fez \\
        --token-env IBM_QUANTUM_TOKEN
    python3 labs/lab53_delay_scan_echo.py --retrieve <job id> --token-env IBM_QUANTUM_TOKEN

Log files
---------
logs/lab53/lab53_<YYYYmmdd_HHMMSS>.log (plain text, never contains the key)
logs/lab53/last_job.json (job id, backend, shots, layout; no secrets)

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
from qiskit.quantum_info import DensityMatrix, Pauli, Statevector

from applog import get_logger

log = get_logger("lab53")

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
    from lab44_noise_aware_on_qpu import attenuation, calibration_circuits
    from lab46_hardware_noise_budget import MAX_SHOTS, pick_backend, transpile_all
    from lab51_herald_and_isolate import (
        BASES,
        LINKS,
        SIGMA,
        check,
        rotate,
        singlet_pair,
    )
    from lab52_gate_count_orientation_idle import (
        depolarize2,
        idle_prediction,
        link_loss,
        plan_time,
        summarise,
        t2_list,
    )
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

JOB_FILE = Path("logs/lab53/last_job.json")
# (kind, idle in ns): F = free idle, E = echoed idle.
VARIANTS = (("F", 0), ("F", 1000), ("F", 2000), ("E", 0), ("E", 1000), ("E", 2000))
LINEAR = (1.4, 2.6)  # r in this range: white dephasing
QUADRATIC = 3.2  # r at or above: quasi-static frequency difference
ECHO_LOW, ECHO_HIGH = 0.3, 0.7
PREP_HIGH, PREP_LOW = 0.5, 0.15


# ----------------------------------------------------------------------------
# Circuits
# ----------------------------------------------------------------------------
def prep_circuit(link):
    """The singlet alone, no measurement: used to read the scheduled duration."""
    qc = QuantumCircuit(N)
    singlet_pair(qc, link)
    return qc


def variant_circuit(kind, idle, link, basis):
    """One single-link circuit: singlet, free or echoed idle, readout."""
    qc = prep_circuit(link)
    a, b = 2 * link, 2 * link + 1
    qc.barrier(a, b)
    if kind == "F":
        if idle:
            qc.delay(idle, [a, b], unit="ns")
    else:
        # Echo: half the idle, X on BOTH qubits, the other half. X x X maps the
        # singlet to minus itself, so only a global sign changes.
        if idle:
            qc.delay(idle // 2, [a, b], unit="ns")
        qc.barrier(a, b)
        qc.x(a)
        qc.x(b)
        qc.barrier(a, b)
        if idle:
            qc.delay(idle // 2, [a, b], unit="ns")
    qc.barrier(a, b)
    rotate(qc, basis, (a, b))
    qc.measure_all()
    return qc


def build_jobs():
    """Keys and circuits in a fixed order: calibration first."""
    keys, circuits = ["cal0", "cal1"], calibration_circuits()
    for kind, idle in VARIANTS:
        for j in LINKS:
            for b in BASES:
                keys.append((kind, idle, j, b))
                circuits.append(variant_circuit(kind, idle, j, b))
    return keys, circuits


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def tabulate(by_key, atten):
    """{(kind, idle, link): (loss, error)} summed over the three bases."""
    table = {}
    for kind, idle in VARIANTS:
        for j in LINKS:
            counts = {b: by_key[(kind, idle, j, b)] for b in BASES}
            table[(kind, idle, j)] = link_loss(counts, atten, j)
    return table


def pooled(table, kind, idle):
    """Mean over the three links of the loss, with its error."""
    vals = [table[(kind, idle, j)] for j in LINKS]
    mean = sum(v[0] for v in vals) / len(vals)
    err = math.sqrt(sum(v[1] ** 2 for v in vals)) / len(vals)
    return mean, err


def diff(x, y):
    """(x - y) with errors added in quadrature, for (value, error) pairs."""
    return x[0] - y[0], math.hypot(x[1], y[1])


def metrics(table):
    """f1, f2, e1, e2, the pulse cost and the plain preparation loss."""
    f0, e0 = pooled(table, "F", 0), pooled(table, "E", 0)
    return {
        "F0": f0,
        "f1": diff(pooled(table, "F", 1000), f0),
        "f2": diff(pooled(table, "F", 2000), f0),
        "e1": diff(pooled(table, "E", 1000), e0),
        "e2": diff(pooled(table, "E", 2000), e0),
        "pulse": diff(e0, f0),
    }


def ratio(x, y):
    """x / y for (value, error) pairs; None when y is not positive."""
    if y[0] <= 0:
        return None
    r = x[0] / y[0]
    err = abs(r) * math.hypot(x[1] / x[0] if x[0] else 0.0, y[1] / y[0])
    return r, err


def shape_verdict(r):
    """The pre-registered SHAPE rule; r = f2 / f1 (None if f1 <= 0)."""
    if r is None:
        return "no idle loss to classify"
    if LINEAR[0] <= r <= LINEAR[1]:
        return "LINEAR in t: white (Markovian) dephasing"
    if r >= QUADRATIC:
        return "QUADRATIC in t: a quasi-static frequency difference"
    return "intermediate between linear and quadratic"


def echo_verdict(q):
    """The pre-registered ECHO rule; q = e2 / f2 (None if f2 <= 0)."""
    if q is None:
        return "no free-idle loss to compare with"
    if q < ECHO_LOW:
        return "the echo REMOVES it: a constant frequency difference (refocusable)"
    if q > ECHO_HIGH:
        return "the echo does NOT help: fast noise (not refocusable)"
    return "the echo helps PARTLY"


def pulse_verdict(p):
    """The PULSE rule on (value, error)."""
    if p[0] > SIGMA * p[1]:
        return "the two X gates cost a significant amount"
    return "the two X gates cost nothing significant"


def prep_prediction(f1, shape_r, tau_ns):
    """Idle loss extrapolated to the preparation time tau_ns, per the SHAPE.

    Linear: f1 * tau / 1000. Quadratic: f1 * (tau / 1000) squared. For an
    intermediate shape both are returned (the linear one is the larger).
    """
    lin = f1 * tau_ns / 1000.0
    quad = f1 * (tau_ns / 1000.0) ** 2
    if shape_r is None:
        return None
    if shape_r >= QUADRATIC:
        return quad, quad
    if shape_r <= LINEAR[1]:
        return lin, lin
    return quad, lin


def prep_verdict(share):
    """The PREP rule on a share (predicted idle loss over the preparation loss)."""
    if share >= PREP_HIGH:
        return "idle-type dephasing during the sequence explains MOST of the preparation loss"
    if share >= PREP_LOW:
        return "idle-type dephasing explains PART of the preparation loss"
    return (
        "idle-type dephasing is NOT the main cause: the loss is in the gates themselves"
    )


# ----------------------------------------------------------------------------
# Exact two-qubit model (to test the analysis against known causes)
# ----------------------------------------------------------------------------
def rz_phase(rho, qubit, phi):
    """Rotate `qubit` about Z by phi (a frequency difference acting for a time)."""
    qc = QuantumCircuit(2)
    qc.rz(phi, qubit)
    return rho.evolve(qc)


def model_counts(kind, idle, link, basis, delta=0.0, gamma=0.0, lam_x=0.0):
    """Exact outcome 'counts' of one link circuit with the named causes.

    delta  constant frequency difference of the two qubits (rad per ns);
    gamma  white dephasing rate of each qubit (1/ns), same on both;
    lam_x  depolarizing strength after the two X gates of an echo.
    """
    rho = DensityMatrix.from_label("00")
    prep = QuantumCircuit(2)
    prep.h(0)
    prep.cx(0, 1)
    prep.z(0)
    prep.x(1)
    rho = rho.evolve(prep)

    def idle_for(rho, t):
        """Free evolution for t ns: the constant detuning and the white noise."""
        rho = rz_phase(rho, 1, delta * t)
        f = math.exp(-gamma * t)
        p = (1 - f) / 2
        for q in (0, 1):
            rho = DensityMatrix(
                (1 - p) * rho.data + p * rho.evolve(Pauli("Z"), qargs=[q]).data
            )
        return rho

    if kind == "F":
        rho = idle_for(rho, idle)
    else:
        rho = idle_for(rho, idle / 2)
        xx = QuantumCircuit(2)
        xx.x(0)
        xx.x(1)
        rho = rho.evolve(xx)
        if lam_x > 0:
            rho = depolarize2(rho, lam_x)
        rho = idle_for(rho, idle / 2)
    rot = QuantumCircuit(2)
    rotate(rot, basis, (0, 1))
    probs = rho.evolve(rot).probabilities()
    out = {}
    for i, pr in enumerate(probs):
        if pr > 1e-15:
            # Place the two bits at the link's qubits in a six-bit string.
            bits = ["0"] * N
            bits[N - 1 - 2 * link] = str(i & 1)
            bits[N - 2 - 2 * link] = str((i >> 1) & 1)
            key = "".join(bits)
            out[key] = out.get(key, 0.0) + float(pr) * 1e9
    return out


def model_table(**causes):
    """The (kind, idle, link) table for a device with the named causes."""
    ones = np.ones(N)
    table = {}
    for kind, idle in VARIANTS:
        for j in LINKS:
            by = {b: model_counts(kind, idle, j, b, **causes) for b in BASES}
            table[(kind, idle, j)] = link_loss(by, ones, j)
    return table


# ----------------------------------------------------------------------------
# Checks with no device
# ----------------------------------------------------------------------------
def check_circuits(r):
    """The circuits are right: ideal state unchanged, gate counts, delays."""
    log.info("=== Check 1: the circuits on an exact simulator ===")
    keys, circuits = build_jobs()
    r.append(
        check(
            len(keys) == 56,
            "56 circuits: 2 calibration + 6 variants x 3 links x 3 bases",
        )
    )
    base = Statevector(
        variant_circuit("F", 0, 1, "Z").remove_final_measurements(inplace=False)
    )
    for kind, idle in VARIANTS[1:]:
        sv = Statevector(
            variant_circuit(kind, idle, 1, "Z").remove_final_measurements(inplace=False)
        )
        r.append(
            check(
                abs(abs(base.inner(sv)) - 1) < 1e-9,
                f"{kind}{idle} leaves the singlet unchanged",
            )
        )
    twoq = [sum(c.count_ops().get(g, 0) for g in ("cx", "cz")) for c in circuits]
    r.append(
        check(twoq == [0, 0] + [1] * 54, "one two-qubit gate in every singlet circuit")
    )
    ones = np.ones(N)
    worst = max(
        abs(link_loss({b: model_counts(k, t, 0, b) for b in BASES}, ones, 0)[0])
        for k, t in VARIANTS
    )
    r.append(check(worst < 1e-9, f"ideal loss is 0 in every variant ({worst:.1e})"))
    xs = [
        c.count_ops().get("x", 0)
        for k, c in zip(keys, circuits)
        if isinstance(k, tuple) and k[0] == "E"
    ]
    r.append(
        check(set(xs) == {3}, "echo circuits carry the singlet X plus X on both qubits")
    )


def check_analysis(r):
    """The analysis and the rules recognise each cause when it is alone."""
    log.info("=== Check 2: the analysis on exact models with known causes ===")
    # (a) constant frequency difference: loss grows as t squared; the echo removes it.
    quasi = metrics(model_table(delta=3e-4))
    rq = ratio(quasi["f2"], quasi["f1"])
    qq = ratio(quasi["e2"], quasi["f2"])
    log.info("  quasi-static: r %.2f, q %.3f", rq[0], qq[0])
    r.append(check(rq[0] >= QUADRATIC, "quasi-static: r is quadratic"))
    r.append(
        check(
            shape_verdict(rq[0]).startswith("QUADRATIC"),
            "rule: quasi-static -> QUADRATIC",
        )
    )
    r.append(
        check(
            abs(qq[0]) < 1e-6 and echo_verdict(qq[0]).startswith("the echo REMOVES"),
            "rule: quasi-static -> echo removes",
        )
    )
    # (b) white dephasing: linear in t; the echo does nothing.
    white = metrics(model_table(gamma=2.6e-5))
    rw = ratio(white["f2"], white["f1"])
    qw = ratio(white["e2"], white["f2"])
    log.info("  white: r %.2f, q %.3f", rw[0], qw[0])
    r.append(check(LINEAR[0] <= rw[0] <= LINEAR[1], "white: r is linear"))
    r.append(check(shape_verdict(rw[0]).startswith("LINEAR"), "rule: white -> LINEAR"))
    r.append(
        check(
            qw[0] > ECHO_HIGH and echo_verdict(qw[0]).startswith("the echo does NOT"),
            "rule: white -> echo does not help",
        )
    )
    # (c) a pulse error is found by E0 - F0.
    pulse = metrics(model_table(lam_x=0.02))
    r.append(
        check(
            abs(pulse["pulse"][0] - 3 * 0.02) < 1e-6, "pulse error: E0 - F0 = 3 lam_x"
        )
    )
    r.append(
        check(
            pulse_verdict((0.06, 0.01)).endswith("significant amount"),
            "rule: pulse cost detected",
        )
    )
    r.append(
        check(
            pulse_verdict((0.01, 0.01)).endswith("nothing significant"),
            "rule: no pulse cost",
        )
    )
    # (d) the preparation prediction follows the shape.
    lin = prep_prediction(0.1, 2.0, 500.0)
    quad = prep_prediction(0.1, 4.0, 500.0)
    r.append(
        check(
            abs(lin[0] - 0.05) < 1e-12 and abs(quad[0] - 0.025) < 1e-12,
            "prep prediction: linear 0.05, quadratic 0.025",
        )
    )
    r.append(
        check(
            prep_verdict(0.6).endswith("MOST of the preparation loss"),
            "rule: share 0.6 -> MOST",
        )
    )
    r.append(check("PART" in prep_verdict(0.3), "rule: share 0.3 -> PART"))
    r.append(
        check(
            "NOT the main cause" in prep_verdict(0.05),
            "rule: share 0.05 -> NOT the main cause",
        )
    )
    r.append(
        check(
            shape_verdict(3.0).startswith("intermediate")
            and echo_verdict(0.5).endswith("PARTLY"),
            "rules: intermediate cases",
        )
    )


# ----------------------------------------------------------------------------
# Report on a finished run
# ----------------------------------------------------------------------------
def prep_durations(backend, layout):
    """Scheduled length (ns) of each link's singlet circuit, or None."""
    try:
        isa, _ = transpile_all(backend, [prep_circuit(j) for j in LINKS], layout)
        return [float(c.estimate_duration(backend.target, unit="s")) * 1e9 for c in isa]
    except Exception as exc:  # noqa: BLE001
        log.warn_("could not estimate the preparation duration: %s", exc)
        return None


def report(table, t2s, taus):
    """Print the tables and the five verdicts."""
    log.info("loss per link, summed over X, Y, Z (positive = lost):")
    log.info("  link  " + "  ".join(f"{k}{t:<5d}" + " " * 9 for k, t in VARIANTS))
    for j in LINKS:
        log.info(
            "  %d   " % j
            + "  ".join("%+.3f+/-%.3f" % table[(k, t, j)] for k, t in VARIANTS)
        )
    m = metrics(table)
    log.info("mean preparation loss L(F0) = %+.3f +/- %.3f", *m["F0"])
    for name in ("f1", "f2", "e1", "e2"):
        log.info("  %s = %+.3f +/- %.3f", name, *m[name])
    r_ = ratio(m["f2"], m["f1"])
    q_ = ratio(m["e2"], m["f2"])
    log.info(
        "SHAPE r = f2/f1 = %s",
        "n/a" if r_ is None else "%.2f +/- %.2f (2 = linear, 4 = quadratic)" % r_,
    )
    log.info("verdict (SHAPE rule): %s", shape_verdict(None if r_ is None else r_[0]))
    log.info("ECHO q = e2/f2 = %s", "n/a" if q_ is None else "%.2f +/- %.2f" % q_)
    log.info("verdict (ECHO rule): %s", echo_verdict(None if q_ is None else q_[0]))
    if t2s:
        pred = float(
            np.mean([idle_prediction(t2s[2 * j], t2s[2 * j + 1], 1000) for j in LINKS])
        )
        log.info(
            "echoed 1 us idle: e1 %+.3f +/- %.3f, predicted from the reported T2 %+.3f",
            *m["e1"],
            pred,
        )
        within = abs(m["e1"][0] - pred) <= SIGMA * m["e1"][1]
        log.info(
            "verdict: the echoed idle %s the calibration",
            "follows" if within else "does NOT follow",
        )
    log.info("PULSE cost of the two X gates: %+.3f +/- %.3f", *m["pulse"])
    log.info("verdict (PULSE rule): %s", pulse_verdict(m["pulse"]))
    if taus and r_ is not None and m["F0"][0] > SIGMA * m["F0"][1]:
        tau = float(np.mean(taus))
        pr = prep_prediction(m["f1"][0], r_[0], tau)
        share = [x / m["F0"][0] for x in pr] if m["F0"][0] > 0 else [0.0, 0.0]
        log.info(
            "PREP: the singlet is scheduled for %.0f ns; extrapolated idle loss %.3f to %.3f = %.0f%% to %.0f%% of %.3f",
            tau,
            min(pr),
            max(pr),
            100 * min(share),
            100 * max(share),
            m["F0"][0],
        )
        log.info(
            "verdict (PREP rule, on the larger share): %s", prep_verdict(max(share))
        )
    else:
        log.warn_(
            "PREP rule not evaluated (no duration, no idle loss, or no significant preparation loss)"
        )


def analyse(r, keys, counts, backend, used, dry):
    """Readout correction, the tables, and the report."""
    by_key = dict(zip(keys, counts))
    if dry:
        log.info("(dry run: the numbers below describe the simulator, not ibm_fez)")
    atten = attenuation(by_key["cal0"], by_key["cal1"])
    log.info(
        "%s %s: readout attenuation %s",
        backend.name,
        "simulated device" if dry else "REAL HARDWARE",
        np.round(atten, 3),
    )
    r.append(
        check(bool(np.all((atten > 0.8) & (atten <= 1.0))), "attenuation in (0.8, 1]")
    )
    table = tabulate(by_key, atten)
    report(table, t2_list(backend, used), prep_durations(backend, used))
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
        backend = service.backend(meta["backend"])
    except Exception as exc:  # noqa: BLE001
        log.fail("retrieve failed: %s: %s", type(exc).__name__, scrub(exc, secret))
        return 1
    analyse(r, keys, counts, backend, meta["layout"], False)
    return summarise(r)


def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 53: delay scan with and without echo")
    ap.add_argument("--live", action="store_true", help="use real IBM hardware")
    ap.add_argument("--retrieve", metavar="JOB_ID", help="analyse an earlier job")
    ap.add_argument("--backend", help="backend name; default: least busy")
    ap.add_argument(
        "--layout",
        default=",".join(str(q) for q in LAYOUT),
        help="six physical qubits (default: the Lab 44 chain) or 'auto'",
    )
    ap.add_argument(
        "--shots", type=int, default=2000, help=f"shots per circuit, 100..{MAX_SHOTS}"
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
        check(twoq == [0, 0] + [1] * 54, "no gate cancelled and no routing overhead")
    )
    delays = [
        c.count_ops().get("delay", 0)
        for k, c in zip(keys, isa)
        if isinstance(k, tuple) and k[1] > 0
    ]
    r.append(check(all(d >= 1 for d in delays), "every idle circuit keeps its delay"))

    # The echo X gates sit between barriers, so they must survive as extra
    # single-qubit gates (x or sx) compared with the same circuit without them.
    def n1q(c):
        """Number of x and sx gates in a transpiled circuit."""
        ops = c.count_ops()
        return ops.get("x", 0) + ops.get("sx", 0)

    f0 = [
        n1q(c) for k, c in zip(keys, isa) if isinstance(k, tuple) and k[:2] == ("F", 0)
    ]
    e0 = [
        n1q(c) for k, c in zip(keys, isa) if isinstance(k, tuple) and k[:2] == ("E", 0)
    ]
    r.append(
        check(
            all(e > f for e, f in zip(e0, f0)),
            "the X gates of E0 survive transpilation",
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
            "python3 labs/lab53_delay_scan_echo.py --retrieve %s",
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
