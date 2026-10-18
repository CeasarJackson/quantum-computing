#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab55_other_pairs_other_device.py

Purpose
-------
Labs 49 to 54 narrowed the ibm_fez depth-0 loss to the singlet-preparation gates
themselves: random (Lab 54: 96% incoherent), Z-weighted, local to each link, and
about 10 times what the reported gate errors imply (0.04 against 0.003 of
singlet error per link). Every one of those experiments used the same six
qubits on the same device. This lab asks whether the gap is a property of those
qubits, of ibm_fez, or of the way a singlet is prepared on a Heron chip. In ONE
job it prepares a singlet, one pair at a time, on EIGHT disjoint links of the
chosen backend, spread across the reported CZ-error ranking, and measures the
XX, YY and ZZ correlators of each. Run it on ibm_fez and again on another
backend (`--live --list` shows what your account can use).

CIRCUITS: 8 links x 3 bases = 24, plus for each link its own two readout
calibration circuits (both qubits |0>, both |1>) = 16, so 40 circuits of the same
shape, at 2000 shots (80000 shots, about 30 to 80 s).

LINK CHOICE: from the backend's own calibration, every link whose two qubits
have a readout error under 0.06 and a known T1 and T2 and a reported CZ error
under 0.05 (a link at 1.0 is not working); the eight disjoint links
nearest the 5th, 20th, 35th, 50th, 65th, 80th, 90th and 97th percentile of the
reported CZ error (greedy, so no qubit is used twice). The spread matters: it
lets the lab ask how the measured loss depends on the reported error.

WHAT THE CALIBRATION IMPLIES for one link (the Lab 50 formula): 3 lambda_cz,
with lambda_cz = r / 0.75 for the link's reported CZ error r, plus the
single-qubit part, 3 x 3 x lambda_sx / 2 per qubit (three sx gates per qubit,
lambda_sx = 2 r_sx), summed over the two qubits.

THE RULES, fixed here before any live run (loss = sum over X, Y, Z of
(value + 1) for one singlet, positive = lost, readout corrected as in Labs 44 to
54; implied = the calibration's number above):

  GAP    R = mean loss / mean implied over the eight links; excess = mean (loss
         - implied) with its error.
         excess > 3 sigma and R >= 3 : the gap exists on THIS device too, so it
         is not specific to the six ibm_fez qubits.
         R < 1.5 or excess within 3 sigma : the loss AGREES with the calibration,
         so the ibm_fez gap is local to those qubits or that device.
         otherwise : intermediate.
  TREND  weighted fit  loss = c + k implied  over the eight links.
         c >= 0.04 and c > 3 sigma_c, with k < 2 : a constant OFFSET, an error
         the calibration does not include, independent of link quality.
         k > 2 and (k - 1) > 3 sigma_k, with c < 0.04 : the loss SCALES with the
         reported error, k times too high.
         otherwise : no clear trend.
  LINKS  the eight losses are listed with the calibration's number, so a single
         bad link can be seen; no rule is attached.

ADDED AFTER THE FIRST LIVE RUNS (2026-10-18; NOT pre-registered). The first three
runs (ibm_kingston, ibm_marrakesh, ibm_fez) were analysed with the rules above and
those outputs stand as recorded. They showed per-link scatter far beyond the
quoted errors and, on two devices, links with a loss below zero by 3 sigma or
more, which no state can have. So the analysis was amended:
  1. The error of each link now includes the uncertainty of its own readout
     calibration (2000 shots per calibration circuit; about 0.02 per link, fully
     correlated across the three bases). The first runs quoted shot noise only.
  2. The correlator is corrected for the readout offset beta as well as the
     attenuation alpha (measured mean m = alpha z + beta).
  3. Each link is classed. UNPHYSICAL: loss below zero by more than 3 sigma (the
     readout correction failed on that link). HIGH: loss - implied > 3 sigma and
     loss >= 3 x implied. BELOW: loss under implied by more than 3 sigma but not
     below zero. CONSISTENT: anything else. The counts are printed, and any
     UNPHYSICAL link marks the device's GAP and TREND verdicts UNRELIABLE.
  4. Per link, alpha, beta, the raw and corrected correlators are printed.
  5. --meta-file lets --retrieve analyse an older job whose links are no longer
     in last_job.json; every job now also saves logs/lab55/job_<backend>_<id>.json.

For reference, the six ibm_fez qubits of Labs 51 to 54 gave a mean loss per link
of 0.10 to 0.11 against an implied 0.012 (R about 9).

LIMITATIONS: one device and one job per run; the links come from the calibration
at the time of the run; the preparation here is the same singlet circuit as Labs
51 to 54 (H, CX as H-CZ-H, Z, X), transpiled for each pair; shot-noise error bars
only (about 0.012 per link); readout corrected by the attenuation of each qubit
alone (Lab 49 showed that is good to a few thousandths on ibm_fez).

Modes
-----
  (default)  DRY RUN on a simulated ibm_fez (no account, no network).
  --live     Real hardware through Qiskit Runtime (needs an IBM account).
  --live --list   Print the backends the account can use and stop (no job).
  --retrieve JOB_ID   Fetch and analyse a job submitted earlier (add
                      --meta-file PATH if it is not the last job).

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
    python3 labs/lab55_other_pairs_other_device.py                   # dry run
    read -rs IBM_QUANTUM_TOKEN; export IBM_QUANTUM_TOKEN             # hidden
    python3 labs/lab55_other_pairs_other_device.py --live --list --token-env IBM_QUANTUM_TOKEN
    python3 labs/lab55_other_pairs_other_device.py --live --backend ibm_torino \\
        --token-env IBM_QUANTUM_TOKEN
    python3 labs/lab55_other_pairs_other_device.py --retrieve <job id> --token-env IBM_QUANTUM_TOKEN

Log files
---------
logs/lab55/lab55_<YYYYmmdd_HHMMSS>.log (plain text, never contains the key)
logs/lab55/last_job.json and logs/lab55/job_<backend>_<id>.json (job id, backend,
shots, links; no secrets)

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
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from applog import get_logger

log = get_logger("lab55")

# Everything shared is imported so each piece of logic exists once.
try:
    from lab17_mirror_benchmark import (
        TERMINAL_BAD,
        connect,
        make_sampler,
        scrub,
        wait_for,
    )
    from lab46_hardware_noise_budget import MAX_SHOTS, pick_backend
    from lab50_calibration_snapshot import (
        SX_PER_QUBIT,
        lambda_from_infidelity,
        link_table,
        qubit_row,
    )
    from lab51_herald_and_isolate import BASES, SIGMA, check, rotate, singlet_pair
    from lab52_gate_count_orientation_idle import (
        depolarize2,
        plan_time,
        summarise,
    )
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

JOB_FILE = Path("logs/lab55/last_job.json")
N_LINKS = 8
PERCENTILES = (5, 20, 35, 50, 65, 80, 90, 97)
READOUT_MAX = 0.06
LINK_MAX = 0.05  # a reported CZ error at or above this is a dead link
GAP_R, AGREE_R = 3.0, 1.5
OFFSET_MIN, SCALE_MIN = 0.04, 2.0
BAD_FACTOR = 3.0  # amendment: a link is HIGH at 3 x its implied loss or more
# Reference from the six ibm_fez qubits of Labs 51 to 54.
FEZ_REFERENCE = {"loss": 0.105, "implied": 0.012}


# ----------------------------------------------------------------------------
# Choosing the links from the calibration
# ----------------------------------------------------------------------------
def eligible_links(target):
    """{(a, b): error} for links whose two qubits have usable readout and T1, T2."""
    links = link_table(target)
    ok = {}
    for q in range(target.num_qubits):
        try:
            t1, t2, sx, ro = qubit_row(target, q)
        except Exception:  # noqa: BLE001
            continue
        if (
            ro is not None
            and ro < READOUT_MAX
            and not math.isnan(t1)
            and not math.isnan(t2)
        ):
            ok[q] = True
    return {
        k: v for k, v in links.items() if k[0] in ok and k[1] in ok and v < LINK_MAX
    }


def select_links(links, count=N_LINKS, percentiles=PERCENTILES):
    """Disjoint links nearest the given percentiles of the CZ-error ranking.

    Returns [((a, b), error)] in increasing error. Greedy: for each percentile
    take the nearest-ranked link whose qubits are still unused.
    """
    ranked = sorted(links.items(), key=lambda kv: kv[1])
    n = len(ranked)
    used, chosen = set(), []
    for pct in percentiles[:count]:
        centre = int(round(pct / 100 * (n - 1)))
        for off in range(n):
            for idx in (centre - off, centre + off):
                if 0 <= idx < n:
                    (a, b), err = ranked[idx]
                    if a not in used and b not in used:
                        used.update((a, b))
                        chosen.append(((a, b), err))
                        break
            else:
                continue
            break
    return sorted(chosen, key=lambda x: x[1])


def implied_link_loss(cz_error, sx_errors):
    """The calibration's implied singlet loss for one link, summed over X, Y, Z."""
    cz = 3 * lambda_from_infidelity(cz_error, 2)
    sx = sum(3 * SX_PER_QUBIT * lambda_from_infidelity(e, 1) / 2 for e in sx_errors)
    return cz + sx


# ----------------------------------------------------------------------------
# Circuits
# ----------------------------------------------------------------------------
def pair_circuit(basis):
    """The singlet on logical qubits 0 and 1, basis change on both, measured."""
    qc = QuantumCircuit(2)
    singlet_pair(qc, 0)
    qc.barrier()
    rotate(qc, basis, (0, 1))
    qc.measure_all()
    return qc


def calibration_pair():
    """|00> and |11> on the pair, measured: the link's own readout calibration."""
    zeros = QuantumCircuit(2)
    zeros.measure_all()
    ones = QuantumCircuit(2)
    ones.x([0, 1])
    ones.measure_all()
    return [zeros, ones]


def build_jobs(chosen):
    """Keys and logical circuits: for each link, its calibration then its bases.

    Every circuit has the same shape (two qubits, two bits), which also keeps
    the local simulator used in the dry run from mixing circuits of different
    widths in one batch.
    """
    keys, circuits = [], []
    for i in range(len(chosen)):
        for name, qc in zip(("cal0", "cal1"), calibration_pair()):
            keys.append((name, i))
            circuits.append(qc)
        for b in BASES:
            keys.append((i, b))
            circuits.append(pair_circuit(b))
    return keys, circuits


def link_of(key):
    """The link index a key belongs to."""
    return key[1] if key[0] in ("cal0", "cal1") else key[0]


def transpile_for(backend, chosen, keys, circuits):
    """Transpile each circuit onto the two physical qubits of its link."""
    kwargs = {"backend": backend, "optimization_level": 3, "seed_transpiler": 3}
    out = []
    for key, qc in zip(keys, circuits):
        layout = list(chosen[link_of(key)][0])
        out.append(
            generate_preset_pass_manager(initial_layout=layout, **kwargs).run(qc)
        )
    return out


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def readout_params(cal0, cal1, nq):
    """Per-qubit readout model m = alpha z + beta, from |0..0> and |1..1> counts.

    f0 = P(read 1 | prepared 0), f1 = P(read 0 | prepared 1); alpha = 1 - f0 - f1
    and beta = f1 - f0 (a perfect qubit has alpha 1, beta 0). Returns arrays
    (alpha, beta, sigma_alpha, f0, f1).
    """
    t0, t1 = float(sum(cal0.values())), float(sum(cal1.values()))
    f0, f1 = np.zeros(nq), np.zeros(nq)
    for q in range(nq):
        pos = nq - 1 - q
        f0[q] = sum(c for s, c in cal0.items() if s[pos] == "1") / t0
        f1[q] = sum(c for s, c in cal1.items() if s[pos] == "0") / t1
    sigma = np.sqrt(f0 * (1 - f0) / t0 + f1 * (1 - f1) / t1)
    return 1 - f0 - f1, f1 - f0, sigma, f0, f1


def alphas(cal0, cal1, nq):
    """Per-logical-qubit readout attenuation a = 1 - P(1|0) - P(0|1)."""
    return readout_params(cal0, cal1, nq)[0]


def pair_detail(counts, ro_a, ro_b):
    """One singlet bond in one basis, readout corrected for alpha AND beta.

    ro_a, ro_b = (alpha, beta, sigma_alpha) of logical qubits 0 and 1. Returns a
    dict: raw (the uncorrected <zz>), corr (the corrected <zz>), shot (shot-noise
    error of corr) and cal (the error that the calibration's alpha adds).
    """
    (al_a, be_a, sa_a), (al_b, be_b, sa_b) = ro_a, ro_b
    total = float(sum(counts.values()))
    m_a = m_b = m_ab = 0.0
    prods, wts = [], []
    for s, c in counts.items():
        z_a, z_b = (-1) ** int(s[-1]), (-1) ** int(s[-2])
        w = c / total
        m_a, m_b, m_ab = m_a + w * z_a, m_b + w * z_b, m_ab + w * z_a * z_b
        prods.append(z_a * z_b)
        wts.append(w)
    prods, wts = np.array(prods), np.array(wts)
    var = float(((prods - m_ab) ** 2) @ wts)
    za, zb = (m_a - be_a) / al_a, (m_b - be_b) / al_b
    corr = (m_ab - al_a * be_b * za - be_a * al_b * zb - be_a * be_b) / (al_a * al_b)
    rel = math.hypot(sa_a / al_a, sa_b / al_b)
    return {
        "raw": m_ab,
        "corr": corr,
        "shot": math.sqrt(var / total) / (al_a * al_b),
        "cal": abs(corr) * rel,
    }


def pair_loss(counts, alpha_a, alpha_b):
    """(loss, shot error) of one bond, attenuation only (the original estimator)."""
    d = pair_detail(counts, (alpha_a, 0.0, 0.0), (alpha_b, 0.0, 0.0))
    return d["corr"] + 1.0, d["shot"]


def link_detail(by_key, i):
    """Everything about link i: loss, errors, readout parameters, per-basis parts."""
    ro = readout_params(by_key[("cal0", i)], by_key[("cal1", i)], 2)
    al, be, sa = ro[0], ro[1], ro[2]
    parts = [
        pair_detail(by_key[(i, b)], (al[0], be[0], sa[0]), (al[1], be[1], sa[1]))
        for b in BASES
    ]
    shot = math.sqrt(sum(p["shot"] ** 2 for p in parts))
    cal = sum(p["cal"] for p in parts)  # the same alpha in all three bases
    return {
        "loss": sum(p["corr"] + 1.0 for p in parts),
        "raw_loss": sum(p["raw"] + 1.0 for p in parts),
        "shot": shot,
        "cal": cal,
        "err": math.hypot(shot, cal),
        "parts": parts,
        "ro": ro,
    }


def link_losses(by_key, n_links):
    """[(loss, error)] per link, summed over the three bases."""
    out = []
    for i in range(n_links):
        d = link_detail(by_key, i)
        out.append((d["loss"], d["err"]))
    return out


def classify(loss, err, implied):
    """The amendment's per-link class (see the docstring)."""
    if loss < -SIGMA * err:
        return "UNPHYSICAL"
    if loss - implied > SIGMA * err and loss >= BAD_FACTOR * implied:
        return "HIGH"
    if implied - loss > SIGMA * err:
        return "BELOW"
    return "CONSISTENT"


def reliability_verdict(n_unphysical):
    """Whether the device's GAP and TREND verdicts can be believed."""
    if n_unphysical:
        return (
            f"UNRELIABLE: {n_unphysical} link(s) lost less than nothing by more than "
            f"{SIGMA:.0f} sigma, which no state can; the readout correction fails "
            "there, so this device's GAP and TREND verdicts are not evidence"
        )
    return "no link is below zero: the readout correction is behaving"


def meta_matches(meta, job_id):
    """True if a saved job file describes the job being retrieved."""
    return meta.get("job_id") == job_id


def weighted_line(x, y, sigma):
    """Weighted least squares y = c + k x; returns (c, sigma_c, k, sigma_k)."""
    x, y, w = (
        np.asarray(x, float),
        np.asarray(y, float),
        1.0 / np.asarray(sigma, float) ** 2,
    )
    a = np.array([[w.sum(), (w * x).sum()], [(w * x).sum(), (w * x * x).sum()]])
    b = np.array([(w * y).sum(), (w * x * y).sum()])
    cov = np.linalg.inv(a)
    c, k = cov @ b
    return float(c), float(math.sqrt(cov[0, 0])), float(k), float(math.sqrt(cov[1, 1]))


def gap_stats(loss, implied):
    """(mean loss, mean implied, R, excess, error of excess) over the links."""
    ml = float(np.mean([x[0] for x in loss]))
    mi = float(np.mean(implied))
    err = math.sqrt(sum(x[1] ** 2 for x in loss)) / len(loss)
    return ml, mi, (ml / mi if mi > 0 else float("inf")), ml - mi, err


def gap_verdict(r_ratio, excess, err):
    """The pre-registered GAP rule."""
    if excess > SIGMA * err and r_ratio >= GAP_R:
        return "the gap exists on THIS device too: it is not specific to the six ibm_fez qubits"
    if r_ratio < AGREE_R or excess <= SIGMA * err:
        return "the loss AGREES with the calibration: the ibm_fez gap is local to those qubits or that device"
    return "intermediate: the loss exceeds the calibration by less than a factor 3"


def trend_verdict(c, sc, k, sk):
    """The pre-registered TREND rule."""
    if c >= OFFSET_MIN and c > SIGMA * sc and k < SCALE_MIN:
        return "a constant OFFSET: an error the calibration does not include, independent of link quality"
    if k > SCALE_MIN and (k - 1) > SIGMA * sk and c < OFFSET_MIN:
        return "the loss SCALES with the reported error, but is several times too high"
    return "no clear trend with the reported error"


# ----------------------------------------------------------------------------
# Exact model (to test the analysis against known causes)
# ----------------------------------------------------------------------------
def model_counts(basis, lam, shots, rng):
    """Sampled counts of the singlet after depolarizing lam, in one basis."""
    prep = QuantumCircuit(2)
    singlet_pair(prep, 0)
    rho = depolarize2(DensityMatrix.from_label("00").evolve(prep), lam)
    rot = QuantumCircuit(2)
    rotate(rot, basis, (0, 1))
    probs = np.clip(np.real(np.diag(rho.evolve(rot).data)), 0, None)
    draws = rng.multinomial(shots, probs / probs.sum())
    return {format(i, "02b"): int(n) for i, n in enumerate(draws) if n}


def confusion2(f0, f1):
    """4 x 4 readout confusion (rows read, columns true) for two equal qubits."""
    c = np.array([[1 - f0, f1], [f0, 1 - f1]])
    return np.kron(c, c)


def model_probs(basis, lam):
    """Outcome probabilities of the singlet after depolarizing lam, in one basis."""
    prep = QuantumCircuit(2)
    singlet_pair(prep, 0)
    rho = depolarize2(DensityMatrix.from_label("00").evolve(prep), lam)
    rot = QuantumCircuit(2)
    rotate(rot, basis, (0, 1))
    probs = np.clip(np.real(np.diag(rho.evolve(rot).data)), 0, None)
    return probs / probs.sum()


def sample(probs, shots, rng):
    """Counts dictionary from a probability vector over two bits."""
    draws = rng.multinomial(shots, probs / probs.sum())
    return {format(i, "02b"): int(n) for i, n in enumerate(draws) if n}


def model_by_key_ro(lam, shots, rng, f_true, f_cal, cal_shots=2000):
    """One link whose real readout is f_true = (f0, f1) but whose calibration saw f_cal."""
    by = {}
    mc = confusion2(*f_cal)
    by[("cal0", 0)] = sample(mc @ np.array([1.0, 0, 0, 0]), cal_shots, rng)
    by[("cal1", 0)] = sample(mc @ np.array([0, 0, 0, 1.0]), cal_shots, rng)
    mt = confusion2(*f_true)
    for b in BASES:
        by[(0, b)] = sample(mt @ model_probs(b, lam), shots, rng)
    return by


def model_by_key(lams, shots, rng):
    """by_key (counts per circuit) for links with the given depolarizing strengths."""
    by = {}
    for i, lam in enumerate(lams):
        by[("cal0", i)] = {"00": shots}
        by[("cal1", i)] = {"11": shots}
        for b in BASES:
            by[(i, b)] = model_counts(b, lam, shots, rng)
    return by


# ----------------------------------------------------------------------------
# Checks with no device
# ----------------------------------------------------------------------------
def check_pure(r):
    """Selection, the implied formula, the fit and the rules, with no device."""
    log.info("=== Check 1: the analysis on exact models with known causes ===")
    rng = np.random.default_rng(5)
    # (a) the singlet bond loss is 3 lambda for depolarizing lambda.
    lams = [0.0, 0.01, 0.03, 0.06]
    got = link_losses(model_by_key(lams, 40000, rng), len(lams))
    ok = all(abs(g[0] - 3 * lam) < 4 * g[1] + 1e-9 for g, lam in zip(got, lams))
    r.append(check(ok, f"loss = 3 lambda for lambda in {lams}"))
    # (b) the implied formula: r = 0.002 -> lambda = 0.002667.
    r.append(
        check(
            abs(lambda_from_infidelity(0.002, 2) - 0.002 * 4 / 3) < 1e-12,
            "lambda = r d/(d-1) for two qubits",
        )
    )
    imp = implied_link_loss(0.002, (0.0002, 0.0003))
    want = (
        3 * 0.002 * 4 / 3
        + 3 * SX_PER_QUBIT * 0.0004 / 2
        + 3 * SX_PER_QUBIT * 0.0006 / 2
    )
    r.append(check(abs(imp - want) < 1e-12, f"implied link loss {imp:.5f}"))
    # (c) the line fit recovers a constant offset and a scale.
    x = np.linspace(0.008, 0.06, 8)
    c, sc, k, sk = weighted_line(x, 0.05 + 1.0 * x, [0.01] * 8)
    r.append(
        check(
            abs(c - 0.05) < 1e-9 and abs(k - 1.0) < 1e-9,
            "line fit recovers c = 0.05, k = 1",
        )
    )
    r.append(
        check(
            trend_verdict(c + 0.0, 0.005, k, 0.3).startswith("a constant OFFSET"),
            "rule: offset",
        )
    )
    r.append(
        check(
            trend_verdict(0.0, 0.01, 5.0, 0.3).startswith("the loss SCALES"),
            "rule: scale",
        )
    )
    r.append(
        check(
            trend_verdict(0.0, 0.01, 1.0, 0.3).startswith("no clear trend"),
            "rule: no trend",
        )
    )
    # (d) the gap rule.
    r.append(
        check(
            gap_verdict(9.0, 0.09, 0.01).startswith("the gap exists"), "rule: gap R = 9"
        )
    )
    r.append(
        check(
            gap_verdict(1.2, 0.002, 0.01).startswith("the loss AGREES"),
            "rule: agrees R = 1.2",
        )
    )
    r.append(
        check(
            gap_verdict(2.0, 0.05, 0.01).startswith("intermediate"),
            "rule: intermediate R = 2",
        )
    )
    # (e) the circuits.
    sv = Statevector(pair_circuit("Z").remove_final_measurements(inplace=False))
    s = np.asarray(sv.data)
    r.append(
        check(
            abs(abs(s[1]) - abs(s[2])) < 1e-12
            and abs(s[0]) < 1e-12
            and abs(s[3]) < 1e-12,
            "the pair circuit prepares a singlet",
        )
    )
    keys, circuits = build_jobs([((0, 1), 0.0)] * N_LINKS)
    r.append(
        check(
            len(keys) == 5 * N_LINKS,
            f"{5 * N_LINKS} circuits for {N_LINKS} links",
        )
    )
    check_amendment(r)


def check_amendment(r):
    """The amended estimator, the classes and the meta-file guard, with no device."""
    log.info("=== Check 1b: the amended analysis on exact models ===")
    rng = np.random.default_rng(11)
    # (a) asymmetric readout (f0 != f1): the beta term recovers 3 lambda.
    d = link_detail(
        model_by_key_ro(0.02, 400000, rng, (0.02, 0.07), (0.02, 0.07), 400000), 0
    )
    r.append(
        check(
            abs(d["loss"] - 0.06) < 4 * d["err"],
            f"asymmetric readout: loss {d['loss']:.3f} +/- {d['err']:.3f} vs 0.060",
        )
    )
    # (b) a calibration that saw worse readout than the experiment: loss < 0.
    d = link_detail(
        model_by_key_ro(0.0, 20000, rng, (0.03, 0.03), (0.15, 0.15), 20000), 0
    )
    r.append(
        check(
            classify(d["loss"], d["err"], 0.01) == "UNPHYSICAL",
            f"a pessimistic calibration is caught (loss {d['loss']:+.2f})",
        )
    )
    # (c) the calibration adds error that shot noise alone does not show.
    d = link_detail(model_by_key_ro(0.02, 2000, rng, (0.03, 0.05), (0.03, 0.05)), 0)
    r.append(
        check(
            d["cal"] > 0.005 and d["err"] > 1.2 * d["shot"],
            f"calibration error {d['cal']:.3f} added to shot error {d['shot']:.3f}",
        )
    )
    # (d) the classes.
    # (loss, error, implied) -> class
    cases = [
        ((0.15, 0.02, 0.015), "HIGH"),
        ((0.03, 0.02, 0.015), "CONSISTENT"),
        ((-0.10, 0.02, 0.015), "UNPHYSICAL"),
        ((-0.01, 0.02, 0.15), "BELOW"),
        ((0.05, 0.02, 0.015), "CONSISTENT"),  # 1.8 sigma over implied
        ((0.07, 0.02, 0.05), "CONSISTENT"),  # high but under 3 x implied
    ]
    r.append(
        check(
            all(classify(*a) == want for a, want in cases),
            "classes: HIGH, CONSISTENT, UNPHYSICAL, BELOW",
        )
    )
    r.append(
        check(
            reliability_verdict(1).startswith("UNRELIABLE")
            and reliability_verdict(0).startswith("no link"),
            "reliability verdict follows the count of unphysical links",
        )
    )
    # (e) the meta-file guard.
    r.append(
        check(
            meta_matches({"job_id": "a"}, "a")
            and not meta_matches({"job_id": "a"}, "b"),
            "a saved job file is only used for its own job",
        )
    )


def check_selection(r, target):
    """The link selection on the backend's own calibration."""
    log.info(
        "=== Check 2: choosing %d disjoint links from the calibration ===", N_LINKS
    )
    links = eligible_links(target)
    chosen = select_links(links)
    r.append(
        check(
            len(chosen) == N_LINKS,
            f"{len(chosen)} links chosen from {len(links)} eligible",
        )
    )
    qubits = [q for (a, b), _ in chosen for q in (a, b)]
    r.append(check(len(set(qubits)) == len(qubits), "no qubit is used twice"))
    errs = [e for _, e in chosen]
    r.append(
        check(
            errs == sorted(errs) and errs[-1] > errs[0],
            "the links span a range of reported errors",
        )
    )
    for (a, b), e in chosen:
        log.info("  link %3d-%3d: reported CZ error %.5f", a, b, e)
    return chosen


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def log_details(chosen, details):
    """Per link: readout parameters and the raw and corrected correlators."""
    log.info(
        "readout and correlators per link (alpha, beta per qubit; <zz> per basis):"
    )
    for ((a, b), _), d in zip(chosen, details):
        al, be = d["ro"][0], d["ro"][1]
        raw = " ".join(f"{p['raw']:+.3f}" for p in d["parts"])
        cor = " ".join(f"{p['corr']:+.3f}" for p in d["parts"])
        log.info(
            "  %3d-%3d  alpha %.3f/%.3f  beta %+.3f/%+.3f  raw %s  corrected %s"
            "  (X Y Z)  raw loss %+.3f",
            a,
            b,
            al[0],
            al[1],
            be[0],
            be[1],
            raw,
            cor,
            d["raw_loss"],
        )


def report(chosen, target, loss, details=None):
    """Print the table, the fit and the verdicts."""
    sx_err = {q: target["sx"][(q,)].error for (a, b), _ in chosen for q in (a, b)}
    implied = [implied_link_loss(e, (sx_err[a], sx_err[b])) for (a, b), e in chosen]
    log.info("per link (loss summed over X, Y, Z; positive = lost):")
    log.info("   link        CZ error   implied    measured")
    for ((a, b), e), imp, (ls, er) in zip(chosen, implied, loss):
        log.info(
            "  %3d-%3d    %.5f    %.4f    %+.3f +/- %.3f   (%.1fx)",
            a,
            b,
            e,
            imp,
            ls,
            er,
            ls / imp if imp else float("nan"),
        )
    ml, mi, ratio, ex, err = gap_stats(loss, implied)
    log.info(
        "mean measured loss %.3f, mean implied %.4f: R = %.1f; excess %+.3f +/- %.3f",
        ml,
        mi,
        ratio,
        ex,
        err,
    )
    log.info(
        "reference, the six ibm_fez qubits (Labs 51-54): measured %.3f, implied %.3f (R about %.0f)",
        FEZ_REFERENCE["loss"],
        FEZ_REFERENCE["implied"],
        FEZ_REFERENCE["loss"] / FEZ_REFERENCE["implied"],
    )
    log.info("verdict (GAP rule): %s", gap_verdict(ratio, ex, err))
    c, sc, k, sk = weighted_line(implied, [x[0] for x in loss], [x[1] for x in loss])
    log.info(
        "fit loss = c + k implied: c = %+.3f +/- %.3f, k = %+.2f +/- %.2f", c, sc, k, sk
    )
    log.info("verdict (TREND rule): %s", trend_verdict(c, sc, k, sk))
    # Everything below was added after the first live runs (see the docstring).
    log.info("--- amendment (not pre-registered) ---")
    classes = [classify(ls, er, imp) for (ls, er), imp in zip(loss, implied)]
    for ((a, b), _), (ls, er), imp, cl in zip(chosen, loss, implied, classes):
        log.info(
            "  %3d-%3d  loss %+.3f +/- %.3f  implied %.4f  %s", a, b, ls, er, imp, cl
        )
    log.info(
        "links: %d HIGH, %d CONSISTENT, %d BELOW, %d UNPHYSICAL (of %d)",
        classes.count("HIGH"),
        classes.count("CONSISTENT"),
        classes.count("BELOW"),
        classes.count("UNPHYSICAL"),
        len(classes),
    )
    log.info("reliability: %s", reliability_verdict(classes.count("UNPHYSICAL")))
    if details is not None:
        log_details(chosen, details)


def analyse(r, backend, chosen, counts, keys, dry):
    """Readout correction and the report."""
    if dry:
        log.info(
            "(dry run: the numbers below describe the simulator, not a real device)"
        )
    by_key = dict(zip(keys, counts))
    details = [link_detail(by_key, i) for i in range(len(chosen))]
    loss = [(d["loss"], d["err"]) for d in details]
    r.append(
        check(all(math.isfinite(x[0]) for x in loss), "every link gave a finite loss")
    )
    report(chosen, backend.target, loss, details)
    if dry:
        log.info("(dry run: those verdicts are about the simulator)")


def list_backends(service, secret):
    """Print the operational real backends the account can use."""
    try:
        for b in service.backends(operational=True, simulator=False):
            status = b.status()
            log.info(
                "  %-16s %3d qubits, %s queued jobs",
                b.name,
                b.num_qubits,
                getattr(status, "pending_jobs", "?"),
            )
    except Exception as exc:  # noqa: BLE001
        log.fail(
            "could not list backends: %s: %s", type(exc).__name__, scrub(exc, secret)
        )
        return 1
    return 0


def finish_retrieve(args, service, secret, r):
    """--retrieve: fetch a finished job and analyse it."""
    try:
        meta_path = Path(args.meta_file) if args.meta_file else JOB_FILE
        meta = json.loads(meta_path.read_text())
        if not meta_matches(meta, args.retrieve):
            log.fail(
                "%s describes job %s, not %s; give --meta-file for this job",
                meta_path,
                meta.get("job_id"),
                args.retrieve,
            )
            return 2
        chosen = [((a, b), e) for a, b, e in meta["links"]]
        keys = build_jobs(chosen)[0]
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
    log.info("backend %s", meta["backend"])
    analyse(r, backend, chosen, counts, keys, False)
    return summarise(r)


def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 55: other pairs, other device")
    ap.add_argument("--live", action="store_true", help="use real IBM hardware")
    ap.add_argument(
        "--list", action="store_true", help="with --live: list backends and stop"
    )
    ap.add_argument("--retrieve", metavar="JOB_ID", help="analyse an earlier job")
    ap.add_argument("--backend", help="backend name; default: least busy")
    ap.add_argument(
        "--meta-file",
        metavar="PATH",
        help="with --retrieve: job file with job_id, backend and links "
        "(default logs/lab55/last_job.json)",
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
    if args.list and not args.live:
        ap.error("--list needs --live.")
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_pure(r)

    log.info("=== Check 3: connect and choose a backend ===")
    service, secret = connect(args)
    if isinstance(service, str):
        return 2
    if args.list:
        return list_backends(service, secret)
    if args.retrieve:
        return finish_retrieve(args, service, secret, r)
    backend = pick_backend(args, service, secret)
    if backend is None:
        return 1
    log.info("backend %s: %d qubits", backend.name, backend.num_qubits)
    chosen = check_selection(r, backend.target)

    log.info("=== Check 4: transpile each circuit onto its pair ===")
    keys, circuits = build_jobs(chosen)
    isa = transpile_for(backend, chosen, keys, circuits)
    twoq = [sum(c.count_ops().get(g, 0) for g in ("cz", "ecr", "cx")) for c in isa]
    r.append(
        check(
            twoq == [0 if isinstance(k[0], str) else 1 for k in keys],
            "one two-qubit gate per pair circuit and no routing",
        )
    )

    log.info("=== Check 5: run %d circuits x %d shots ===", len(isa), args.shots)
    plan_time(args.shots, len(isa))
    if args.live:
        log.warn_("LIVE: this uses real QPU time. cap=%d s", args.max_seconds)
    job = make_sampler(args, backend).run(isa)
    job_id = job.job_id()
    # Save the id and the links (never the key) for --retrieve.
    JOB_FILE.parent.mkdir(parents=True, exist_ok=True)
    meta_text = json.dumps(
        {
            "job_id": job_id,
            "backend": backend.name,
            "shots": args.shots,
            "links": [[a, b, e] for (a, b), e in chosen],
        }
    )
    JOB_FILE.write_text(meta_text)
    (JOB_FILE.parent / f"job_{backend.name}_{job_id}.json").write_text(meta_text)
    log.info("job id: %s (saved to %s)", job_id, JOB_FILE)
    state = wait_for(job, args.wait if args.live else 600, poll=5)
    if state == "TIMEOUT":
        log.warn_(
            "still queued after %d s. Fetch it later with: "
            "python3 labs/lab55_other_pairs_other_device.py --retrieve %s",
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
    analyse(r, backend, chosen, counts, keys, not args.live)
    return summarise(r)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
