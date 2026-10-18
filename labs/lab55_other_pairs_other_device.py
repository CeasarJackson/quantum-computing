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
    python3 labs/lab55_other_pairs_other_device.py                   # dry run
    read -rs IBM_QUANTUM_TOKEN; export IBM_QUANTUM_TOKEN             # hidden
    python3 labs/lab55_other_pairs_other_device.py --live --list --token-env IBM_QUANTUM_TOKEN
    python3 labs/lab55_other_pairs_other_device.py --live --backend ibm_torino \\
        --token-env IBM_QUANTUM_TOKEN
    python3 labs/lab55_other_pairs_other_device.py --retrieve <job id> --token-env IBM_QUANTUM_TOKEN

Log files
---------
logs/lab55/lab55_<YYYYmmdd_HHMMSS>.log (plain text, never contains the key)
logs/lab55/last_job.json (job id, backend, shots, links; no secrets)

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
def alphas(cal0, cal1, nq):
    """Per-logical-qubit readout attenuation a = 1 - P(1|0) - P(0|1)."""
    t0, t1 = float(sum(cal0.values())), float(sum(cal1.values()))
    a = np.ones(nq)
    for q in range(nq):
        pos = nq - 1 - q
        f0 = sum(c for s, c in cal0.items() if s[pos] == "1") / t0
        f1 = sum(c for s, c in cal1.items() if s[pos] == "0") / t1
        a[q] = 1 - f0 - f1
    return a


def pair_loss(counts, alpha_a, alpha_b):
    """(loss, error) of one singlet bond from a two-qubit counts dictionary."""
    total = float(sum(counts.values()))
    vals, wts = [], []
    for s, c in counts.items():
        sign = (-1) ** (int(s[-1]) + int(s[-2]))
        vals.append(sign / (alpha_a * alpha_b))
        wts.append(c / total)
    vals, wts = np.array(vals), np.array(wts)
    mean = float(vals @ wts)
    var = float(((vals - mean) ** 2) @ wts)
    return mean + 1.0, math.sqrt(var / total)


def link_losses(by_key, n_links):
    """[(loss, error)] per link, summed over the three bases."""
    out = []
    for i in range(n_links):
        al = alphas(by_key[("cal0", i)], by_key[("cal1", i)], 2)
        parts = [pair_loss(by_key[(i, b)], al[0], al[1]) for b in BASES]
        out.append((sum(p[0] for p in parts), math.sqrt(sum(p[1] ** 2 for p in parts))))
    return out


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
def report(chosen, target, loss):
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


def analyse(r, backend, chosen, counts, keys, dry):
    """Readout correction and the report."""
    if dry:
        log.info(
            "(dry run: the numbers below describe the simulator, not a real device)"
        )
    by_key = dict(zip(keys, counts))
    loss = link_losses(by_key, len(chosen))
    r.append(
        check(all(math.isfinite(x[0]) for x in loss), "every link gave a finite loss")
    )
    report(chosen, backend.target, loss)
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
        meta = json.loads(JOB_FILE.read_text())
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
    JOB_FILE.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "backend": backend.name,
                "shots": args.shots,
                "links": [[a, b, e] for (a, b), e in chosen],
            }
        )
    )
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
