#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab44_noise_aware_on_qpu.py

Purpose
-------
Lab 21 trained Heisenberg HVA angles against a MODELLED noise level and found
that, at realistic noise, angles trained under noise beat angles trained on a
perfect simulator, at least for deep circuits. Lab 17 then MEASURED the noise
of the real machine (ibm_fez: 0.281 % error per CNOT, equivalent to a
depolarizing strength of 0.0037). Lab 44 joins the two.

  1. It takes the measured ibm_fez mirror-benchmark decay of Lab 17 (recorded
     from the live run of 2026-10-13, qubits 44, 45, 37, 25), converts it to a
     depolarizing strength with the Lab 17 calibration, and trains the angles
     of a 6-qubit chain under THAT noise (Lab 21 model).
  2. It checks the energy estimator on simulators first: basis rotations and
     bit order (ideal simulator), and readout-error correction (simulator with
     a known readout error).
  3. It builds, for each depth, one circuit set for the ideal-trained angles
     and one for the noise-trained angles, measured in the X, Y and Z bases
     (the Heisenberg energy needs all three), plus two readout-calibration
     circuits, all on ONE fixed set of physical qubits.
  4. It runs them (dry run on a simulated ibm_fez by default, the real machine
     with --live) and compares the measured energies of the two angle sets with
     each other and with the model.

Honest limit: the predicted gain from noise-aware angles is small compared with
shot noise at the shot counts allowed here. The lab prints the shots that would
be needed to resolve it, and reports the measured difference in standard
deviations without claiming a win it cannot see.

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
    python3 labs/lab44_noise_aware_on_qpu.py                  # dry run
    read -rs IBM_QUANTUM_API_KEY; export IBM_QUANTUM_API_KEY  # hidden prompt
    python3 labs/lab44_noise_aware_on_qpu.py --live --backend ibm_fez
    python3 labs/lab44_noise_aware_on_qpu.py --retrieve <job id>

Log files
---------
logs/lab44/lab44_<YYYYmmdd_HHMMSS>.log (plain text, never contains the key)
logs/lab44/last_job.json (job id, backend, shots, depth, angles; no secrets)

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
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from applog import get_logger

log = get_logger("lab44")

# The cloud plumbing, the Lab 17 calibration and the Lab 21 model are reused,
# not copied: the security-critical code then exists in exactly one place.
try:
    from qiskit_ibm_runtime.fake_provider import FakeFez

    from lab17_mirror_benchmark import (
        TERMINAL_BAD,
        calibrate,
        connect,
        fit_rate,
        lambda_equiv,
        make_sampler,
        scrub,
        wait_for,
    )
    from lab21_noise_aware_training import (
        N,
        NoisyChain,
        aer_energy,
        hva_circuit,
        ideal_angles,
        train,
    )
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

JOB_FILE = Path("logs/lab44/last_job.json")
MAX_SHOTS = 4000  # cap: a typo must not burn the monthly quota
# Lab 17, live ibm_fez run of 2026-10-13: mirror circuits with 0, 12, 24, 36, 48
# CNOTs, 2000 shots each; recorded probability of returning to |0000>.
FEZ_CNOTS = [0, 12, 24, 36, 48]
FEZ_P0 = [0.977, 0.945, 0.917, 0.887, 0.849]
FEZ_LAMBDA_RECORDED = 0.0037  # the equivalent depolarizing strength Lab 17 reported
BASES = ("Z", "X", "Y")


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


# ----------------------------------------------------------------------------
# Circuits and the energy estimator
# ----------------------------------------------------------------------------
def basis_circuit(params, basis):
    """HVA state followed by a rotation into `basis`, then measurement.

    Measuring sigma_x needs a Hadamard first; sigma_y needs S-dagger then H.
    The Heisenberg energy is the sum of XX, YY and ZZ over the bonds, so all
    three bases are needed.
    """
    # Same state preparation for all three bases; only the rotation differs.
    qc = hva_circuit(N, params)
    for q in range(N):
        if basis == "Y":
            qc.sdg(q)
        if basis in ("X", "Y"):
            qc.h(q)
    qc.measure_all()
    return qc


def calibration_circuits():
    """Prepare |000000> and |111111> and measure: readout-error calibration."""
    zeros = QuantumCircuit(N)
    zeros.measure_all()
    ones = QuantumCircuit(N)
    ones.x(range(N))
    ones.measure_all()
    return [zeros, ones]


def attenuation(counts_zero, counts_one):
    """Per-qubit readout attenuation a_q = 1 - P(1|0) - P(0|1).

    A noisy readout multiplies <sigma_z> of qubit q by a_q, and a two-qubit
    correlator by a_q a_(q+1); dividing by them undoes the error.
    """
    # Flip probabilities are counted separately for each qubit.
    shots0, shots1 = sum(counts_zero.values()), sum(counts_one.values())
    a = np.ones(N)
    for q in range(N):
        pos = N - 1 - q  # Qiskit bit strings put qubit 0 on the right
        flip0 = sum(c for s, c in counts_zero.items() if s[pos] == "1") / shots0
        flip1 = sum(c for s, c in counts_one.items() if s[pos] == "0") / shots1
        a[q] = 1 - flip0 - flip1
    return a


def basis_value(counts, atten):
    """(mean, standard error) of sum over bonds of the correlator in one basis."""
    # One value per distinct bit string, weighted by how often it occurred.
    total = sum(counts.values())
    values = np.empty(len(counts))
    weights = np.empty(len(counts))
    for k, (bits, c) in enumerate(counts.items()):
        # Each bond contributes +/-1 from its two measured bits, rescaled by the
        # readout attenuation of both qubits.
        v = 0.0
        for q in range(N - 1):
            sign = (-1) ** (int(bits[N - 1 - q]) + int(bits[N - 2 - q]))
            v += sign / (atten[q] * atten[q + 1])
        values[k], weights[k] = v, c / total
    mean = float(values @ weights)
    var = float(((values - mean) ** 2) @ weights)
    return mean, math.sqrt(var / total)


def energy_from_counts(counts_by_basis, atten):
    """Heisenberg energy and its standard error from the three bases' counts."""
    mean, var = 0.0, 0.0
    for basis in BASES:
        m, se = basis_value(counts_by_basis[basis], atten)
        mean += m
        # The bases come from separate runs, so their errors add in quadrature.
        var += se**2
    return mean, math.sqrt(var)


def run_ideal(circuits, shots, seed, noise=None):
    """Run circuits on Aer (ideal, or with the given noise model); return counts."""
    from qiskit_aer import AerSimulator  # noqa: PLC0415

    # A fixed simulator seed makes the estimator checks reproducible.
    sim = AerSimulator(noise_model=noise, seed_simulator=seed)
    return [sim.run(c, shots=shots).result().get_counts() for c in circuits]


# ----------------------------------------------------------------------------
# Model side: the measured noise and the trained angles
# ----------------------------------------------------------------------------
def fez_lambda(rng):
    """Depolarizing strength equivalent to the recorded ibm_fez mirror decay."""
    # Decay rate per CNOT of the recorded points (fit through the origin).
    s = fit_rate(FEZ_CNOTS, FEZ_P0)
    # Same calibration as Lab 17: known depolarizing noise -> fitted decay.
    table = calibrate(4, 0.25, 4, rng)
    return s, lambda_equiv(table, s)


def train_sets(chain, lam, depth_max, rng):
    """Ideal-trained and noise-trained angles for depths 1..depth_max."""
    # Many-restart ideal baseline (a weak one would flatter the aware angles).
    ideal = ideal_angles(chain, depth_max, rng)
    sets = {}
    for p in range(1, depth_max + 1):
        _, x_ideal = ideal[p]
        # Warm start from the ideal angles: noise-aware training is a
        # refinement, and Lab 21 showed this finds the better optimum.
        _, x_aware = train(chain, p, lam, x_ideal, 4, rng)
        sets[p] = {"ideal": np.asarray(x_ideal), "aware": np.asarray(x_aware)}
    return sets


def build_jobs(sets, depth_max):
    """Circuits in a fixed order: calibration, then (depth, kind, basis)."""
    # Calibration first, then every (depth, angle set, basis) in a fixed order.
    keys, circuits = ["cal0", "cal1"], calibration_circuits()
    for p in range(1, depth_max + 1):
        for kind in ("ideal", "aware"):
            for basis in BASES:
                keys.append((p, kind, basis))
                circuits.append(basis_circuit(sets[p][kind], basis))
    return keys, circuits


def analyse_counts(keys, counts, depth_max):
    """Energies (mean, se) per (depth, kind) from counts in `keys` order."""
    # Counts arrive in the same order the circuits were built.
    by_key = dict(zip(keys, counts))
    atten = attenuation(by_key["cal0"], by_key["cal1"])
    out = {}
    for p in range(1, depth_max + 1):
        for kind in ("ideal", "aware"):
            bases = {b: by_key[(p, kind, b)] for b in BASES}
            out[(p, kind)] = energy_from_counts(bases, atten)
    return out, atten


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_calibration(r, ctx):
    """Check 1: the recorded ibm_fez decay becomes the Lab 17 depolarizing strength."""
    log.info("=== Check 1: measured ibm_fez noise ===")
    s, lam = fez_lambda(ctx["rng"])
    eff = 1 - math.exp(-s)
    log.info(
        "recorded P0 %s at %s CNOTs -> s = %.5f, %.3f%% per CNOT, lambda = %.4f",
        FEZ_P0,
        FEZ_CNOTS,
        s,
        100 * eff,
        lam,
    )
    r.append(
        check(abs(eff - 0.00281) < 2e-4, "0.28% error per CNOT, as Lab 17 reported")
    )
    r.append(
        check(
            abs(lam - FEZ_LAMBDA_RECORDED) < 0.1 * FEZ_LAMBDA_RECORDED,
            f"equivalent depolarizing strength {lam:.4f} ~ recorded 0.0037",
        )
    )
    ctx["lam"] = lam


def check_training(r, ctx):
    """Check 2: noise-aware angles under the measured noise."""
    log.info("=== Check 2: training under the measured noise ===")
    chain, lam = ctx["chain"], ctx["lam"]
    sets = train_sets(chain, lam, ctx["depth_max"], ctx["rng"])
    ctx["sets"] = sets
    # Tallies: predicted gains, ordering violations, energies below E0.
    gains, bad_order, below = [], 0, 0
    for p, pair in sets.items():
        e_ideal_angles = chain.energy(pair["ideal"], lam)
        e_aware_angles = chain.energy(pair["aware"], lam)
        e_noiseless = chain.energy(pair["ideal"], 0.0)
        gains.append(e_ideal_angles - e_aware_angles)
        bad_order += int(e_aware_angles > e_ideal_angles + 1e-9)
        below += int(min(e_ideal_angles, e_aware_angles, e_noiseless) < chain.e0 - 1e-9)
        log.info(
            "depth %d: E0 %.4f | noiseless %.4f | noisy, ideal angles %.4f"
            " | noisy, aware angles %.4f | gain %.4f",
            p,
            chain.e0,
            e_noiseless,
            e_ideal_angles,
            e_aware_angles,
            gains[-1],
        )
    r.append(check(bad_order == 0, "aware angles are never worse than ideal angles"))
    r.append(check(below == 0, "no energy below the exact ground energy"))
    r.append(check(max(gains) >= 0, f"largest predicted gain {max(gains):.4f}"))
    ctx["gains"] = gains
    # The model against Aer on the transpiled circuit, for one aware set.
    p = min(2, ctx["depth_max"])
    model = chain.energy(sets[p]["aware"], lam)
    aer = aer_energy(chain, sets[p]["aware"], lam)
    log.info("depth %d aware angles: model %.5f, Aer %.5f", p, model, aer)
    r.append(check(abs(model - aer) < 2e-3, "NumPy noise model agrees with Aer"))


def check_estimator(r, ctx):
    """Check 3: the shot-based energy estimator, on simulators."""
    log.info("=== Check 3: energy estimator on simulators ===")
    from qiskit_aer.noise import NoiseModel, ReadoutError  # noqa: PLC0415

    chain = ctx["chain"]
    p = min(2, ctx["depth_max"])
    x = ctx["sets"][p]["aware"]
    # The noiseless energy of these angles is the target of the estimator.
    exact = chain.energy(x, 0.0)
    keys = [("cal", 0), ("cal", 1)] + list(BASES)
    circuits = calibration_circuits() + [basis_circuit(x, b) for b in BASES]

    def estimate(noise):
        """Run the set on Aer and return (energy, se, attenuations)."""
        counts = run_ideal(circuits, 20000, 5, noise)
        by_key = dict(zip(keys, counts))
        atten = attenuation(by_key[("cal", 0)], by_key[("cal", 1)])
        e, se = energy_from_counts({b: by_key[b] for b in BASES}, atten)
        return e, se, atten

    e, se, atten = estimate(None)
    log.info("ideal simulator: E = %.4f +/- %.4f, exact %.4f", e, se, exact)
    r.append(check(abs(e - exact) < 4 * se, "ideal simulator: estimator is unbiased"))
    r.append(check(float(np.abs(atten - 1).max()) == 0, "ideal readout: attenuation 1"))
    # A known symmetric readout error of 4 percent per qubit; the mean
    # attenuation should then be 1 - 0.04 - 0.04 = 0.92.
    noise = NoiseModel()
    noise.add_all_qubit_readout_error(ReadoutError([[0.96, 0.04], [0.04, 0.96]]))
    raw_atten = np.ones(N)
    e_fix, se_fix, atten = estimate(noise)
    # Without the correction the energy shrinks towards zero by about 0.92.
    counts = run_ideal(circuits, 20000, 5, noise)
    by_key = dict(zip(keys, counts))
    e_raw, _ = energy_from_counts({b: by_key[b] for b in BASES}, raw_atten)
    log.info(
        "4%% readout error: raw E = %.4f, corrected %.4f +/- %.4f, exact %.4f",
        e_raw,
        e_fix,
        se_fix,
        exact,
    )
    r.append(
        check(abs(atten.mean() - 0.92) < 0.01, "calibration recovers attenuation 0.92")
    )
    r.append(
        check(abs(e_fix - exact) < 4 * se_fix, "correction restores the exact energy")
    )
    r.append(
        check(abs(e_raw - exact) > 4 * se_fix, "...and the raw estimate was biased")
    )
    # How many shots would resolve the predicted gain? sigma scales as 1/sqrt(N).
    gain = max(ctx["gains"])
    sigma_two = (
        math.sqrt(2) * se * math.sqrt(20000)
    )  # difference of two energies, N = 1
    need = (3 * sigma_two / max(gain, 1e-9)) ** 2
    log.info(
        "predicted gain %.4f: a 3-sigma detection needs about %.0f shots per basis",
        gain,
        need,
    )
    r.append(check(need > 0, "shot budget for a 3-sigma detection computed"))
    ctx["shots_needed"] = need


def pick_backend(args, service, secret):
    """Simulated ibm_fez (dry run) or the named / least-busy real backend."""
    if not args.live:
        log.info("DRY RUN: simulated copy of ibm_fez (noise model only)")
        return FakeFez()
    try:
        if args.backend:
            return service.backend(args.backend)
        return service.least_busy(operational=True, simulator=False, min_num_qubits=N)
    except Exception as exc:  # noqa: BLE001
        log.fail(
            "could not select a backend: %s: %s", type(exc).__name__, scrub(exc, secret)
        )
        return None


def transpile_all(backend, circuits, keys):
    """Transpile every circuit onto the SAME physical qubits."""
    # First pass: let the transpiler choose good physical qubits.
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=3, seed_transpiler=3
    )
    # Place the deepest circuit (last in order) and reuse its layout, so the
    # ideal and aware angle sets see identical qubits and only the angles differ.
    deepest = pm.run(circuits[-1])
    layout = deepest.layout.initial_index_layout(filter_ancillas=True)
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=3, seed_transpiler=3, initial_layout=layout
    )
    return [pm.run(c) for c in circuits], layout


def report(r, ctx, results, atten, backend_name, dry):
    """Compare measured energies with the model and with each other."""
    # Wording only: tells the reader whether this was real hardware.
    chain, lam = ctx["chain"], ctx["lam"]
    where = "simulated device" if dry else "REAL HARDWARE"
    log.info("readout attenuation per qubit: %s", np.round(atten, 3))
    r.append(
        check(
            bool(np.all(atten > 0.8) and np.all(atten <= 1.0)),
            "readout attenuation in (0.8, 1]",
        )
    )
    log.info(
        "%-5s %-6s %10s %8s %10s %10s",
        "depth",
        "angles",
        "E measured",
        "sigma",
        "E model",
        "E noiseless",
    )
    # inside: all energies physical; ratios: measured / modelled shortfall from E0.
    inside, ratios = True, []
    for p in range(1, ctx["depth_max"] + 1):
        for kind in ("ideal", "aware"):
            e, se = results[(p, kind)]
            x = ctx["sets"][p][kind]
            model, free = chain.energy(x, lam), chain.energy(x, 0.0)
            log.info(
                "%-5d %-6s %10.4f %8.4f %10.4f %10.4f", p, kind, e, se, model, free
            )
            # Variational bound and the fully mixed state (energy 0) bound it.
            inside &= chain.e0 - 4 * se < e < 4 * se
            if p == ctx["depth_max"]:
                ratios.append(float((e - chain.e0) / (model - chain.e0)))
    r.append(
        check(inside, f"{where}: every energy lies between E0 and 0 (within 4 sigma)")
    )
    r.append(
        check(
            all(0.25 < x < 4 for x in ratios),
            "deepest circuits: shortfall vs model ratio "
            f"{[round(x, 2) for x in ratios]} within a factor 4",
        )
    )
    # Does the noise-aware set beat the ideal set, and by how many sigma?
    for p in range(1, ctx["depth_max"] + 1):
        (ei, si), (ea, sa) = results[(p, "ideal")], results[(p, "aware")]
        z = (ei - ea) / math.hypot(si, sa)
        pred = chain.energy(ctx["sets"][p]["ideal"], lam) - chain.energy(
            ctx["sets"][p]["aware"], lam
        )
        log.info(
            "depth %d: ideal - aware = %+.4f (%+.1f sigma); the model predicts %+.4f",
            p,
            ei - ea,
            z,
            pred,
        )
    # Say plainly whether the shot budget could resolve the predicted gain.
    enough = ctx["shots"] >= ctx["shots_needed"]
    log.info(
        "backend %s: %d shots per basis; a 3-sigma detection of the predicted "
        "gain needs about %.0f, so the comparison %s.",
        backend_name,
        ctx["shots"],
        ctx["shots_needed"],
        "is resolvable" if enough else "is limited by shot noise",
    )


def summarise(r):
    """Log the tally and return 0/1."""
    bad = r.count(False)
    if bad:
        log.fail("%d of %d checks failed", bad, len(r))
        return 1
    log.ok("All %d checks passed", len(r))
    return 0


def finish_retrieve(args, service, secret, r, ctx):
    """--retrieve: fetch a finished job and analyse it with the saved angles."""
    try:
        meta = json.loads(JOB_FILE.read_text())
        ctx["depth_max"], ctx["shots"] = meta["depth_max"], meta["shots"]
        ctx["sets"] = {
            int(p): {k: np.asarray(v) for k, v in pair.items()}
            for p, pair in meta["angles"].items()
        }
        keys = build_jobs(ctx["sets"], ctx["depth_max"])[0]
        job = service.job(args.retrieve)
        state = wait_for(job, args.wait, poll=10)
        if state != "DONE":
            log.warn_("job %s is %s; try again later", args.retrieve, state)
            return 0 if all(r) else 1
        res = job.result()
        counts = [res[i].data.meas.get_counts() for i in range(len(keys))]
        name = job.backend().name if hasattr(job, "backend") else "device"
    except Exception as exc:  # noqa: BLE001
        log.fail("retrieve failed: %s: %s", type(exc).__name__, scrub(exc, secret))
        return 1
    results, atten = analyse_counts(keys, counts, ctx["depth_max"])
    report(r, ctx, results, atten, name, False)
    return summarise(r)


def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 44: noise-aware angles on a QPU")
    ap.add_argument("--live", action="store_true", help="use real IBM hardware")
    ap.add_argument("--retrieve", metavar="JOB_ID", help="analyse an earlier job")
    ap.add_argument("--backend", help="backend name; default: least busy")
    ap.add_argument("--depth-max", type=int, default=3, help="HVA depths 1..this (1-3)")
    ap.add_argument(
        "--shots", type=int, default=2000, help=f"shots per circuit, 100..{MAX_SHOTS}"
    )
    ap.add_argument("--max-seconds", type=int, default=120, help="QPU time cap")
    ap.add_argument("--wait", type=int, default=900, help="queue wait (0 = submit)")
    ap.add_argument(
        "--token-env",
        default="IBM_QUANTUM_API_KEY",
        help="NAME of the environment variable holding the key",
    )
    ap.add_argument("--sampler", choices=("v2", "client"), default="v2")
    ap.add_argument("--seed", type=int, default=7, help="training seed")
    args = ap.parse_args(argv)
    if not 1 <= args.depth_max <= 3:
        ap.error(f"--depth-max must be 1..3. Got {args.depth_max}.")
    if not 100 <= args.shots <= MAX_SHOTS:
        ap.error(f"--shots must be 100..{MAX_SHOTS}. Got {args.shots}.")
    if args.max_seconds < 10:
        ap.error("--max-seconds must be >= 10.")
    if args.live and args.retrieve:
        ap.error("use --live OR --retrieve, not both.")
    ctx = {
        "rng": np.random.default_rng(args.seed),
        "chain": NoisyChain(N),
        "depth_max": args.depth_max,
        "shots": args.shots,
    }
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_calibration(r, ctx)
    check_training(r, ctx)
    check_estimator(r, ctx)

    log.info("=== Check 4: connect and choose a backend ===")
    service, secret = connect(args)
    if isinstance(service, str):
        return 2
    if args.retrieve:
        return finish_retrieve(args, service, secret, r, ctx)
    backend = pick_backend(args, service, secret)
    if backend is None:
        return 1
    log.info("backend %s: %d qubits", backend.name, backend.num_qubits)

    log.info("=== Check 5: transpile onto one set of physical qubits ===")
    keys, circuits = build_jobs(ctx["sets"], ctx["depth_max"])
    isa, layout = transpile_all(backend, circuits, keys)
    log.info("physical qubits used (same for every circuit): %s", layout)
    same = all(
        c.layout.initial_index_layout(filter_ancillas=True) == layout for c in isa
    )
    r.append(check(same, "all circuits run on identical physical qubits"))
    allowed = set(backend.operation_names) | {"measure", "barrier", "delay"}
    r.append(
        check(
            set().union(*(c.count_ops() for c in isa)) <= allowed,
            "only native gates after transpilation",
        )
    )
    twoq = {
        k: sum(c.count_ops().get(g, 0) for g in ("cz", "ecr", "cx"))
        for k, c in zip(keys, isa)
        if isinstance(k, tuple) and k[2] == "Z"
    }
    log.info("two-qubit gates per (depth, angles, basis): %s", twoq)

    log.info("=== Check 6: run %d circuits x %d shots ===", len(isa), args.shots)
    if args.live:
        log.warn_("LIVE: this uses real QPU time. cap=%d s", args.max_seconds)
    job = make_sampler(args, backend).run(isa)
    job_id = job.job_id()
    # Save the id, the run parameters and the angles (never the key).
    JOB_FILE.parent.mkdir(parents=True, exist_ok=True)
    JOB_FILE.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "backend": backend.name,
                "shots": args.shots,
                "depth_max": args.depth_max,
                "angles": {
                    str(p): {k: [float(v) for v in x] for k, x in pair.items()}
                    for p, pair in ctx["sets"].items()
                },
            }
        )
    )
    log.info("job id: %s (saved to %s)", job_id, JOB_FILE)
    state = wait_for(job, args.wait if args.live else 600, poll=5)
    if state == "TIMEOUT":
        log.warn_(
            "still queued after %d s. Fetch it later with: "
            "python3 labs/lab44_noise_aware_on_qpu.py --retrieve %s",
            args.wait,
            job_id,
        )
        return 0 if all(r) else 1
    if state in TERMINAL_BAD:
        log.fail("job ended with status %s", state)
        return 1
    result = job.result()

    log.info("=== Check 7: results ===")
    counts = [result[i].data.meas.get_counts() for i in range(len(isa))]
    r.append(
        check(
            all(sum(c.values()) == args.shots for c in counts),
            f"every circuit returned {args.shots} shots",
        )
    )
    results, atten = analyse_counts(keys, counts, args.depth_max)
    report(r, ctx, results, atten, backend.name, not args.live)
    return summarise(r)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
