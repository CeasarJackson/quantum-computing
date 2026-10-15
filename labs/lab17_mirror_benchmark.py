#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab17_mirror_benchmark.py

Purpose
-------
Lab 17 measures how much error a quantum computer adds per CNOT, using a
MIRROR circuit (a Loschmidt echo): run k Trotter steps of the Lab 14/15 Ising
evolution, then run the exact INVERSE circuit, and measure the probability P0
of returning to |00..0>. Ideal hardware gives P0 = 1 for every k, so ANY
shortfall is hardware error, with no Trotter error and no need to know the
physics. This is the stricter test promised at the end of Lab 15.

Because P0 decays roughly as exp(-s * N_cx), a fit of ln(P0 / P0(k=0)) against
the CNOT count N_cx gives a decay rate s and an "effective error per CNOT"
1 - exp(-s). To express it in the units of Lab 14, the lab first calibrates the
same fit on a simulator with a KNOWN depolarizing error, then reads off the
depolarizing strength that would reproduce the device's decay.

Modes
-----
  (default)  DRY RUN on a fake IBM backend with its noise model (no account).
  --live     Real hardware through Qiskit Runtime (needs an IBM account).
  --retrieve JOB_ID   Fetch and analyse a job submitted earlier.

Security
--------
The API key is read ONLY from an environment variable (default name
IBM_QUANTUM_API_KEY; change the NAME with --token-env). There is no --token
flag because command-line arguments leak into shell history. The key is never
logged, never written to a file, and is scrubbed from error text. The optional
instance CRN comes from IBM_QUANTUM_INSTANCE. Nothing secret touches disk.

Usage
-----
    python3 labs/lab17_mirror_benchmark.py                  # dry run
    read -rs IBM_QUANTUM_API_KEY; export IBM_QUANTUM_API_KEY  # hidden prompt
    python3 labs/lab17_mirror_benchmark.py --live --backend ibm_fez
    python3 labs/lab17_mirror_benchmark.py --retrieve <job id>

Log files
---------
logs/lab17/lab17_<YYYYmmdd_HHMMSS>.log (plain text, never contains the key)
logs/lab17/last_job.json (job id, backend, shots, n, dt, steps; no secrets)

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
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from applog import get_logger

log = get_logger("lab17")

# The runtime package is needed in BOTH modes (the dry run borrows its fake
# backends), so fail early with the fix instead of a traceback.
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

J = 1.0  # Ising coupling
H_FIELD = 1.0  # transverse field
JOB_FILE = Path("logs/lab17/last_job.json")
MAX_SHOTS = 4000  # cap: a typo must not burn the monthly quota
TERMINAL_OK = {"DONE"}
TERMINAL_BAD = {"ERROR", "CANCELLED", "FAILED"}
BASIS = ["cx", "rz", "sx", "x"]


def scrub(text, secret):
    """Remove the API key from any text before it is logged."""
    # HTTP libraries sometimes echo request headers; replacing the exact
    # secret guarantees it can never reach a log file.
    return str(text).replace(secret, "***") if secret else str(text)


# ----------------------------------------------------------------------------
# Circuits
# ----------------------------------------------------------------------------
def forward_circuit(n, dt, k):
    """k second-order Trotter steps of the Ising chain (no measurement)."""
    qc = QuantumCircuit(n)
    for _ in range(k):
        for q in range(n):
            qc.rx(-H_FIELD * dt, q)  # half step of the X part
        for q in range(n - 1):
            qc.rzz(-2 * J * dt, q, q + 1)  # full step of the ZZ part
        for q in range(n):
            qc.rx(-H_FIELD * dt, q)  # second half step
    return qc


def mirror_circuit(n, dt, k):
    """U_k, a barrier, U_k^dagger, then measure. Ideal outcome: all zeros.

    The barrier stops the transpiler from noticing that U and its inverse
    cancel; without it the whole circuit could collapse to the identity and
    the benchmark would measure nothing (Lab 10 showed the optimiser is
    inconsistent about such cancellations, so we remove the doubt).
    """
    # Build U once, then reuse it and its exact inverse so the two halves
    # are guaranteed to be mirror images.
    fwd = forward_circuit(n, dt, k)
    qc = QuantumCircuit(n)
    qc.compose(fwd, inplace=True)
    qc.barrier()
    qc.compose(fwd.inverse(), inplace=True)
    qc.measure_all()
    return qc


def expected_cx(n, k):
    """Two CNOTs per RZZ, (n-1) RZZ per step, forward plus inverse."""
    return 2 * (n - 1) * k * 2


def p_zero(counts, n):
    """Probability of the all-zeros outcome from a counts dictionary."""
    # Qiskit keys are bit strings, so all-zeros is simply n '0' characters.
    total = sum(counts.values())
    return counts.get("0" * n, 0) / total


# ----------------------------------------------------------------------------
# Cloud plumbing (same pattern as Lab 11)
# ----------------------------------------------------------------------------
def connect(args):
    """Return (service, secret). Dry run needs neither."""
    if not args.live and not args.retrieve:
        return None, ""
    secret = os.environ.get(args.token_env, "").strip()
    if not secret:
        log.fail(
            "environment variable %s is empty or unset. Create an API key "
            "on the IBM Quantum Platform, then run: "
            "read -rs %s; export %s",
            args.token_env,
            args.token_env,
            args.token_env,
        )
        return "missing", ""
    instance = os.environ.get("IBM_QUANTUM_INSTANCE", "").strip() or None
    try:
        # Never print or store `secret`; the service object keeps it in memory.
        svc = QiskitRuntimeService(token=secret, instance=instance)
    except Exception as exc:  # noqa: BLE001
        log.fail("login failed: %s: %s", type(exc).__name__, scrub(exc, secret))
        return "failed", secret
    log.ok("connected to IBM Quantum Platform (credentials not logged)")
    return svc, secret


def pick_backend(args, service, secret, n):
    """Fake backend (dry run) or a real one (named, else least busy)."""
    if not args.live:
        log.info("DRY RUN: simulated copy of ibm_sherbrooke (noise model only)")
        return FakeSherbrooke()
    try:
        if args.backend:
            return service.backend(args.backend)
        return service.least_busy(operational=True, simulator=False, min_num_qubits=n)
    except Exception as exc:  # noqa: BLE001
        log.fail(
            "could not select a backend: %s: %s", type(exc).__name__, scrub(exc, secret)
        )
        return None


def make_sampler(args, backend):
    """Sampler bound to the backend, shots and a hard execution-time cap."""
    cls = SamplerV2
    if args.sampler == "client" and args.live:
        from qiskit_ibm_runtime.executor_sampler import Sampler  # noqa: PLC0415

        cls = Sampler
    with warnings.catch_warnings():
        # SamplerV2 is deprecated in 0.50 but still supported.
        warnings.simplefilter("ignore", DeprecationWarning)
        sampler = cls(mode=backend)
    sampler.options.default_shots = args.shots
    if args.live:
        # Hard stop so a stuck job cannot consume the plan's QPU minutes.
        sampler.options.max_execution_time = args.max_seconds
    return sampler


def wait_for(job, timeout, poll=10):
    """Poll a job until it finishes or `timeout` seconds pass."""
    start, last = time.time(), None
    while True:
        status = str(getattr(job.status(), "name", job.status())).upper()
        if status != last:
            log.info("job status: %s (%.0f s)", status, time.time() - start)
            last = status
        if status in TERMINAL_OK or status in TERMINAL_BAD:
            return status
        if time.time() - start > timeout:
            return "TIMEOUT"
        time.sleep(poll)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def fit_rate(ncx, p0):
    """Decay rate s from ln(P0(k)/P0(0)) = -s * N_cx (fit through the origin).

    Using the ratio to the k=0 point removes readout error, which does not
    grow with circuit depth. Points with zero counts are dropped.
    """
    # Least squares through the origin: slope = sum(x*y) / sum(x*x).
    x = np.asarray(ncx[1:], dtype=float)
    y = -np.log(np.maximum(np.asarray(p0[1:]), 1e-9) / max(p0[0], 1e-9))
    return float(np.sum(x * y) / np.sum(x * x))


def fit_with_error(ncx, counts, n, rng, boot=300):
    """Return (s, sigma_s) with a binomial bootstrap over the shot noise."""
    # Point estimate first, then the spread of re-fits to resampled data.
    shots = sum(counts[0].values())
    p0 = [p_zero(c, n) for c in counts]
    s = fit_rate(ncx, p0)
    draws = []
    for _ in range(boot):
        # Each P0 is a binomial proportion; redraw it with its own shot noise.
        resampled = [rng.binomial(shots, p) / shots for p in p0]
        draws.append(fit_rate(ncx, resampled))
    return s, float(np.std(draws))


def calibrate(n, dt, steps, rng):
    """Fit the same decay on a simulator with KNOWN depolarizing errors.

    Returns {lambda: s}. The device's s is later mapped to the lambda that
    would give the same decay, by linear interpolation.
    """
    from qiskit import transpile  # noqa: PLC0415
    from qiskit_aer import AerSimulator  # noqa: PLC0415
    from qiskit_aer.noise import NoiseModel, depolarizing_error  # noqa: PLC0415

    # lambda = 0 is included so the table also proves the zero-noise baseline.
    table = {}
    ncx = [expected_cx(n, k) for k in range(steps + 1)]
    for lam in (0.0, 0.005, 0.01, 0.02):
        # Only the CNOTs are noisy, exactly as in Lab 14's noise study.
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(lam, 2), ["cx"])
        sim = AerSimulator(noise_model=nm, seed_simulator=11)
        # Transpile to the same gate set the hardware uses, then run.
        circs = [
            transpile(
                mirror_circuit(n, dt, k),
                basis_gates=BASIS,
                optimization_level=1,
                seed_transpiler=3,
            )
            for k in range(steps + 1)
        ]
        res = [sim.run(c, shots=20000).result().get_counts() for c in circs]
        table[lam] = fit_with_error(ncx, res, n, rng, boot=20)[0]
    return table


def lambda_equiv(table, s):
    """Depolarizing strength whose calibrated decay rate equals s."""
    # np.interp needs increasing x values: the rates grow with lambda.
    lams = sorted(table)
    rates = [table[x] for x in lams]
    return float(np.interp(s, rates, lams))


def self_test(n, dt, steps, rng):
    """Check the benchmark on simulators before trusting hardware."""
    from qiskit_aer import AerSimulator  # noqa: PLC0415

    ideal = AerSimulator(seed_simulator=1)
    out = []
    # An ideal machine must return to |0..0> every time, for every k.
    worst = min(
        p_zero(ideal.run(mirror_circuit(n, dt, k), shots=2000).result().get_counts(), n)
        for k in range(steps + 1)
    )
    out.append(
        check(
            worst == 1.0,
            "ideal simulator: mirror returns to |0..0> "
            f"with probability {worst:.4f} for every k",
        )
    )
    table = calibrate(n, dt, steps, rng)
    log.info(
        "calibration (known depolarizing lambda -> fitted decay s per " "CNOT): %s",
        {k: round(v, 5) for k, v in table.items()},
    )
    rates = [table[k] for k in sorted(table)]
    out.append(
        check(
            abs(rates[0]) < 5e-4 and all(b > a for a, b in zip(rates, rates[1:])),
            "no noise gives s ~ 0 and s grows with the true error",
        )
    )
    # A pure global depolarizing channel would lose lambda per CNOT; one CNOT
    # only touches 2 of the n qubits, so the observed ratio is below 1.
    ratio = [table[k] / k for k in sorted(table) if k > 0]
    log.info(
        "s / lambda = %s (a global-depolarizing estimate would be 1)",
        [round(x, 3) for x in ratio],
    )
    out.append(
        check(
            0.3 < min(ratio) and max(ratio) < 1.2 and max(ratio) / min(ratio) < 1.4,
            "s is proportional to lambda (ratio within 0.3..1.2, "
            "spread < 1.4x): the fit is a usable error meter",
        )
    )
    return out, table


def analyse(counts, n, dt, steps, shots, backend_name, dry, table, rng):
    """Compare device counts with the ideal (P0=1); return check outcomes."""
    # CNOT count, return probability and shot-noise error for every k.
    ncx = [expected_cx(n, k) for k in range(steps + 1)]
    p0 = [p_zero(c, n) for c in counts]
    floor = 1 / 2**n
    sig = [math.sqrt(max(p * (1 - p), 1e-6) / shots) for p in p0]
    s, ds = fit_with_error(ncx, counts, n, rng)
    log.info("%-3s %5s %8s %8s %8s", "k", "CX", "P0", "sigma", "model")
    for k in range(steps + 1):
        model = p0[0] * math.exp(-s * ncx[k])
        log.info("%-3d %5d %8.4f %8.4f %8.4f", k, ncx[k], p0[k], sig[k], model)
    # Fraction of the signal lost per CNOT, and the matching Lab 14 lambda.
    eff = 1 - math.exp(-s)
    lam = lambda_equiv(table, s)
    log.info(
        "decay s = %.5f +/- %.5f per CNOT => effective error %.3f%% per " "CNOT",
        s,
        ds,
        100 * eff,
    )
    log.info("equivalent depolarizing lambda = %.4f (Lab 14 assumed 0.0100)", lam)
    # Wording only: tells the reader whether this was real hardware.
    where = "simulated device" if dry else "REAL HARDWARE"
    out = [
        check(
            p0[0] > 0.95, f"k=0 (readout only): P0 = {p0[0]:.4f} " f"on {backend_name}"
        ),
        check(
            all(
                b <= a + 3 * math.hypot(sa, sb)
                for a, b, sa, sb in zip(p0, p0[1:], sig, sig[1:])
            ),
            "P0 falls (never rises beyond shot noise) as circuits " "get deeper",
        ),
        check(
            p0[-1] > floor + 3 * sig[-1],
            f"{where}: even the deepest "
            f"circuit ({ncx[-1]} CNOTs) beats the random floor "
            f"{floor:.4f}: P0 = {p0[-1]:.4f}",
        ),
        check(
            s > 3 * ds and 0 < eff < 0.1,
            f"decay is measurable "
            f"({s / max(ds, 1e-12):.1f} sigma) and plausible "
            f"({100 * eff:.2f}% per CNOT)",
        ),
    ]
    # Largest deviation from the fitted curve, in units of that point's sigma.
    resid = max(
        abs(p - p0[0] * math.exp(-s * x)) / max(sg, 1e-3)
        for p, x, sg in zip(p0, ncx, sig)
    )
    out.append(
        check(
            resid < 4.5,
            f"single-exponential model fits (largest " f"residual {resid:.1f} sigma)",
        )
    )
    return out


def summarise(r):
    """Log the tally and return 0/1."""
    # Any False in the list means at least one check failed.
    bad = r.count(False)
    if bad:
        log.fail("%d of %d checks failed", bad, len(r))
        return 1
    log.ok("All %d checks passed", len(r))
    return 0


def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 17: mirror-circuit benchmark")
    ap.add_argument(
        "--live", action="store_true", help="use real IBM hardware (default: dry run)"
    )
    ap.add_argument(
        "--retrieve", metavar="JOB_ID", help="analyse a job submitted earlier"
    )
    ap.add_argument("--backend", help="backend name; default: least busy")
    ap.add_argument("--qubits", type=int, default=4, help="chain length 3..8")
    ap.add_argument("--dt", type=float, default=0.25, help="Trotter step size")
    ap.add_argument("--steps", type=int, default=4, help="max Trotter steps 2..6")
    ap.add_argument(
        "--shots", type=int, default=2000, help=f"shots per circuit, 100..{MAX_SHOTS}"
    )
    ap.add_argument(
        "--max-seconds",
        type=int,
        default=60,
        help="hard cap on QPU execution time for the job",
    )
    ap.add_argument(
        "--wait",
        type=int,
        default=900,
        help="seconds to wait for the queue (0 = submit and exit)",
    )
    ap.add_argument(
        "--token-env",
        default="IBM_QUANTUM_API_KEY",
        help="NAME of the environment variable holding the key",
    )
    ap.add_argument(
        "--sampler",
        choices=("v2", "client"),
        default="v2",
        help="v2 = SamplerV2 (default); client = new 0.50 sampler",
    )
    ap.add_argument("--seed", type=int, default=5, help="bootstrap seed")
    args = ap.parse_args(argv)
    if not 3 <= args.qubits <= 8:
        ap.error(f"--qubits must be 3..8. Got {args.qubits}.")
    if not 0.05 <= args.dt <= 0.5:
        ap.error(f"--dt must be 0.05..0.5. Got {args.dt}.")
    if not 2 <= args.steps <= 6:
        ap.error(f"--steps must be 2..6. Got {args.steps}.")
    if not 100 <= args.shots <= MAX_SHOTS:
        ap.error(f"--shots must be 100..{MAX_SHOTS}. Got {args.shots}.")
    if args.max_seconds < 10:
        ap.error("--max-seconds must be >= 10.")
    if args.live and args.retrieve:
        ap.error("use --live OR --retrieve, not both.")
    # Short aliases, and one seeded generator so bootstraps are reproducible.
    n, dt, steps = args.qubits, args.dt, args.steps
    rng = np.random.default_rng(args.seed)

    log.info("=== Check 1: benchmark self-test (no network) ===")
    r, table = self_test(n, dt, steps, rng)

    log.info("=== Check 2: connect and choose a backend ===")
    service, secret = connect(args)
    if isinstance(service, str):
        return 2
    if args.retrieve:
        return finish_retrieve(args, service, secret, r, table, rng)
    backend = pick_backend(args, service, secret, n)
    if backend is None:
        return 1
    log.info("backend %s: %d qubits", backend.name, backend.num_qubits)

    log.info("=== Check 3: transpile (barrier keeps the mirror intact) ===")
    # Level 3 places the chain on good physical qubits. Transpiling each
    # circuit on its own would let the layout CHANGE with k (a real run did
    # exactly that and produced a non-exponential curve), mixing a change in
    # qubit quality with the effect of depth. So: place the deepest circuit
    # first, then force every circuit onto the SAME physical qubits.
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=3, seed_transpiler=3
    )
    deepest = pm.run(mirror_circuit(n, dt, steps))
    layout = deepest.layout.initial_index_layout(filter_ancillas=True)
    log.info("physical qubits used (same for every circuit): %s", layout)
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=3, seed_transpiler=3, initial_layout=layout
    )
    isa = [pm.run(mirror_circuit(n, dt, k)) for k in range(steps + 1)]
    same = all(
        c.layout.initial_index_layout(filter_ancillas=True) == layout for c in isa
    )
    r.append(
        check(
            same,
            "all circuits run on identical physical qubits, "
            "so only depth changes between them",
        )
    )
    twoq = [sum(c.count_ops().get(g, 0) for g in ("cz", "ecr", "cx")) for c in isa]
    want = [expected_cx(n, k) for k in range(steps + 1)]
    log.info("two-qubit gates (k=0..%d): %s, expected %s", steps, twoq, want)
    allowed = set(backend.operation_names) | {"measure", "barrier", "delay"}
    r.append(
        check(
            set().union(*(c.count_ops() for c in isa)) <= allowed,
            "only native gates after transpilation",
        )
    )
    # Fewer gates than expected would mean the mirror was partly optimised
    # away; more would mean SWAPs were needed to fit the chain on the chip.
    r.append(
        check(
            twoq == want,
            "CNOT counts equal 4(n-1)k: nothing " "cancelled, no routing overhead",
        )
    )

    log.info("=== Check 4: run %d circuits x %d shots ===", len(isa), args.shots)
    if args.live:
        log.warn_("LIVE: this uses real QPU time (seconds). cap=%d s", args.max_seconds)
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
                "n": n,
                "dt": dt,
                "steps": steps,
            }
        )
    )
    log.info("job id: %s (saved to %s)", job_id, JOB_FILE)
    state = wait_for(job, args.wait if args.live else 600, poll=5)
    if state == "TIMEOUT":
        log.warn_(
            "still queued after %d s. Fetch it later with: "
            "python3 labs/lab17_mirror_benchmark.py --retrieve %s",
            args.wait,
            job_id,
        )
        return 0 if all(r) else 1
    if state in TERMINAL_BAD:
        log.fail("job ended with status %s", state)
        return 1
    # The job is already DONE here, so this returns immediately.
    result = job.result()

    log.info("=== Check 5: results ===")
    counts = [result[i].data.meas.get_counts() for i in range(len(isa))]
    r.append(
        check(
            all(sum(c.values()) == args.shots for c in counts),
            f"every circuit returned {args.shots} shots",
        )
    )
    r += analyse(
        counts, n, dt, steps, args.shots, backend.name, not args.live, table, rng
    )
    return summarise(r)


def finish_retrieve(args, service, secret, r, table, rng):
    """--retrieve: fetch a finished job and analyse it with saved parameters."""
    try:
        meta = json.loads(JOB_FILE.read_text())
        n, dt, steps, shots = (meta["n"], meta["dt"], meta["steps"], meta["shots"])
        job = service.job(args.retrieve)
        state = wait_for(job, args.wait, poll=10)
        if state != "DONE":
            log.warn_("job %s is %s; try again later", args.retrieve, state)
            return 0 if all(r) else 1
        res = job.result()
        counts = [res[i].data.meas.get_counts() for i in range(steps + 1)]
        name = job.backend().name if hasattr(job, "backend") else "device"
    except Exception as exc:  # noqa: BLE001
        log.fail("retrieve failed: %s: %s", type(exc).__name__, scrub(exc, secret))
        return 1
    r += analyse(counts, n, dt, steps, shots, name, False, table, rng)
    return summarise(r)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
