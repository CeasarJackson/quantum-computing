#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab15_hamiltonian_on_qpu.py

Purpose
-------
Lab 15 runs the Lab 14 time-evolution experiment on a REAL IBM quantum
computer (or, by default, on a simulated copy of one) and compares it with
theory. A chain of n qubits starts in |00..0>; the transverse-field Ising
Hamiltonian H = -J sum Z_i Z_{i+1} - h sum X_i then evolves it. After k
second-order Trotter steps (k = 0..K, step size dt) we measure the mean
magnetisation  M(t) = (1/n) sum_i <Z_i>.

Three curves are compared:
  exact    exp(-iHt) applied to |0..0>          (what nature does)
  ideal    the noiseless Trotter circuit          (what the algorithm gives)
  device   counts from the QPU or the fake QPU     (what you actually get)

The device curve decays too fast because every CNOT adds noise. The k=0
circuit has NO gates, so it measures readout error alone; dividing by it is a
simple, honest correction (readout rescaling). Nothing else is mitigated.

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
    python3 labs/lab15_hamiltonian_on_qpu.py                  # dry run
    read -rs IBM_QUANTUM_API_KEY; export IBM_QUANTUM_API_KEY  # hidden prompt
    python3 labs/lab15_hamiltonian_on_qpu.py --live
    python3 labs/lab15_hamiltonian_on_qpu.py --live --backend ibm_fez --steps 5
    python3 labs/lab15_hamiltonian_on_qpu.py --retrieve <job id>

Log files
---------
logs/lab15/lab15_<YYYYmmdd_HHMMSS>.log (plain text, never contains the key)
logs/lab15/last_job.json (job id, backend, shots, n, dt, steps; no secrets)

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
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.linalg import expm

from applog import get_logger

log = get_logger("lab15")

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
JOB_FILE = Path("logs/lab15/last_job.json")
MAX_SHOTS = 4000  # cap: a typo must not burn the monthly quota
TERMINAL_OK = {"DONE"}
TERMINAL_BAD = {"ERROR", "CANCELLED", "FAILED"}


def scrub(text, secret):
    """Remove the API key from any text before it is logged."""
    # HTTP libraries sometimes echo request headers; replacing the exact
    # secret guarantees it can never reach a log file.
    return str(text).replace(secret, "***") if secret else str(text)


# ----------------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------------
def trotter_circuit(n, dt, k, measure=True):
    """k second-order Trotter steps: RX(half) RZZ(full) RX(half), then measure.

    RZZ(theta) = exp(-i theta/2 ZZ) and RX(theta) = exp(-i theta/2 X), so
    exp(-i A dt) with A = -J sum ZZ needs theta = -2 J dt (same for X with h).
    """
    qc = QuantumCircuit(n)
    for _ in range(k):
        for q in range(n):
            qc.rx(-H_FIELD * dt, q)  # half step of the X part
        for q in range(n - 1):
            qc.rzz(-2 * J * dt, q, q + 1)  # full step of the ZZ part
        for q in range(n):
            qc.rx(-H_FIELD * dt, q)  # second half step
    if measure:
        qc.measure_all()
    return qc


def hamiltonian(n):
    """Dense H as a NumPy matrix (Qiskit ordering, qubit 0 rightmost)."""
    zz = [("ZZ", [q, q + 1], -J) for q in range(n - 1)]
    xx = [("X", [q], -H_FIELD) for q in range(n)]
    return (
        SparsePauliOp.from_sparse_list(zz, num_qubits=n)
        + SparsePauliOp.from_sparse_list(xx, num_qubits=n)
    ).to_matrix()


def mean_z_from_probs(probs, n):
    """Mean of <Z_i> over qubits from a probability vector (index = bitstring)."""
    idx = np.arange(len(probs))
    # Bit q of the index is qubit q's outcome; Z eigenvalue is +1 for 0.
    z = [1 - 2 * ((idx >> q) & 1) for q in range(n)]
    return float(np.mean([np.dot(probs, zq) for zq in z]))


def mean_z_from_counts(counts, n):
    """Mean magnetisation from a counts dictionary (keys: qubit 0 on RIGHT)."""
    total = sum(counts.values())
    acc = 0.0
    for key, c in counts.items():
        ones = key.replace(" ", "").count("1")
        acc += c * (n - 2 * ones) / n
    return acc / total


def references(n, dt, steps):
    """Return (exact, ideal) magnetisation lists for k = 0..steps."""
    h = hamiltonian(n)
    psi0 = np.zeros(2**n, dtype=complex)
    psi0[0] = 1.0
    exact, ideal = [], []
    # One exact and one circuit-based value per step count k (time = k*dt).
    for k in range(steps + 1):
        psi = expm(-1j * h * dt * k) @ psi0
        exact.append(mean_z_from_probs(np.abs(psi) ** 2, n))
        sv = Statevector.from_instruction(trotter_circuit(n, dt, k, False))
        ideal.append(mean_z_from_probs(sv.probabilities(), n))
    return exact, ideal


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


def analyse(results, n, shots, exact, ideal, backend_name, dry):
    """Compare device counts with theory; return the list of check outcomes."""
    # Uncorrected magnetisation at every Trotter step count k.
    raw = [mean_z_from_counts(c, n) for c in results]
    m0 = raw[0]
    # k=0 has no gates, so m0 = 1 - (readout flip rate)*2 on average.
    cor = [x / m0 for x in raw]
    sigma = 1 / math.sqrt(shots)  # conservative bound on one point
    log.info(
        "%-3s %8s %8s %8s %8s %8s", "k", "exact", "ideal", "raw", "corr", "|corr-ex|"
    )
    for k, (e, i, a, c) in enumerate(zip(exact, ideal, raw, cor)):
        log.info("%-3d %+8.4f %+8.4f %+8.4f %+8.4f %8.4f", k, e, i, a, c, abs(c - e))
    rms = math.sqrt(np.mean([(c - e) ** 2 for c, e in zip(cor[1:], exact[1:])]))
    # "No dynamics" baseline: predict that nothing ever changes (M stays 1).
    rms_static = math.sqrt(np.mean([(1 - e) ** 2 for e in exact[1:]]))
    rms_raw = math.sqrt(np.mean([(a - e) ** 2 for a, e in zip(raw[1:], exact[1:])]))
    log.info(
        "RMS error vs exact: raw %.4f, readout-corrected %.4f, "
        "'nothing happens' baseline %.4f (1-sigma per point <= %.3f)",
        rms_raw,
        rms,
        rms_static,
        sigma,
    )
    where = "simulated device" if dry else "REAL HARDWARE"
    out = [
        check(
            m0 > 0.85,
            f"readout alone leaves M(0) = {m0:.4f} (> 0.85) " f"on {backend_name}",
        ),
        check(
            rms < rms_static,
            f"{where} sees the dynamics: corrected "
            f"RMS {rms:.3f} < static baseline {rms_static:.3f}",
        ),
        check(rms < 0.25, f"corrected RMS error {rms:.3f} < 0.25"),
        check(
            rms <= rms_raw + 0.01,
            "readout rescaling does not make " "the agreement worse (within 0.01)",
        ),
    ]
    # (The 0.01 slack is about one shot-noise sigma; the correction is tiny
    # when readout error is already small.)
    # Noise does NOT simply shrink <Z> toward 0: amplitude damping (T1) pulls
    # qubits toward |0>, which RAISES the magnetisation. A live ibm_fez run
    # showed exactly that (device above the ideal curve), so the check is
    # two-sided: the device may deviate from ideal, but only by a bounded amount.
    dev = max(abs(c - i) for c, i in zip(cor, ideal))
    signed = [c - i for c, i in zip(cor, ideal)]
    log.info(
        "device minus ideal Trotter, per k: %s", " ".join(f"{x:+.3f}" for x in signed)
    )
    out.append(
        check(
            dev < 4 * sigma + 0.02,
            f"hardware stays within "
            f"{4 * sigma + 0.02:.3f} of the ideal curve (largest "
            f"deviation {dev:.4f})",
        )
    )
    return out


def self_test(n, dt, steps, shots=4000):
    """Check the pipeline on an ideal simulator before trusting hardware."""
    from qiskit_aer import AerSimulator  # noqa: PLC0415

    sim = AerSimulator(seed_simulator=1)
    exact, ideal = references(n, dt, steps)
    # Run each noiseless circuit and keep its measured magnetisation.
    got = []
    for k in range(steps + 1):
        counts = sim.run(trotter_circuit(n, dt, k), shots=shots).result().get_counts()
        got.append(mean_z_from_counts(counts, n))
    dev = max(abs(g - i) for g, i in zip(got, ideal))
    ok = dev < 4 / math.sqrt(shots)
    (log.ok if ok else log.fail)(
        "self-test: noiseless simulator reproduces the ideal Trotter curve "
        "(max deviation %.4f)",
        dev,
    )
    # Trotter error itself must be small at this step size, else the
    # exact-vs-device comparison would mix two effects.
    terr = max(abs(a - b) for a, b in zip(exact, ideal))
    ok2 = terr < 0.05
    (log.ok if ok2 else log.fail)(
        "self-test: Trotter error in M(t) is %.4f (< 0.05), so remaining "
        "differences come from hardware",
        terr,
    )
    return [ok, ok2]


def summarise(r):
    """Log the tally and return 0/1."""
    bad = r.count(False)
    if bad:
        log.fail("%d of %d checks failed", bad, len(r))
        return 1
    log.ok("All %d checks passed", len(r))
    return 0


def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 15: Hamiltonian on a QPU")
    ap.add_argument(
        "--live", action="store_true", help="use real IBM hardware (default: dry run)"
    )
    ap.add_argument(
        "--retrieve", metavar="JOB_ID", help="analyse a job submitted earlier"
    )
    ap.add_argument("--backend", help="backend name; default: least busy")
    ap.add_argument("--qubits", type=int, default=4, help="chain length 3..8")
    ap.add_argument("--dt", type=float, default=0.25, help="Trotter step size")
    ap.add_argument("--steps", type=int, default=4, help="max Trotter steps 1..6")
    ap.add_argument(
        "--shots", type=int, default=1000, help=f"shots per circuit, 100..{MAX_SHOTS}"
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
    args = ap.parse_args(argv)
    if not 3 <= args.qubits <= 8:
        ap.error(f"--qubits must be 3..8. Got {args.qubits}.")
    if not 0.05 <= args.dt <= 0.5:
        ap.error(f"--dt must be 0.05..0.5. Got {args.dt}.")
    if not 1 <= args.steps <= 6:
        ap.error(f"--steps must be 1..6. Got {args.steps}.")
    if not 100 <= args.shots <= MAX_SHOTS:
        ap.error(f"--shots must be 100..{MAX_SHOTS}. Got {args.shots}.")
    if args.max_seconds < 10:
        ap.error("--max-seconds must be >= 10.")
    if args.live and args.retrieve:
        ap.error("use --live OR --retrieve, not both.")
    n, dt, steps = args.qubits, args.dt, args.steps
    r = []

    log.info("=== Check 1: pipeline self-test (no network) ===")
    r += self_test(n, dt, steps)

    log.info("=== Check 2: connect and choose a backend ===")
    service, secret = connect(args)
    if isinstance(service, str):
        return 2
    if args.retrieve:
        return finish_retrieve(args, service, secret, r)
    backend = pick_backend(args, service, secret, n)
    if backend is None:
        return 1
    log.info("backend %s: %d qubits", backend.name, backend.num_qubits)

    log.info("=== Check 3: transpile to the device's native gates ===")
    # Level 3 places the chain on good physical qubits. Placing each circuit
    # separately lets the layout drift with k, mixing qubit quality with the
    # effect of depth (Lab 17 caught this on real hardware). So place the
    # deepest circuit first and reuse its qubits for every circuit.
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=3, seed_transpiler=3
    )
    deepest = pm.run(trotter_circuit(n, dt, steps))
    layout = deepest.layout.initial_index_layout(filter_ancillas=True)
    log.info("physical qubits used (same for every circuit): %s", layout)
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=3, seed_transpiler=3, initial_layout=layout
    )
    isa = [pm.run(trotter_circuit(n, dt, k)) for k in range(steps + 1)]
    twoq = [sum(c.count_ops().get(g, 0) for g in ("cz", "ecr", "cx")) for c in isa]
    log.info(
        "two-qubit gates per circuit (k=0..%d): %s; depth %s",
        steps,
        twoq,
        [c.depth() for c in isa],
    )
    # Anything outside the native set would be rejected by a real device.
    allowed = set(backend.operation_names) | {"measure", "barrier", "delay"}
    r.append(
        check(
            set().union(*(c.count_ops() for c in isa)) <= allowed,
            "only native gates after transpilation",
        )
    )
    r.append(
        check(
            twoq[0] == 0 and all(b > a for a, b in zip(twoq, twoq[1:])),
            "k=0 circuit has no two-qubit gates; cost grows with k",
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
            "python3 labs/lab15_hamiltonian_on_qpu.py --retrieve %s",
            args.wait,
            job_id,
        )
        return 0 if all(r) else 1
    if state in TERMINAL_BAD:
        log.fail("job ended with status %s", state)
        return 1
    result = job.result()

    log.info("=== Check 5: results ===")
    counts = [result[i].data.meas.get_counts() for i in range(len(isa))]
    r.append(
        check(
            all(sum(c.values()) == args.shots for c in counts),
            f"every circuit returned {args.shots} shots",
        )
    )
    exact, ideal = references(n, dt, steps)
    r += analyse(counts, n, args.shots, exact, ideal, backend.name, not args.live)
    return summarise(r)


def finish_retrieve(args, service, secret, r):
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
    exact, ideal = references(n, dt, steps)
    r += analyse(counts, n, shots, exact, ideal, name, False)
    return summarise(r)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
