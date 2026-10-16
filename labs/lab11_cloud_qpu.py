#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab11_cloud_qpu.py

Purpose
-------
Lab 11 runs the same experiment on a REAL IBM quantum computer (or, by
default, on a simulated copy of one) and certifies what it produced.

The experiment is a 3-qubit GHZ state, (|000> + |111>)/sqrt(2). Z-basis counts
alone cannot tell GHZ from a classical coin that picked 000 or 111, so the lab
measures five bases (ZZZ, XXX, XYY, YXY, YYX) and combines them into the
exact GHZ fidelity
    F = (1/8) (1 + <ZZI> + <IZZ> + <ZIZ> + <XXX> - <XYY> - <YXY> - <YYX>).
Any F > 0.5 proves genuine three-qubit entanglement (a fidelity witness), no
matter how noisy the device is. That is the pass/fail test.

Modes
-----
  (default)  DRY RUN: the identical pipeline on a fake IBM backend with its
             noise model. No account, no network, no QPU time.
  --live     Real hardware through Qiskit Runtime (needs an IBM account).
  --retrieve JOB_ID   Fetch and analyse a job submitted earlier.

Security
--------
The API key is read ONLY from an environment variable (default name
IBM_QUANTUM_API_KEY; change the NAME with --token-env). There is deliberately
no --token flag, because command-line arguments leak into shell history and
process listings. The key is never logged, never written to a file, and is
scrubbed from any error text before it reaches the log. The optional instance
CRN comes from IBM_QUANTUM_INSTANCE. No credentials are saved to disk.

Usage
-----
    python3 labs/lab11_cloud_qpu.py                       # dry run
    read -rs IBM_QUANTUM_API_KEY; export IBM_QUANTUM_API_KEY   # hidden prompt
    python3 labs/lab11_cloud_qpu.py --live                # submit, wait
    python3 labs/lab11_cloud_qpu.py --live --backend ibm_brisbane --shots 1000
    python3 labs/lab11_cloud_qpu.py --retrieve <job id>   # later, if needed

Log files
---------
logs/lab11/lab11_<YYYYmmdd_HHMMSS>.log (plain text, never contains the key)
logs/lab11/last_job.json (job id, backend and shots only, for --retrieve)

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

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from applog import get_logger

log = get_logger("lab11")

# The runtime package is required in BOTH modes (the dry run borrows its fake
# backends), so fail early with the fix instead of a traceback.
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

# The five measurement settings, one letter per qubit. Z gives populations;
# the X/Y mixtures give the coherence between |000> and |111>.
BASES = ("ZZZ", "XXX", "XYY", "YXY", "YYX")
JOB_FILE = Path("logs/lab11/last_job.json")
MAX_SHOTS = 4000  # cap: a typo must not burn the monthly QPU quota
TERMINAL_OK = {"DONE"}
TERMINAL_BAD = {"ERROR", "CANCELLED", "FAILED"}


def scrub(text, secret):
    """Remove the API key from any text before it is logged."""
    # Errors from HTTP libraries sometimes echo request headers; replacing the
    # exact secret guarantees it can never reach a log file.
    return str(text).replace(secret, "***") if secret else str(text)


def ghz_circuit(basis):
    """GHZ(3) followed by the rotations that make `basis` readable in Z."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    for q, b in enumerate(basis):
        # Y = rotate with S-dagger then H; X = just H; Z needs nothing.
        # Qubit q gets the letter basis[q] (qubit 0 is the first letter).
        if b == "Y":
            qc.sdg(q)
        if b in "XY":
            qc.h(q)
    qc.measure_all()
    return qc


def parity(counts, qubits):
    """<product of Paulis on `qubits>` from counts: P(even) - P(odd)."""
    total = sum(counts.values())
    acc = 0
    # Qiskit keys are bit strings with qubit 0 on the RIGHT.
    for key, c in counts.items():
        bits = key[::-1]  # bits[k] is the result of qubit k
        ones = sum(bits[q] == "1" for q in qubits)
        acc += c if ones % 2 == 0 else -c
    return acc / total


def analyse(counts_by_basis):
    """Turn five count dictionaries into the GHZ fidelity and its terms."""
    z = counts_by_basis["ZZZ"]
    terms = {
        "ZZI": parity(z, (0, 1)),
        "IZZ": parity(z, (1, 2)),
        "ZIZ": parity(z, (0, 2)),
    }
    for b in BASES[1:]:
        terms[b] = parity(counts_by_basis[b], (0, 1, 2))
    # Signs come from the stabilizer expansion of |GHZ><GHZ|: XXX positive,
    # each XYY-type term negative (two Y's give an extra minus sign).
    fid = (
        1
        + terms["ZZI"]
        + terms["IZZ"]
        + terms["ZIZ"]
        + terms["XXX"]
        - terms["XYY"]
        - terms["YXY"]
        - terms["YYX"]
    ) / 8
    shots = sum(z.values())
    # Each term is an average of +/-1 values, so var <= 1/shots. Adding the
    # seven variances treats the terms as independent: an approximation (the
    # three ZZ terms share one dataset) that is good enough for a 3-sigma test.
    sigma = math.sqrt(7 / shots) / 8
    pop = (z.get("000", 0) + z.get("111", 0)) / shots
    return {"terms": terms, "fidelity": fid, "sigma": sigma, "z_success": pop}


def counts_from_result(result):
    """List of five PUB results -> {basis: counts}; the register is 'meas'."""
    return {b: result[i].data.meas.get_counts() for i, b in enumerate(BASES)}


def pick_backend(args, service, secret):
    """Fake backend (dry run) or a real one (named, else least busy)."""
    if not args.live:
        log.info("DRY RUN: simulated copy of ibm_sherbrooke (noise model only)")
        return FakeSherbrooke()
    try:
        if args.backend:
            return service.backend(args.backend)
        # operational + real hardware + big enough, then shortest queue.
        return service.least_busy(operational=True, simulator=False, min_num_qubits=3)
    except Exception as exc:  # noqa: BLE001
        log.fail(
            "could not select a backend: %s: %s", type(exc).__name__, scrub(exc, secret)
        )
        return None


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
        # Never print or store `secret`. The service object keeps it in memory.
        svc = QiskitRuntimeService(token=secret, instance=instance)
    except Exception as exc:  # noqa: BLE001
        log.fail("login failed: %s: %s", type(exc).__name__, scrub(exc, secret))
        log.info(
            "Check the key (44 characters, an API key, not a bearer "
            "token) and, if you have several instances, the CRN."
        )
        return "failed", secret
    log.ok("connected to IBM Quantum Platform (credentials not logged)")
    return svc, secret


def make_sampler(args, backend):
    """Sampler bound to the backend, shots and a hard execution-time cap."""
    sampler_cls = SamplerV2
    if args.sampler == "client" and args.live:
        # Newer client-side sampler introduced in qiskit-ibm-runtime 0.50.
        from qiskit_ibm_runtime.executor_sampler import Sampler  # noqa: PLC0415

        sampler_cls = Sampler
    with warnings.catch_warnings():
        # SamplerV2 is deprecated in 0.50 but supported for at least 3 more
        # months; say so once in the log instead of spamming warnings.
        warnings.simplefilter("ignore", DeprecationWarning)
        sampler = sampler_cls(mode=backend)
    sampler.options.default_shots = args.shots
    if args.live:
        # Hard stop so a stuck job cannot consume the plan's QPU minutes.
        sampler.options.max_execution_time = args.max_seconds
    return sampler


def wait_for(job, timeout, poll=10):
    """Poll a job until it finishes or `timeout` seconds pass."""
    start = time.time()
    last = None
    # Loop until the job reaches a terminal state or the clock runs out.
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


def report(res, backend_name, dry):
    """Log the analysis and return the list of check outcomes."""
    out = []
    fid, sig, pop = res["fidelity"], res["sigma"], res["z_success"]
    for name, val in res["terms"].items():
        log.info("<%s> = %+.4f", name, val)
    log.info("Z-basis success P(000 or 111) = %.4f (uniform noise gives 0.25)", pop)
    log.info("GHZ fidelity F = %.4f +/- %.4f on %s", fid, sig, backend_name)
    # 0.4 is far above the 0.25 garbage floor yet below any working device,
    # so it separates "dead" from "noisy but alive" without being fragile.
    out.append(pop > 0.4)
    (
        log.ok("population check")
        if out[-1]
        else log.fail("Z-basis success is not clearly above the 0.25 garbage floor")
    )
    witness = fid > 0.5 + 3 * sig
    out.append(witness)
    if witness:
        log.ok(
            "F > 0.5 + 3 sigma: genuine 3-qubit entanglement certified%s",
            " (simulated device)" if dry else " ON REAL HARDWARE",
        )
    else:
        log.fail(
            "F is not above 0.5: entanglement NOT certified (device too "
            "noisy for this circuit right now)"
        )
    return out


def self_test():
    """Check the analysis code on an ideal simulator before trusting it."""
    from qiskit_aer import AerSimulator  # noqa: PLC0415

    # Seeded ideal simulator: any deviation from F=1 would be an analysis bug.
    sim = AerSimulator(seed_simulator=1)
    counts = {
        b: sim.run(ghz_circuit(b), shots=4000).result().get_counts() for b in BASES
    }
    res = analyse(counts)
    ok = abs(res["fidelity"] - 1) < 4 * res["sigma"] + 1e-9
    (log.ok if ok else log.fail)(
        "self-test: noiseless GHZ gives F = %.4f (expect 1.0)", res["fidelity"]
    )
    # A classical 000/111 coin must score about 0.5 or less, never above.
    # Build the counts a classical coin would give: perfect Z, random X/Y.
    mix = {"ZZZ": {"000": 2000, "111": 2000}}
    for b in BASES[1:]:
        # Random X-basis parities for a coin: half even, half odd.
        mix[b] = {"000": 1000, "001": 1000, "011": 1000, "111": 1000}
    cres = analyse(mix)
    cok = cres["fidelity"] < 0.5 + 3 * cres["sigma"]
    (log.ok if cok else log.fail)(
        "self-test: classical coin gives F = %.4f (cannot exceed 0.5)", cres["fidelity"]
    )
    return [ok, cok]


def main(argv=None):
    """Run the lab; return the process exit code."""
    # Every knob is a flag except the secret, which only the environment may hold.
    ap = argparse.ArgumentParser(description="Lab 11: a real quantum computer")
    ap.add_argument(
        "--live", action="store_true", help="use real IBM hardware (default: dry run)"
    )
    ap.add_argument(
        "--retrieve", metavar="JOB_ID", help="analyse a job submitted earlier"
    )
    ap.add_argument("--backend", help="backend name; default: least busy")
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
    if not 100 <= args.shots <= MAX_SHOTS:
        ap.error(f"--shots must be 100..{MAX_SHOTS}. Got {args.shots}.")
    if args.max_seconds < 10:
        ap.error("--max-seconds must be >= 10.")
    # These modes are alternatives: one submits, the other only reads.
    if args.live and args.retrieve:
        ap.error("use --live OR --retrieve, not both.")
    r = []

    log.info("=== Check 1: analysis code self-test (no network) ===")
    r += self_test()

    log.info("=== Check 2: connect and choose a backend ===")
    # connect() returns an error string instead of raising, to keep exit codes tidy.
    service, secret = connect(args)
    if isinstance(service, str):
        return 2
    if args.retrieve:
        return finish_retrieve(args, service, secret, r)
    backend = pick_backend(args, service, secret)
    if backend is None:
        return 1
    # Pending-job count tells you how long the queue is likely to be.
    status = backend.status() if hasattr(backend, "status") else None
    log.info(
        "backend %s: %d qubits, pending jobs: %s",
        backend.name,
        backend.num_qubits,
        getattr(status, "pending_jobs", "n/a"),
    )

    log.info("=== Check 3: transpile to the device's native gates ===")
    # Level 3 lets the transpiler choose good physical qubits on a big chip.
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=3, seed_transpiler=3
    )
    isa = [pm.run(ghz_circuit(b)) for b in BASES]
    # Anything outside the native set would be rejected by the real device.
    allowed = set(backend.operation_names) | {"measure", "barrier", "delay"}
    ops = set().union(*(c.count_ops() for c in isa))
    log.info(
        "two-qubit gates per circuit: %s; depth %s",
        [
            c.count_ops().get("cz", 0)
            + c.count_ops().get("ecr", 0)
            + c.count_ops().get("cx", 0)
            for c in isa
        ],
        [c.depth() for c in isa],
    )
    r.append(ops <= allowed)
    (log.ok if r[-1] else log.fail)("only native gates after transpilation")

    log.info("=== Check 4: run %d circuits x %d shots ===", len(isa), args.shots)
    if args.live:
        log.warn_(
            "LIVE: this uses real QPU time (a few seconds). cap=%d s", args.max_seconds
        )
    # One job holds all five circuits, so we queue only once.
    sampler = make_sampler(args, backend)
    job = sampler.run(isa)
    job_id = job.job_id()
    # Save the id (not the key) so --retrieve works after closing the terminal.
    JOB_FILE.parent.mkdir(parents=True, exist_ok=True)
    JOB_FILE.write_text(
        json.dumps({"job_id": job_id, "backend": backend.name, "shots": args.shots})
    )
    log.info("job id: %s (saved to %s)", job_id, JOB_FILE)
    # Queue time can be minutes to hours; wait politely, then give a retrieve command.
    state = wait_for(job, args.wait if args.live else 600, poll=5)
    if state == "TIMEOUT":
        log.warn_(
            "still queued after %d s. Fetch it later with: "
            "python3 labs/lab11_cloud_qpu.py --retrieve %s",
            args.wait,
            job_id,
        )
        return 0 if all(r) else 1
    if state in TERMINAL_BAD:
        log.fail("job ended with status %s", state)
        return 1
    # Blocks only briefly: the job is already DONE here.
    result = job.result()

    log.info("=== Check 5: results ===")
    # One count dictionary per measurement basis.
    counts = counts_from_result(result)
    r.append(all(sum(c.values()) == args.shots for c in counts.values()))
    (log.ok if r[-1] else log.fail)("every circuit returned %d shots", args.shots)
    r += report(analyse(counts), backend.name, not args.live)
    return summarise(r)


def finish_retrieve(args, service, secret, r):
    """--retrieve: fetch a finished job and analyse it."""
    try:
        job = service.job(args.retrieve)
        state = wait_for(job, args.wait, poll=10)
        if state != "DONE":
            log.warn_("job %s is %s; try again later", args.retrieve, state)
            return 0 if all(r) else 1
        counts = counts_from_result(job.result())
        name = job.backend().name if hasattr(job, "backend") else "device"
    except Exception as exc:  # noqa: BLE001
        log.fail("retrieve failed: %s: %s", type(exc).__name__, scrub(exc, secret))
        return 1
    r += report(analyse(counts), name, False)
    return summarise(r)


def summarise(r):
    """Log the tally and return 0/1."""
    bad = r.count(False)
    if bad:
        log.fail("%d of %d checks failed", bad, len(r))
        return 1
    log.ok("All %d checks passed", len(r))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
