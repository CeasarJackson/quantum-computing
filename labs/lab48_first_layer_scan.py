#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab48_first_layer_scan.py

Purpose
-------
Lab 46 found that the ibm_fez energies of the Heisenberg circuits lose about
0.26 before any bond gate and a FURTHER 0.42 once the first layer of bond gates
starts, with per-gate noise at the modelled strength. That three-part fit was
chosen after the data were seen, so it describes the numbers but does not test
a mechanism. Lab 48 asks the next question: WHERE in the first layer does the
extra 0.42 appear?

The circuit is the three singlets followed by the first k bond gates of the
Lab 44 depth-1 ideal angles, for k = 0..5 (the odd bonds first, then the even
bonds, as in the HVA layer). Each k is measured in the X, Y and Z bases and the
energy loss against the NOISELESS energy of the same circuit is compared with
the loss the Lab 21 noise model expects.

TWO PREDICTIONS, fixed here BEFORE any run, both built from the Lab 46 refit
(c0 = 0.259, c1 = 0.416, s = 0.934; see fezdata.LAB46_FIT):

  STEP    the extra loss c1 appears with the FIRST bond gate and stays:
              loss(k) = c0 + c1 * [k >= 1] + s * model(k)
  GRADUAL the extra loss grows with every bond gate, reaching c1 at k = 5:
              loss(k) = c0 + c1 * k / 5 + s * model(k)

They agree at k = 0 and k = 5 and differ most at k = 1 (by 0.33, about 6 sigma
at the default shot count). The lab prints the chi-squared of each prediction,
the increment of the excess loss at every added gate (which shows WHICH gate
adds it), and a free fit excess(k) = a + b*[k >= 1] + g*k.

--order reverse applies the gates within each group in the opposite order
(bonds 3,1 then 4,2,0). That is the MIRROR IMAGE of the forward order (the
chain reflection q -> 5 - q maps the singlet pairs onto themselves and bond b
onto bond 4 - b), so the noise model gives identical numbers for both orders
and any difference on the machine is specific to the physical qubits and
links. A loss tied to "the first gate" stays at k = 1 in both orders; a loss
tied to one particular link moves with that link.

The lab claims only what its checks test: the circuit builder against an ideal
simulator, the noise model against Lab 21, the two predictions, and the
verdict logic on synthetic data. What the machine does is REPORTED, not
asserted. The Lab 46 limitations apply: one device, one chain, depolarizing
model only, shot-noise error bars, and mirror circuits cannot see coherent
error.

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
    python3 labs/lab48_first_layer_scan.py                       # dry run
    read -rs IBM_QUANTUM_API_KEY; export IBM_QUANTUM_API_KEY     # hidden prompt
    python3 labs/lab48_first_layer_scan.py --live --backend ibm_fez
    python3 labs/lab48_first_layer_scan.py --retrieve <job id>
    python3 labs/lab48_first_layer_scan.py --live --order reverse   # control

Log files
---------
logs/lab48/lab48_<YYYYmmdd_HHMMSS>.log (plain text, never contains the key)
logs/lab48/last_job.json (job id, backend, shots, layout, order; no secrets)

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
from scipy.linalg import expm
from scipy.stats import chi2 as chi2_dist

from applog import get_logger

log = get_logger("lab48")

# Cloud plumbing, the estimator, the noise model and the recorded data are all
# reused, not copied: the security-critical code then exists in one place.
try:
    from fezdata import (
        ANGLES,
        LAB46_FIT,
        LAMBDA_FEZ,
        LAYOUT,
        RUNS,
        fit_design,
    )
    from lab17_mirror_benchmark import (
        TERMINAL_BAD,
        connect,
        make_sampler,
        scrub,
        wait_for,
    )
    from lab21_noise_aware_training import (
        J,
        N,
        NoisyChain,
        bond,
        bond_gate,
        depolarize,
        singlets,
    )
    from lab44_noise_aware_on_qpu import (
        BASES,
        attenuation,
        calibration_circuits,
        energy_from_counts,
        run_ideal,
    )
    from lab46_hardware_noise_budget import MAX_SHOTS, pick_backend, transpile_all
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

JOB_FILE = Path("logs/lab48/last_job.json")
# Bond-gate order: the start qubit of each bond. Odd bonds first, then even,
# as in one HVA layer; "reverse" flips the order inside each group.
ORDERS = {"forward": (1, 3, 0, 2, 4), "reverse": (3, 1, 4, 2, 0)}
ANGLES1 = ANGLES[(1, "ideal")]  # [odd-group angle, even-group angle]
KS = tuple(range(len(ORDERS["forward"]) + 1))  # 0..5 bond gates
PREP_GATES = 3  # one two-qubit gate per singlet
GATES_PER_BOND = 3  # rxx, ryy and rzz each cost one two-qubit gate
SIGMA_NOMINAL = 0.05  # typical error bar of one energy at 4000 shots


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


# ----------------------------------------------------------------------------
# Circuits
# ----------------------------------------------------------------------------
def gate_angle(q, angles=ANGLES1):
    """Angle of the bond starting on qubit q: odd bonds use the first angle."""
    return angles[0] if q % 2 else angles[1]


def scan_circuit(k, order, basis, angles=ANGLES1):
    """Three singlets, the first k bond gates in `order`, then a basis change."""
    qc = singlets(N)
    for q in order[:k]:
        bond_gate(qc, q, q + 1, J * gate_angle(q, angles))
    # Same rotations as Lab 44: X needs H, Y needs S-dagger then H.
    for q in range(N):
        if basis == "Y":
            qc.sdg(q)
        if basis in ("X", "Y"):
            qc.h(q)
    qc.measure_all()
    return qc


def expected_twoq(key):
    """Two-qubit gates a circuit should keep after transpilation."""
    if key in ("cal0", "cal1"):
        return 0
    return PREP_GATES + GATES_PER_BOND * key[0]


def build_jobs(order):
    """Circuits in a fixed order: calibration, then k = 0..5 in X, Y, Z."""
    keys, circuits = ["cal0", "cal1"], calibration_circuits()
    for k in KS:
        for basis in BASES:
            keys.append((k, basis))
            circuits.append(scan_circuit(k, order, basis))
    return keys, circuits


# ----------------------------------------------------------------------------
# Model: the same density-matrix noise as Lab 21, one bond gate at a time
# ----------------------------------------------------------------------------
def scan_energy(chain, order, k, lam, angles=ANGLES1):
    """Energy after the singlets and k bond gates, with depolarizing strength lam."""
    rho = np.outer(chain.psi0, chain.psi0.conj())
    # Singlet preparation: one noisy two-qubit gate per pair (as in Lab 21).
    if lam > 0:
        for q in range(0, N, 2):
            rho = depolarize(rho, N, q, 1 - lam)
    for q in order[:k]:
        # Exact bond unitary exp(-i angle * bond); bond() already contains J.
        u = expm(-1j * gate_angle(q, angles) * bond(N, q))
        rho = u @ rho @ u.conj().T
        # A bond gate is three two-qubit gates, so it retains (1 - lam)^3.
        rho = depolarize(rho, N, q, (1 - lam) ** 3)
    return float(np.trace(rho @ chain.h).real)


def model_table(chain, order):
    """Per k: noiseless energy and the modelled loss at the ibm_fez strength."""
    free = np.array([scan_energy(chain, order, k, 0.0) for k in KS])
    noisy = np.array([scan_energy(chain, order, k, LAMBDA_FEZ) for k in KS])
    return free, noisy - free


def predictions(model_loss):
    """Losses expected by the STEP and GRADUAL stories (fixed beforehand)."""
    c0, c1, s = (LAB46_FIT[x][0] for x in ("c0", "c1", "s"))
    k = np.array(KS, dtype=float)
    step = c0 + c1 * (k >= 1) + s * model_loss
    gradual = c0 + c1 * k / KS[-1] + s * model_loss
    return step, gradual


# ----------------------------------------------------------------------------
# Verdict logic (pure functions, tested on synthetic data)
# ----------------------------------------------------------------------------
def free_fit(excess, sig):
    """Fit excess(k) = a + b*[k >= 1] + g*k; returns (beta, errors, chi2)."""
    k = np.array(KS, dtype=float)
    cols = [np.ones_like(k), (k >= 1).astype(float), k]
    beta, cov, chi2 = fit_design(cols, excess, sig)
    return beta, np.sqrt(np.diag(cov)), chi2


def verdict(chi_step, chi_gradual, dof):
    """Which pre-registered story the data allow (99% chi-squared limit)."""
    limit = float(chi2_dist.ppf(0.99, dof))
    ok_step, ok_grad = chi_step < limit, chi_gradual < limit
    if ok_step and not ok_grad:
        return "agrees with STEP: the extra loss appears with the first bond gate"
    if ok_grad and not ok_step:
        return "agrees with GRADUAL: the extra loss grows with every bond gate"
    if ok_step and ok_grad:
        return "cannot separate STEP from GRADUAL at this precision"
    return "matches NEITHER story: read the per-gate increments below"


# ----------------------------------------------------------------------------
# Checks that need no network
# ----------------------------------------------------------------------------
def check_models(r, ctx):
    """The scan model against Lab 21, the two orders and the circuit builder."""
    log.info("=== Check 1: the scan model ===")
    chain, fwd, rev = ctx["chain"], ORDERS["forward"], ORDERS["reverse"]
    free, loss = model_table(chain, fwd)
    log.info("noiseless energy by k: %s", np.round(free, 4))
    log.info("modelled loss by k:    %s", np.round(loss, 4))
    r.append(check(abs(free[0] + 9) < 1e-9, "k = 0 is three singlets: energy -9"))
    # With all five gates the scan is exactly the Lab 21 depth-1 circuit.
    same = max(
        abs(scan_energy(chain, fwd, KS[-1], lam) - chain.energy(ANGLES1, lam))
        for lam in (0.0, LAMBDA_FEZ)
    )
    r.append(check(same < 1e-9, f"k = 5 equals the Lab 21 depth-1 model ({same:.1e})"))
    # Gates inside a group commute, so both orders end in the same state ...
    end = abs(
        scan_energy(chain, fwd, 5, LAMBDA_FEZ) - scan_energy(chain, rev, 5, LAMBDA_FEZ)
    )
    r.append(check(end < 1e-9, f"both orders end at the same energy ({end:.1e})"))
    # ... and every intermediate circuit of one order is the chain reflection
    # of the other's, so the MODEL gives equal numbers at every k. Any
    # difference on hardware is therefore specific to the physical links.
    mirror = max(
        abs(
            scan_energy(chain, fwd, k, LAMBDA_FEZ)
            - scan_energy(chain, rev, k, LAMBDA_FEZ)
        )
        for k in KS
    )
    r.append(
        check(
            mirror < 1e-9, f"the orders are mirror images in the model ({mirror:.1e})"
        )
    )
    r.append(
        check(
            fwd != rev and set(fwd) == set(rev),
            "the orders use the same bonds, reversed",
        )
    )

    log.info("=== Check 2: circuits on an ideal simulator ===")
    order = ctx["order"]
    keys, circuits = build_jobs(order)
    counts = run_ideal(circuits, 4000, 11)
    by_key = dict(zip(keys, counts))
    flat = np.ones(N)  # ideal simulator: no readout error to undo
    free_o, _ = model_table(chain, order)
    worst = 0.0
    for k in KS:
        e, s = energy_from_counts({b: by_key[(k, b)] for b in BASES}, flat)
        worst = max(worst, abs(e - free_o[k]) / s)
    log.info("largest |ideal simulator - exact| over k: %.2f sigma", worst)
    r.append(check(worst < 4, "circuits reproduce the exact energies for every k"))


def check_predictions(r, ctx):
    """The two stories agree at the ends and are separable at k = 1."""
    log.info("=== Check 3: the pre-registered predictions ===")
    _, loss = model_table(ctx["chain"], ctx["order"])
    step, gradual = predictions(loss)
    log.info("STEP    extra-loss prediction by k: %s", np.round(step, 3))
    log.info("GRADUAL extra-loss prediction by k: %s", np.round(gradual, 3))
    gap = np.abs(step - gradual)
    r.append(check(gap[0] < 1e-12, "the stories agree at k = 0 (loss before any gate)"))
    r.append(check(gap[-1] < 1e-12, "the stories agree at k = 5 (the Lab 46 depth 1)"))
    sep = gap[1] / SIGMA_NOMINAL
    r.append(check(sep > 5, f"they differ at k = 1 by {sep:.1f} sigma (> 5)"))

    log.info("=== Check 4: the verdict logic on synthetic data ===")
    c0, c1 = LAB46_FIT["c0"][0], LAB46_FIT["c1"][0]
    k = np.array(KS, dtype=float)
    sig = np.full(len(KS), SIGMA_NOMINAL)
    cases = {
        "step": c0 + c1 * (k >= 1),
        "gradual": c0 + c1 * k / KS[-1],
        "neither": c0 + 0.0 * k,
    }
    want = {"step": "STEP:", "gradual": "GRADUAL:", "neither": "NEITHER"}
    for name, excess in cases.items():
        # Excess loss means loss minus the modelled loss, so predictions are
        # compared as pure excess here (model part removed on both sides).
        cs = float((((excess - (c0 + c1 * (k >= 1))) / sig) ** 2).sum())
        cg = float((((excess - (c0 + c1 * k / KS[-1])) / sig) ** 2).sum())
        text = verdict(cs, cg, len(KS))
        r.append(check(want[name] in text, f"synthetic {name} data -> {text[:28]}"))
    # An exact free fit must return the injected numbers.
    beta, err, chi2 = free_fit(c0 + c1 * (k >= 1) + 0.02 * k, sig)
    ok = np.allclose(beta, [c0, c1, 0.02]) and chi2 < 1e-9
    r.append(check(ok, "free fit recovers the injected a, b, g exactly"))
    log.info(
        "free fit errors at %.2f nominal sigma: a %.3f, b %.3f, g %.3f", sig[0], *err
    )
    r.append(
        check(c1 / err[1] > 5 and c1 / 5 / err[2] > 4, "b and g are both resolvable")
    )


# ----------------------------------------------------------------------------
# Analysis of a finished run
# ----------------------------------------------------------------------------
def analyse(r, ctx, keys, counts, backend_name, dry):
    """Energies, excess loss per gate, both stories, the free fit and a verdict."""
    where = "simulated device" if dry else "REAL HARDWARE"
    by_key = dict(zip(keys, counts))
    atten = attenuation(by_key["cal0"], by_key["cal1"])
    log.info("%s %s: readout attenuation %s", backend_name, where, np.round(atten, 3))
    r.append(
        check(bool(np.all((atten > 0.8) & (atten <= 1.0))), "attenuation in (0.8, 1]")
    )
    order = ctx["order"]
    free, mloss = model_table(ctx["chain"], order)
    step, gradual = predictions(mloss)
    # Energies and their shot-noise errors, one per k.
    en = [energy_from_counts({b: by_key[(k, b)] for b in BASES}, atten) for k in KS]
    e = np.array([x[0] for x in en])
    sig = np.array([x[1] for x in en])
    r.append(
        check(
            bool(np.all(e > ctx["chain"].e0 - 4 * sig)),
            f"{where}: no energy below its ground-state bound",
        )
    )
    loss = e - free
    excess = loss - mloss  # what the noise model does not explain
    if dry:
        log.info("(dry run: the numbers below describe the simulator, not ibm_fez)")
    log.info("k  gate  measured E     loss   model  excess | STEP  z   GRADUAL  z")
    for k in KS:
        gate = f"({order[k-1]},{order[k-1]+1})" if k else "none "
        log.info(
            "%d  %s  %8.3f +/- %.3f %6.3f %6.3f %6.3f | %6.3f %+5.1f %6.3f %+5.1f",
            k,
            gate,
            e[k],
            sig[k],
            loss[k],
            mloss[k],
            excess[k],
            step[k],
            (loss[k] - step[k]) / sig[k],
            gradual[k],
            (loss[k] - gradual[k]) / sig[k],
        )
    # Which gate adds the extra loss: the change of the excess at each step.
    for k in KS[1:]:
        d = excess[k] - excess[k - 1]
        ds = math.hypot(sig[k], sig[k - 1])
        log.info(
            "  gate %d (bond %d-%d): excess changes by %+.3f +/- %.3f (%+.1f sigma)",
            k,
            order[k - 1],
            order[k - 1] + 1,
            d,
            ds,
            d / ds,
        )
    chi_s = float((((loss - step) / sig) ** 2).sum())
    chi_g = float((((loss - gradual) / sig) ** 2).sum())
    log.info(
        "chi2 (%d points, no free parameter): STEP %.1f, GRADUAL %.1f",
        len(KS),
        chi_s,
        chi_g,
    )
    beta, err, chi_f = free_fit(excess, sig)
    log.info(
        "free fit a + b*[k>=1] + g*k: a = %.3f +/- %.3f, b = %.3f +/- %.3f, "
        "g = %.3f +/- %.3f, chi2 = %.1f (%d dof)",
        beta[0],
        err[0],
        beta[1],
        err[1],
        beta[2],
        err[2],
        chi_f,
        len(KS) - 3,
    )
    log.info("verdict: the scan %s", verdict(chi_s, chi_g, len(KS)))
    if dry:
        log.info("(dry run: that verdict is about the simulator, not ibm_fez)")


def summarise(r):
    """Log the tally and return 0/1."""
    bad = r.count(False)
    if bad:
        log.fail("%d of %d checks failed", bad, len(r))
        return 1
    log.ok("All %d checks passed", len(r))
    return 0


def plan_time(shots):
    """Log the QPU time this job should take, from the two recorded rates."""
    rates = [secs / (circ * sh) for circ, sh, secs in RUNS]
    total = (2 + 3 * len(KS)) * shots
    log.info(
        "plan: %d circuits x %d shots = %d shots, about %.0f to %.0f s running "
        "(recorded rates %.2f and %.2f ms per shot)",
        2 + 3 * len(KS),
        shots,
        total,
        min(rates) * total,
        max(rates) * total,
        min(rates) * 1e3,
        max(rates) * 1e3,
    )


def finish_retrieve(args, service, secret, r, ctx):
    """--retrieve: fetch a finished job and analyse it (order from the job file)."""
    try:
        meta = json.loads(JOB_FILE.read_text())
        ctx["order"] = ORDERS[meta.get("order", "forward")]
        keys = build_jobs(ctx["order"])[0]
        job = service.job(args.retrieve)
        state = wait_for(job, args.wait, poll=10)
        if state != "DONE":
            log.warn_("job %s is %s; try again later", args.retrieve, state)
            return 0 if all(r) else 1
        res = job.result()
        counts = [res[i].data.meas.get_counts() for i in range(len(keys))]
    except Exception as exc:  # noqa: BLE001
        log.fail("retrieve failed: %s: %s", type(exc).__name__, scrub(exc, secret))
        return 1
    analyse(r, ctx, keys, counts, meta.get("backend", "device"), False)
    return summarise(r)


def main(argv=None):
    """Run the lab; return the process exit code."""
    ap = argparse.ArgumentParser(description="Lab 48: where the first layer loses")
    ap.add_argument("--live", action="store_true", help="use real IBM hardware")
    ap.add_argument("--retrieve", metavar="JOB_ID", help="analyse an earlier job")
    ap.add_argument("--backend", help="backend name; default: least busy")
    ap.add_argument("--order", choices=tuple(ORDERS), default="forward")
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
    # The chain model is built once (a 64 x 64 density-matrix simulator).
    ctx = {"chain": NoisyChain(N), "order": ORDERS[args.order]}
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_models(r, ctx)
    check_predictions(r, ctx)

    log.info("=== Check 5: connect and choose a backend ===")
    service, secret = connect(args)
    if isinstance(service, str):
        return 2
    if args.retrieve:
        return finish_retrieve(args, service, secret, r, ctx)
    backend = pick_backend(args, service, secret)
    if backend is None:
        return 1
    log.info("backend %s: %d qubits", backend.name, backend.num_qubits)

    log.info("=== Check 6: transpile onto one set of physical qubits ===")
    keys, circuits = build_jobs(ctx["order"])
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

    log.info("=== Check 7: run %d circuits x %d shots ===", len(isa), args.shots)
    plan_time(args.shots)
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
                "order": args.order,
            }
        )
    )
    log.info("job id: %s (saved to %s)", job_id, JOB_FILE)
    state = wait_for(job, args.wait if args.live else 600, poll=5)
    if state == "TIMEOUT":
        log.warn_(
            "still queued after %d s. Fetch it later with: "
            "python3 labs/lab48_first_layer_scan.py --retrieve %s",
            args.wait,
            job_id,
        )
        return 0 if all(r) else 1
    if state in TERMINAL_BAD:
        log.fail("job ended with status %s", state)
        return 1
    result = job.result()

    log.info("=== Check 8: results ===")
    counts = [result[i].data.meas.get_counts() for i in range(len(isa))]
    r.append(
        check(
            all(sum(c.values()) == args.shots for c in counts),
            f"every circuit returned {args.shots} shots",
        )
    )
    analyse(r, ctx, keys, counts, backend.name, not args.live)
    return summarise(r)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
