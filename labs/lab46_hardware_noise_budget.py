#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab46_hardware_noise_budget.py

Purpose
-------
Lab 44 ran noise-aware Heisenberg angles on ibm_fez (qubits 129,118,109,110,
111,112) and found the machine WORSE than the Lab 21 model. The first summary
said "1.6 to 1.75 times the modelled shortfall". Lab 46 looks at that number
properly and then designs the measurement that explains it.

  1. RECORDED DATA (no network). The six measured Lab 44 energies are fitted as
         shortfall_measured = c + s * shortfall_model
     where the shortfall is the energy lost against the NOISELESS energy of
     the same circuit. A pure scale s (gate noise higher than modelled) is
     REJECTED (chi-squared 98 for 5 degrees of freedom; it would need s = 2 and
     still miss depth 1 by a factor 3.7). A depth-independent loss c = 0.79
     plus gate noise s = 0.79 of the model fits (chi-squared 4.9 for 4). So
     the missing energy is mostly NOT per-gate noise.
  2. TWO COMPETING EXPLANATIONS, with numbers fixed BEFORE the live run:
       A. "offset": the loss is already there before any Heisenberg gate
          (state preparation, readout, basis rotation). The depth-0 circuit
          (three singlets, exact energy -9) should then measure
          about -8.18 +/- 0.08.
       B. "first layer": nothing is wrong at depth 0 and the loss appears
          with the first bond gates. Depth 0 should then measure about -8.97.
     Lab 46 measures the depth-0 energy on the SAME qubits and prints which
     prediction it lands on.
  3. REPLICATION. The Lab 44 depth-1 circuits are measured again. At depth 1
     the ideal and noise-trained angles are the same angles with their signs
     flipped (the model predicts equal energies), yet Lab 44 measured them
     0.21 apart (2 sigma). A second measurement says whether that gap repeats
     (a real hardware asymmetry) or not (shot noise).
  4. MIRROR DECAY on these six qubits: the Lab 44 depth-3 circuit followed by
     its inverse (U then U-dagger), 6 + 30 k two-qubit gates for k = 0..3 layers.
     The decay per gate is converted to a depolarizing strength with a
     simulator table built for THIS circuit family, to compare with 0.0037
     (Lab 17, four other qubits) and with the 0.0030 the offset fit prefers.

FIRST LIVE RUN (ibm_fez, 2026-10-17): neither story held. The depth-0 energy
was -8.709 +/- 0.044, 6 sigma from both predictions, so about 0.26 of the
loss is present before any bond gate and the rest appears with the first
layer. The mirror gave lambda = 0.0036 (0.0034 .. 0.0038), the same as Lab 17
on other qubits, so the per-gate noise of the model is right. The depth-1
sign-flip gap did not repeat (+0.021 +/- 0.074 against +0.208 +/- 0.103).

LIMITATIONS of the model and of what the live run shows
-------------------------------------------------------
  * One device, one chain of six qubits, one job per measurement. Lab 44 and
    Lab 46 ran on different days; drift between calibrations is not measured,
    and a common offset between the two jobs would be absorbed into c0 / c1.
  * The noise model is per-CNOT depolarizing only. Coherent over-rotation,
    amplitude damping, crosstalk, correlated readout error and leakage are
    NOT modelled, and readout is corrected by independent per-qubit
    attenuation only.
  * Mirror circuits (U then U-dagger) measure INCOHERENT error per gate.
    Coherent errors largely cancel there, so lambda = 0.0036 says nothing
    about them; it does not rule them out as the source of the step.
  * The three-part fit c0 + c1*[depth >= 1] + s*model was chosen AFTER the
    data were seen (nine points, three parameters). chi-squared 8.5 for 6
    degrees of freedom shows it is consistent with the data, not that the
    step is a separate mechanism; the 6-sigma size of c1 holds only inside
    that parametrisation. The cause of the step is unknown.
  * Error bars are shot noise only (sampling variance plus the attenuation
    correction as written); they contain no allowance for calibration drift.

The lab claims only what its checks test: the recorded-data fit, the circuit
structure, the estimator and calibration on simulators. What the machine does
is REPORTED, not asserted.

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
    python3 labs/lab46_hardware_noise_budget.py                  # dry run
    read -rs IBM_QUANTUM_API_KEY; export IBM_QUANTUM_API_KEY     # hidden prompt
    python3 labs/lab46_hardware_noise_budget.py --live --backend ibm_fez
    python3 labs/lab46_hardware_noise_budget.py --retrieve <job id>

Log files
---------
logs/lab46/lab46_<YYYYmmdd_HHMMSS>.log (plain text, never contains the key)
logs/lab46/last_job.json (job id, backend, shots, layout; no secrets)

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
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.optimize import minimize_scalar

from applog import get_logger

log = get_logger("lab46")

# Cloud plumbing, the estimator, the noise model and the recorded data are all
# reused, not copied: the security-critical code then exists in one place.
try:
    from qiskit_ibm_runtime.fake_provider import FakeFez

    from fezdata import (
        ANGLES,
        LAMBDA_FEZ,
        LAYOUT,
        MEASURED,
        MODEL_ENERGY,
        fit_design,
        fit_lines,
        shortfalls,
    )
    from lab17_mirror_benchmark import (
        BASIS,
        TERMINAL_BAD,
        connect,
        fit_with_error,
        lambda_equiv,
        make_sampler,
        p_zero,
        scrub,
        wait_for,
    )
    from lab21_noise_aware_training import N, NoisyChain, hva_circuit
    from lab44_noise_aware_on_qpu import (
        BASES,
        attenuation,
        basis_circuit,
        calibration_circuits,
        energy_from_counts,
        run_ideal,
    )
except ImportError:  # pragma: no cover
    log.fail("qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime")
    sys.exit(2)

JOB_FILE = Path("logs/lab46/last_job.json")
MAX_SHOTS = 20000  # cap: a typo must not burn the monthly quota
KS = (0, 1, 2, 3)  # mirror depths: layers of the depth-3 circuit
MIRROR_ANGLES = ANGLES[(3, "ideal")]
GATES_PER_LAYER = 15  # five bond gates, three two-qubit gates each
PREP_GATES = 3  # one CNOT per singlet


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


# ----------------------------------------------------------------------------
# Circuits
# ----------------------------------------------------------------------------
def mirror_circuit(k):
    """k layers of the Lab 44 depth-3 circuit, a barrier, then the exact inverse.

    The ideal outcome is all zeros. The barrier stops the transpiler from
    cancelling U against U-dagger (Lab 17 explains why that matters).
    """
    # The same angles for every k, so only the gate count changes. The depth-3
    # ideal angles are used because they are the deepest circuit Lab 44 ran.
    fwd = hva_circuit(N, MIRROR_ANGLES[: 2 * k])
    qc = QuantumCircuit(N)
    qc.compose(fwd, inplace=True)
    qc.barrier()
    qc.compose(fwd.inverse(), inplace=True)
    qc.measure_all()
    return qc


def expected_twoq(key):
    """Two-qubit gates a circuit should keep after transpilation."""
    # Calibration circuits have none; energy circuits are prep plus layers; a
    # mirror is the forward half plus its inverse.
    if key in ("cal0", "cal1"):
        return 0
    if key[0] == "M":
        return 2 * (PREP_GATES + GATES_PER_LAYER * key[1])
    return PREP_GATES + GATES_PER_LAYER * key[0]


def build_jobs():
    """Circuits in a fixed order: calibration, energies, then mirrors."""
    keys, circuits = ["cal0", "cal1"], calibration_circuits()
    # Depth 0 (three singlets) once; depth 1 for both angle sets (replication).
    for depth, kind in ((0, "ideal"), (1, "ideal"), (1, "aware")):
        params = ANGLES[(depth, kind)] if depth else []
        for basis in BASES:
            keys.append((depth, kind, basis))
            circuits.append(basis_circuit(params, basis))
    for k in KS:
        keys.append(("M", k))
        circuits.append(mirror_circuit(k))
    return keys, circuits


# ----------------------------------------------------------------------------
# Recorded data: where did Lab 44's energy go?
# ----------------------------------------------------------------------------
def best_gate_only(chain, sig):
    """Smallest chi-squared reachable by changing ONLY the gate noise lambda."""

    # No offset allowed: this is explanation "gates are noisier than Lab 17".
    def chi2(lam):
        """Chi-squared of the six measured energies for a given lambda."""
        mod, mea, _ = shortfalls(chain, lam)
        return float((((mea - mod) / sig) ** 2).sum())

    res = minimize_scalar(chi2, bounds=(0.001, 0.03), method="bounded")
    return float(res.x), float(res.fun)


def check_recorded(r, ctx):
    """Check 1: what the recorded Lab 44 energies say."""
    log.info("=== Check 1: the recorded Lab 44 energies ===")
    chain = ctx["chain"]
    # The stored angles must still reproduce the energies Lab 44 printed.
    dev = max(
        abs(chain.energy(ANGLES[k], LAMBDA_FEZ) - MODEL_ENERGY[k]) for k in ANGLES
    )
    log.info("stored angles vs printed model energies: max difference %.1e", dev)
    r.append(check(dev < 2e-4, "stored angles reproduce the Lab 44 model energies"))
    # Shortfall = energy lost against the NOISELESS energy of the same circuit,
    # which removes the (depth-dependent) distance of the ansatz from E0.
    mod, mea, sig = shortfalls(chain, LAMBDA_FEZ)
    log.info("shortfall from the noiseless energy (model | measured):")
    for key, m, e in zip(ANGLES, mod, mea):
        log.info("  depth %d %-5s  %.3f | %.3f", key[0], key[1], m, e)
    s1, chi_scale, beta, cov, chi_both = fit_lines(mod, mea, sig)
    # beta = (offset c, scale s); cov gives their uncertainties.
    c, s = beta
    sc, ss = math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])
    lam_gate, chi_gate = best_gate_only(chain, sig)
    log.info("scale only:        s = %.2f, chi2 = %.1f (5 dof)", s1, chi_scale)
    log.info("gate noise only:   lambda = %.4f, chi2 = %.1f", lam_gate, chi_gate)
    log.info(
        "offset + scale:    c = %.3f +/- %.3f, s = %.2f +/- %.2f, chi2 = %.2f (4 dof)",
        c,
        sc,
        s,
        ss,
        chi_both,
    )
    r.append(
        check(
            chi_scale > 25,
            f"a pure scale on the model is rejected (chi2 {chi_scale:.0f})",
        )
    )
    r.append(
        check(chi_gate > 20, f"so is a pure change of gate noise (chi2 {chi_gate:.0f})")
    )
    r.append(
        check(chi_both < 9.49, f"offset + scale fits (chi2 {chi_both:.1f} < 9.49)")
    )
    r.append(check(c > 5 * sc, f"the offset is {c / sc:.1f} sigma from zero"))
    r.append(
        check(0.3 < s < 1.3, f"the depth-dependent loss is about {s:.2f} of the model")
    )
    # Keep the fit: the live analysis and the predictions use it.
    ctx["fit"] = (c, s, cov)
    # Predictions for the depth-0 circuit (exact energy -9) before any run.
    # Depth 0 has no Heisenberg gate, so the model shortfall is tiny (0.033);
    # the offset story therefore predicts almost the whole offset c.
    mod0 = chain.energy([], LAMBDA_FEZ) - chain.energy([], 0.0)
    v = np.array([1.0, mod0])
    ctx["pred_offset"] = (-9.0 + float(v @ beta), math.sqrt(float(v @ cov @ v)))
    ctx["pred_first_layer"] = chain.energy([], LAMBDA_FEZ)
    log.info(
        "depth-0 energy predicted: offset story %.3f +/- %.3f; "
        "first-layer story %.3f",
        *ctx["pred_offset"],
        ctx["pred_first_layer"],
    )
    ctx["lam_offset"] = lam_best_with_offset(chain, sig)
    log.info("offset + gate noise prefers lambda = %.4f", ctx["lam_offset"])


def lam_best_with_offset(chain, sig):
    """Lambda that fits best when a free offset c is allowed."""

    def chi2(lam):
        """Chi-squared after the best offset for this lambda."""
        mod, mea, _ = shortfalls(chain, lam)
        w = 1 / sig**2
        # The weighted mean residual is the best constant offset.
        c = (w * (mea - mod)).sum() / w.sum()
        return float((w * (mea - mod - c) ** 2).sum())

    return float(minimize_scalar(chi2, bounds=(0.0005, 0.02), method="bounded").x)


# ----------------------------------------------------------------------------
# Simulator checks: structure, estimator, calibration
# ----------------------------------------------------------------------------
def check_structure(r, ctx):
    """Check 2: circuits, the depth-0 energy and the ideal mirror."""
    log.info("=== Check 2: circuits on an ideal simulator ===")
    chain = ctx["chain"]
    r.append(
        check(
            abs(chain.energy([], 0.0) + 9.0) < 1e-9,
            "three singlets have energy exactly -9",
        )
    )
    r.append(
        check(
            abs(chain.energy([], LAMBDA_FEZ) + 9.0 * (1 - LAMBDA_FEZ)) < 1e-9,
            "model: one noisy CNOT per singlet shrinks it to -9 (1 - lambda)",
        )
    )
    # The estimator on the depth-0 circuits with a perfect device.
    # One build gives both the circuits and their keys, in hardware order.
    keys, circuits = build_jobs()
    counts = run_ideal(circuits, 20000, 5)
    by_key = dict(zip(keys, counts))
    atten = attenuation(by_key["cal0"], by_key["cal1"])
    e, se = energy_from_counts({b: by_key[(0, "ideal", b)] for b in BASES}, atten)
    log.info("ideal simulator, depth 0: E = %.4f +/- %.4f (exact -9)", e, se)
    r.append(check(abs(e + 9.0) < 4 * se, "depth-0 energy measured without bias"))
    # U then U-dagger returns every shot to all zeros on a perfect device.
    p0 = [p_zero(by_key[("M", k)], N) for k in KS]
    log.info("ideal simulator mirror P0 for k = 0..3: %s", p0)
    r.append(check(min(p0) == 1.0, "mirror circuits return |000000> exactly"))
    # Gate counts after compilation to CNOTs: nothing cancelled, nothing routed.
    # Compile to CNOTs exactly as the calibration will, and count them.
    got = []
    for key, c in zip(keys, circuits):
        t = transpile(c, basis_gates=BASIS, optimization_level=3, seed_transpiler=3)
        got.append(t.count_ops().get("cx", 0))
    want = [expected_twoq(k) for k in keys]
    r.append(
        check(
            got == want,
            "CNOT counts are 3 + 15 d (energy) and 2 (3 + 15 k) (mirror)",
        )
    )
    ctx["keys"] = keys


def mirror_rates(seed, rng, lam):
    """Decay rate s for the mirror family under depolarizing strength `lam`."""
    from qiskit_aer import AerSimulator  # noqa: PLC0415
    from qiskit_aer.noise import NoiseModel, depolarizing_error  # noqa: PLC0415

    # Only the CNOTs are noisy, as in the Lab 17 calibration; single-qubit gates
    # and readout are perfect here, so the table maps gate noise alone.
    nm = NoiseModel()
    if lam > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(lam, 2), ["cx"])
    sim = AerSimulator(noise_model=nm, seed_simulator=seed)
    counts, ncx = [], []
    for k in KS:
        t = transpile(
            mirror_circuit(k),
            basis_gates=BASIS,
            optimization_level=3,
            seed_transpiler=3,
        )
        ncx.append(t.count_ops().get("cx", 0) - expected_twoq(("M", 0)))
        counts.append(sim.run(t, shots=20000).result().get_counts())
    return fit_with_error(ncx, counts, N, rng, boot=20)[0]


def check_calibration(r, ctx):
    """Check 3: decay rate to depolarizing strength, for THIS circuit family."""
    log.info("=== Check 3: calibration table for the mirror family ===")
    # Five strengths bracket the values of interest (0.0029 .. 0.0077).
    rng = ctx["rng"]
    table = {
        lam: mirror_rates(11, rng, lam) for lam in (0.0, 0.002, 0.004, 0.007, 0.012)
    }
    log.info("lambda -> decay per gate: %s", {k: round(v, 5) for k, v in table.items()})
    rates = [table[k] for k in sorted(table)]
    r.append(check(abs(table[0.0]) < 3e-4, "no noise: no decay"))
    r.append(check(all(np.diff(rates) > 0), "decay rises with the error strength"))
    # An independent simulation with a known strength, read back through the table.
    # A different simulator seed makes this a genuine out-of-sample test.
    s_test = mirror_rates(23, rng, 0.005)
    est = lambda_equiv(table, s_test)
    log.info("known lambda 0.005 -> s = %.5f -> read back %.4f", s_test, est)
    r.append(check(abs(est - 0.005) < 0.00075, "known 0.005 recovered within 15%"))
    ctx["table"] = table


# ----------------------------------------------------------------------------
# Cloud side
# ----------------------------------------------------------------------------
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


def parse_layout(text):
    """'auto' -> None; otherwise six comma-separated physical qubit numbers."""
    if text == "auto":
        return None
    layout = [int(x) for x in text.split(",")]
    if len(layout) != N or len(set(layout)) != N:
        raise ValueError(f"--layout needs {N} different qubits or 'auto'")
    return layout


def transpile_all(backend, circuits, layout):
    """Transpile every circuit onto the SAME physical qubits."""
    # With 'auto', place the deepest circuit and reuse its layout, as in Lab 44.
    # Level 3 and a fixed seed make the compilation reproducible between runs.
    kwargs = {"backend": backend, "optimization_level": 3, "seed_transpiler": 3}
    if layout is None:
        deepest = generate_preset_pass_manager(**kwargs).run(circuits[-1])
        layout = deepest.layout.initial_index_layout(filter_ancillas=True)
    pm = generate_preset_pass_manager(initial_layout=layout, **kwargs)
    return [pm.run(c) for c in circuits], list(layout)


def analyse(r, ctx, keys, counts, backend_name, dry):
    """Energies, replication, mirror decay and the verdict on the two stories."""
    # Wording only: tells the reader whether this was real hardware.
    where = "simulated device" if dry else "REAL HARDWARE"
    # Counts arrive in the order the circuits were built.
    by_key = dict(zip(keys, counts))
    # Readout attenuation from the |0..0> and |1..1> circuits, as in Lab 44.
    atten = attenuation(by_key["cal0"], by_key["cal1"])
    log.info("%s %s: readout attenuation %s", backend_name, where, np.round(atten, 3))
    r.append(
        check(bool(np.all((atten > 0.8) & (atten <= 1.0))), "attenuation in (0.8, 1]")
    )
    # en maps (depth, angle set) to (energy, standard error).
    en = {}
    for depth, kind in ((0, "ideal"), (1, "ideal"), (1, "aware")):
        bases = {b: by_key[(depth, kind, b)] for b in BASES}
        en[(depth, kind)] = energy_from_counts(bases, atten)
    # Depth 0 is the cleanest test: its exact energy is -9 and no bond gate acts.
    e0, s0 = en[(0, "ideal")]
    inside = e0 > -9.0 - 4 * s0 and all(
        v > ctx["chain"].e0 - 4 * s for v, s in en.values()
    )
    r.append(check(inside, f"{where}: no energy below its ground-state bound"))
    # Story A or B? Distance of the depth-0 measurement from each prediction.
    pa, sa = ctx["pred_offset"]
    pb = ctx["pred_first_layer"]
    za, zb = (e0 - pa) / math.hypot(s0, sa), (e0 - pb) / s0
    log.info(
        "depth 0 (exact -9): measured %.3f +/- %.3f; offset story %.3f (%+.1f sigma), "
        "first-layer story %.3f (%+.1f sigma)",
        e0,
        s0,
        pa,
        za,
        pb,
        zb,
    )
    if dry:
        log.info("(dry run: the verdict below describes the simulator, not ibm_fez)")
    # Within 3 sigma of a story counts as agreeing with it; otherwise neither.
    if abs(za) < 3:
        near = "agrees with the OFFSET story (loss before any bond gate)"
    elif abs(zb) < 3:
        near = "agrees with the FIRST-LAYER story (no loss at depth 0)"
    else:
        near = "matches NEITHER story: the loss is split"
    log.info("verdict: the depth-0 energy %s", near)
    # How much of the depth-0 shortfall the model does not explain.
    free = ctx["chain"].energy([], 0.0)
    extra = (e0 - free) - (ctx["pred_first_layer"] - free)
    log.info(
        "depth-0 loss %.3f +/- %.3f against %.3f modelled: %.3f +/- %.3f unexplained",
        e0 - free,
        s0,
        ctx["pred_first_layer"] - free,
        extra,
        s0,
    )
    # Replication of the Lab 44 depth-1 circuits and of their sign-flip gap.
    for kind in ("ideal", "aware"):
        old, sold = MEASURED[(1, kind)]
        new, snew = en[(1, kind)]
        log.info(
            "depth 1 %-5s: Lab 44 %.3f, now %.3f (%+.1f sigma apart)",
            kind,
            old,
            new,
            (new - old) / math.hypot(sold, snew),
        )
    # The two depth-1 angle sets differ only by a sign, so the model says equal.
    gap = en[(1, "ideal")][0] - en[(1, "aware")][0]
    sgap = math.hypot(en[(1, "ideal")][1], en[(1, "aware")][1])
    log.info(
        "depth 1 ideal - aware: now %+.3f +/- %.3f; Lab 44 +0.208 +/- 0.103; "
        "model +0.000",
        gap,
        sgap,
    )
    components_report(ctx, en, dry)
    mirror_report(r, ctx, by_key)


def components_report(ctx, en, dry):
    """Refit every energy with a depth-0 loss, a first-layer step and a scale."""
    # Recorded Lab 44 points first, then this run's three (energy, error) pairs.
    chain = ctx["chain"]
    pts = [(k[0], k[1], *MEASURED[k]) for k in ANGLES]
    pts += [(d, kind, *en[(d, kind)]) for (d, kind) in en]
    mod, mea, sig, dep = [], [], [], []
    for depth, kind, energy, err in pts:
        x = [] if depth == 0 else ANGLES[(depth, kind)]
        free = chain.energy(x, 0.0)
        mod.append(chain.energy(x, LAMBDA_FEZ) - free)
        mea.append(energy - free)
        sig.append(err)
        dep.append(depth)
    mod = np.array(mod)
    one = np.ones_like(mod)
    # step = 1 for every circuit that contains at least one bond-gate layer.
    step = (np.array(dep) >= 1).astype(float)
    if dry:
        log.info("(dry run: the three new points come from the simulator)")
    log.info("refit of all %d energies (loss against the noiseless energy):", len(pts))
    models = {
        "c + s*model": ([one, mod], ("c", "s")),
        "c0 + c1*[depth>=1] + s*model": ([one, step, mod], ("c0", "c1", "s")),
    }
    for name, (cols, labels) in models.items():
        beta, cov, chi2 = fit_design(cols, mea, sig)
        parts = ", ".join(
            f"{n} = {b:.3f} +/- {math.sqrt(cov[i, i]):.3f}"
            for i, (n, b) in enumerate(zip(labels, beta))
        )
        log.info(
            "  %-28s %s, chi2 = %.1f (%d dof)", name, parts, chi2, len(pts) - len(beta)
        )


def mirror_report(r, ctx, by_key):
    """Fit the mirror decay on the device and convert it to a lambda."""
    # Probability of returning to |000000> at each mirror depth.
    counts = [by_key[("M", k)] for k in KS]
    p0 = [p_zero(c, N) for c in counts]
    ncx = [GATES_PER_LAYER * 2 * k for k in KS]
    # Rate per gate relative to k = 0, which cancels readout and the singlet
    # preparation (same fit as Lab 17).
    s, ds = fit_with_error(ncx, counts, N, ctx["rng"], boot=300)
    lam = lambda_equiv(ctx["table"], s)
    # The one-sigma band on s, carried through the same table, gives a lambda range.
    lo = lambda_equiv(ctx["table"], max(s - ds, 0.0))
    hi = lambda_equiv(ctx["table"], s + ds)
    log.info("mirror P0 for k = 0..3: %s", np.round(p0, 3))
    log.info(
        "decay per gate %.5f +/- %.5f -> lambda = %.4f (%.4f .. %.4f); "
        "Lab 17: 0.0037, offset fit: %.4f",
        s,
        ds,
        lam,
        lo,
        hi,
        ctx["lam_offset"],
    )
    r.append(check(p0[-1] < p0[0], "mirror survival falls with depth"))
    r.append(check(s >= 0 and math.isfinite(lam), "decay rate and lambda are finite"))


def summarise(r):
    """Log the tally and return 0/1."""
    bad = r.count(False)
    if bad:
        log.fail("%d of %d checks failed", bad, len(r))
        return 1
    log.ok("All %d checks passed", len(r))
    return 0


def finish_retrieve(args, service, secret, r, ctx):
    """--retrieve: fetch a finished job and analyse it."""
    try:
        meta = json.loads(JOB_FILE.read_text())
        keys = build_jobs()[0]
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
    ap = argparse.ArgumentParser(description="Lab 46: where Lab 44's energy went")
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
    ap.add_argument("--max-seconds", type=int, default=120, help="QPU time cap")
    ap.add_argument("--wait", type=int, default=900, help="queue wait (0 = submit)")
    ap.add_argument(
        "--token-env",
        default="IBM_QUANTUM_API_KEY",
        help="NAME of the environment variable holding the key",
    )
    ap.add_argument("--sampler", choices=("v2", "client"), default="v2")
    ap.add_argument("--seed", type=int, default=5, help="bootstrap seed")
    args = ap.parse_args(argv)
    if not 100 <= args.shots <= MAX_SHOTS:
        ap.error(f"--shots must be 100..{MAX_SHOTS}. Got {args.shots}.")
    if args.max_seconds < 10:
        ap.error("--max-seconds must be >= 10.")
    if args.live and args.retrieve:
        ap.error("use --live OR --retrieve, not both.")
    try:
        layout = parse_layout(args.layout)
    except ValueError as exc:
        ap.error(str(exc))
    # The chain model is built once (a 64 x 64 density-matrix simulator).
    ctx = {"rng": np.random.default_rng(args.seed), "chain": NoisyChain(N)}
    # r collects every check's boolean so the exit code reflects all of them.
    r = []
    check_recorded(r, ctx)
    check_structure(r, ctx)
    check_calibration(r, ctx)

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
    keys, circuits = build_jobs()
    # `used` is the layout actually applied (the Lab 44 chain unless --layout).
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
    # Native two-qubit gate names differ by device (cz on Heron, ecr on Eagle).
    twoq = [sum(c.count_ops().get(g, 0) for g in ("cz", "ecr", "cx")) for c in isa]
    want = [expected_twoq(k) for k in keys]
    log.info("two-qubit gates: %s (expected %s)", twoq, want)
    r.append(check(twoq == want, "no gate cancelled and no routing overhead"))

    log.info("=== Check 6: run %d circuits x %d shots ===", len(isa), args.shots)
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
            "python3 labs/lab46_hardware_noise_budget.py --retrieve %s",
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
    analyse(r, ctx, keys, counts, backend.name, not args.live)
    return summarise(r)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
