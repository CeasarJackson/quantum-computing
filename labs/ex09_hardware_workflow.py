#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/ex09_hardware_workflow.py

Purpose
-------
Lab 9 optional exercises, each as predict-then-verify:
  1. Other calibration days (BACKEND_SEED 1, 2, 7): does the error-sum
     layout heuristic pick the layout the benchmark picks?
  2. A longer CNOT fold (noise scales 1, 3, 5, 7): linear versus exponential
     zero-noise extrapolation on <Z0 Z4> of GHZ(5), averaged over seeds.
  3. <XXX> on GHZ(3) (H on every qubit, then Z parity) versus the Z-basis
     success rate: what each one can and cannot certify.
  4. Lab 6's dense-unitary Shor circuit pushed through to_hardware and the
     noisy simulator: CNOT cost and success versus the hand-compiled circuit.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/ex09_hardware_workflow.py [--shots 50000]

Log files
---------
logs/ex09/ex09_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid command-line usage.
130 User interrupted execution.
===============================================================================
"""

import sys

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator

import lab09_hardware_workflow as lab09
import lab10_capstone_shor as lab10
from applog import get_logger
from exlib import Tally, base_parser

log = get_logger("ex09")


def backend_for(seed):
    """Lab 9's 5-qubit line chip, but with another calibration day."""
    # A different seed re-draws the simulated calibration (gate and readout
    # errors), standing in for the same chip on another day.
    return GenericBackendV2(
        num_qubits=5, coupling_map=lab09.LINE, basis_gates=lab09.BASIS, seed=seed
    )


def ex1(t, shots):
    """Exercise 1: layout heuristic versus benchmark on three chips."""
    t.ask("Exercise 1: layout heuristic vs benchmark, seeds 1, 2, 7")
    t.predict(
        "the error-sum heuristic ignores thermal relaxation, so it"
        " will NOT always pick the benchmark winner (Lab 9 saw one"
        " disagreement already); only measurement settles it"
    )
    layouts = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    agree = 0
    spreads = []
    for seed in (1, 2, 7):
        # Each seed is a different simulated calibration day.
        be = backend_for(seed)
        sim = AerSimulator.from_backend(be, seed_simulator=42)
        rep = lab09.device_report(be)

        def cost(lay, rep=rep):
            """Calibration-sheet cost: CNOT errors plus readout errors."""
            # A layout is a path of physical qubits: sum the CNOT error of each
            # neighbouring pair along the path.
            edges = sum(rep["cx"][tuple(sorted(p))] for p in zip(lay, lay[1:]))
            return edges + sum(rep["readout"][q] for q in lay)

        # Benchmark: run the same circuit on every candidate layout.
        measured = {}
        for lay in layouts:
            counts = (
                sim.run(
                    lab09.to_hardware(be, lab09.chain_ghz(3), layout=lay),
                    shots=4 * shots,
                )
                .result()
                .get_counts()
            )
            measured[tuple(lay)] = lab09.ghz_success(counts, 3)
        pick, best = tuple(min(layouts, key=cost)), max(measured, key=measured.get)
        # Best-minus-worst success is the practical value of choosing well.
        spread = max(measured.values()) - min(measured.values())
        spreads.append(spread)
        agree += pick == best
        t.note(
            "seed %d: heuristic %s, benchmark %s (%s); P = %s; spread %.4f",
            seed,
            list(pick),
            list(best),
            "agree" if pick == best else "DISAGREE",
            {k: round(v, 4) for k, v in measured.items()},
            spread,
        )
    t.note("heuristic matched the benchmark on %d of 3 chips", agree)
    t.check(
        agree < 3,
        "the heuristic fails to match the benchmark on at"
        " least one chip: calibration sheets are a guide, not an oracle",
    )
    t.check(
        all(s > 0.001 for s in spreads),
        "every chip shows a measurable layout spread (> 0.001)",
    )


def zne(vals, scales):
    """Linear and exponential zero-noise intercepts from (scale, value) pairs."""
    # Order 1 polynomial fit: index [1] is the intercept, i.e. the noise
    # extrapolated back to scale 0.
    lin = float(np.polyfit(scales, vals, 1)[1])
    # v = A exp(-b s) is a straight line in log v; its intercept is log A.
    expo = float(np.exp(np.polyfit(scales, np.log(vals), 1)[1]))
    return lin, expo


def ex2(t, shots):
    """Exercise 2: longer fold, linear versus exponential extrapolation."""
    t.ask("Exercise 2: fold k = 0..3 (noise x1, x3, x5, x7), <Z0 Z4> on GHZ(5)")
    t.predict(
        "noise acts multiplicatively per gate, so the decay is closer to"
        " exponential than linear and the exponential fit should land"
        " nearer 1; both beat the raw value; a 4th point adds little"
        " because extrapolation amplifies shot noise"
    )
    be = lab09.make_backend()
    base = lab09.to_hardware(be, lab09.chain_ghz(5))
    # Folding k times multiplies the CNOT count by 2k+1.
    scales = [1, 3, 5, 7]
    errs = {"raw": [], "lin3": [], "exp3": [], "lin4": [], "exp4": []}
    for seed in range(1, 6):
        # A new simulator seed per repeat averages out shot noise.
        sim = AerSimulator.from_backend(be, seed_simulator=seed)
        vals = [
            lab09.zz_expectation(
                sim.run(lab09.fold_cx(base, k), shots=shots).result().get_counts(), 0, 4
            )
            for k in range(4)
        ]
        l3, e3 = zne(vals[:3], scales[:3])
        l4, e4 = zne(vals, scales)
        for key, v in (
            ("raw", vals[0]),
            ("lin3", l3),
            ("exp3", e3),
            ("lin4", l4),
            ("exp4", e4),
        ):
            errs[key].append(abs(1 - v))
        t.note(
            "seed %d: values %s -> lin %.4f exp %.4f (4 pts)",
            seed,
            np.round(vals, 4),
            l4,
            e4,
        )
    # Average over seeds: one run is too noisy to rank the fits.
    mean = {k: float(np.mean(v)) for k, v in errs.items()}
    t.note("mean |error| vs ideal 1: %s", {k: round(v, 4) for k, v in mean.items()})
    t.check(
        all(mean[k] < mean["raw"] for k in ("lin3", "exp3", "lin4", "exp4")),
        "every extrapolation beats the raw (unmitigated) value",
    )
    t.check(
        mean["exp4"] < mean["lin4"],
        "exponential extrapolation is closer to 1 than linear (4 points)",
    )
    t.note(
        "adding the x7 point changes the mean error by %+.4f (linear) and"
        " %+.4f (exponential)",
        mean["lin4"] - mean["lin3"],
        mean["exp4"] - mean["exp3"],
    )


def parity(counts):
    """<Z...Z> from counts: P(even number of 1s) minus P(odd)."""
    # Total shots, to turn the counts into probabilities below.
    total = sum(counts.values())
    return (
        sum(
            c * (1 if bin(int(k, 2)).count("1") % 2 == 0 else -1)
            for k, c in counts.items()
        )
        / total
    )


def ex3(t, shots):
    """Exercise 3: <XXX> versus Z-basis success."""
    t.ask("Exercise 3: <XXX> on GHZ(3)")
    t.predict(
        "ideal GHZ has <XXX> = +1, while a classical 50/50 mix of 000"
        " and 111 has <XXX> = 0 even though its Z-basis success is also"
        " 100%. So <XXX> near 1 certifies coherence; it should sit"
        " slightly below the Z-basis success because dephasing hits it"
    )
    be = lab09.make_backend()
    sim = AerSimulator.from_backend(be, seed_simulator=42)
    # Same GHZ circuit read in two bases.
    ghz = QuantumCircuit(3)
    ghz.h(0)
    ghz.cx(0, 1)
    ghz.cx(1, 2)
    zc, xc = ghz.copy(), ghz.copy()
    zc.measure_all()
    # H maps the X basis onto Z so ordinary readout measures X.
    xc.h([0, 1, 2])  # rotate X onto Z, then read parities
    xc.measure_all()
    z_counts = sim.run(lab09.to_hardware(be, zc), shots=shots).result().get_counts()
    x_counts = sim.run(lab09.to_hardware(be, xc), shots=shots).result().get_counts()
    succ, xxx = lab09.ghz_success(z_counts, 3), parity(x_counts)
    # Shot noise scale for a +/-1 observable.
    sigma = 1 / np.sqrt(shots)
    t.note("Z-basis success %.4f; <XXX> = %.4f (shot sigma ~ %.4f)", succ, xxx, sigma)
    # Exact contrast from linear algebra: the mixture has zero off-diagonals.
    # Exact linear algebra for the classical mixture.
    xmat = np.array([[0, 1], [1, 0]])
    xxx_op = np.kron(np.kron(xmat, xmat), xmat)
    mix = np.zeros((8, 8))
    mix[0, 0] = mix[7, 7] = 0.5
    t.check(
        abs(np.trace(mix @ xxx_op)) < 1e-12,
        "a classical 000/111 mixture has <XXX> = 0 (but 100% Z success)",
    )
    t.check(xxx > 0.9, f"<XXX> = {xxx:.3f} > 0.9: genuine GHZ coherence")
    t.check(
        xxx < succ - 2 * sigma,
        "<XXX> sits below the Z-basis success: coherence is more fragile",
    )
    t.check(
        abs(parity(z_counts)) < 0.1,
        "<ZZZ> ~ 0 on GHZ(3), as theory says (odd operator)",
    )


def ex4(t, shots):
    """Exercise 4: dense-unitary Shor through the hardware workflow."""
    t.ask("Exercise 4: Lab 6's dense Shor circuit on the noisy 7-qubit chip")
    t.predict(
        "compiling dense unitaries costs thousands of CNOTs and drives"
        " success to the 0.25 garbage floor, versus ~100 CNOTs for the"
        " hand-compiled circuit (we will see how much of this holds)"
    )
    # The 7-qubit chip is needed: 3 counting + 4 target qubits.
    dev = lab10.make_device()
    sim = AerSimulator.from_backend(dev, seed_simulator=42)
    # Lab 6's circuit: each controlled multiplier is a dense unitary.
    dense = lab10.shor_15_dense()
    dense.add_register(ClassicalRegister(lab10.T))
    dense.measure(range(lab10.T), range(lab10.T))  # counting qubits only
    # Compile the dense version through the same workflow.
    t_dense = lab09.to_hardware(dev, dense)
    comp = lab10.shor_15_compiled()
    t_comp = lab09.to_hardware(dev, comp)
    # Gate count with NO coupling constraint: the cost of the algorithm
    # itself, before the router adds SWAPs for the chip's topology.
    flat = transpile(
        comp, basis_gates=lab09.BASIS, optimization_level=1, seed_transpiler=3
    )
    cx_dense = t_dense.count_ops().get("cx", 0)
    cx_comp = t_comp.count_ops().get("cx", 0)
    t.note(
        "CNOTs after routing: dense %d (depth %d) vs compiled %d (depth %d);"
        " the compiled circuit needs %d before routing",
        cx_dense,
        t_dense.depth(),
        cx_comp,
        t_comp.depth(),
        flat.count_ops().get("cx", 0),
    )
    # Success = probability mass on outcomes that yield factors.
    p_dense = lab10.success_prob(lab10.run_probs(sim, t_dense, shots))
    p_comp = lab10.success_prob(lab10.run_probs(sim, t_comp, shots))
    # Binomial shot noise at the garbage floor.
    sigma = float(np.sqrt(0.25 / shots))
    t.note(
        "success: dense %.4f, compiled %.4f, floor 0.25, ideal 0.5 (sigma" " ~ %.4f)",
        p_dense,
        p_comp,
        sigma,
    )
    t.note(
        "prediction check: 'thousands of CNOTs' was %s; 'dense at the" " floor' was %s",
        "RIGHT" if cx_dense >= 1000 else "TOO PESSIMISTIC",
        "RIGHT" if p_dense < 0.25 + 4 * sigma else "WRONG (dense is above it)",
    )
    t.check(
        cx_dense > 5 * cx_comp,
        f"dense costs {cx_dense / cx_comp:.1f}x the CNOTs of the compiled circuit",
    )
    t.check(
        p_dense < p_comp - 4 * sigma,
        "dense success is measurably below the compiled success",
    )
    t.check(
        p_comp > 0.25 + 4 * sigma,
        "compiled circuit stays clearly above the 0.25 garbage floor",
    )


def main(argv=None):
    """Run all four Lab 9 exercises; return the exit code."""
    ap = base_parser("Lab 9 exercises")
    ap.add_argument("--shots", type=int, default=50000)
    args = ap.parse_args(argv)
    if args.shots < 5000:
        ap.error(f"--shots must be >= 5000. Got {args.shots}.")
    t = Tally(log)
    ex1(t, args.shots)
    ex2(t, args.shots)
    ex3(t, args.shots)
    ex4(t, args.shots)
    return t.finish()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
