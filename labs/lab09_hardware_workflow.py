#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/lab09_hardware_workflow.py

Purpose
-------
Lab 9 walks the complete workflow for running on a real quantum processor, but
against a SIMULATED device, so it needs no account, token or network. The
device is Qiskit's GenericBackendV2: 5 qubits on a line (0-1-2-3-4), a native
gate set {cx, id, rz, sx, x}, and per-qubit / per-gate calibration data
(T1, T2, readout error, gate error). Aer turns that calibration into a noise
model, so noisy runs here behave like a small superconducting chip.

Everything below carries over to real hardware unchanged except make_backend(),
which would instead fetch a device from a cloud service. That service needs a
credential, which must be read from an environment variable and never be
logged, printed, or passed on a command line.

The lab verifies:
  1. Calibration data is physical (probabilities in range, T2 <= 2*T1).
  2. Transpilation: only native gates, only coupled qubit pairs; routing a
     star-shaped circuit onto a line costs extra CNOTs.
  3. Transpiling preserves the circuit's ideal output distribution.
  4. Optimization levels 0-3 and their effect on gate count.
  5. Noise grows with circuit size and with routing overhead.
  6. Qubit choice matters, and a naive error-sum heuristic can mis-rank them.
  7. Readout-error mitigation (calibration matrix inversion) helps.
  8. Zero-noise extrapolation (CNOT folding) helps against gate errors.

Security
--------
Handles no secrets and makes no network calls. Cloud use would need an API
token supplied through the environment only.

Usage
-----
    python3 labs/lab09_hardware_workflow.py
    python3 labs/lab09_hardware_workflow.py --shots 200000 --seed 7

Log files
---------
logs/lab09/lab09_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid command-line usage.
130 User interrupted execution.
===============================================================================
"""

import argparse
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from applog import get_logger

log = get_logger("lab09")

# A real chip executes only a few native gates; everything else is compiled
# into them. This set is typical of IBM superconducting devices.
BASIS = ["cx", "id", "rz", "sx", "x"]  # the chip's native gates
# Two-qubit gates exist only between physically coupled neighbours. A line is
# the sparsest useful topology, which makes routing costs easy to see.
LINE = [(0, 1), (1, 2), (2, 3), (3, 4)]  # which qubit pairs can interact
BACKEND_SEED = 7  # fixes the 'calibration day'
NATIVE_OPS = set(BASIS) | {"measure", "barrier"}


def make_backend():
    """The simulated device. A cloud backend would be returned here instead."""
    return GenericBackendV2(
        num_qubits=5, coupling_map=LINE, basis_gates=BASIS, seed=BACKEND_SEED
    )


def device_report(backend):
    """Read calibration data out of the backend's Target, like a real chip."""
    # The Target is the modern, uniform place for per-gate error rates; the
    # qubit properties hold the coherence times in seconds (we print in us).
    tgt = backend.target
    rep = {"readout": {}, "cx": {}, "sx": {}, "t1": {}, "t2": {}}
    for q in range(backend.num_qubits):
        rep["readout"][q] = tgt["measure"][(q,)].error
        rep["sx"][q] = tgt["sx"][(q,)].error
        props = backend.qubit_properties(q)
        rep["t1"][q], rep["t2"][q] = props.t1 * 1e6, props.t2 * 1e6  # us
    for edge in LINE:
        rep["cx"][edge] = tgt["cx"][edge].error
    return rep


def chain_ghz(n):
    """GHZ by a CNOT chain 0-1, 1-2, ...: every gate is between neighbours."""
    qc = QuantumCircuit(n)
    qc.h(0)  # superposition on qubit 0
    for i in range(n - 1):  # then copy it down the chain
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def star_ghz(n):
    """GHZ by fan-out from qubit 0: needs qubit 0 to touch every other one.

    On a line that is impossible without SWAPs (each = 3 CNOTs), so this is
    the same state at a much higher cost in noise.
    """
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    qc.measure_all()
    return qc


def mixed_circuit():
    """A lopsided 4-qubit circuit with non-adjacent CNOTs (no GHZ symmetry)."""
    # Pairs (0,3), (1,3), (2,0), (3,1) are far apart on a line, forcing the
    # router to insert SWAPs. Unequal outcome probabilities mean a wrong
    # qubit permutation would show up as a large distance, unlike GHZ.
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 3)
    qc.ry(0.7, 1)
    qc.cx(1, 3)
    qc.rz(0.4, 3)
    qc.cx(2, 0)
    qc.h(2)
    qc.cx(3, 1)
    return qc


def to_hardware(backend, qc, layout=None, level=1):
    """Transpile = pick physical qubits, route, and rewrite in native gates."""
    # seed_transpiler fixes the stochastic layout/routing passes so reruns
    # produce the same circuit and our numbers stay reproducible.
    return transpile(
        qc, backend, initial_layout=layout, optimization_level=level, seed_transpiler=3
    )


def probs_from_counts(counts, n):
    """Counts dict -> length-2^n vector; index bit k is classical bit k."""
    # Keys are bit strings with classical bit 0 rightmost (Qiskit order), so
    # int(key, 2) puts qubit k at bit k, matching Statevector.probabilities.
    vec = np.zeros(2**n)
    for key, c in counts.items():
        vec[int(key, 2)] += c
    return vec / vec.sum()


def ghz_success(counts, n):
    """Fraction of shots that landed on the two GHZ outcomes 0..0 and 1..1."""
    return (counts.get("0" * n, 0) + counts.get("1" * n, 0)) / sum(counts.values())


def zz_expectation(counts, a, b):
    """<Z_a Z_b> = P(bits equal) - P(bits differ), from measured counts."""
    total = 0
    for key, c in counts.items():
        bits = key[::-1]  # bits[k] is classical bit k
        total += c if bits[a] == bits[b] else -c
    return total / sum(counts.values())


def fold_cx(tqc, k):
    """Noise scaling by gate folding: each CNOT becomes CNOT (CNOT CNOT)^k.

    CNOT is its own inverse, so ideally nothing changes, but the device now
    applies 2k+1 times as many noisy CNOTs. Measuring at several noise scales
    lets us extrapolate back to the zero-noise value.
    """
    # copy_empty_like keeps registers and the layout record, so the folded
    # circuit still maps onto the same physical qubits as the original.
    out = tqc.copy_empty_like()
    for inst in tqc.data:
        reps = 2 * k + 1 if inst.operation.name == "cx" else 1
        for _ in range(reps):
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def check(cond, label):
    """Log PASS/FAIL and return the boolean for tallying."""
    (log.ok if cond else log.fail)("%s", label)
    return bool(cond)


def check_device(rep, r):
    """Check 1: print the calibration sheet and sanity-check it."""
    log.info("=== Check 1: calibration data ===")
    for q in range(5):
        log.info(
            "qubit %d: T1 %6.1f us  T2 %6.1f us  readout err %.4f  " "sx err %.5f",
            q,
            rep["t1"][q],
            rep["t2"][q],
            rep["readout"][q],
            rep["sx"][q],
        )
    for edge, err in rep["cx"].items():
        log.info("cx %s error %.4f", edge, err)
    # Gather every error rate into one list so a single range test covers all.
    errs = (
        list(rep["readout"].values())
        + list(rep["sx"].values())
        + list(rep["cx"].values())
    )
    r.append(
        check(
            all(0 < e < 0.5 for e in errs),
            "every error rate is a probability in (0, 0.5)",
        )
    )
    # Dephasing cannot be faster than twice the relaxation time (T2 <= 2*T1);
    # a violation would mean unphysical calibration data.
    r.append(
        check(
            all(rep["t2"][q] <= 2 * rep["t1"][q] for q in range(5)),
            "T2 <= 2*T1 on every qubit",
        )
    )
    r.append(
        check(
            max(rep["cx"].values()) > 3 * min(rep["cx"].values()),
            "CNOT quality varies by >3x across the chip: layout matters",
        )
    )


def check_transpile(backend, sim, shots, r):
    """Checks 2-4: constraints, correctness and optimization levels."""
    # Allowed pairs, in either order: the transpiler fixes CNOT direction by
    # wrapping with Hadamards, so direction itself is not a constraint here.
    pairs = {tuple(e) for e in backend.coupling_map.get_edges()}
    log.info("=== Check 2: native gates, coupled pairs, routing cost ===")
    stats = {}
    for name, make in (("chain", chain_ghz), ("star", star_ghz)):
        tqc = to_hardware(backend, make(5))
        ops = set(tqc.count_ops())
        ok_pairs = True
        # find_bit maps a circuit qubit to its index, i.e. the physical qubit.
        for inst in tqc.data:
            if inst.operation.num_qubits == 2 and inst.operation.name != "barrier":
                a, b = (tqc.find_bit(q).index for q in inst.qubits)
                ok_pairs &= (a, b) in pairs or (b, a) in pairs
        stats[name] = (tqc.count_ops().get("cx", 0), tqc.depth())
        log.info("GHZ(5) %-5s: %2d cx, depth %2d", name, *stats[name])
        r.append(check(ops <= NATIVE_OPS, f"{name}: only native gates {sorted(ops)}"))
        r.append(check(ok_pairs, f"{name}: every 2-qubit gate on a coupled pair"))
    r.append(
        check(
            stats["star"][0] > stats["chain"][0],
            f"routing the star costs CNOTs: {stats['star'][0]} vs "
            f"{stats['chain'][0]}",
        )
    )

    log.info("=== Check 3: transpiling preserves the answer ===")
    # Compare the exact ideal distribution of the original circuit with an
    # ideal (noise-free) simulation of the transpiled one. They must agree up
    # to sampling noise, otherwise routing or basis translation broke it.
    qc = mixed_circuit()
    exact = Statevector.from_instruction(qc).probabilities()
    meas = qc.copy()
    meas.measure_all()
    counts = (
        AerSimulator()
        .run(to_hardware(backend, meas, level=2), shots=4 * shots)
        .result()
        .get_counts()
    )
    tvd = 0.5 * float(np.abs(probs_from_counts(counts, 4) - exact).sum())
    log.info("total variation distance (ideal) = %.4f", tvd)
    r.append(check(tvd < 0.01, "ideal distribution unchanged by transpilation"))

    log.info("=== Check 4: optimization levels 0-3 on GHZ(5) star ===")
    # Level 0 does only what is required; higher levels also optimise the
    # layout and cancel gates. Fewer CNOTs means fewer chances for error.
    cx_counts = []
    for level in range(4):
        tqc = to_hardware(backend, star_ghz(5), level=level)
        cx_counts.append(tqc.count_ops().get("cx", 0))
        log.info("level %d: %2d cx, depth %2d", level, cx_counts[-1], tqc.depth())
    r.append(
        check(
            cx_counts[3] <= cx_counts[0], "level 3 never uses more CNOTs than level 0"
        )
    )


def check_noise(backend, sim, shots, rep, r):
    """Checks 5-6: how noise scales, and calibration-aware qubit choice."""
    log.info("=== Check 5: noise grows with size and routing ===")
    # Each extra CNOT is another chance for error, so success must fall with n.
    succ = {}
    for n in (2, 3, 4, 5):
        counts = (
            sim.run(to_hardware(backend, chain_ghz(n)), shots=shots)
            .result()
            .get_counts()
        )
        succ[n] = ghz_success(counts, n)
        log.info("GHZ(%d) chain: P(0..0 or 1..1) = %.4f", n, succ[n])
    counts = (
        sim.run(to_hardware(backend, star_ghz(5)), shots=shots).result().get_counts()
    )
    star5 = ghz_success(counts, 5)
    log.info("GHZ(5) star : P(0..0 or 1..1) = %.4f", star5)
    r.append(check(succ[2] > succ[5], "bigger GHZ is noisier (2 > 5 qubits)"))
    r.append(
        check(
            succ[5] > star5,
            "routing overhead costs fidelity "
            f"(chain {succ[5]:.4f} > star {star5:.4f})",
        )
    )

    log.info("=== Check 6: qubit choice matters; benchmark, don't guess ===")
    layouts = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]

    def cost(lay):
        """Naive heuristic: sum of CNOT and readout error on the chosen qubits."""
        edges = sum(rep["cx"][tuple(sorted(p))] for p in zip(lay, lay[1:]))
        return edges + sum(rep["readout"][q] for q in lay)

    # Run the SAME circuit on every candidate chain with many shots, so the
    # shot noise (about 0.0003) is far below the differences we look for.
    measured = {}
    for lay in layouts:
        counts = (
            sim.run(to_hardware(backend, chain_ghz(3), layout=lay), shots=4 * shots)
            .result()
            .get_counts()
        )
        measured[tuple(lay)] = ghz_success(counts, 3)
        log.info(
            "layout %s: heuristic cost %.4f, measured P = %.4f",
            lay,
            cost(lay),
            measured[tuple(lay)],
        )
    spread = max(measured.values()) - min(measured.values())
    pick = min(layouts, key=cost)
    best = max(measured, key=measured.get)
    # The device model also includes thermal relaxation (T1/T2 over each
    # gate's duration), which the error sum ignores. If the two rankings
    # disagree, that is the lesson: calibration numbers are a guide only.
    log.info(
        "heuristic picks %s, benchmark picks %s -> %s",
        pick,
        list(best),
        "agree" if tuple(pick) == best else "DISAGREE",
    )
    r.append(
        check(
            spread > 0.002, f"layouts differ measurably: best-worst spread {spread:.4f}"
        )
    )


def calibration_matrix(backend, sim, layout, shots):
    """A[i, j] = P(read integer i | prepared integer j), one run per column.

    Preparing each basis state with X gates and measuring it isolates the
    readout channel: any other outcome than the prepared one is an error.
    """
    # 2^n calibration circuits: affordable for 3 qubits, impossible at 50,
    # which is why real tools assume independent per-qubit errors instead.
    n = len(layout)
    mat = np.zeros((2**n, 2**n))
    for j in range(2**n):
        qc = QuantumCircuit(n)
        for k in range(n):
            if (j >> k) & 1:  # bit k of j sets qubit k
                qc.x(k)
        qc.measure_all()
        # Level 0 so the optimizer cannot remove the X gates or reorder qubits.
        tqc = to_hardware(backend, qc, layout=layout, level=0)
        counts = sim.run(tqc, shots=shots).result().get_counts()
        mat[:, j] = probs_from_counts(counts, n)
    return mat


def mitigate(mat, probs):
    """Undo readout error: solve A p_true = p_measured, keep a valid p."""
    sol, *_ = np.linalg.lstsq(mat, probs, rcond=None)
    sol = np.clip(sol, 0, None)  # noise can make entries slightly <0
    return sol / sol.sum()


def check_mitigation(backend, sim, shots, rep, r):
    """Check 7: readout-error mitigation on a 3-qubit GHZ state."""
    log.info("=== Check 7: readout-error mitigation ===")
    # Calibrate on exactly the qubits the experiment will use: readout error
    # differs per qubit, so a matrix from other qubits would be wrong.
    layout = [2, 3, 4]
    mat = calibration_matrix(backend, sim, layout, shots)
    r.append(
        check(
            np.allclose(mat.sum(axis=0), 1),
            "each column of the calibration matrix sums to 1",
        )
    )
    # |000> uses no gates, so P(read 000) should be the product of the
    # per-qubit readout fidelities from the calibration sheet.
    expect = np.prod([1 - rep["readout"][q] for q in layout])
    log.info("A[000,000] = %.4f (calibration sheet predicts %.4f)", mat[0, 0], expect)
    r.append(
        check(
            abs(mat[0, 0] - expect) < 0.003,
            "measured readout matches the calibration sheet",
        )
    )
    counts = (
        sim.run(to_hardware(backend, chain_ghz(3), layout=layout), shots=shots)
        .result()
        .get_counts()
    )
    # Mitigation fixes only readout error. Gate error remains, which is why
    # the mitigated value rises toward, but not all the way to, 1.
    raw = probs_from_counts(counts, 3)
    fixed = mitigate(mat, raw)
    p_raw, p_fix = raw[0] + raw[7], fixed[0] + fixed[7]
    log.info("GHZ(3): raw P = %.4f, mitigated P = %.4f (ideal 1.0000)", p_raw, p_fix)
    r.append(
        check(1 - p_fix < 1 - p_raw, "mitigation moves the result toward the ideal")
    )


def check_zne(backend, sim, shots, r):
    """Check 8: zero-noise extrapolation by CNOT folding."""
    log.info("=== Check 8: zero-noise extrapolation, <Z0 Z4> on GHZ(5) ===")
    # Fold the already-transpiled circuit: transpiling again could cancel the
    # extra CNOT pairs and silently remove the noise we are adding.
    base = to_hardware(backend, chain_ghz(5))
    scales, vals = [1, 3, 5], []
    for k in range(3):
        counts = sim.run(fold_cx(base, k), shots=shots).result().get_counts()
        vals.append(zz_expectation(counts, 0, 4))
        log.info("noise x%d: <Z0 Z4> = %.4f", scales[k], vals[-1])
    # The value falls roughly linearly with the noise scale, so the fitted
    # line's intercept at scale 0 estimates the noise-free result.
    _, icept = np.polyfit(scales, vals, 1)
    log.info("linear fit to zero noise: %.4f (ideal 1.0000, raw %.4f)", icept, vals[0])
    r.append(
        check(
            vals[0] > vals[1] > vals[2],
            "more folded CNOTs means more noise (monotone decay)",
        )
    )
    r.append(
        check(
            abs(1 - icept) < abs(1 - vals[0]),
            f"extrapolated {icept:.4f} is closer to 1 than raw " f"{vals[0]:.4f}",
        )
    )


def main(argv=None):
    """Run every Lab 9 check; return the process exit code (0 = all passed)."""
    ap = argparse.ArgumentParser(description="Lab 9: hardware workflow")
    ap.add_argument("--shots", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if args.shots < 5000:
        ap.error(f"--shots must be >= 5000. Got {args.shots}.")
    # AerSimulator.from_backend builds a noise model (gate errors, thermal
    # relaxation, readout error) from the device's calibration data.
    backend = make_backend()
    sim = AerSimulator.from_backend(backend, seed_simulator=args.seed)
    rep = device_report(backend)
    r = []
    check_device(rep, r)
    check_transpile(backend, sim, args.shots, r)
    check_noise(backend, sim, args.shots, rep, r)
    check_mitigation(backend, sim, args.shots, rep, r)
    check_zne(backend, sim, args.shots, r)

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
