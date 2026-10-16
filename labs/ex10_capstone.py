#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/ex10_capstone.py

Purpose
-------
Lab 10 optional exercises, each as predict-then-verify:
  1. Other bases (a = 2, 4, 8, 11, 13, 14 for N = 15): compile each as bit
     rotations (+ NOT), verify against the dense unitary, then run noisy.
  2. Keep the "identity" multiplier: insert barriers so its cswap pairs
     survive transpilation, and measure the cost in CNOTs and success.
  3. Scale the chip's CNOT error until success falls to the 0.25 floor.
  4. Is the sign of the readout-mitigation effect stable across simulator
     seeds and with a very large shot count?

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/ex10_capstone.py [--shots 100000]

Log files
---------
logs/ex10/ex10_<YYYYmmdd_HHMMSS>.log

Exit Codes
----------
0   Success (all checks passed).
1   One or more checks failed.
2   Invalid command-line usage.
130 User interrupted execution.
===============================================================================
"""

import math
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

import lab10_capstone_shor as lab10
from applog import get_logger
from exlib import Tally, base_parser
from lab05_qft_qpe import qft_circuit
from lab06_shor import factors_from_measurement, modmult_matrix
from lab09_hardware_workflow import to_hardware

log = get_logger("ex10")

N, T, W = 15, lab10.T, lab10.W


def multiplier_plan(m):
    """(rotation, negate) such that y -> m*y mod 15 is rotate-left then NOT.

    Powers of two are pure rotations (2y shifts every bit up, wrapping bit 3
    to bit 0 because 16 = 1 mod 15). Negation y -> 15 - y is a NOT on all
    four bits, so m = -2^k is a rotation followed by NOT.
    """
    # Multiplying by 1 needs no gates at all.
    if m == 1:
        return 0, False
    # Check m against +2^k and its negation 15 - 2^k.
    for k in range(4):
        if m == 2**k:
            return k, False
        if m == 15 - 2**k:
            return k, True
    raise ValueError(f"{m} is not +-2^k mod 15")


def rotate_left(qc, ctl, tq, k):
    """Controlled rotate-left by k places (k = 1, 2, 3) with cswaps."""
    # k=1 is new here; k=2 and 3 come from Lab 10.
    if k == 1:
        # Move bit 3 down to bit 0 by three neighbour swaps from the top.
        for a, b in ((2, 3), (1, 2), (0, 1)):
            qc.cswap(ctl, tq[a], tq[b])
    else:
        lab10.controlled_rotations(qc, ctl, tq, k)  # the lab's k = 2, 3


def compiled_for(a, measure=True, keep_identity=False):
    """Compiled order-finding circuit for base a, N = 15, 3 counting qubits."""
    # Registers: 3 counting + 4 target qubits, 3 classical bits.
    qc = QuantumCircuit(T + W, T if measure else 0)
    tq = list(range(T, T + W))
    qc.x(tq[0])
    qc.h(range(T))
    for j in range(T):
        # Counting qubit j controls multiplication by a^(2^j) mod 15.
        rot, neg = multiplier_plan(pow(a, 2**j, N))
        if rot == 0 and not neg and keep_identity:
            # Multiplying by 1 as two identical cswap layers: the optimizer
            # would cancel them, so a barrier between the layers pins them.
            for _ in range(2):
                qc.cswap(j, tq[0], tq[2])
                qc.cswap(j, tq[1], tq[3])
                qc.barrier()
        if rot:
            rotate_left(qc, j, tq, rot)
        if neg:
            for q in tq:
                qc.cx(j, q)
    # Inverse QFT turns the phase pattern into a readable integer.
    qc.compose(qft_circuit(T, inverse=True), qubits=list(range(T)), inplace=True)
    if measure:
        qc.measure(range(T), range(T))
    return qc


def dense_for(a):
    """Lab 6 style circuit with dense modular-multiplication unitaries."""
    # T counting qubits plus W target qubits; the counting register sits
    # on the low indices, the modular-multiplication target above it.
    qc = QuantumCircuit(T + W)
    tq = list(range(T, T + W))
    qc.x(tq[0])
    qc.h(range(T))
    for j in range(T):
        gate = UnitaryGate(modmult_matrix(pow(a, 2**j, N), N, W))
        qc.append(gate.control(1), [j] + tq)
    qc.compose(qft_circuit(T, inverse=True), qubits=list(range(T)), inplace=True)
    return qc


def good_outcomes(a):
    """Counting outcomes m whose post-processing yields factors for base a."""
    # Post-processing decides which outcomes count as success.
    return [m for m in range(2**T) if factors_from_measurement(m, T, a, N)]


def success(probs, a):
    """Probability that one run with base a yields factors."""
    # Sum the probability of every counting outcome that yields a factor;
    # good_outcomes() lists those outcomes for this base a.
    return float(sum(probs[m] for m in good_outcomes(a)))


def noisy_sim(backend, seed=42):
    """Density-matrix simulator carrying the device's noise model."""
    # Density-matrix simulation is exact for the noise model (no sampling
    # noise from trajectories), at the price of 4^n memory.
    return AerSimulator.from_backend(
        backend, method="density_matrix", seed_simulator=seed
    )


def ex1(t, shots):
    """Exercise 1: other bases."""
    t.ask("Exercise 1: a = 2, 4, 7, 8, 11, 13, 14 for N = 15")
    t.predict(
        "every multiplier is a rotation (+ NOT), so each compiled"
        " circuit equals its dense unitary exactly. Ideal success is"
        " 0.5 for every base except a = 14 (a^(r/2) = -1, so 0). CNOT"
        " cost follows the gate count: a=14 < a=4 < a=11 < a=2,8 <"
        " a=7,13, and noisy success falls as cost rises"
    )
    dev = lab10.make_device()
    sim = noisy_sim(dev)
    rows = {}
    for a in (2, 4, 7, 8, 11, 13, 14):
        # Ideal run first: if this disagrees, the compilation is wrong.
        ideal = Statevector.from_instruction(compiled_for(a, False))
        dense = Statevector.from_instruction(dense_for(a))
        pm = ideal.probabilities(qargs=list(range(T)))
        pd = dense.probabilities(qargs=list(range(T)))
        t.check(
            np.allclose(pm, pd, atol=1e-9),
            f"a={a}: compiled == dense unitary (counting distribution)",
        )
        t.check(
            abs(success(pm, a) - (0.0 if a == 14 else 0.5)) < 1e-9,
            f"a={a}: ideal success {success(pm, a):.3f}",
        )
        # Level 3 gives the cheapest routing found by the transpiler.
        tqc = to_hardware(dev, compiled_for(a), level=3)
        cx = tqc.count_ops().get("cx", 0)
        probs = lab10.run_probs(sim, tqc, shots)
        rows[a] = (cx, success(probs, a))
        t.note(
            "a=%2d: plan %s, %3d CNOTs, noisy success %.4f",
            a,
            [multiplier_plan(pow(a, 2**j, N)) for j in range(T)],
            cx,
            rows[a][1],
        )
    # a=14 has zero ideal success, so exclude it from rankings.
    good = [a for a in rows if a != 14]
    cheap = min(good, key=lambda a: rows[a][0])
    dear = max(good, key=lambda a: rows[a][0])
    t.check(rows[14][0] <= rows[cheap][0], "a=14 (NOT only) is the cheapest circuit")
    t.check(
        rows[cheap][1] > rows[dear][1],
        f"cheapest good base (a={cheap}, {rows[cheap][0]} CX) beats the"
        f" costliest (a={dear}, {rows[dear][0]} CX) in noisy success",
    )
    t.check(
        all(rows[a][1] > 0.25 for a in good),
        "every good base still beats the 0.25 garbage floor",
    )


def ex2(t, shots):
    """Exercise 2: keep the identity multiplier with barriers."""
    t.ask("Exercise 2: barriers keep the 'identity' multiplier alive")
    t.predict(
        "I expect the optimizer to cancel un-barriered cswap pairs, so"
        " only the barrier version pays for the 4 extra cswaps (about 8"
        " CNOTs each, plus routing) and loses success"
    )
    dev = lab10.make_device()
    sim = noisy_sim(dev)
    out = {}
    # Two variants differ only in the identity multiplier.
    for name, keep in (("compiled", False), ("kept identity", True)):
        tqc = to_hardware(dev, compiled_for(7, keep_identity=keep), level=3)
        probs = lab10.run_probs(sim, tqc, shots)
        out[name] = (tqc.count_ops().get("cx", 0), success(probs, 7))
        t.note("%-14s: %3d CNOTs, noisy success %.4f", name, *out[name])
    sigma = math.sqrt(0.25 / shots)
    t.check(
        out["kept identity"][0] > out["compiled"][0] + 20,
        "barrier-pinned identity pairs add dozens of CNOTs",
    )
    t.check(
        out["kept identity"][1] < out["compiled"][1] - 3 * sigma,
        "and cost measurable success: skipping gates is worth it",
    )
    # Control experiment: the same circuit with every barrier removed.
    plain = compiled_for(7, keep_identity=True)
    free = QuantumCircuit(*plain.qregs, *plain.cregs)
    for inst in plain.data:
        if inst.operation.name != "barrier":
            free.append(inst.operation, inst.qubits, inst.clbits)
    cx_free = to_hardware(dev, free, level=3).count_ops().get("cx", 0)
    t.note("same circuit WITHOUT barriers: %d CNOTs", cx_free)
    # Finding: the transpiler does NOT recognize a pair of controlled swaps
    # as the identity, with or without barriers, even at optimization_level=3.
    t.note(
        "prediction check: 'optimizer cancels the pairs' was %s",
        (
            "RIGHT"
            if cx_free <= out["compiled"][0] + 2
            else "WRONG: redundant gates survive, so hand-compiling them away"
            " (Lab 10's j=2 qubit) is real work no optimizer does for you"
        ),
    )
    t.check(
        cx_free > out["compiled"][0] + 20,
        "the optimizer does not remove the identity pairs on its own",
    )


def ex3(t, shots):
    """Exercise 3: scale CNOT error up."""
    # The sweep multiplies the device's CNOT error; the success
    # probability should fall smoothly toward the uniform-guess floor.
    t.ask("Exercise 3: scale the CNOT error until success hits the floor")
    t.predict(
        "success decays smoothly toward the garbage floor 0.25 as CNOT"
        " error grows (base errors are ~0.002, so expect the floor near"
        " x100); relaxation may bias outputs toward 0 and undershoot it"
    )
    sigma = math.sqrt(0.25 / shots)
    # Remember the first factor at which success hits the floor.
    rows, crossing = {}, None
    for factor in (1, 4, 16, 64, 128, 256):
        dev = lab10.make_device()
        for edge in lab10.H_TOPOLOGY:
            err = dev.target["cx"][edge].error
            # Cap below 0.75 (the two-qubit depolarizing maximum) to stay valid.
            dev.target["cx"][edge].error = min(err * factor, 0.7)
        tqc = to_hardware(dev, lab10.shor_15_compiled(), level=3)
        probs = lab10.run_probs(noisy_sim(dev), tqc, shots)
        rows[factor] = success(probs, 7)
        t.note(
            "CNOT error x%-2d: success %.4f  P(m)=%s",
            factor,
            rows[factor],
            np.round(probs, 3).tolist(),
        )
        if crossing is None and rows[factor] < 0.25 + 3 * sigma:
            crossing = factor
    # Success listed in order of increasing error.
    seq = [rows[f] for f in sorted(rows)]
    t.check(
        all(x > y - 3 * sigma for x, y in zip(seq, seq[1:])),
        "success never improves as the CNOT error grows (monotone decay)",
    )
    t.check(
        crossing is not None,
        f"success reaches the floor (within 3 sigma) at x{crossing}",
    )
    t.note(
        "lowest success seen %.4f vs floor 0.25 (%s)",
        min(seq),
        (
            "UNDERSHOOTS the floor"
            if min(seq) < 0.25 - 3 * sigma
            else "approaches the floor from above and never undershoots it"
        ),
    )
    t.check(
        min(seq) > 0.25 - 3 * sigma,
        "success does not undershoot the 0.25 floor (uniform noise wins)",
    )


def mitigation_effect(seed, shots):
    """(p_raw, p_mitigated, sigma) for one simulator seed."""
    # Fresh device per call so each seed sees an independent calibration.
    dev = lab10.make_device()
    sim = noisy_sim(dev, seed)
    tqc = to_hardware(dev, lab10.shor_15_compiled(), level=3)
    raw = lab10.run_probs(sim, tqc, shots)
    # Calibrate readout on exactly the qubits that were measured.
    mat = lab10.readout_matrix(dev, sim, lab10.measured_physical_qubits(tqc), shots)
    # Least squares inverts the readout confusion matrix.
    sol, *_ = np.linalg.lstsq(mat, raw, rcond=None)
    sol = np.clip(sol, 0, None)
    sol /= sol.sum()
    p_raw = lab10.success_prob(raw)
    return p_raw, lab10.success_prob(sol), math.sqrt(p_raw * (1 - p_raw) / shots)


def ex4(t, shots, big):
    """Exercise 4: stability of the mitigation effect."""
    t.ask("Exercise 4: readout-mitigation effect across seeds and shots")
    t.predict(
        "readout error is ~0.3% per qubit, so the true gain is small and"
        " positive (a fraction of a percent); with few shots its sign"
        " flips from seed to seed, with a huge shot count it settles"
    )
    for label, n in (("normal shots", shots), ("large shots", big)):
        diffs = []
        for seed in (1, 7, 99):
            p_raw, p_mit, sigma = mitigation_effect(seed, n)
            diffs.append(p_mit - p_raw)
            t.note(
                "%s=%d seed %2d: raw %.4f mitigated %.4f diff %+.4f" " (sigma %.4f)",
                label,
                n,
                seed,
                p_raw,
                p_mit,
                p_mit - p_raw,
                sigma,
            )
        t.note(
            "%s: differences %s, mean %+.4f",
            label,
            [f"{d:+.4f}" for d in diffs],
            float(np.mean(diffs)),
        )
        if label == "normal shots":
            t.check(
                max(abs(d) for d in diffs) < 0.02,
                "normal shots: effect is tiny, bounded by 0.02",
            )
        else:
            t.check(
                all(d > 0 for d in diffs) or all(d < 0 for d in diffs),
                "large shots: all three seeds agree on the sign",
            )
            t.check(
                abs(float(np.mean(diffs))) < 0.02,
                "large shots: the mean effect is still tiny (<0.02)",
            )


def main(argv=None):
    """Run all four Lab 10 exercises; return the exit code."""
    ap = base_parser("Lab 10 exercises")
    ap.add_argument("--shots", type=int, default=100000)
    ap.add_argument(
        "--big-shots",
        type=int,
        default=1000000,
        help="shot count for the 'large shots' arm of exercise 4",
    )
    args = ap.parse_args(argv)
    if args.shots < 20000 or args.big_shots < args.shots:
        ap.error("--shots must be >= 20000 and --big-shots >= --shots.")
    t = Tally(log)
    ex1(t, args.shots)
    ex2(t, args.shots)
    ex3(t, args.shots)
    ex4(t, args.shots, args.big_shots)
    return t.finish()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
