#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/ex06_shor.py

Purpose
-------
Lab 6 optional exercises, each as predict-then-verify:
  1. Other bases for N = 15: a = 2 (order 4) and a = 4 (order 2).
  2. Fewer counting qubits: N = 15, a = 7 with t = width and below, then
     N = 21, a = 2 where a too-small t blurs the peaks.
  3. Bases that share a factor with N (a = 5, a = 3): gcd already wins.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/ex06_shor.py            # full (about 30-60 s)
    python3 labs/ex06_shor.py --quick    # skips the N = 21 sweep above t = 7

Log files
---------
logs/ex06/ex06_<YYYYmmdd_HHMMSS>.log

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

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

from applog import get_logger
from exlib import Tally, base_parser
from lab05_qft_qpe import qft_circuit
from lab06_shor import (
    classical_order,
    distribution,
    factors_from_measurement,
    modmult_matrix,
    success_probability,
)

log = get_logger("ex06")


def dist_with_t(a, n, t):
    """Counting-register distribution of Lab 6's circuit with t counting qubits."""
    # Width is the number of bits needed to hold a residue mod n; the target
    # register must be that wide so every residue y < n fits.
    width = n.bit_length()
    qc = QuantumCircuit(t + width)
    target = list(range(t, t + width))
    qc.x(target[0])  # target starts in |1>
    qc.h(range(t))
    for j in range(t):
        # Same controlled modular multiplication as Lab 6; only t differs.
        gate = UnitaryGate(modmult_matrix(pow(a, 2**j, n), n, width))
        # Same controlled multiplication as Lab 6, only t differs.
        qc.append(gate.control(1), [j] + target)
    qc.compose(qft_circuit(t, inverse=True), qubits=list(range(t)), inplace=True)
    return Statevector.from_instruction(qc).probabilities(qargs=list(range(t)))


def ex1(t):
    """Exercise 1: a = 2 and a = 4."""
    t.ask("Exercise 1: a = 2 (order 4) and a = 4 (order 2) for N = 15")
    t.predict(
        "each succeeds with probability 0.5: half the peaks (m = 0 and,"
        " for even s, the s/r = 1/2 peak) carry no information"
    )
    # Both bases are tested against the classical order first.
    for a, order, facts in ((2, 4, (3, 5)), (4, 2, (3, 5))):
        t.check(classical_order(a, 15) == order, f"order of {a} mod 15 is {order}")
        # Exact distribution from the statevector, no sampling noise.
        dist, tt = distribution(a, 15)
        p = success_probability(dist, tt, a, 15)
        t.note("a=%d: P(one run factors 15) = %.4f", a, p)
        t.check(abs(p - 0.5) < 1e-9, f"a={a}: success probability 0.5")
        # Collect every non-empty post-processing result over all peaks.
        found = {
            factors_from_measurement(m, tt, a, 15)
            for m in range(2**tt)
            if dist[m] > 1e-9
        } - {None}
        t.check(found == {facts}, f"a={a}: every success returns (3, 5)")
    # Bonus: the whole table of coprime bases, to see the pattern.
    # The whole table of coprime bases shows the pattern at a glance.
    for a in (7, 8, 11, 13, 14):
        dist, tt = distribution(a, 15)
        p = success_probability(dist, tt, a, 15)
        t.note("a=%d (order %d): P = %.3f", a, classical_order(a, 15), p)
        t.check(
            abs(p - (0.0 if a == 14 else 0.5)) < 1e-9,
            f"a={a}: P = {'0 (a^(r/2) = -1)' if a == 14 else '0.5'}",
        )


def ex2(t, quick):
    """Exercise 2: fewer counting qubits."""
    t.ask("Exercise 2a: N = 15, a = 7 with t = 4, 3, 2, 1")
    t.predict(
        "r = 4 divides 2^t for t >= 2, so peaks stay exact and P = 0.5"
        " down to t = 2; at t = 1 the only fractions are 0 and 1/2, so"
        " r = 4 cannot appear and P = 0"
    )
    # Shrink the counting register one qubit at a time.
    for tt in (4, 3, 2, 1):
        p = success_probability(dist_with_t(7, 15, tt), tt, 7, 15)
        t.note("t=%d: P = %.4f", tt, p)
        t.check(
            abs(p - (0.0 if tt == 1 else 0.5)) < 1e-9,
            f"t={tt}: P = {0.0 if tt == 1 else 0.5}",
        )
    t.ask("Exercise 2b: N = 21, a = 2 (r = 6), sweeping t")
    t.predict(
        "r = 6 never divides 2^t, so peaks blur. While 2^t <= 21 every"
        " fraction m/2^t has a power-of-two denominator, which can"
        " never equal 6: P = 0 for t <= 4. From t = 5 it becomes"
        " possible and P rises with t overall (not necessarily monotonically: the"
        " fractions that happen to land near s/6 change with t)"
    )
    # t=9 means 14 qubits; --quick caps memory and time.
    top = 7 if quick else 9
    # Collect P(t) so the checks can compare across the sweep.
    res = {}
    for tt in range(3, top + 1):
        res[tt] = success_probability(dist_with_t(2, 21, tt), tt, 2, 21)
        t.note("t=%d (2^t=%4d): P = %.4f", tt, 2**tt, res[tt])
    t.check(
        all(res[tt] == 0.0 for tt in res if 2**tt <= 21),
        "P = 0 whenever 2^t <= N (denominators are powers of two)",
    )
    t.check(
        res[top] > res[5] and res[top] > 0.2,
        f"P(t={top}) = {res[top]:.3f} beats t = 5 and exceeds 0.2",
    )


def ex3(t):
    """Exercise 3: a shares a factor with N."""
    t.ask("Exercise 3: a = 5 and a = 3 (gcd > 1)")
    t.predict(
        "no quantum step needed: gcd(a, 15) is already a factor, and"
        " classical_order refuses (no order exists)"
    )
    # gcd(a, 15) is the factor itself when a is 3 or 5.
    for a, f in ((5, 5), (3, 3)):
        g = math.gcd(a, 15)
        t.check(g == f and 15 % g == 0, f"gcd({a}, 15) = {g}: factor found")
        try:
            classical_order(a, 15)
            refused = False
        except ValueError:
            refused = True
        t.check(refused, f"classical_order({a}, 15) raises: no order exists")
    # Count the lucky bases: they share a factor with 15.
    shared = [a for a in range(2, 15) if math.gcd(a, 15) > 1]
    t.note("bases 2..14 sharing a factor with 15: %s", shared)
    t.check(
        len(shared) == 6 and len(range(2, 15)) == 13,
        "6 of the 13 bases are lucky classically (and trivially so)",
    )


def main(argv=None):
    """Run all three Lab 6 exercises; return the exit code."""
    ap = base_parser("Lab 6 exercises")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="stop the N=21 sweep at t=7 (saves time and memory)",
    )
    args = ap.parse_args(argv)
    t = Tally(log)
    ex1(t)
    ex2(t, args.quick)
    ex3(t)
    return t.finish()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
