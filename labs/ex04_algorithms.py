#!/usr/bin/env python3
"""
===============================================================================
Project : Quantum Computing Course
Author  : Dr. Ceasar Jackson Jr.
File    : labs/ex04_algorithms.py

Purpose
-------
Lab 4 optional exercises, each as predict-then-verify:
  1. Run Grover with the marked item 0 and 7; why does mark() need X gates
     for some items and none for 7?
  2. Set N_BITS = 4 and predict which Lab 4 check breaks.
  3. Two marked items out of 8: predict the best iteration count and the
     peak probability, then build the oracle by calling mark() twice.

Security
--------
Handles no secrets and makes no network calls.

Usage
-----
    python3 labs/ex04_algorithms.py

Log files
---------
logs/ex04/ex04_<YYYYmmdd_HHMMSS>.log

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
from qiskit.quantum_info import Operator, Statevector

import lab04_algorithms as lab04
from applog import get_logger
from exlib import Tally, base_parser

log = get_logger("ex04")


def mark_diag(n, w):
    """Diagonal of the oracle circuit mark(n, w): should be -1 only at w."""
    qc = QuantumCircuit(n)
    lab04.mark(qc, n, w)
    return np.real(np.diag(Operator(qc).data)), qc


def ex1(t):
    """Exercise 1: marked item 0 and 7, and the X-gate count."""
    t.ask("Exercise 1: marked = 0 and 7, and why mark() needs X gates")
    t.predict(
        "both runs pass; the X count is 2 * (number of ZERO bits in w):"
        " w=7 (111) needs none, w=0 (000) needs 6"
    )
    # The two extremes: all-zeros needs the most X gates, all-ones none.
    for w in (0, 7):
        # main() returns its exit code, so we can reuse the whole lab as a test.
        rc = lab04.main(["--marked", str(w), "--shots", "4000"])
        t.check(rc == 0, f"Lab 4 passes end to end with --marked {w}")
    for w in range(8):
        # The oracle is diagonal, so its diagonal is its whole behavior.
        diag, qc = mark_diag(3, w)
        # Expected diagonal: +1 everywhere except -1 on the marked item.
        want = np.ones(8)
        want[w] = -1
        xs = qc.count_ops().get("x", 0)
        zero_bits = 3 - bin(w).count("1")
        t.note(
            "w=%d (%s): X gates = %d, zero bits = %d",
            w,
            format(w, "03b"),
            xs,
            zero_bits,
        )
        # H.MCX.H is a controlled-Z that only fires on |111>. X on the zero
        # bits relocates |w> onto |111>, and the second X layer undoes it.
        t.check(
            np.allclose(diag, want) and xs == 2 * zero_bits,
            f"w={w}: phase -1 only on |w>, X gates = 2 x zero bits",
        )


class _Recorder:
    """Swap-in for lab04.check that remembers which labels failed."""

    def __init__(self, original):
        """Keep the real check so logging still happens."""
        self.original = original
        self.failed = []

    def __call__(self, cond, label):
        """Forward to the real check and note any failure."""
        if not cond:
            self.failed.append(label)
        return self.original(cond, label)


def ex2(t):
    """Exercise 2: N_BITS = 4 breaks a hard-coded check."""
    t.ask("Exercise 2: N_BITS = 4")
    t.predict(
        "best k becomes floor(pi/4 * 4) = 3, so the hard-coded check"
        " 'k=3 OVERSHOOTS' compares k=3 with itself and fails; nothing"
        " else should break"
    )
    # Patch the module-level check so we can see WHICH check fails.
    rec = _Recorder(lab04.check)
    old_bits, old_check = lab04.N_BITS, lab04.check
    lab04.N_BITS, lab04.check = 4, rec
    try:
        rc = lab04.main(["--marked", "5", "--shots", "4000"])
    finally:
        # Restore module state so later exercises see the original lab.
        lab04.N_BITS, lab04.check = old_bits, old_check
    t.note("exit code %d; failed checks: %s", rc, rec.failed)
    t.check(
        rc == 1 and len(rec.failed) == 1 and "OVERSHOOTS" in rec.failed[0],
        "exactly one check fails: the hard-coded k=3 overshoot test",
    )
    # The physics itself is fine: N=16 simply wants 3 rounds, not 2.
    # Direct formula check: the algorithm is fine, only the test is rigid.
    p3 = lab04.grover_theory(4, 3)
    t.check(p3 > 0.95, f"Grover on N=16 is healthy: P(k=3) = {p3:.4f}")


def two_item_circuit(n, w1, w2, k):
    """Grover with two marked items: mark() is simply called twice."""
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(k):
        # Each mark() flips one item's sign, so calling it twice flips both.
        lab04.mark(qc, n, w1)
        lab04.mark(qc, n, w2)
        lab04.diffuse(qc, n)
    return qc


def ex3(t):
    """Exercise 3: M = 2 marked items out of N = 8."""
    t.ask("Exercise 3: two marked items, N = 8")
    t.predict(
        "theta = asin(sqrt(M/N)) = pi/6; k_opt = pi/(4 theta) - 1/2 = 1;"
        " P(success) = sin^2(3 pi/6) = 1 exactly, 50% on each item"
    )
    # With M marked items the rotation angle grows to asin(sqrt(M/N)).
    theta = math.asin(math.sqrt(2 / 8))
    t.check(abs(theta - math.pi / 6) < 1e-12, "theta = pi/6")
    # Any two distinct items work; these differ in all three bits.
    w1, w2 = 2, 5
    for k in (0, 1, 2, 3):
        probs = Statevector.from_instruction(
            two_item_circuit(3, w1, w2, k)
        ).probabilities()
        p = float(probs[w1] + probs[w2])
        want = math.sin((2 * k + 1) * theta) ** 2
        t.note("k=%d: P(w1 or w2) = %.4f (theory %.4f)", k, p, want)
        t.check(abs(p - want) < 1e-9, f"k={k} matches sin^2((2k+1) theta)")
    probs = Statevector.from_instruction(two_item_circuit(3, w1, w2, 1)).probabilities()
    t.check(
        abs(probs[w1] - 0.5) < 1e-9 and abs(probs[w2] - 0.5) < 1e-9,
        "k=1: probability 1 split evenly, 0.5 on each marked item",
    )
    # k=2 shows overshoot again: past the peak the state rotates away.
    t.check(
        abs(math.sin(5 * theta) ** 2 - 0.25) < 1e-9,
        "k=2 overshoots to 0.25: stopping at the right k matters",
    )


def main(argv=None):
    """Run all three Lab 4 exercises; return the exit code."""
    base_parser("Lab 4 exercises").parse_args(argv)
    t = Tally(log)
    ex1(t)
    ex2(t)
    ex3(t)
    return t.finish()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warn_("Interrupted by user")
        sys.exit(130)
